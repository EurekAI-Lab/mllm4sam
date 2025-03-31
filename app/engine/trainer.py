# app/engine/trainer.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.
#
# We do NOT rename "Trainer" but we incorporate TRL's SFTTrainer logic here
# to remain consistent with user demands. We also keep your existing approach for
# evaluation. The user is free to add or remove references to bitsandbytes or QLoRA
# if desired.

import os
import math
import csv
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

from app.util.logger import log_validation_samples
from app.util.utils import set_seed, ensure_dir
from app.engine.evaluator import Evaluator

from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class Trainer:
    """
    This Trainer merges your approach with the TRL SFTTrainer for supervised fine-tuning
    of Qwen2-VL. We keep your prior code's style of logging and shape debugging as is.

    Key Changes for shape mismatch fix:
    - In `my_data_collator`, we add special vision tokens if `images` is present in the batch:
      1) `vision_start_token_id` once.
      2) `image_token_id` repeated as many times as the final image embeddings (e.g. 256).
    - We automatically compute the needed number of image tokens based on
      (temporal_patch_size * (height // patch_size) * (width // patch_size)) / (spatial_merge_size^2).
    - We set the appended tokens in `labels` to `-100` so as not to compute loss on them.

    New enhancement:
    - Custom model saving logic to properly handle shared tensors between model components
    """

    def __init__(
            self,
            model,
            train_dataloader,
            val_dataloader=None,
            lr=1e-4,
            max_epochs=3,
            grad_acc_steps=1,
            scheduler_type="linear",
            warmup_steps=1000,
            early_stop_patience=5,
            clip_grad_norm=1.0,
            output_dir="runs",
            run_name="sam4mllm",
            device="cuda",
            use_amp=True,
            log_interval=10
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Basic hyperparams
        self.lr = lr
        self.max_epochs = max_epochs
        self.grad_acc_steps = grad_acc_steps
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.early_stop_patience = early_stop_patience
        self.clip_grad_norm = clip_grad_norm
        self.output_dir = output_dir
        self.device = device
        self.use_amp = use_amp
        self.log_interval = log_interval

        dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"{run_name}_{dt_string}")
        ensure_dir(self.run_dir)

        # Setup wandb
        wandb.init(project="SAM4MLLM", name=run_name, dir=self.run_dir)

        # Logging CSV
        self.train_log_path = os.path.join(self.run_dir, "train_log.csv")
        self.val_log_path = os.path.join(self.run_dir, "val_log.csv")
        with open(self.train_log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "loss", "lr"])
        with open(self.val_log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "val_loss"])

        # Create a TRL SFT config
        self.sft_config = SFTConfig(
            num_train_epochs=max_epochs,
            per_device_train_batch_size=1,  # We'll pass actual batch via our custom logic if needed
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_acc_steps,
            max_seq_length=1536,
            learning_rate=lr,
            lr_scheduler_type=self.scheduler_type,
            warmup_ratio=0.0,
            output_dir=self.run_dir,
            logging_steps=10,
            eval_steps=50,
            save_steps=200,
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,  # Keep 'text_input' / 'image' keys
            save_safetensors=False,  # To avoid shared tensor issues with safetensors
        )

        # Prepare evaluator (placeholder)
        self.evaluator = Evaluator(self.model, self.device) if self.val_dataloader else None

        # Global step tracking
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.should_stop = False

        # Move model to device
        self.model.to(self.device)

    def train(self):
        """
        We create a SFTTrainer from TRL and feed it the existing train and val dataset.
        We'll rely on SFTTrainer's standard train() loop for advanced SFT logic.
        We keep your prior code's style of logging and shape debugging as is.
        """

        print("[Trainer] Creating TRL SFTTrainer...")

        def _compute_n_img_tokens_for_sample(img_tensor: torch.Tensor, backbone) -> int:
            """
            Computes how many final patch tokens Qwen2-VL will produce for a single image
            given the user-specified patch_size, temporal_patch_size, and spatial_merge_size.

            For example, for a 512×512 image with patch_size=16, spatial_merge_size=2,
            temporal_patch_size=1 => final 256 tokens.
            """
            if img_tensor is None or img_tensor.shape[0] == 0:
                return 0

            # shape = [channels, H, W]
            _, h, w = img_tensor.shape
            patch_size = getattr(backbone, 'override_patch_size', 14)
            temporal_patch_size = getattr(backbone, 'override_temporal_patch_size', 1)

            # Attempt to read from vision_config as well, if it exists
            if hasattr(backbone.qwen_model.config, "vision_config"):
                vc = backbone.qwen_model.config.vision_config
                if hasattr(vc, "spatial_merge_size"):
                    spatial_merge = vc.spatial_merge_size
                else:
                    spatial_merge = 2
            else:
                spatial_merge = 2

            # T x (H//patch_size) x (W//patch_size), then / (spatial_merge^2)
            t_count = temporal_patch_size  # for a single image
            h_count = h // patch_size
            w_count = w // patch_size
            total_patches = t_count * h_count * w_count
            merged_count = total_patches // (spatial_merge * spatial_merge)
            return merged_count

        # This custom collator handles text + images from your dataset
        def my_data_collator(batch):
            """
            Each element in 'batch' is a dict with keys:
              "text_input", "image", "points_str"
            We'll tokenize 'text_input' using Qwen tokenizer,
            then if images is present, we'll add the correct # of vision tokens:
              - 1 vision_start_token_id
              - repeated image_token_id (match final patch count).
            We'll also adjust `labels` so that appended tokens have label=-100.
            """
            text_list = [item["text_input"] for item in batch]
            images = [item["image"] for item in batch]

            # Qwen's tokenizer
            tokenizer = self.model.backbone.qwen_tokenizer

            # 1) Basic text tokenization
            encoded = tokenizer(
                text_list,
                padding=False,  # We'll pad manually below
                truncation=True,
                max_length=self.sft_config.max_seq_length,
                return_tensors="pt",
            )

            # Convert images to float
            pixel_values = torch.stack(images, dim=0).float()

            # Identify special token IDs from config
            vs_id = self.model.backbone.qwen_model.config.vision_start_token_id
            vi_id = self.model.backbone.qwen_model.config.image_token_id
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []

            # 2) Build final sequences, possibly adding vision tokens
            for i in range(len(batch)):
                input_ids_i = encoded["input_ids"][i]
                attention_mask_i = encoded["attention_mask"][i]

                # We'll clone for labels, initially identical
                labels_i = input_ids_i.clone()

                # Check if there's an actual image. If so, compute how many patch tokens are expected
                if images[i] is not None:
                    n_img_tokens = _compute_n_img_tokens_for_sample(images[i], self.model.backbone)
                else:
                    n_img_tokens = 0

                if n_img_tokens > 0:
                    # Insert 1 vision_start_token, then n_img_tokens of image_token_id
                    # Typically appended at the end
                    vs_tensor = torch.tensor([vs_id], dtype=torch.long)
                    vi_tensor = torch.tensor([vi_id] * n_img_tokens, dtype=torch.long)
                    # Concat them
                    input_ids_i = torch.cat([input_ids_i, vs_tensor, vi_tensor], dim=0)

                    # For attention mask
                    mask_append = torch.ones(1 + n_img_tokens, dtype=attention_mask_i.dtype)
                    attention_mask_i = torch.cat([attention_mask_i, mask_append], dim=0)

                    # We do not train the model to generate these tokens, so we set them to -100
                    labels_append = torch.full((1 + n_img_tokens,), -100, dtype=labels_i.dtype)
                    labels_i = torch.cat([labels_i, labels_append], dim=0)

                batch_input_ids.append(input_ids_i)
                batch_attention_mask.append(attention_mask_i)
                batch_labels.append(labels_i)

            # 3) Now pad them to max length in this batch
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                batch_input_ids,
                batch_first=True,
                padding_value=pad_id
            )
            padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
                batch_attention_mask,
                batch_first=True,
                padding_value=0
            )
            padded_labels = torch.nn.utils.rnn.pad_sequence(
                batch_labels,
                batch_first=True,
                padding_value=-100
            )

            # Return a standard batch
            batch_dict = {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "images": pixel_values,
                "labels": padded_labels,
            }
            return batch_dict

        # Custom save function to handle shared tensors
        def custom_save_model(trainer, output_dir, _internal_call=True):
            """
            Custom save_model implementation that handles shared tensors properly.
            This replaces the default SFTTrainer's save_model to avoid safetensors errors.
            """
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Saving model checkpoint to {output_dir}")

            # Save the model state dict with proper handling of shared tensors
            # For LoRA or PEFT models, use PEFT-specific saving
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'qwen_model'):
                # Save Qwen model and SAM model separately
                if hasattr(self.model.backbone, 'qwen_model'):
                    qwen_output_dir = os.path.join(output_dir, "qwen")
                    os.makedirs(qwen_output_dir, exist_ok=True)
                    print(f"[INFO] Saving Qwen model to {qwen_output_dir}")

                    # Check if the model is a PEFT model
                    if hasattr(self.model.backbone.qwen_model, 'save_pretrained'):
                        # For PEFT models, only save the PEFT model adapter
                        if hasattr(self.model.backbone.qwen_model, 'peft_config'):
                            print(f"[INFO] Detected PEFT model, saving adapters")
                            self.model.backbone.qwen_model.save_pretrained(qwen_output_dir)
                        else:
                            # For full models, save with PyTorch's save to avoid safetensors errors
                            print(f"[INFO] Saving full model with PyTorch")
                            torch.save(
                                self.model.backbone.qwen_model.state_dict(),
                                os.path.join(qwen_output_dir, "pytorch_model.bin")
                            )
                            # Save config as well
                            if hasattr(self.model.backbone.qwen_model, 'config'):
                                self.model.backbone.qwen_model.config.save_pretrained(qwen_output_dir)

                    # Save tokenizer
                    if hasattr(self.model.backbone, 'qwen_tokenizer'):
                        self.model.backbone.qwen_tokenizer.save_pretrained(qwen_output_dir)

                # Save SAM model if present
                if hasattr(self.model.backbone, 'sam_model') and self.model.backbone.sam_model is not None:
                    sam_output_dir = os.path.join(output_dir, "sam")
                    os.makedirs(sam_output_dir, exist_ok=True)
                    print(f"[INFO] Saving SAM model to {sam_output_dir}")

                    if hasattr(self.model.backbone.sam_model, 'save_pretrained'):
                        self.model.backbone.sam_model.save_pretrained(sam_output_dir)
                    else:
                        # Use PyTorch's save
                        torch.save(
                            self.model.backbone.sam_model.state_dict(),
                            os.path.join(sam_output_dir, "pytorch_model.bin")
                        )
                        # Save config if available
                        if hasattr(self.model.backbone.sam_model, 'config'):
                            self.model.backbone.sam_model.config.save_pretrained(sam_output_dir)

                # Save overview file to help with reloading
                overview_path = os.path.join(output_dir, "model_overview.txt")
                with open(overview_path, "w") as f:
                    f.write("SAM4MLLM Model Components:\n")
                    if hasattr(self.model.backbone, 'qwen_model'):
                        f.write(f"- Qwen Model: {type(self.model.backbone.qwen_model).__name__}\n")
                        if hasattr(self.model.backbone.qwen_model, 'peft_config'):
                            f.write("  - Uses PEFT/LoRA adapters\n")
                    if hasattr(self.model.backbone, 'sam_model'):
                        f.write(f"- SAM Model: {type(self.model.backbone.sam_model).__name__}\n")
            else:
                # Fallback for simpler models - use PyTorch save
                print(f"[INFO] Using PyTorch save for model without recognized backbone structure")
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_dir, "pytorch_model.bin")
                )

            print(f"[INFO] Model checkpoint saved successfully to {output_dir}")
            return output_dir

        # Create and initialize SFT trainer
        sft_trainer = SFTTrainer(
            model=self.model,
            args=self.sft_config,
            train_dataset=self.train_dataloader.dataset,
            eval_dataset=self.val_dataloader.dataset if self.val_dataloader else None,
            data_collator=my_data_collator,
        )

        # Replace the save_model method to handle shared tensors correctly
        sft_trainer.save_model = lambda output_dir, _internal_call=True: custom_save_model(
            sft_trainer, output_dir, _internal_call
        )

        print("[Trainer] Start SFT training with TRL SFTTrainer...")
        sft_trainer.train()  # The standard training loop from TRL

        # After training, let's do final validation with your old approach if desired
        if self.val_dataloader is not None:
            print("[Trainer] Doing final validation pass with your code.")
            val_loss = self._eval_validation()
            print(f"[Trainer] Final val loss: {val_loss}")

        # Save final model with our custom save function
        final_model_path = os.path.join(self.run_dir, "final_model")
        custom_save_model(sft_trainer, final_model_path)
        print(f"[Trainer] SFT training complete. Model saved at {final_model_path}")

    def _eval_validation(self):
        """
        We replicate a minimal approach for cross-entropy on val_dataloader.
        Optionally parse points and do SAM-based IoU if needed.
        """

        self.model.eval()
        losses = []
        for i, batch in enumerate(tqdm(self.val_dataloader, desc="Validation")):
            if isinstance(batch["text_input"], list):
                text_list = batch["text_input"]
            else:
                text_list = [batch["text_input"]]

            images = batch["image"]
            if isinstance(images, torch.Tensor):
                pixel_values = images.to(self.device)
            else:
                pixel_values = None

            encoded = self.model.backbone.qwen_tokenizer(
                text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.sft_config.max_seq_length
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=pixel_values,
                    labels=input_ids,  # standard cross-entropy
                )
                loss_val = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                losses.append(loss_val.item())

        val_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
        wandb.log({"val_loss": val_loss})
        return val_loss