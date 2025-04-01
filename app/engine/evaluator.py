import os

import PIL
import torch
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb
import random
from datetime import datetime

from app.util.logger import compute_point_match, parse_points_from_text
from app.util.metrics import compute_iou, compute_dice


class Evaluator:
    def __init__(self, model: nn.Module, device="cuda", output_dir="./validation_viz"):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self, dataloader):
        """
        Evaluate model on validation data with multiple metrics:
        - Loss: Cross-entropy loss on text generation
        - Point accuracy: Match between predicted and ground truth points
        - IoU: If SAM is available, compute IoU between predicted and ground truth masks
        - Dice: If SAM is available, compute Dice coefficient
        """
        self.model.eval()
        metrics = {
            "loss": [],
            "point_match": [],
            "iou": [],
            "dice": []
        }

        # Store validation samples for visualization
        val_samples = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                # 1. Get inputs from batch
                if isinstance(batch["text_input"], list):
                    text_list = batch["text_input"]
                else:
                    text_list = [batch["text_input"]]

                images = batch["image"]
                if isinstance(images, torch.Tensor):
                    pixel_values = images.to(self.device)
                else:
                    pixel_values = None

                # 2. Prepare inputs with proper image tokens
                tokenizer = self.model.backbone.qwen_tokenizer

                # Instead of directly tokenizing and forwarding, we need to handle image tokens properly
                # This is similar to what's done in the data_collator in Trainer

                # First tokenize the text
                encoded = tokenizer(
                    text_list,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1536
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                # Get vision token IDs from model config
                vision_start_token_id = self.model.backbone.qwen_model.config.vision_start_token_id
                image_token_id = self.model.backbone.qwen_model.config.image_token_id

                # Compute number of image tokens needed
                if pixel_values is not None:
                    # Extract grid dimensions
                    batch_size, channels, height, width = pixel_values.shape
                    patch_size = getattr(self.model.backbone, 'override_patch_size', 16)
                    h_grid = height // patch_size
                    w_grid = width // patch_size

                    # Determine spatial_merge_size (default 2)
                    spatial_merge_size = 2
                    if hasattr(self.model.backbone.qwen_model.config, "vision_config"):
                        vc = self.model.backbone.qwen_model.config.vision_config
                        if hasattr(vc, "spatial_merge_size"):
                            spatial_merge_size = vc.spatial_merge_size

                    # Calculate number of image tokens after merging
                    n_image_tokens = (h_grid * w_grid) // (spatial_merge_size * spatial_merge_size)

                    # Calculate image grid THW for the model
                    image_grid_thw = torch.tensor(
                        [[1, h_grid, w_grid] for _ in range(batch_size)],
                        device=self.device,
                        dtype=torch.long
                    )

                    # Now append vision tokens to input_ids and update attention_mask
                    batch_input_ids = []
                    batch_attention_mask = []
                    batch_labels = []

                    for i in range(batch_size):
                        # Append vision_start_token_id followed by n_image_tokens of image_token_id
                        new_input_ids = torch.cat([
                            input_ids[i],
                            torch.tensor([vision_start_token_id], device=self.device),
                            torch.tensor([image_token_id] * n_image_tokens, device=self.device)
                        ])

                        # Update attention mask
                        new_attention_mask = torch.cat([
                            attention_mask[i],
                            torch.ones(1 + n_image_tokens, device=self.device, dtype=attention_mask.dtype)
                        ])

                        # For labels, mark vision tokens as -100 to ignore in loss
                        new_labels = torch.cat([
                            input_ids[i],  # Use input_ids as labels for text tokens
                            torch.full((1 + n_image_tokens,), -100, device=self.device, dtype=torch.long)
                        ])

                        batch_input_ids.append(new_input_ids)
                        batch_attention_mask.append(new_attention_mask)
                        batch_labels.append(new_labels)

                    # Pad sequences to max length in batch
                    max_len = max(len(ids) for ids in batch_input_ids)
                    padded_input_ids = []
                    padded_attention_mask = []
                    padded_labels = []

                    for i in range(batch_size):
                        padding_len = max_len - len(batch_input_ids[i])
                        padded_input_ids.append(torch.cat([
                            batch_input_ids[i],
                            torch.full((padding_len,), tokenizer.pad_token_id, device=self.device, dtype=torch.long)
                        ]))
                        padded_attention_mask.append(torch.cat([
                            batch_attention_mask[i],
                            torch.zeros(padding_len, device=self.device, dtype=attention_mask.dtype)
                        ]))
                        padded_labels.append(torch.cat([
                            batch_labels[i],
                            torch.full((padding_len,), -100, device=self.device, dtype=torch.long)
                        ]))

                    # Stack tensors
                    input_ids = torch.stack(padded_input_ids)
                    attention_mask = torch.stack(padded_attention_mask)
                    labels = torch.stack(padded_labels)

                    # Now call the model with the properly prepared inputs
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=labels
                    )
                else:
                    # Text-only case
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )

                loss_val = outputs.loss.item() if hasattr(outputs, "loss") else outputs[0].item()
                metrics["loss"].append(loss_val)

                # 3. Generate point predictions
                user_text = "Point out the wound area in up to 10 points."
                prompts = [f"You are a helpful segmentation assistant.\n[USER]: {user_text}\n[ASSISTANT]:"] * len(
                    text_list)

                # Tokenize prompts
                prompt_ids = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1536
                ).to(self.device)

                # For generation, we'll use a similar approach to ensure correct image token handling
                batch_size = prompt_ids.input_ids.shape[0]

                if pixel_values is not None:
                    # For generation we need to append vision tokens to the prompt
                    batch_prompt_ids = []
                    batch_prompt_attn = []

                    for i in range(batch_size):
                        # Append vision_start_token_id followed by n_image_tokens of image_token_id
                        new_prompt_ids = torch.cat([
                            prompt_ids.input_ids[i],
                            torch.tensor([vision_start_token_id], device=self.device),
                            torch.tensor([image_token_id] * n_image_tokens, device=self.device)
                        ])

                        # Update attention mask
                        new_prompt_attn = torch.cat([
                            prompt_ids.attention_mask[i],
                            torch.ones(1 + n_image_tokens, device=self.device, dtype=prompt_ids.attention_mask.dtype)
                        ])

                        batch_prompt_ids.append(new_prompt_ids)
                        batch_prompt_attn.append(new_prompt_attn)

                    # Pad sequences to max length in batch
                    max_len = max(len(ids) for ids in batch_prompt_ids)
                    padded_prompt_ids = []
                    padded_prompt_attn = []

                    for i in range(batch_size):
                        padding_len = max_len - len(batch_prompt_ids[i])
                        padded_prompt_ids.append(torch.cat([
                            batch_prompt_ids[i],
                            torch.full((padding_len,), tokenizer.pad_token_id, device=self.device, dtype=torch.long)
                        ]))
                        padded_prompt_attn.append(torch.cat([
                            batch_prompt_attn[i],
                            torch.zeros(padding_len, device=self.device, dtype=prompt_ids.attention_mask.dtype)
                        ]))

                    # Stack tensors
                    gen_input_ids = torch.stack(padded_prompt_ids)
                    gen_attention_mask = torch.stack(padded_prompt_attn)

                    # Generate responses
                    gen_ids = self.model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        images=pixel_values,
                        image_grid_thw=image_grid_thw,
                        max_new_tokens=128,
                        do_sample=False
                    )
                else:
                    # Text-only generation
                    gen_ids = self.model.generate(
                        input_ids=prompt_ids.input_ids,
                        attention_mask=prompt_ids.attention_mask,
                        max_new_tokens=128,
                        do_sample=False
                    )

                # Decode generated text
                gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                # For each example, extract points and compute metrics
                for i, gen_text in enumerate(gen_texts):
                    # Extract model's response
                    response = gen_text.split("[ASSISTANT]:")[-1].strip()

                    # Parse predicted points
                    pred_points_set = parse_points_from_text(response)

                    # Get ground truth points
                    gt_points_str = batch["points_str"][i] if isinstance(batch["points_str"], list) else batch[
                        "points_str"]

                    # Compute point match metric
                    point_match = compute_point_match(pred_points_set, gt_points_str)
                    metrics["point_match"].append(point_match)

                    # If SAM is available and we have mask paths, compute segmentation metrics
                    if hasattr(self.model.backbone, "sam_model") and self.model.backbone.sam_model is not None:
                        # Try to get the mask path
                        mask_path = batch["mask_path"][i] if isinstance(batch["mask_path"], list) else batch[
                            "mask_path"]

                        if mask_path and os.path.exists(mask_path):
                            # Load ground truth mask
                            gt_mask = np.array(PIL.Image.open(mask_path).convert("L"))
                            gt_mask = (gt_mask >= 128).astype(np.uint8)

                            # Skip if no points were found
                            if len(pred_points_set) == 0:
                                metrics["iou"].append(0.0)
                                metrics["dice"].append(0.0)
                                continue

                            # Convert set of points to list format for SAM
                            pred_points = [[x, y] for x, y in pred_points_set]

                            # Generate segmentation with SAM
                            try:
                                result = self.model.backbone.predict_segmentation(
                                    image=images[i] if isinstance(images, torch.Tensor) else images,
                                    prompt=None,  # We're providing points directly
                                    return_points=True
                                )

                                if result["mask"] is not None:
                                    # Resize predicted mask to match ground truth if needed
                                    pred_mask = result["mask"]
                                    if pred_mask.shape != gt_mask.shape:
                                        import cv2
                                        pred_mask = cv2.resize(
                                            pred_mask.astype(np.uint8),
                                            (gt_mask.shape[1], gt_mask.shape[0]),
                                            interpolation=cv2.INTER_NEAREST
                                        )

                                    # Compute IoU
                                    iou = compute_iou(pred_mask > 0, gt_mask > 0)
                                    metrics["iou"].append(iou)

                                    # Compute Dice coefficient
                                    dice = compute_dice(pred_mask > 0, gt_mask > 0)
                                    metrics["dice"].append(dice)

                                    # Store sample for visualization
                                    val_samples.append({
                                        "image": images[i].cpu() if isinstance(images, torch.Tensor) else images,
                                        "gt_mask": gt_mask,
                                        "pred_mask": pred_mask,
                                        "pred_points": pred_points,
                                        "gt_points_str": gt_points_str,
                                        "iou": iou,
                                        "dice": dice,
                                        "batch_idx": batch_idx,
                                        "sample_idx": i
                                    })
                                else:
                                    # No valid mask generated
                                    metrics["iou"].append(0.0)
                                    metrics["dice"].append(0.0)
                            except Exception as e:
                                print(f"[WARNING] Error generating segmentation: {e}")
                                metrics["iou"].append(0.0)
                                metrics["dice"].append(0.0)

        # Compute mean metrics
        results = {}
        for metric_name, values in metrics.items():
            if values:
                results[metric_name] = sum(values) / len(values)
            else:
                results[metric_name] = 0.0

        # Visualize random samples
        if val_samples:
            self.visualize_random_samples(val_samples)

        return results

    def visualize_random_samples(self, val_samples, num_samples=2):
        """
        Visualize random samples from validation set, log them to wandb,
        and save them to the output directory.

        Args:
            val_samples (list): List of validation samples with images, masks, and points
            num_samples (int): Number of random samples to visualize
        """
        # If we have fewer samples than requested, use all available samples
        num_samples = min(num_samples, len(val_samples))
        if num_samples == 0:
            print("[WARNING] No validation samples available for visualization.")
            return

        # Select random samples
        random_samples = random.sample(val_samples, num_samples)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, sample in enumerate(random_samples):
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 1. Original image with predicted points
            if isinstance(sample["image"], torch.Tensor):
                img_np = sample["image"].permute(1, 2, 0).numpy()
                # Normalize if needed
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = np.array(sample["image"])

            axes[0].imshow(img_np)
            axes[0].set_title("Image with Predicted Points")

            # Plot predicted points
            if sample["pred_points"]:
                points = np.array(sample["pred_points"])
                axes[0].scatter(points[:, 0], points[:, 1], c='red', marker='x', s=100)

            # 2. Ground truth mask
            axes[1].imshow(img_np)
            gt_mask = sample["gt_mask"]
            gt_mask_overlay = np.zeros_like(img_np)
            gt_mask_overlay[:, :, 1] = 255  # Green channel
            mask_alpha = 0.5

            # Create overlay
            img_with_gt_mask = img_np.copy()
            img_with_gt_mask[gt_mask > 0] = img_with_gt_mask[gt_mask > 0] * (1 - mask_alpha) + gt_mask_overlay[
                gt_mask > 0] * mask_alpha

            axes[1].imshow(img_with_gt_mask)
            axes[1].set_title("Ground Truth Mask")

            # 3. Predicted mask
            axes[2].imshow(img_np)
            pred_mask = sample["pred_mask"]
            pred_mask_overlay = np.zeros_like(img_np)
            pred_mask_overlay[:, :, 0] = 255  # Red channel

            # Create overlay
            img_with_pred_mask = img_np.copy()
            img_with_pred_mask[pred_mask > 0] = img_with_pred_mask[pred_mask > 0] * (1 - mask_alpha) + \
                                                pred_mask_overlay[pred_mask > 0] * mask_alpha

            axes[2].imshow(img_with_pred_mask)
            axes[2].set_title(f"Predicted Mask (IoU: {sample['iou']:.3f}, Dice: {sample['dice']:.3f})")

            # Adjust layout
            plt.tight_layout()

            # Save figure to disk
            filename = f"validation_sample_{timestamp}_{i}.png"
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[INFO] Saved visualization to {save_path}")

            # Log to wandb
            try:
                wandb.log({
                    f"validation_sample_{i}": wandb.Image(fig),
                    f"validation_sample_{i}_iou": sample["iou"],
                    f"validation_sample_{i}_dice": sample["dice"]
                })
            except Exception as e:
                print(f"[WARNING] Failed to log to wandb: {e}")

            plt.close(fig)