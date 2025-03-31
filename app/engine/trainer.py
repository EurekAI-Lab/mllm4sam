# /home/dbcloud/PycharmProjects/mllm4sam/app/engine/trainer.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.

import os
import time
import math
import csv
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

from util.logger import log_validation_samples
from util.utils import set_seed, ensure_dir
from engine.evaluator import Evaluator

class Trainer:
    def __init__(self,
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
                 log_interval=10):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.lr = lr
        self.max_epochs = max_epochs
        self.grad_acc_steps = grad_acc_steps
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.early_stop_patience = early_stop_patience
        self.clip_grad_norm = clip_grad_norm

        self.output_dir = output_dir
        dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"{run_name}_{dt_string}")
        ensure_dir(self.run_dir)

        self.device = device
        self.use_amp = use_amp
        self.log_interval = log_interval

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.should_stop = False

        # Move model to device
        self.model.to(self.device)

        # Setup wandb
        wandb.init(project="SAM4MLLM", name=run_name, dir=self.run_dir)

        # Setup logging files
        self.train_log_path = os.path.join(self.run_dir, "train_log.csv")
        self.val_log_path = os.path.join(self.run_dir, "val_log.csv")

        with open(self.train_log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "loss", "lr"])
        with open(self.val_log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "val_loss"])

        # Setup optimizer
        # Only LoRA parameters are "trainable" in the Qwen model
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=float(self.lr))

        # Setup scheduler
        num_training_steps = len(self.train_dataloader) // self.grad_acc_steps * self.max_epochs
        self.lr_scheduler = self._create_scheduler(self.optimizer, num_training_steps)

        # Setup grad scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Evaluator
        if self.val_dataloader is not None:
            self.evaluator = Evaluator(self.model, self.device)
        else:
            self.evaluator = None

        # Folder for saving sample predictions
        self.val_samples_dir = os.path.join(self.run_dir, "val_samples")
        ensure_dir(self.val_samples_dir)

    def _create_scheduler(self, optimizer, num_training_steps):
        if self.scheduler_type == "linear":
            def lr_lambda(current_step: int):
                if current_step < self.warmup_steps:
                    return float(current_step) / float(max(1, self.warmup_steps))
                return max(
                    0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - self.warmup_steps))
                )
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.scheduler_type == "cosine":
            def lr_lambda(current_step: int):
                if current_step < self.warmup_steps:
                    return float(current_step) / float(max(1, self.warmup_steps))
                progress = float(current_step - self.warmup_steps) / float(max(1, num_training_steps - self.warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_type}")
        return scheduler

    def train(self):
        for epoch in range(self.max_epochs):
            print(f"\n=== [Epoch {epoch+1}/{self.max_epochs}] ===")
            self.model.train()
            epoch_loss = 0.0
            step_count = 0

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Training")):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                images = batch.get("image", None)
                if images is not None and isinstance(images, torch.Tensor):
                    images = images.to(self.device)

                loss_val = self._training_step(input_ids, attention_mask, images)
                epoch_loss += loss_val
                step_count += 1

                if (step + 1) % self.log_interval == 0:
                    avg_loss = epoch_loss / step_count
                    lr_now = self.optimizer.param_groups[0]["lr"]
                    print(f"Epoch [{epoch+1}] Step [{step+1}/{len(self.train_dataloader)}], "
                          f"Loss: {avg_loss:.4f}, LR: {lr_now:.6f}")
                    wandb.log({"train/loss": avg_loss, "train/lr": lr_now, "step": self.global_step})
                    with open(self.train_log_path, "a", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.global_step, epoch+1, avg_loss, lr_now])

            # End of epoch
            if self.val_dataloader is not None:
                val_loss = self.validate(epoch+1)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    self._save_checkpoint(is_best=True, epoch=epoch+1)
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.early_stop_patience:
                        print("Early stopping triggered!")
                        self.should_stop = True
                        break
            else:
                self._save_checkpoint(is_best=False, epoch=epoch+1)

            if self.should_stop:
                break

        # Save final
        self._save_checkpoint(is_best=False, epoch=self.max_epochs, last=True)

    def _training_step(self, input_ids, attention_mask, images):
        # We let the huggingface trainer do cross-entropy if we pass `labels`.
        # So we'll pass `labels=input_ids` in a typical "teacher forcing" next token style
        # but it requires we shift the tokens ourselves or rely on the built-in logic.
        with autocast(enabled=self.use_amp):
            # The HF CausalLM automatically uses next-token if we pass labels=input_ids
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                labels=input_ids
            )
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        loss_for_backward = loss / self.grad_acc_steps
        self.scaler.scale(loss_for_backward).backward()

        if (self.global_step + 1) % self.grad_acc_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

        self.global_step += 1
        return loss.item()

    def validate(self, epoch):
        self.model.eval()
        val_losses = []
        # We'll store the samples for later logging
        sample_storage = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_dataloader, desc=f"Validation Epoch {epoch}")):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                images = batch.get("image", None)
                if images is not None and isinstance(images, torch.Tensor):
                    images = images.to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=images,
                        labels=input_ids
                    )
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    val_losses.append(loss.item())

                # Store the first sample in this batch for logging
                if i < 3:  # we log up to 3
                    sample_storage.append({
                        "input_ids": input_ids[0].detach().cpu(),
                        "attention_mask": attention_mask[0].detach().cpu(),
                        "points_str": batch.get("points_str", None)
                    })

        val_loss = sum(val_losses) / len(val_losses) if len(val_losses) > 0 else 0.0
        with open(self.val_log_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.global_step, epoch, val_loss])

        print(f"[Validation] Epoch: {epoch}, Loss: {val_loss:.4f}")
        wandb.log({"val/loss": val_loss, "epoch": epoch, "step": self.global_step})

        # Now log the 3 samples
        log_validation_samples(
            model=self.model,
            val_samples=sample_storage,
            device=self.device,
            global_step=self.global_step,
            out_dir=self.val_samples_dir,
            tokenizer=getattr(self.model.backbone, "qwen_tokenizer", None)
        )
        return val_loss

    def _save_checkpoint(self, is_best=False, epoch=0, last=False):
        name = "best_model.pt" if is_best else "last_model.pt" if last else f"checkpoint_epoch_{epoch}.pt"
        save_path = os.path.join(self.run_dir, name)
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict()
        }, save_path)
        print(f"Checkpoint saved to {save_path}")
