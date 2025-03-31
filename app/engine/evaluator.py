# /home/dbcloud/PycharmProjects/mllm4sam/app/engine/evaluator.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.

import torch
import torch.nn as nn

class Evaluator:
    """
    Minimal evaluator skeleton for SAM4MLLM.
    We do not do advanced metrics here. We let the trainer handle
    the cross-entropy loss. This class can be expanded if needed.
    """
    def __init__(self, model: nn.Module, device="cuda"):
        self.model = model
        self.device = device

    def evaluate(self, dataloader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                images = batch.get("image", None)
                if images is not None and isinstance(images, torch.Tensor):
                    images = images.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    labels=input_ids
                )
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                losses.append(loss.item())

        mean_loss = sum(losses) / len(losses) if len(losses) > 0 else 0
        return {"loss": mean_loss}
