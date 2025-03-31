# /home/dbcloud/PycharmProjects/mllm4sam/app/models/model.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.

import torch
import torch.nn as nn

###############################################################################
# SAM4MLLM Model
###############################################################################
class SAM4MLLMModel(nn.Module):
    """
    This wrapper now holds a QwenSamBackbone, which uses Qwen-VL and optionally
    SAM synergy. We do NOT train SAM, only Qwen (with LoRA).
    """

    def __init__(self, backbone, projector=None):
        super().__init__()
        self.backbone = backbone
        self.projector = projector  # e.g. if you want extra layers

    def forward(self, input_ids, attention_mask, images=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            **kwargs
        )
        return outputs

    def generate(self, input_ids, attention_mask, images=None, max_new_tokens=128, **kwargs):
        if hasattr(self.backbone, "generate"):
            return self.backbone.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        else:
            print(f'[INFO] Model does not support generation.')
            return None
