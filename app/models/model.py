# app/models/model.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.
#
# This class wraps your synergy backbone (Qwen + SAM) with a standard forward / generate interface.

import torch
import torch.nn as nn

class SAM4MLLMModel(nn.Module):
    """
    A synergy wrapper that holds a QwenSamBackbone (Qwen2-VL + possible LoRA),
    plus optional projection layers if you want them.
    We do not rename or skip anything to preserve your project structure.
    """

    def __init__(self, backbone, projector=None):
        super().__init__()
        self.backbone = backbone
        self.projector = projector  # optional

        # Expose the necessary attributes for TRL compatibility
        if hasattr(backbone, 'qwen_model') and hasattr(backbone.qwen_model, 'config'):
            self.config = backbone.qwen_model.config
            # Also expose other necessary attributes that might be needed
            self.model_name = backbone.qwen_model.__class__.__name__
            # Expose tokenizer if it exists
            if hasattr(backbone, 'qwen_tokenizer'):
                self.tokenizer = backbone.qwen_tokenizer

    def forward(self, input_ids=None, attention_mask=None, images=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            **kwargs
        )
        return outputs

    def generate(self, input_ids=None, attention_mask=None, images=None, max_new_tokens=128, **kwargs):
        if hasattr(self.backbone, "generate"):
            return self.backbone.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        else:
            print(f"[INFO] Model does not support generation.")
            return None

    # Make sure we pass through other important attributes for TRL
    def get_input_embeddings(self):
        if hasattr(self.backbone.qwen_model, "get_input_embeddings"):
            return self.backbone.qwen_model.get_input_embeddings()
        return None

    def get_output_embeddings(self):
        if hasattr(self.backbone.qwen_model, "get_output_embeddings"):
            return self.backbone.qwen_model.get_output_embeddings()
        return None