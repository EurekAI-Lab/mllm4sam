# /home/dbcloud/PycharmProjects/mllm4sam/app/models/blocks/qwen_sam_backbone.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# -------------------------------------------------------------------------
# This code provides a synergy backbone that integrates Qwen2-VL and (optionally)
# a SAM model from Hugging Face Transformers. It gracefully handles the situation
# where importing Qwen2VLConfig or Qwen2VLVisionConfig fails, by performing a
# "monkey patch" of the Qwen2-VL visual patch embedding to avoid shape mismatch.
#
# Important notes:
#  1) By default, Qwen2-VL expects patch_size=14, temporal_patch_size=2 for 224x224
#     or 448x448, etc. If your data is 512x512 single-frame images, that leads to
#     an invalid shape `'[-1,3,2,14,14]'`.
#  2) If we cannot import Qwen2VLVisionConfig (the environment lacks it, or a version
#     mismatch), we will forcibly do a "monkey patch" where we override the patch
#     size and stride in Qwen2-VL's `PatchEmbed` module. This ensures shape alignment
#     for e.g. 512x512 single-frame images.
#  3) For 512x512, you can pick override_patch_size=16 (then 512/16=32).
#     override_temporal_patch_size=1 if you do not have a time dimension.
#  4) Full debugging prints are present, so you can confirm patch shapes.
#  5) LoRA is attempted if environment is OK with `peft`; if it fails, it will
#     log a warning and proceed.
#
# Example usage in your main code:
#   from models.blocks.qwen_sam_backbone import QwenSamBackbone
#   backbone = QwenSamBackbone(
#       qwen_model_path="path_or_repo_to_Qwen2-VL",
#       sam_model_path="path_or_repo_to_SAM",
#       device="cuda",
#       override_patch_size=16,
#       override_temporal_patch_size=1
#   )
#   ...
#
# -------------------------------------------------------------------------

import os
import sys
import torch
import torch.nn as nn

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Attempt to import SamModel, SamProcessor from HF:
try:
    from transformers import SamModel, SamProcessor
    HF_SAM_AVAILABLE = True
except ImportError as e:
    print("[WARNING] Could not import SamModel, SamProcessor from transformers. Reason:", e)
    HF_SAM_AVAILABLE = False

# Attempt to import PEFT for LoRA:
try:
    from peft import LoraConfig, get_peft_model
    LOADING_PEFT_OK = True
except Exception as e:
    print("[WARNING] Could not import peft or bitsandbytes. Reason:", e)
    LOADING_PEFT_OK = False

# Attempt to import Qwen2VLConfig, Qwen2VLVisionConfig (some envs lack it):
try:
    from transformers.models.qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig
    QWEN_CONFIG_AVAILABLE = True
except ImportError as e:
    print("[WARNING] Could not import Qwen2VLConfig or Qwen2VLVisionConfig. Reason:", e)
    QWEN_CONFIG_AVAILABLE = False


class QwenSamBackbone(nn.Module):
    """
    A synergy backbone that loads Qwen2-VL with Qwen2VLForConditionalGeneration
    and optionally Hugging Face SAM.

    Key features:
      - If we cannot override the patch size via the config (import error),
        we do a "monkey patch" on the model's `patch_embed` module to forcibly
        set patch_size, temporal_patch_size, and the Conv3d kernel/stride.
      - We do extensive debug prints to help identify shape mismatch issues.
      - If environment does not allow LoRA (missing dev libraries, etc.), we skip it.

    Args:
        qwen_model_path (str): Path or repo ID for Qwen2-VL model.
        sam_model_path (str): Path or repo ID for SAM model (optional).
        device (str): "cuda" or "cpu".
        override_patch_size (int): Desired patch_size for Qwen2-VL.
                                   If images are 512x512, pick e.g. 16 or 32.
        override_temporal_patch_size (int): Usually 1 for single images.
    """

    def __init__(
        self,
        qwen_model_path: str,
        sam_model_path: str,
        device="cuda",
        override_patch_size: int = 16,
        override_temporal_patch_size: int = 1,
    ):
        super().__init__()
        self.device = device
        self.override_patch_size = override_patch_size
        self.override_temporal_patch_size = override_temporal_patch_size

        # 1) Try to load Qwen2-VL with a custom config if possible:
        if QWEN_CONFIG_AVAILABLE:
            print("[INFO] Creating a custom Qwen2VLConfig to override patch sizes.")
            try:
                base_config = Qwen2VLConfig.from_pretrained(qwen_model_path, trust_remote_code=True)
                vision_conf = base_config.vision_config
                if not isinstance(vision_conf, Qwen2VLVisionConfig):
                    print("[WARNING] vision_config not a Qwen2VLVisionConfig. We'll attempt to forcibly re-instantiate.")
                    vision_conf = Qwen2VLVisionConfig()
                print(f"[INFO] Overriding patch_size => {override_patch_size}, temporal_patch_size => {override_temporal_patch_size}")
                vision_conf.patch_size = override_patch_size
                vision_conf.temporal_patch_size = override_temporal_patch_size
                base_config.vision_config = vision_conf

                print(f"[DEBUG] Summarizing Qwen2VLVisionConfig:\n"
                      f"  patch_size={base_config.vision_config.patch_size}, "
                      f"  temporal_patch_size={base_config.vision_config.temporal_patch_size}")

                print(f"[INFO] Loading Qwen2-VL model from {qwen_model_path} with custom config ...")
                self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    qwen_model_path,
                    config=base_config,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(device)
            except Exception as e:
                print(f"[WARNING] Something went wrong using custom config: {e}\n"
                      f"Falling back to normal load and monkey patch.")
                self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    qwen_model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(device)
                self._monkey_patch_patch_embed()
        else:
            # 2) If we can't import QWEN2VLVisionConfig, we do a normal load then monkey patch
            print("[INFO] Loading Qwen2-VL model from the original config (cannot override patch sizes).")
            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                qwen_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(device)
            self._monkey_patch_patch_embed()

        # Load the processor (tokenizer + image processor)
        print("[INFO] Loading Qwen2-VL processor (tokenizer + image processor) ...")
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_model_path, trust_remote_code=True)
        self.qwen_tokenizer = self.qwen_processor.tokenizer
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        # Attempt LoRA
        if LOADING_PEFT_OK:
            try:
                print("[INFO] Attempting to set up LoRA for Qwen2-VL ...")
                self._setup_lora()
            except Exception as e:
                print("[WARNING] LoRA setup failed due to environment error. Disabling LoRA. Error:", e)
        else:
            print("[WARNING] LoRA is disabled because we cannot import PEFT properly.")

        # Optionally load SAM
        self.sam_model = None
        self.sam_processor = None
        if HF_SAM_AVAILABLE:
            try:
                print(f"[INFO] Attempting to load SAM model from {sam_model_path} via HF Transformers SamModel...")
                self.sam_model = SamModel.from_pretrained(
                    sam_model_path,
                    torch_dtype=torch.float16
                ).eval().to(device)
                self.sam_processor = SamProcessor.from_pretrained(sam_model_path)
            except Exception as e:
                print(f"[WARNING] Could not load huggingface SamModel from {sam_model_path}. Reason: {e}")
                self.sam_model = None
                self.sam_processor = None
        else:
            print("[WARNING] SamModel is not available from transformers package. Skipping HF-SAM usage.")

    def _monkey_patch_patch_embed(self):
        """
        If we can't override the Qwen2VL config or if it fails for some reason,
        forcibly adjust Qwen2-VL's patch_embed to have the new patch_size
        and temporal_patch_size. Then also fix the underlying conv's kernel/stride.
        This ensures shape alignment for e.g. 512x512 input images.
        """
        # The Qwen2-VL code sets:
        #    self.visual.patch_embed.patch_size
        #    self.visual.patch_embed.temporal_patch_size
        #    self.visual.patch_embed.proj = nn.Conv3d(in_channels=3, out_channels=embed_dim, kernel_size=..., stride=..., bias=False)
        #
        # We can attempt to do the same override now:
        print("[INFO] Performing monkey patch for Qwen2-VL patch embedding to fix shape mismatch.")
        patch_embed_module = getattr(self.qwen_model.visual, "patch_embed", None)
        if patch_embed_module is None:
            print("[WARNING] Could not locate patch_embed in self.qwen_model.visual. Skipping patch fix.")
            return

        # Patch in the new fields:
        old_patch_size = patch_embed_module.patch_size
        old_temporal = patch_embed_module.temporal_patch_size
        embed_dim = patch_embed_module.embed_dim
        in_channels = patch_embed_module.in_channels

        print(f"[DEBUG] Original patch_embed.patch_size={old_patch_size}, patch_embed.temporal_patch_size={old_temporal}")
        patch_embed_module.patch_size = self.override_patch_size
        patch_embed_module.temporal_patch_size = self.override_temporal_patch_size

        # Now reconstruct the conv layer with new kernel=[temporal, patch, patch], stride=same
        kernel_size = [self.override_temporal_patch_size, self.override_patch_size, self.override_patch_size]
        stride = [self.override_temporal_patch_size, self.override_patch_size, self.override_patch_size]
        print(f"[DEBUG] Re-building patch_embed.proj with kernel_size={kernel_size}, stride={stride}, in_channels={in_channels}, out_channels={embed_dim}")
        new_conv = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        # Copy old weights if shapes match the new
        if patch_embed_module.proj.weight.shape == new_conv.weight.shape:
            with torch.no_grad():
                new_conv.weight.copy_(patch_embed_module.proj.weight)
            print("[INFO] Copied old patch_embed.proj weights to new shape (identical shape).")
        else:
            # If shape mismatch, we reset
            nn.init.normal_(new_conv.weight, mean=0.0, std=0.02)
            print("[WARNING] Old patch_embed.proj weights cannot be copied due to shape mismatch. Re-initialized randomly.")

        patch_embed_module.proj = new_conv

    def _setup_lora(self):
        """
        Create and apply a LoRA adapter config to Qwen2-VL if environment permits.
        If it fails, we skip LoRA.
        """
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.qwen_model = get_peft_model(self.qwen_model, lora_config)
        self.qwen_model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, images=None, labels=None, **kwargs):
        """
        Standard forward pass for Qwen2VLForConditionalGeneration:
          - If images is not None, we pass them as pixel_values (multi-modal).
          - If labels is given, we get cross-entropy loss from Qwen2-VL.
          - Print shapes for debug.
        """
        print(f"[DEBUG QwenSamBackbone] forward() input_ids shape: {input_ids.shape}")
        print(f"[DEBUG QwenSamBackbone] forward() attention_mask shape: {attention_mask.shape}")
        if images is not None:
            print(f"[DEBUG QwenSamBackbone] forward() images shape: {images.shape}")

        # The Qwen2-VL forward expects pixel_values=images
        # The code will do an internal check to create the patch embeddings
        # with patch size, etc. as we've forced or configured.
        if images is not None:
            outputs = self.qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=images,
                labels=labels,
                **kwargs
            )
        else:
            outputs = self.qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        return outputs

    def generate(self, input_ids, attention_mask, images=None, max_new_tokens=128, **kwargs):
        """
        For text generation (multi-modal or text-only).
        We do the same "pixel_values=images" approach if images are provided.
        """
        print(f"[DEBUG QwenSamBackbone] generate() input_ids shape: {input_ids.shape}")
        print(f"[DEBUG QwenSamBackbone] generate() attention_mask shape: {attention_mask.shape}")
        if images is not None:
            print(f"[DEBUG QwenSamBackbone] generate() images shape: {images.shape}")

        if images is not None:
            outputs = self.qwen_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=images,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        else:
            outputs = self.qwen_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        return outputs
