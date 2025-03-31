# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn

class VisualEncoder(nn.Module):
    """
    A placeholder for a custom visual encoder that might process images
    before feeding into your main backbone. For instance,
    you can incorporate SAM ViT or any other custom logic here.
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        # In practice, you might load a pretrained SAM encoder or similar
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=7, stride=2, padding=3)

    def forward(self, x: torch.Tensor):
        """
        x shape: (B, 3, H, W)
        """
        # Dummy forward
        feats = self.conv(x)
        return feats
