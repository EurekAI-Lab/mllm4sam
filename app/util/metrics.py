# /home/dbcloud/PycharmProjects/mllm4sam/app/util/metrics.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.

def compute_iou(pred_mask, gt_mask):
    """
    Example IoU computation placeholder.
    pred_mask, gt_mask: (H, W) boolean or 0/1 arrays
    """
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)
