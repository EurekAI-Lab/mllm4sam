import numpy as np


def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union (IoU) between predicted and ground truth masks.

    Args:
        pred_mask (numpy.ndarray): Binary predicted mask (0/1)
        gt_mask (numpy.ndarray): Binary ground truth mask (0/1)

    Returns:
        float: IoU score between 0 and 1
    """
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    if union == 0:
        return 1.0  # Both masks are empty, consider it a perfect match
    return float(intersection) / float(union)


def compute_dice(pred_mask, gt_mask):
    """
    Compute Dice coefficient between predicted and ground truth masks.
    The Dice coefficient is defined as 2*|X∩Y|/(|X|+|Y|).

    Args:
        pred_mask (numpy.ndarray): Binary predicted mask (0/1)
        gt_mask (numpy.ndarray): Binary ground truth mask (0/1)

    Returns:
        float: Dice coefficient between 0 and 1
    """
    # Ensure inputs are boolean or binary
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    # Calculate intersection and sums
    intersection = (pred_mask & gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()

    # Handle edge case where both masks are empty
    if pred_sum + gt_sum == 0:
        return 1.0  # Both masks are empty, consider it a perfect match

    # Compute Dice coefficient: 2*|X∩Y|/(|X|+|Y|)
    dice = (2.0 * intersection) / (pred_sum + gt_sum)

    return float(dice)