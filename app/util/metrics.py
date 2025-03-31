# app/util/metrics.py
# Example placeholder for segmentation IoU if you decode predicted points and pass them to SAM.

def compute_iou(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)
