# app/util/logger.py
# We keep your parse_points_from_text approach and optional logging.

import wandb
import torch
import re

def parse_points_from_text(text):
    pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
    matches = re.findall(pattern, text)
    points = set()
    for match in matches:
        x = int(match[0])
        y = int(match[1])
        points.add((x, y))
    return points

def compute_point_match(pred_set, gt_str):
    if not gt_str or gt_str == "NoValidPoints":
        return 1.0 if (len(pred_set) == 0) else 0.0
    gt_points = parse_points_from_text(gt_str)
    if len(gt_points) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    correct = len(pred_set.intersection(gt_points))
    total = len(gt_points)
    return float(correct) / float(total) if total > 0 else 0.0

def log_validation_samples(model, val_samples, device, global_step, out_dir, tokenizer=None):
    pass  # We rely on the new SFT-based approach now, or you can adapt as needed.
