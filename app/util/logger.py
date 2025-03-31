# /home/dbcloud/PycharmProjects/mllm4sam/app/util/logger.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.

import wandb
import torch

def parse_points_from_text(text):
    """
    Attempt to parse lines like '(12,34), (56,78)' from the predicted text
    Return a set of (x, y) tuples.
    """
    # naive parse
    import re
    pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
    matches = re.findall(pattern, text)
    points = set()
    for match in matches:
        x = int(match[0])
        y = int(match[1])
        points.add((x, y))
    return points

def compute_point_match(pred_set, gt_str):
    """
    Count how many points in pred_set also appear in the ground truth.
    The ground truth is a string like '(x1,y1), (x2,y2)...' or "NoValidPoints".
    """
    if not gt_str or gt_str == "NoValidPoints":
        return 1.0 if (len(pred_set) == 0) else 0.0

    gt_points = parse_points_from_text(gt_str)
    if len(gt_points) == 0:
        # No GT
        if len(pred_set) == 0:
            return 1.0
        else:
            return 0.0

    # measure fraction of GT that was predicted
    correct = len(pred_set.intersection(gt_points))
    total = len(gt_points)
    return float(correct) / float(total) if total > 0 else 0.0

def log_validation_samples(model, val_samples, device, global_step, out_dir, tokenizer=None):
    """
    For each sample in val_samples, decode the model's predictions and compute
    a naive "point match" with the ground truth string.
    """
    if not val_samples:
        return

    model.eval()
    for idx, sample in enumerate(val_samples):
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        gt_str = sample.get("points_str", None)
        if gt_str is not None and isinstance(gt_str, list):
            gt_str = gt_str[0]  # if it was batched

        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50
            )
        if tokenizer is not None:
            pred_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        else:
            # fallback
            pred_text = str(gen_tokens[0].tolist())

        # parse points
        pred_points = parse_points_from_text(pred_text)
        match_score = compute_point_match(pred_points, gt_str)

        # Log to wandb
        log_dict = {
            f"val_sample_{idx}/pred_text": pred_text,
            f"val_sample_{idx}/ground_truth_points": gt_str,
            f"val_sample_{idx}/match_score": match_score,
            "global_step": global_step
        }
        wandb.log(log_dict)
