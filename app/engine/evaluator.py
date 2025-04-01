# app/engine/evaluator.py
# Minimal placeholder; we do not rename it.
# You can optionally add advanced segmentation metrics here (e.g. IoU) if you run SAM
# from the predicted points in the text output.

import torch
import torch.nn as nn


class Evaluator:
    def __init__(self, model: nn.Module, device="cuda"):
        self.model = model
        self.device = device

    def evaluate(self, dataloader):
        """
        Evaluate model on validation data with multiple metrics:
        - Loss: Cross-entropy loss on text generation
        - Point accuracy: Match between predicted and ground truth points
        - IoU: If SAM is available, compute IoU between predicted and ground truth masks
        - Dice: If SAM is available, compute Dice coefficient
        """
        self.model.eval()
        metrics = {
            "loss": [],
            "point_match": [],
            "iou": [],
            "dice": []
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # 1. Get inputs from batch
                if isinstance(batch["text_input"], list):
                    text_list = batch["text_input"]
                else:
                    text_list = [batch["text_input"]]

                images = batch["image"]
                if isinstance(images, torch.Tensor):
                    pixel_values = images.to(self.device)
                else:
                    pixel_values = None

                # 2. Compute text generation loss
                tokenizer = self.model.backbone.qwen_tokenizer
                encoded = tokenizer(
                    text_list,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1536  # Adjust as needed
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=pixel_values,
                    labels=input_ids
                )

                loss_val = outputs.loss.item() if hasattr(outputs, "loss") else outputs[0].item()
                metrics["loss"].append(loss_val)

                # 3. Generate point predictions
                user_text = "Point out the wound area in up to 10 points."
                prompts = [f"You are a helpful segmentation assistant.\n[USER]: {user_text}\n[ASSISTANT]:"] * len(
                    text_list)

                # Tokenize prompts
                prompt_ids = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1536
                ).to(self.device)

                # Generate responses
                gen_ids = self.model.generate(
                    input_ids=prompt_ids.input_ids,
                    attention_mask=prompt_ids.attention_mask,
                    images=pixel_values,
                    max_new_tokens=128,
                    do_sample=False
                )

                # Decode generated text
                gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                # For each example, extract points and compute metrics
                for i, gen_text in enumerate(gen_texts):
                    # Extract model's response
                    response = gen_text.split("[ASSISTANT]:")[-1].strip()

                    # Parse predicted points
                    pred_points_set = parse_points_from_text(response)

                    # Get ground truth points
                    gt_points_str = batch["points_str"][i] if isinstance(batch["points_str"], list) else batch[
                        "points_str"]

                    # Compute point match metric
                    point_match = compute_point_match(pred_points_set, gt_points_str)
                    metrics["point_match"].append(point_match)

                    # If SAM is available and we have mask paths, compute segmentation metrics
                    if hasattr(self.model.backbone, "sam_model") and self.model.backbone.sam_model is not None:
                        # Try to get the mask path
                        mask_path = batch["mask_path"][i] if isinstance(batch["mask_path"], list) else batch[
                            "mask_path"]

                        if mask_path and os.path.exists(mask_path):
                            # Load ground truth mask
                            gt_mask = np.array(PIL.Image.open(mask_path).convert("L"))
                            gt_mask = (gt_mask >= 128).astype(np.uint8)

                            # Skip if no points were found
                            if len(pred_points_set) == 0:
                                metrics["iou"].append(0.0)
                                metrics["dice"].append(0.0)
                                continue

                            # Convert set of points to list format for SAM
                            pred_points = [[x, y] for x, y in pred_points_set]

                            # Generate segmentation with SAM
                            try:
                                result = self.model.backbone.predict_segmentation(
                                    image=images[i] if isinstance(images, torch.Tensor) else images,
                                    prompt=None,  # We're providing points directly
                                    return_points=True
                                )

                                if result["mask"] is not None:
                                    # Resize predicted mask to match ground truth if needed
                                    pred_mask = result["mask"]
                                    if pred_mask.shape != gt_mask.shape:
                                        import cv2
                                        pred_mask = cv2.resize(
                                            pred_mask.astype(np.uint8),
                                            (gt_mask.shape[1], gt_mask.shape[0]),
                                            interpolation=cv2.INTER_NEAREST
                                        )

                                    # Compute IoU
                                    iou = compute_iou(pred_mask > 0, gt_mask > 0)
                                    metrics["iou"].append(iou)

                                    # Compute Dice coefficient
                                    dice = compute_dice(pred_mask > 0, gt_mask > 0)
                                    metrics["dice"].append(dice)
                                else:
                                    # No valid mask generated
                                    metrics["iou"].append(0.0)
                                    metrics["dice"].append(0.0)
                            except Exception as e:
                                print(f"[WARNING] Error generating segmentation: {e}")
                                metrics["iou"].append(0.0)
                                metrics["dice"].append(0.0)

        # Compute mean metrics
        results = {}
        for metric_name, values in metrics.items():
            if values:
                results[metric_name] = sum(values) / len(values)
            else:
                results[metric_name] = 0.0

        return results


def compute_dice(pred_mask, gt_mask):
    """
    Compute Dice coefficient between predicted and ground truth masks.

    Args:
        pred_mask (numpy.ndarray): Binary prediction mask
        gt_mask (numpy.ndarray): Binary ground truth mask

    Returns:
        float: Dice coefficient (0-1)
    """
    intersection = (pred_mask & gt_mask).sum()
    if intersection == 0:
        return 0.0
    return (2.0 * intersection) / (pred_mask.sum() + gt_mask.sum())