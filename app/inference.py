# inference.py
import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.blocks.qwen_sam_backbone import QwenSamBackbone
from models.model import SAM4MLLMModel


def main(args):
    # 1. Load model
    print(f"[INFO] Loading model from {args.model_path}")

    # First check if we have separate qwen and sam directories
    qwen_path = os.path.join(args.model_path, "qwen")
    sam_path = os.path.join(args.model_path, "sam")

    if os.path.exists(qwen_path) and os.path.exists(sam_path):
        print(f"[INFO] Found separate Qwen and SAM directories")
        backbone = QwenSamBackbone(
            qwen_model_path=qwen_path,
            sam_model_path=sam_path,
            device=args.device,
            override_patch_size=16,
            override_temporal_patch_size=1
        )
    else:
        # Try to load from main path
        backbone = QwenSamBackbone(
            qwen_model_path=args.model_path,
            sam_model_path=args.sam_path,
            device=args.device,
            override_patch_size=16,
            override_temporal_patch_size=1
        )

    model = SAM4MLLMModel(backbone=backbone)
    model.to(args.device)
    model.eval()

    # 2. Load image
    print(f"[INFO] Loading image from {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")

    # 3. Run end-to-end inference
    print("[INFO] Running inference")
    result = backbone.predict_segmentation(
        image=image,
        text=args.text,
        prompt=args.prompt,
        return_points=True
    )

    # 4. Visualize results
    plt.figure(figsize=(12, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    # If we have points, plot them
    if "points" in result and result["points"]:
        points = np.array(result["points"])
        plt.scatter(points[:, 0], points[:, 1], c='red', marker='x', s=100)

    # Segmentation mask
    plt.subplot(1, 2, 2)
    if result["mask"] is not None:
        # Combine image and mask
        image_np = np.array(image)
        mask = result["mask"]

        # Overlay mask
        masked_img = image_np.copy()
        mask_color = np.zeros_like(image_np)
        mask_color[:, :, 0] = 255  # Red channel
        mask_color[mask > 0] = [0, 255, 0]  # Green for the mask

        alpha = 0.5
        masked_img = (1 - alpha) * image_np + alpha * mask_color
        masked_img = masked_img.astype(np.uint8)

        plt.imshow(masked_img)
        plt.title(f"Segmentation (Confidence: {result['confidence']:.3f})")
    else:
        plt.imshow(image)
        plt.title("No valid segmentation generated")

    # Save or show
    if args.output_path:
        plt.savefig(args.output_path, bbox_inches='tight')
        print(f"[INFO] Results saved to {args.output_path}")
    else:
        plt.show()

    # Print points
    if "points" in result and result["points"]:
        print("\nPredicted points:")
        for i, (x, y) in enumerate(result["points"]):
            print(f"Point {i + 1}: ({x}, {y})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run wound segmentation inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--sam_path", type=str, default=None, help="Path to SAM model (if not included in model_path)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--text", type=str, default=None, help="Text context for guided segmentation")
    parser.add_argument("--prompt", type=str, default=None, help="Specific prompt to use (overrides text)")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save output visualization")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()
    main(args)