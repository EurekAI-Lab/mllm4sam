# app/dataloader/dataset_sam4mllm.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.

import os
import csv
import random
import torch
import numpy as np
import PIL
from torch.utils.data import Dataset


def mask_to_points(mask_array, max_points=10):
    """
    Convert a binary segmentation mask to a list of (x,y) pixel coordinates
    lying inside the mask. Return up to max_points of them.
    """
    coords = np.argwhere(mask_array == 1)  # shape (N, 2)
    if len(coords) == 0:
        return []
    coords_list = coords.tolist()
    random.shuffle(coords_list)
    coords_list = coords_list[:max_points]
    return coords_list


class BaseSAM4MLLMDataset(Dataset):
    """
    A base dataset for SAM4MLLM demonstration that also
    extracts random points from the segmentation mask as "ground truth."
    We provide short text prompts about wound segmentation (or a dummy conversation).

    This dataset is used for Supervised Fine Tuning (SFT). The actual text
    tokenization is done by TRL's collator or similar inside the Trainer.
    We only store the raw text in "prompt_text" or "gt_points_str," etc.
    """

    def __init__(
            self,
            data_list=None,
            tokenizer=None,
            transforms=None,
            max_len=1536,
            img_size=(224, 224),
            img_dir='./data_images/',
            system_prompt="You are a helpful segmentation assistant.",
            use_data="dummy",
            root_dir="",
            split="train"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.max_len = max_len
        self.img_size = img_size
        self.img_dir = img_dir
        self.system_prompt = system_prompt
        self.use_data = use_data
        self.root_dir = root_dir
        self.split = split

        if data_list is not None:
            self.data_list = data_list
        else:
            self.data_list = []
            if self.use_data == "woundsegmentation":
                self.data_list = self._load_woundsegmentation_data()
            else:
                raise ValueError("Unknown use_data argument")

        self._prune_missing_files()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        item = self.data_list[idx]
        image_path = item.get("image_path", None)
        mask_path = item.get("mask_path", None)
        conversation = item.get("conversation", "")

        if not image_path:
            raise ValueError(f"No 'image_path' found for index {idx} in data_list.")

        # 1. Load image
        full_image_path = self._resolve_path(image_path, subfolder="images")
        if not os.path.exists(full_image_path):
            raise FileNotFoundError(f"Image file not found even after pruning: {full_image_path}")
        # print(f"[DEBUG Dataset] Loading image: {full_image_path}")

        image = PIL.Image.open(full_image_path).convert("RGB")

        # 2. Load mask if available
        mask = None
        if mask_path:
            full_mask_path = self._resolve_path(mask_path, subfolder="masks")
            if os.path.exists(full_mask_path):
                # print(f"[DEBUG Dataset] Loading mask: {full_mask_path}")
                mask = PIL.Image.open(full_mask_path).convert("L")
            else:
                # print(f"[DEBUG Dataset] No mask found at {full_mask_path}, skipping mask.")
                mask = None

        # 3. transforms or resizing to (img_size)
        if self.transforms:
            image = self.transforms(image)
            if mask is not None:
                mask = self.transforms(mask)
        else:
            image = image.resize(self.img_size)
            if mask is not None:
                mask = mask.resize(self.img_size)

        # 4. Convert mask to up to 10 points
        max_points = 10
        points_str = "NoValidPoints"
        if mask is not None:
            mask_np = np.array(mask, dtype=np.uint8)
            mask_np = (mask_np >= 128).astype(np.uint8)
            coords_list = mask_to_points(mask_np, max_points=max_points)
            if len(coords_list) > 0:
                # format: "(x,y), (x2,y2), ..."
                pts_text_list = []
                for (rowY, colX) in coords_list:
                    pts_text_list.append(f"({colX},{rowY})")
                points_str = ", ".join(pts_text_list)

        # 5. Build short text for SFT
        # We store the user prompt and system prompt, letting the collator handle
        # the final arrangement. For example:
        user_text = f"Point out the wound area in up to {max_points} points."
        system_text = self.system_prompt
        assistant_text = points_str

        # We'll store these raw texts to be used by the TRL collator or in the custom collate function.
        # The "conversation" can also be appended if you want multi-turn text. For now we keep it simple.
        sample_text = (
            f"{system_text}\n"
            f"[USER]: {conversation}\n"  # if you want multiple lines
            f"[USER]: {user_text}\n"
            f"[ASSISTANT]: {assistant_text}\n"
        )

        # 6. Convert image to tensor if still PIL
        if isinstance(image, PIL.Image.Image):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1)

        # Debug shape
        # print(f"[DEBUG Dataset] Final image shape: {image.shape}")

        # 7. Return a dict with text+image
        return {
            "text_input": sample_text,
            "image": image,
            "points_str": points_str,  # ground truth for validation usage
        }

    def _resolve_path(self, filename, subfolder="images"):
        fix_list = [
            "train/images/", "test/images/",
            "train/masks/", "test/masks/"
        ]
        for fix_token in fix_list:
            if fix_token in filename:
                filename = filename.replace(fix_token, "")

        if os.path.isabs(filename):
            return filename

        potential_path = os.path.join(self.img_dir, filename)
        if os.path.exists(potential_path):
            return potential_path

        alt = os.path.join(self.root_dir, self.split, subfolder, filename)
        return alt

    def _load_woundsegmentation_data(self):
        data_list = []
        csv_name = "train_labels.csv" if self.split == "train" else "test_labels.csv"
        csv_path = os.path.join(self.root_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"[WARNING] CSV file not found at {csv_path}. Returning empty data_list.")
            return data_list

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                filename = row[0].strip()
                data_list.append({
                    "image_path": filename,
                    "mask_path": filename,
                    "conversation": "Help me segment this wound."
                })
        return data_list

    def _prune_missing_files(self):
        kept = []
        for item in self.data_list:
            resolved = self._resolve_path(item["image_path"], subfolder="images")
            if not os.path.exists(resolved):
                print(f"[WARNING] Skipping nonexistent file: {resolved}")
                continue
            kept.append(item)
        self.data_list = kept
        print(f"[INFO] After pruning, we have {len(self.data_list)} valid samples.")
