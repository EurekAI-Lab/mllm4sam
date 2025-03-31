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
    We will present these points in the "assistant" portion to do
    next-token cross-entropy training on Qwen's text output.

    This version addresses:
     - Forcing the image to 224 x 224 to match Qwen2-VL shape expectations.
     - Additional debug prints for shape mismatch issues.
     - Optional CSV loading for 'woundsegmentation' scenario.
     - Using system_prompt to unify conversation text.

    If the user sets `use_data: "woundsegmentation"`, then we attempt to load
    a CSV with image/mask pairs from root_dir. Otherwise, we fallback to a
    user-provided data_list or a dummy example.
    """

    def __init__(self,
                 data_list=None,
                 tokenizer=None,
                 transforms=None,
                 max_len=1536,
                 img_size=(224, 224),   # We now default to 224x224
                 img_dir='./data_images/',
                 system_prompt="You are a helpful segmentation assistant.",
                 use_data="dummy",
                 root_dir="",
                 split="train"):
        super().__init__()
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.max_len = max_len
        # We force 224 x 224 by default for Qwen2-VL
        self.img_size = img_size
        self.img_dir = img_dir
        self.system_prompt = system_prompt
        self.use_data = use_data
        self.root_dir = root_dir
        self.split = split

        # Load or set data_list
        if data_list is not None:
            self.data_list = data_list
        else:
            self.data_list = []
            if self.use_data == "woundsegmentation":
                self.data_list = self._load_woundsegmentation_data()
            else:
                # fallback dummy
                self.data_list = [
                    {"image_path": "dummy1.png", "mask_path": "dummy1_mask.png", "conversation": "Segment?"},
                    {"image_path": "dummy2.png", "mask_path": "dummy2_mask.png", "conversation": "Segment?"}
                ]

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
        print(f"[DEBUG Dataset] Loading image: {full_image_path}")

        image = PIL.Image.open(full_image_path).convert("RGB")

        # 2. Load mask if mask_path is available
        mask = None
        if mask_path:
            full_mask_path = self._resolve_path(mask_path, subfolder="masks")
            if os.path.exists(full_mask_path):
                print(f"[DEBUG Dataset] Loading mask: {full_mask_path}")
                mask = PIL.Image.open(full_mask_path).convert("L")
            else:
                print(f"[DEBUG Dataset] No mask found at {full_mask_path}, skipping mask.")
                mask = None

        # 3. transforms or resizing to 224x224
        #    Qwen2-VL’s patch-embedding requires (B, 3, 224, 224) by default
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
        points_str = ""
        if mask is not None:
            mask_np = np.array(mask, dtype=np.uint8)
            mask_np = (mask_np >= 128).astype(np.uint8)
            coords_list = mask_to_points(mask_np, max_points=max_points)
            if len(coords_list) > 0:
                points_str_list = []
                for (y, x) in coords_list:
                    points_str_list.append(f"({x},{y})")
                points_str = ", ".join(points_str_list)
            else:
                points_str = "NoValidPoints"
        else:
            # If no mask, we have no valid points
            points_str = "NoValidPoints"

        # 5. Build textual prompt
        user_text = f"Please provide up to 10 points that cover the object region."
        assistant_text = points_str
        full_text = (
            f"{self.system_prompt}\n[USER]: {conversation}\n"
            f"[USER]: {user_text}\n"
            f"[ASSISTANT]: {assistant_text}"
        )

        # 6. Tokenize if available
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_len
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
        else:
            # placeholders
            input_ids = torch.tensor([0])
            attention_mask = torch.tensor([1])

        # 7. Convert image to tensor if still PIL
        if isinstance(image, PIL.Image.Image):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        # Debug shape
        print(f"[DEBUG Dataset] Final image shape: {image.shape}")

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "points_str": points_str
        }
        return sample

    def _resolve_path(self, filename, subfolder="images"):
        """
        1) Fix common issues if 'train/images/' or 'test/images/' are already in 'filename'
           so we don't double them.
        2) Attempt to see if it's an absolute path. If so, return it.
        3) Otherwise, check if it's in self.img_dir
        4) Else, fallback to root_dir/split/subfolder/filename
        """
        fix_list = [
            "train/images/", "test/images/",
            "train/masks/", "test/masks/"
        ]
        for fix_token in fix_list:
            if fix_token in filename:
                # print(f"[DEBUG Dataset] Stripping '{fix_token}' from filename: {filename}")
                filename = filename.replace(fix_token, "")

        # Now proceed
        if os.path.isabs(filename):
            return filename

        potential_path = os.path.join(self.img_dir, filename)
        if os.path.exists(potential_path):
            return potential_path

        alt = os.path.join(self.root_dir, self.split, subfolder, filename)
        return alt

    def _load_woundsegmentation_data(self):
        """
        If `use_data == "woundsegmentation"`, we read from:
          root_dir/train_labels.csv or root_dir/test_labels.csv
        to build data_list with image_path, mask_path, conversation, etc.
        """
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
                data_item = {
                    "image_path": filename,
                    "mask_path": filename,
                    "conversation": "Help me segment this wound."
                }
                data_list.append(data_item)
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
