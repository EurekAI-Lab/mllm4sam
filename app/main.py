# /home/dbcloud/PycharmProjects/mllm4sam/app/main.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from dataloader.dataset_sam4mllm import BaseSAM4MLLMDataset
from models.model import SAM4MLLMModel
from engine.trainer import Trainer
from util.utils import set_seed


###############################################################################
# Example main script
###############################################################################
def main(args):
    # -------------------------------------------------------------------------
    # 1. Load Config
    # -------------------------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # fix seed
    set_seed(config.get("seed", 42))

    # Retrieve from config
    sam_model_path = config["model"]["sam"]
    qwen_model_path = config["model"]["qwen"]
    print(f"[DEBUG] SAM model path: {sam_model_path}")
    print(f"[DEBUG] Qwen model path: {qwen_model_path}")

    # -------------------------------------------------------------------------
    # 2. Decide how to load data
    # -------------------------------------------------------------------------
    # If the user sets `use_data: "woundsegmentation"`, we let the dataset
    # load from CSV automatically by passing None for data_list.
    # If the user sets `use_data: "dummy"`, we show a small example list.
    #
    # This ensures we do NOT hit the FileNotFoundError from "dummy1.png"
    # unless you intentionally choose "dummy".
    #
    if config["dataset"]["use_data"] == "woundsegmentation":
        train_data_list = None
        val_data_list = None
        print("[INFO] Using the woundsegmentation CSV data from root_dir.")
    else:
        # fallback dummy
        print("[INFO] Using dummy data_list since use_data != woundsegmentation.")
        train_data_list = [
            # Example placeholders. In real usage, you might rely on `use_data`
            # to auto-load from disk with segmentation maps, etc.
            {"image_path": "dummy1.png", "mask_path": "dummy1_mask.png", "conversation": "Segment the object."},
            {"image_path": "dummy2.png", "mask_path": "dummy2_mask.png", "conversation": "Segment the big region."}
        ]
        val_data_list = [
            {"image_path": "dummy3.png", "mask_path": "dummy3_mask.png", "conversation": "Find me the object."}
        ]

    # -------------------------------------------------------------------------
    # 3. Prepare Dataset / Dataloader
    # -------------------------------------------------------------------------
    train_dataset = BaseSAM4MLLMDataset(
        data_list=train_data_list,   # can be None if we want CSV-based loading
        tokenizer=None,             # The actual Qwen tokenizer is loaded inside the backbone
        transforms=None,
        max_len=config["train"]["max_len"],
        img_size=tuple(config["train"]["img_size"]),
        img_dir=config["train"]["img_dir"],
        system_prompt="You are a helpful segmentation assistant.",
        use_data=config["dataset"]["use_data"],
        root_dir=config["dataset"]["root_dir"],
        split=config["dataset"]["split"]
    )

    # For validation dataset, we typically do "split='test'" if there's a separate CSV/test set.
    # But we rely on config["dataset"]["split"] = "train" or "test" as you prefer.
    # For demonstration, we keep it the same. You might point it to a test CSV if you have one.
    val_dataset = BaseSAM4MLLMDataset(
        data_list=val_data_list,     # can be None if we want CSV-based loading for val
        tokenizer=None,
        transforms=None,
        max_len=config["train"]["max_len"],
        img_size=tuple(config["train"]["img_size"]),
        img_dir=config["train"]["img_dir"],
        system_prompt="You are a helpful segmentation assistant.",
        use_data=config["dataset"]["use_data"],
        root_dir=config["dataset"]["root_dir"],
        split=config["dataset"]["split"]  # or "test" if you want separate test set
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=0
    )

    # -------------------------------------------------------------------------
    # 4. Build Model
    # -------------------------------------------------------------------------
    from models.blocks.qwen_sam_backbone import QwenSamBackbone

    # This backbone will internally load Qwen + SAM (but SAM is not trained).
    backbone = QwenSamBackbone(
        qwen_model_path=qwen_model_path,
        sam_model_path=sam_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model = SAM4MLLMModel(backbone=backbone)

    # -------------------------------------------------------------------------
    # 5. Trainer
    # -------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr=config["optimizer"]["lr"],
        max_epochs=config["train"]["epochs"],
        grad_acc_steps=config["train"]["grad_acc_steps"],
        scheduler_type=config["optimizer"]["scheduler_type"],
        warmup_steps=config["optimizer"]["warmup_steps"],
        early_stop_patience=config["train"]["early_stop_patience"],
        clip_grad_norm=config["train"]["clip_grad_norm"],
        output_dir=config["train"]["output_dir"],
        run_name=config["train"]["run_name"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=config["train"]["use_amp"],
        log_interval=config["train"]["log_interval"]
    )

    # -------------------------------------------------------------------------
    # 6. Train
    # -------------------------------------------------------------------------
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config")
    args = parser.parse_args()
    main(args)
