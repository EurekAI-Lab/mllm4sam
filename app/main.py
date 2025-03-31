# app/main.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.
#
# This main script loads your config, prepares the dataset, sets up the SFT-based Trainer,
# and launches the fine-tuning for Qwen2-VL with TRL. The code supports pointing out wound areas
# using a minimal example of "point-based" supervision, then in validation we can decode
# and optionally run SAM for segmentation metrics.

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from dataloader.dataset_sam4mllm import BaseSAM4MLLMDataset
from models.model import SAM4MLLMModel
from engine.trainer import Trainer
from util.utils import set_seed

def main(args):
    # 1. Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 2. Fix seed
    set_seed(config.get("seed", 42))

    # 3. Retrieve model paths
    sam_model_path = config["model"]["sam"]
    qwen_model_path = config["model"]["qwen"]
    print(f"[DEBUG] SAM model path: {sam_model_path}")
    print(f"[DEBUG] Qwen model path: {qwen_model_path}")

    # 4. Decide how to load data
    use_data_mode = config["dataset"]["use_data"]
    if use_data_mode == "woundsegmentation":
        train_data_list = None
        val_data_list = None
        print("[INFO] Using the woundsegmentation CSV data from root_dir.")
    else:
        # fallback dummy data
        print("[INFO] Using dummy data_list since use_data != woundsegmentation.")
        train_data_list = [
            {"image_path": "dummy1.png", "mask_path": "dummy1_mask.png", "conversation": "Segment the object."},
            {"image_path": "dummy2.png", "mask_path": "dummy2_mask.png", "conversation": "Segment the big region."}
        ]
        val_data_list = [
            {"image_path": "dummy3.png", "mask_path": "dummy3_mask.png", "conversation": "Find me the object."}
        ]

    # 5. Prepare Dataset / DataLoader
    train_dataset = BaseSAM4MLLMDataset(
        data_list=train_data_list,
        tokenizer=None,     # We do not use HF tokenizer here, we'll let SFT handle text
        transforms=None,
        max_len=config["train"]["max_len"],
        img_size=tuple(config["train"]["img_size"]),
        img_dir=config["train"]["img_dir"],
        system_prompt="You are a helpful segmentation assistant.",
        use_data=use_data_mode,
        root_dir=config["dataset"]["root_dir"],
        split=config["dataset"]["split"]
    )

    val_dataset = BaseSAM4MLLMDataset(
        data_list=val_data_list,
        tokenizer=None,
        transforms=None,
        max_len=config["train"]["max_len"],
        img_size=tuple(config["train"]["img_size"]),
        img_dir=config["train"]["img_dir"],
        system_prompt="You are a helpful segmentation assistant.",
        use_data=use_data_mode,
        root_dir=config["dataset"]["root_dir"],
        split=config["dataset"]["split"]
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

    # 6. Build the synergy backbone for Qwen + SAM
    from models.blocks.qwen_sam_backbone import QwenSamBackbone
    backbone = QwenSamBackbone(
        qwen_model_path=qwen_model_path,
        sam_model_path=sam_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        override_patch_size=16,           # or 32 if you prefer
        override_temporal_patch_size=1    # single-frame images
    )
    model = SAM4MLLMModel(backbone=backbone)

    # 7. Set up our trainer (which internally uses TRL’s SFT)
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

    # 8. Train
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config")
    args = parser.parse_args()
    main(args)
