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
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in dataloader:
                # We do standard cross-entropy on entire batch if needed
                # but code is replaced by your new SFT logic.
                # We keep a simple placeholder here to avoid refactoring your calls.
                pass
        mean_loss = 0.0
        return {"loss": mean_loss}
