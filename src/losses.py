"""Compound loss: BCE (per-pixel) + soft-Dice (overlap of the whole mask).
   L = w_bce * BCE(p, y)  +  w_dice * (1 - (2*sum(p*y) + s) / (sum(p) + sum(y) + s))     (Dice term per image, then averaged)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.wb, self.wd, self.s = bce_weight, dice_weight, smooth

    def forward(self, logits, target):
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        denom = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
        dice = 1.0 - (2.0 * inter + self.s) / (denom + self.s)
        return self.wb * bce + self.wd * dice.mean()


def build_loss(cfg):
    if cfg["name"] == "bce_dice":
        return BCEDiceLoss(cfg.get("bce_weight", 1.0), cfg.get("dice_weight", 1.0))
    if cfg["name"] == "bce":
        return BCEDiceLoss(1.0, 0.0)
    raise ValueError(cfg["name"])
