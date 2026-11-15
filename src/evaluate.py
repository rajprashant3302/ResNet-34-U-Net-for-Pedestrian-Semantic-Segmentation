"""Evaluate a trained run ONCE on the held-out test split (+ val for threshold selection)."""
import numpy as np
import torch

from .metrics import evaluate_masks
from .model import build_model
from .train import predict_probs
from .utils import save_json

THR_GRID = np.round(np.arange(0.30, 0.701, 0.05), 2)


def evaluate_run(cfg, run_dir, data, split, device):
    ck = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
    vx, vy = data["images"][split["val"]], data["masks"][split["val"]]
    tx, ty = data["images"][split["test"]], data["masks"][split["test"]]

    p_val_tta = predict_probs(model, vx, device, tta=True)
    thr = max(THR_GRID, key=lambda t: evaluate_masks(p_val_tta, vy, t)["mean"]["iou"])   # chosen on VAL only
    p_test = predict_probs(model, tx, device, tta=False)
    p_test_tta = predict_probs(model, tx, device, tta=True)
    p_val = predict_probs(model, vx, device, tta=False)

    out = {
        "test_names": [str(n) for n in data["names"][split["test"]]],
        "variants": {
            "plain": {"thr": 0.5, **evaluate_masks(p_test, ty, 0.5)},                    # primary (required) result
            "tta": {"thr": 0.5, **evaluate_masks(p_test_tta, ty, 0.5)},                  # + flip test-time augmentation
            "tta_thr": {"thr": float(thr), **evaluate_masks(p_test_tta, ty, float(thr))},  # + val-tuned threshold
        },
        "val_plain": evaluate_masks(p_val, vy, 0.5)["mean"],
    }
    save_json(out, run_dir / "metrics.json")
    np.save(run_dir / "test_probs.npy", p_test.astype(np.float16))
    return out
