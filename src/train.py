"""Train ONE (experiment, seed) run. Experiments differ only in the training-augmentation settings."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SegDataset, load_dataset, make_split, to_tensor_batch
from .losses import build_loss
from .metrics import counts, from_counts
from .model import build_model, count_params
from .utils import load_config, save_json, seed_all


@torch.no_grad()
def predict_probs(model, images, device, bs=16, tta=False):
    """images: (N,H,W,3) uint8 -> probabilities (N,H,W). tta=True averages with the horizontally flipped pass."""
    model.eval()
    out = []
    for i in range(0, len(images), bs):
        x = to_tensor_batch(images[i:i + bs]).to(device)
        p = torch.sigmoid(model(x).float())
        if tta:
            p = (p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1)) / 2
        out.append(p[:, 0].cpu().numpy())
    return np.concatenate(out)


def run_dir_for(cfg, exp, seed):
    return Path(cfg["out_dir"]) / "runs" / f"{exp}_seed{seed}"


def run_train(cfg, exp, seed, data, split, device, force=False):
    run_dir = run_dir_for(cfg, exp, seed)
    if (run_dir / "train_summary.json").exists() and not force:
        print(f"[skip] {exp} seed{seed} already trained")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    tc = cfg["train"]
    seed_all(seed)                                    # same seed => same init & batch order for every experiment
    aug = cfg["experiments"][exp]
    tr = SegDataset(data["images"], data["masks"], split["train"], aug=aug, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(tr, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                        num_workers=tc.get("num_workers", 0), generator=gen)
    model = build_model(cfg["model"]).to(device)
    criterion = build_loss(cfg["loss"])
    enc, dec = model.encoder_parameters(), model.decoder_parameters()
    groups = [{"params": dec, "lr": tc["lr"]}]
    if enc:
        groups.append({"params": enc, "lr": tc["lr"] * tc.get("encoder_lr_mult", 1.0)})
    opt = torch.optim.AdamW(groups, weight_decay=tc["weight_decay"])
    steps_per_epoch = len(loader)
    total, warm = tc["epochs"] * steps_per_epoch, tc.get("warmup_epochs", 0) * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = bool(tc.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    val_x = data["images"][split["val"]]
    val_y = data["masks"][split["val"]]
    val_t = torch.from_numpy(val_y.astype(np.float32))[:, None]
    hist, best, bad, t0 = [], {"val_iou": -1.0}, 0, time.time()
    for ep in range(1, tc["epochs"] + 1):
        model.train()
        run_loss = 0.0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
        probs = predict_probs(model, val_x, device)
        tp, fp, fn = counts(probs >= 0.5, val_y)
        val_iou = float(from_counts(tp, fp, fn)[3].mean())
        with torch.no_grad():
            val_loss = float(criterion(torch.logit(torch.from_numpy(probs).clamp(1e-6, 1 - 1e-6))[:, None], val_t))
        hist.append({"epoch": ep, "train_loss": run_loss / steps_per_epoch, "val_loss": val_loss, "val_iou": val_iou})
        if val_iou > best["val_iou"]:
            best, bad = {"val_iou": val_iou, "epoch": ep}, 0
            torch.save({"model": model.state_dict(), "epoch": ep}, run_dir / "best.pt")
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  [{exp} s{seed}] ep {ep:3d} train {hist[-1]['train_loss']:.4f} val_loss {val_loss:.4f} "
                  f"val_IoU {val_iou:.4f} (best {best['val_iou']:.4f} @ {best['epoch']})")
        if bad >= tc.get("patience", 10**9):
            print(f"  early stop at epoch {ep}")
            break
    summary = {"exp": exp, "seed": seed, "aug": aug, "best_epoch": best["epoch"], "best_val_iou": best["val_iou"],
               "epochs_run": len(hist), "train_time_s": time.time() - t0, "params": count_params(model),
               "lr": tc["lr"], "batch_size": tc["batch_size"]}
    save_json(hist, run_dir / "history.json")
    save_json(summary, run_dir / "train_summary.json")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.override)
    data = load_dataset(cfg["data_root"], cfg["img_size"], cfg["cache_dir"])
    split = make_split(data, cfg["split"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_train(cfg, a.exp, a.seed, data, split, device, a.force)


if __name__ == "__main__":
    main()
