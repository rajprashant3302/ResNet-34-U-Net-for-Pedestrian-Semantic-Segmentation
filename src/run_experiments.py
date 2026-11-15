"""Run every experiment x seed (train + evaluate), then build tables / figures / README results."""
import argparse
import shutil

import torch

from .analyze import analyze_all
from .data import load_dataset, make_split
from .evaluate import evaluate_run
from .train import run_dir_for, run_train
from .utils import env_info, load_config, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="e.g. train.epochs=5 model.encoder=resnet18")
    ap.add_argument("--experiments", nargs="*", default=None, help="subset of experiment names")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="retrain even if a finished run exists")
    ap.add_argument("--skip-analysis", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config, a.override)
    exps = a.experiments or list(cfg["experiments"])
    seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | experiments: {exps} | seeds: {seeds}")

    data = load_dataset(cfg["data_root"], cfg["img_size"], cfg["cache_dir"])
    split = make_split(data, cfg["split"])
    print({k: len(v) for k, v in split.items()}, "images in train/val/test")
    out = cfg["out_dir"]
    save_json({k: [str(data["names"][i]) for i in v] for k, v in split.items()}, f"{out}/splits.json")
    save_json(env_info(), f"{out}/environment.json")
    shutil.copy(a.config, f"{out}/config_used.yaml")

    for exp in exps:
        for seed in seeds:
            rd = run_dir_for(cfg, exp, seed)
            run_train(cfg, exp, seed, data, split, device, a.force)
            if a.force or not (rd / "metrics.json").exists():
                r = evaluate_run(cfg, rd, data, split, device)
                m = r["variants"]["plain"]["mean"]
                print(f"[test] {exp} seed{seed}: IoU {m['iou']:.4f} F1 {m['f1']:.4f} "
                      f"P {m['precision']:.4f} R {m['recall']:.4f}")
    if not a.skip_analysis:
        analyze_all(cfg, data, split)


if __name__ == "__main__":
    main()
