"""Dataset audit + EDA (descriptive only; no modelling decision uses test data)."""
import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_dataset, make_split, source_of
from .utils import load_config, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.override)
    d = load_dataset(cfg["data_root"], cfg["img_size"], cfg["cache_dir"])
    split = make_split(d, cfg["split"])
    out = Path(cfg["out_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    n = len(d["names"])
    fg = d["masks"].reshape(n, -1).mean(1)
    src = np.array([source_of(x) for x in d["names"]])
    hashes = [hashlib.md5(x.tobytes()).hexdigest() for x in d["images"]]
    audit = {
        "n_images": n, "image_size_after_resize": cfg["img_size"],
        "n_by_source": {s: int((src == s).sum()) for s in np.unique(src)},
        "persons_per_image": {"min": int(d["n_instances"].min()), "max": int(d["n_instances"].max()),
                              "mean": float(d["n_instances"].mean()), "total": int(d["n_instances"].sum())},
        "foreground_fraction": {"min": float(fg.min()), "median": float(np.median(fg)), "max": float(fg.max()),
                                "mean": float(fg.mean())},
        "background_pixel_share": float(1 - fg.mean()),
        "images_with_empty_mask": int((fg == 0).sum()),
        "exact_duplicate_images": int(n - len(set(hashes))),
        "mask_values_after_merge": sorted(int(v) for v in np.unique(d["masks"])),
        "original_size_range_hw": {"min": d["orig_hw"].min(0).tolist(), "max": d["orig_hw"].max(0).tolist()},
        "split_sizes": {k: int(len(v)) for k, v in split.items()},
        "split_fg_mean": {k: float(fg[v].mean()) for k, v in split.items()},
        "split_source_counts": {k: {s: int((src[v] == s).sum()) for s in np.unique(src)} for k, v in split.items()},
        "missing_values": "not applicable (image/mask pairs; unreadable files raise an error)",
    }
    save_json(audit, Path(cfg["out_dir"]) / "audit_summary.json")
    print(audit)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    axs[0].hist(fg, bins=25, color="#2c7fb8"); axs[0].set_title("Person-pixel fraction per image\n(target imbalance)")
    axs[0].set_xlabel("fraction of pixels = person")
    axs[1].hist(d["n_instances"], bins=np.arange(0.5, d["n_instances"].max() + 1.5), color="#7fcdbb")
    axs[1].set_title("Persons per image (merged to 1 mask)"); axs[1].set_xlabel("# persons")
    axs[2].scatter(d["orig_hw"][:, 1], d["orig_hw"][:, 0], c=(src == "Fudan"), cmap="coolwarm", s=12)
    axs[2].set_title("Original image sizes (colour = source)"); axs[2].set_xlabel("width"); axs[2].set_ylabel("height")
    fig.tight_layout(); fig.savefig(out / "eda_distributions.png", dpi=150); plt.close(fig)
    rng = np.random.default_rng(0)
    pick = rng.choice(n, 6, replace=False)
    fig, axs = plt.subplots(2, 6, figsize=(14, 5))
    for j, i in enumerate(pick):
        axs[0, j].imshow(d["images"][i]); axs[0, j].set_title(d["names"][i], fontsize=7)
        axs[1, j].imshow(d["masks"][i], cmap="gray"); axs[0, j].axis("off"); axs[1, j].axis("off")
    fig.suptitle("Samples after 256x256 resize: image (top) and merged binary person mask (bottom)")
    fig.tight_layout(); fig.savefig(out / "eda_samples.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
