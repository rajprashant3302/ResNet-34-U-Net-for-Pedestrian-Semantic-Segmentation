"""Aggregate all runs: tables, paired statistics, figures, experiment log, README injection."""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import load_json, save_json

KEYS = ("precision", "recall", "f1", "iou", "bf1")
LABEL = {"precision": "Precision", "recall": "Recall", "f1": "F1", "iou": "IoU", "bf1": "Boundary-F1"}
NICE = {"baseline": "Baseline (no aug)", "proposed": "Proposed (flip+zoom)", "flip_only": "Flip only",
        "zoom_only": "Zoom only", "proposed_plus": "Proposed+ (extended aug)"}


def _runs(cfg, exp):
    out = []
    for d in sorted((Path(cfg["out_dir"]) / "runs").glob(f"{exp}_seed*")):
        if (d / "metrics.json").exists() and (d / "train_summary.json").exists():
            out.append((d, load_json(d / "metrics.json"), load_json(d / "train_summary.json")))
    return out


def _ms(vals):
    v = np.array(vals, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def _fmt(vals):
    m, s = _ms(vals)
    return f"{m:.4f} ± {s:.4f}"


def _perimg_iou(runs, variant="plain"):
    return np.mean([r[1]["variants"][variant]["per_image"]["iou"] for r in runs], axis=0)


def analyze_all(cfg, data, split):
    out = Path(cfg["out_dir"])
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    exps = [e for e in cfg["experiments"] if _runs(cfg, e)]
    R = {e: _runs(cfg, e) for e in exps}
    if not exps:
        print("no finished runs to analyse")
        return
    nseeds = {e: len(R[e]) for e in exps}

    # ---------- tables ----------
    lines = ["| Model | Aug. | Precision | Recall | F1 | IoU | Boundary-F1 |", "|---|---|---|---|---|---|---|"]
    for e in exps:
        aug = ", ".join(f"{k}={v}" for k, v in cfg["experiments"][e].items() if v) or "none"
        cells = [_fmt([r[1]["variants"]["plain"]["mean"][k] for r in R[e]]) for k in KEYS]
        lines.append(f"| {NICE.get(e, e)} | {aug} | " + " | ".join(cells) + " |")
    main_tbl = "\n".join(lines)

    lines = ["| Model | Variant | Precision | Recall | F1 | IoU | Boundary-F1 |", "|---|---|---|---|---|---|---|"]
    for e in exps:
        for v, vn in (("plain", "single pass, thr 0.5"), ("tta", "+ flip TTA"), ("tta_thr", "+ flip TTA + val-tuned thr")):
            cells = [_fmt([r[1]["variants"][v]["mean"][k] for r in R[e]]) for k in KEYS]
            lines.append(f"| {NICE.get(e, e)} | {vn} | " + " | ".join(cells) + " |")
    tta_tbl = "\n".join(lines)

    lines = ["| Model | Val IoU (best epoch) | Test IoU | Pooled-pixel IoU | Train time / run (s) | Epochs run | Params (M) |",
             "|---|---|---|---|---|---|---|"]
    for e in exps:
        lines.append(f"| {NICE.get(e, e)} | {_fmt([r[2]['best_val_iou'] for r in R[e]])} | "
                     f"{_fmt([r[1]['variants']['plain']['mean']['iou'] for r in R[e]])} | "
                     f"{_fmt([r[1]['variants']['plain']['pooled']['iou'] for r in R[e]])} | "
                     f"{np.mean([r[2]['train_time_s'] for r in R[e]]):.0f} | "
                     f"{np.mean([r[2]['epochs_run'] for r in R[e]]):.0f} | {R[e][0][2]['params'] / 1e6:.2f} |")
    guide_tbl = "\n".join(lines)

    # ---------- paired statistics (baseline vs each other experiment) ----------
    stats, stat_lines = {}, ["| Comparison | ΔIoU (mean over test images) | 95% bootstrap CI | Wilcoxon p | Seeds where better |",
                             "|---|---|---|---|---|"]
    base = cfg["primary_comparison"][0]
    rng = np.random.default_rng(0)
    if base in R:
        b = _perimg_iou(R[base])
        for e in exps:
            if e == base:
                continue
            p_ = _perimg_iou(R[e])
            d = p_ - b
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            try:
                from scipy.stats import wilcoxon
                pval = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            except Exception:
                pval = float("nan")
            wins = sum(R[e][i][1]["variants"]["plain"]["mean"]["iou"] > R[base][i][1]["variants"]["plain"]["mean"]["iou"]
                       for i in range(min(len(R[e]), len(R[base]))))
            stats[e] = {"delta_iou": float(d.mean()), "ci95": [float(lo), float(hi)], "wilcoxon_p": pval,
                        "seeds_better": int(wins), "n_seeds": min(len(R[e]), len(R[base])), "n_test_images": int(len(d))}
            stat_lines.append(f"| {NICE.get(e, e)} vs {NICE.get(base, base)} | {d.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                              f"{pval:.3f} | {wins}/{stats[e]['n_seeds']} |")
    stats_tbl = "\n".join(stat_lines)
    save_json(stats, out / "paired_stats.json")

    tables = (f"#### Main result (test split, mean ± std over {max(nseeds.values())} seeds, per-image average)\n\n{main_tbl}\n\n"
              f"#### Guideline table (validation vs test, cost, size)\n\n{guide_tbl}\n\n"
              f"#### Extras: test-time augmentation and validation-tuned threshold\n\n{tta_tbl}\n\n"
              f"#### Paired significance (same test images, seed-averaged)\n\n{stats_tbl}\n")
    (out / "results_table.md").write_text(tables)

    # ---------- experiment log ----------
    with open(out / "experiment_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "split_seed", "model_seed", "augmentation", "lr", "batch", "best_epoch",
                    "val_iou", "test_iou", "runtime_s"])
        for e in exps:
            for r in R[e]:
                w.writerow([f"{e}_seed{r[2]['seed']}", cfg["split"]["seed"], r[2]["seed"], r[2]["aug"], r[2]["lr"],
                            r[2]["batch_size"], r[2]["best_epoch"], f"{r[2]['best_val_iou']:.4f}",
                            f"{r[1]['variants']['plain']['mean']['iou']:.4f}", f"{r[2]['train_time_s']:.0f}"])

    # ---------- figures ----------
    pc = [e for e in cfg["primary_comparison"] if e in R]
    colors = {"baseline": "#7f8c8d", "proposed": "#2c7fb8", "flip_only": "#a1dab4", "zoom_only": "#fec44f",
              "proposed_plus": "#d95f0e"}
    # 1 bar chart of the four required metrics
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ks = ("precision", "recall", "f1", "iou")
    w = 0.8 / len(pc)
    for i, e in enumerate(pc):
        ms = [_ms([r[1]["variants"]["plain"]["mean"][k] for r in R[e]]) for k in ks]
        ax.bar(np.arange(4) + i * w, [m[0] for m in ms], w, yerr=[m[1] for m in ms], capsize=3,
               label=NICE.get(e, e), color=colors.get(e))
    ax.set_xticks(np.arange(4) + w * (len(pc) - 1) / 2)
    ax.set_xticklabels([LABEL[k] for k in ks])
    lo = min(m for e in pc for k in ks for m in [np.mean([r[1]["variants"]["plain"]["mean"][k] for r in R[e]])])
    ax.set_ylim(max(0, lo - 0.1), 1.0)
    ax.set_ylabel("score (test split)")
    ax.set_title("Baseline vs proposed - test metrics (mean ± std over seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)

    # 2 ablation IoU with per-seed dots
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, e in enumerate(exps):
        v = [r[1]["variants"]["plain"]["mean"]["iou"] for r in R[e]]
        ax.bar(i, np.mean(v), color=colors.get(e), alpha=0.85)
        ax.scatter([i] * len(v), v, color="k", s=14, zorder=3)
    ax.set_xticks(range(len(exps)))
    ax.set_xticklabels([NICE.get(e, e).replace(" (", "\n(") for e in exps], fontsize=8)
    allv = [r[1]["variants"]["plain"]["mean"]["iou"] for e in exps for r in R[e]]
    ax.set_ylim(max(0, min(allv) - 0.08), min(1.0, max(allv) + 0.04))
    ax.set_ylabel("test IoU (dots = seeds)")
    ax.set_title("Ablation: which augmentation matters?")
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation_iou.png", dpi=150)
    plt.close(fig)

    # 3 training curves
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for e in pc:
        for j, r in enumerate(R[e]):
            h = load_json(r[0] / "history.json")
            ep = [x["epoch"] for x in h]
            axs[0].plot(ep, [x["val_iou"] for x in h], color=colors.get(e), alpha=0.7, label=NICE.get(e, e) if j == 0 else None)
            axs[1].plot(ep, [x["train_loss"] for x in h], color=colors.get(e), alpha=0.7, label=NICE.get(e, e) + " train" if j == 0 else None)
            axs[1].plot(ep, [x["val_loss"] for x in h], color=colors.get(e), alpha=0.4, ls="--")
    axs[0].set_title("Validation IoU"); axs[1].set_title("Loss (solid: train, dashed: val)")
    for a in axs:
        a.set_xlabel("epoch"); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    if len(pc) == 2:
        b, p_ = _perimg_iou(R[pc[0]]), _perimg_iou(R[pc[1]])
        names = R[pc[0]][0][1]["test_names"]
        ty = data["masks"][split["test"]]
        fgr = ty.reshape(len(ty), -1).mean(1)
        # 4 scatter
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.scatter(b, p_, c=fgr, cmap="viridis", s=36)
        lim = [min(b.min(), p_.min()) - 0.03, 1.0]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"IoU - {NICE[pc[0]]}"); ax.set_ylabel(f"IoU - {NICE[pc[1]]}")
        ax.set_title("Per test image (above line = proposed better)\ncolour = person area fraction", fontsize=9)
        fig.tight_layout(); fig.savefig(fig_dir / "per_image_scatter.png", dpi=150); plt.close(fig)
        # 5 IoU by person size
        bins = np.digitize(fgr, np.quantile(fgr, [1 / 3, 2 / 3]))
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for i, (e, arr) in enumerate(((pc[0], b), (pc[1], p_))):
            ax.bar(np.arange(3) + i * 0.38, [arr[bins == t].mean() if np.any(bins == t) else 0 for t in range(3)], 0.38,
                   label=NICE[e], color=colors.get(e))
        ax.set_xticks(np.arange(3) + 0.19); ax.set_xticklabels(["small persons", "medium", "large"])
        ax.set_ylabel("mean test IoU"); ax.set_title("Error analysis by person size (tertiles of mask area)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(fig_dir / "iou_by_size.png", dpi=150); plt.close(fig)
        # 6 qualitative gallery: 3 biggest gains + 2 worst cases for the proposed model
        pb = np.load(R[pc[0]][0][0] / "test_probs.npy").astype(np.float32)
        pp = np.load(R[pc[1]][0][0] / "test_probs.npy").astype(np.float32)
        d = p_ - b
        order = list(np.argsort(-d)[:3]) + [i for i in np.argsort(p_)[:2] if i not in np.argsort(-d)[:3]]
        tx = data["images"][split["test"]]

        def overlay(img, pred, gt):
            o = img.astype(np.float32).copy()
            tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
            for m, c in ((tp, (0, 200, 0)), (fp, (230, 0, 0)), (fn, (0, 90, 255))):
                o[m] = 0.45 * o[m] + 0.55 * np.array(c)
            return o.astype(np.uint8)

        fig, axs = plt.subplots(len(order), 4, figsize=(11, 2.8 * len(order)))
        axs = np.atleast_2d(axs)
        for r_, i in enumerate(order):
            gt = ty[i].astype(bool)
            tag = "largest gain" if i in np.argsort(-d)[:3] else "worst case (proposed)"
            axs[r_, 0].imshow(tx[i]); axs[r_, 0].set_title(f"{names[i]}\n[{tag}]", fontsize=8)
            axs[r_, 1].imshow(gt, cmap="gray"); axs[r_, 1].set_title("ground truth", fontsize=8)
            axs[r_, 2].imshow(overlay(tx[i], pb[i] >= .5, gt)); axs[r_, 2].set_title(f"{NICE[pc[0]]}  IoU {b[i]:.3f}", fontsize=8)
            axs[r_, 3].imshow(overlay(tx[i], pp[i] >= .5, gt)); axs[r_, 3].set_title(f"{NICE[pc[1]]}  IoU {p_[i]:.3f}", fontsize=8)
            for a in axs[r_]:
                a.axis("off")
        fig.suptitle("Green = correct person pixels, Red = false positive, Blue = missed (false negative)", fontsize=9)
        fig.tight_layout(); fig.savefig(fig_dir / "qualitative.png", dpi=130); plt.close(fig)

    print("\n" + tables)
    if cfg.get("update_readme", False):
        inject_readme(tables, [p.name for p in sorted(fig_dir.glob("*.png"))], out)


def inject_readme(tables, figs, out):
    readme = Path("README.md")
    if not readme.exists():
        return
    cap = {"metrics_bar.png": "Required metrics, baseline vs proposed", "ablation_iou.png": "Ablation (flip / zoom / both / extended)",
           "training_curves.png": "Validation IoU and loss during training", "per_image_scatter.png": "Per-image IoU comparison",
           "iou_by_size.png": "IoU by person size", "qualitative.png": "Test photographs: biggest gains and worst cases"}
    body = tables + "\n" + "\n".join(f"**{cap.get(f, f)}**\n\n![{f}]({out}/figures/{f})\n" for f in figs if f in cap)
    txt = readme.read_text()
    new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                 lambda m: "<!-- RESULTS:START -->\n" + body + "\n<!-- RESULTS:END -->", txt, flags=re.S)
    readme.write_text(new)
    print("README.md results section updated")
