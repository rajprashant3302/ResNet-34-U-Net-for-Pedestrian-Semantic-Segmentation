#### Main result (test split, mean ± std over 1 seeds, per-image average)

| Model | Aug. | Precision | Recall | F1 | IoU | Boundary-F1 |
|---|---|---|---|---|---|---|
| Baseline (no aug) | none | 0.8578 ± 0.0000 | 0.9396 ± 0.0000 | 0.8920 ± 0.0000 | 0.8110 ± 0.0000 | 0.7985 ± 0.0000 |
| Proposed (flip+zoom) | flip=True, zoom=0.2 | 0.8600 ± 0.0000 | 0.9336 ± 0.0000 | 0.8918 ± 0.0000 | 0.8089 ± 0.0000 | 0.7916 ± 0.0000 |

#### Guideline table (validation vs test, cost, size)

| Model | Val IoU (best epoch) | Test IoU | Pooled-pixel IoU | Train time / run (s) | Epochs run | Params (M) |
|---|---|---|---|---|---|---|
| Baseline (no aug) | 0.7877 ± 0.0000 | 0.8110 ± 0.0000 | 0.8085 ± 0.0000 | 823 | 15 | 24.44 |
| Proposed (flip+zoom) | 0.7788 ± 0.0000 | 0.8089 ± 0.0000 | 0.8080 ± 0.0000 | 760 | 15 | 24.44 |

#### Extras: test-time augmentation and validation-tuned threshold

| Model | Variant | Precision | Recall | F1 | IoU | Boundary-F1 |
|---|---|---|---|---|---|---|
| Baseline (no aug) | single pass, thr 0.5 | 0.8578 ± 0.0000 | 0.9396 ± 0.0000 | 0.8920 ± 0.0000 | 0.8110 ± 0.0000 | 0.7985 ± 0.0000 |
| Baseline (no aug) | + flip TTA | 0.8597 ± 0.0000 | 0.9464 ± 0.0000 | 0.8967 ± 0.0000 | 0.8183 ± 0.0000 | 0.8077 ± 0.0000 |
| Baseline (no aug) | + flip TTA + val-tuned thr | 0.8795 ± 0.0000 | 0.9290 ± 0.0000 | 0.8992 ± 0.0000 | 0.8222 ± 0.0000 | 0.8155 ± 0.0000 |
| Proposed (flip+zoom) | single pass, thr 0.5 | 0.8600 ± 0.0000 | 0.9336 ± 0.0000 | 0.8918 ± 0.0000 | 0.8089 ± 0.0000 | 0.7916 ± 0.0000 |
| Proposed (flip+zoom) | + flip TTA | 0.8598 ± 0.0000 | 0.9416 ± 0.0000 | 0.8950 ± 0.0000 | 0.8147 ± 0.0000 | 0.8050 ± 0.0000 |
| Proposed (flip+zoom) | + flip TTA + val-tuned thr | 0.8822 ± 0.0000 | 0.9223 ± 0.0000 | 0.8980 ± 0.0000 | 0.8192 ± 0.0000 | 0.8102 ± 0.0000 |

#### Paired significance (same test images, seed-averaged)

| Comparison | ΔIoU (mean over test images) | 95% bootstrap CI | Wilcoxon p | Seeds where better |
|---|---|---|---|---|
| Proposed (flip+zoom) vs Baseline (no aug) | -0.0021 | [-0.0125, +0.0099] | 0.247 | 0/1 |
