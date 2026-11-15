# Effect of Flip and Zoom Augmentation on U-Net Person Segmentation in Penn-Fudan Images
*Skeleton following the IMPACT Lab guidelines (max 8 pages). Fill every `[ ]` AFTER running the experiments; never write a number you did not measure.*

## 1. Abstract and contributions
[ ] 4-5 sentences: problem, method, main measured result (IoU baseline vs proposed, mean ± std, seeds), honest limitation.
Contributions: (i) controlled comparison of no-augmentation vs flip+zoom with identical split/loss/seeds; (ii) ablation (flip-only, zoom-only);
(iii) paired statistics and error analysis; (iv) fully reproducible pipeline.

## 2. Introduction and problem statement
Given an RGB street/campus photograph resized to 256×256, predict a binary mask of all persons (overlapping people merged), for the Penn-Fudan
pedestrian set, measured by precision, recall, F1 and IoU on held-out photographs. Hypothesis: flip + small zoom improves test scores of the same U-Net.
[ ] motivation (tiny dataset, 170 photos; scale/orientation variety).

## 3. Related work and gap
[ ] Comparison table (Study | Dataset | Method | Evaluation | Gap). Start from: Ronneberger 2015 (U-Net); Iglovikov 2018 (pretrained-encoder U-Net);
Milletari 2016 (Dice); Shorten & Khoshgoftaar 2019 (augmentation survey); Lau et al. 2020 (ResNet34-U-Net, cracks); Penn-Fudan IoU reports (e.g. Kim & Lee 2019, 80.91 % IoU with extra COCO training data).
Gap: effect of simple geometric augmentation on a *very small* pedestrian set under a controlled, multi-seed protocol.

## 4. Dataset and audit
[ ] Use `results/audit_summary.json` and `results/figures/eda_*.png` (counts, person-pixel fraction, persons per image, sizes, duplicates, split balance). Cite `docs/dataset_card.md`.

## 5. Split protocol and methodology
70/15/15 stratified (source × person-size), fixed seed 42, indices in `results/splits.json`. No statistic fitted on any split (fixed ImageNet normalisation).
Model: ResNet34-encoder U-Net + full-resolution skip (diagram `docs/architecture.png`). Loss: BCE + Dice (formula in README §2). Augmentation: see `src/augment.py`.
Algorithm: README §2.

## 6. Experimental setup and baselines
[ ] Table of hyper-parameters (`results/config_used.yaml`), environment (`results/environment.json`), seeds, hardware, run time. Baseline = same U-Net without augmentation.

## 7. Results
[ ] Paste `results/results_table.md` and the figures `metrics_bar.png`, `ablation_iou.png`, `training_curves.png`. One-paragraph interpretation per figure (state whether CI/Wilcoxon support the difference).

## 8. Error analysis and ablations
[ ] `iou_by_size.png`, `per_image_scatter.png`, `qualitative.png` (green/red/blue). Discuss where flip/zoom helps or fails (small persons? occlusion? clutter?).

## 9. Limitations, ethics and conclusion
Small test set (~26 images); possible scene overlap across splits; ImageNet-pretrained encoder is external data; privacy of pedestrian imagery. [ ] conclusion strictly matching the evidence.

## 10. References and reproducibility notes
[ ] Full references (README §8). Commands: README §3-4. Fresh clone + `python -m src.run_experiments` regenerates all tables/figures.
