# Dataset card - Penn-Fudan Pedestrian Database
* **Source:** University of Pennsylvania, https://www.cis.upenn.edu/~jshi/ped_html/ (research use; cite Wang, Shi, Song, Shen, ACCV 2007).
* **Content:** 170 photographs of pedestrians in street/campus scenes, two photo sources (files named `PennPed*` and `FudanPed*`), 345 labelled pedestrians.
* **Unit of observation:** one photograph. **Target:** per-pixel person mask. The released masks hold one integer id per pedestrian (0 = background); this project merges ids > 0 into ONE foreground class (overlapping people are not separated).
* **Image sizes:** vary; all images and masks are resized to 256 × 256 (area interpolation; masks re-thresholded at 0.5).
* **Class balance:** background dominates (see `results/audit_summary.json` → `background_pixel_share`, produced by `python -m src.eda`).
* **Quality checks performed (`src/eda.py`):** unreadable files, missing masks, empty masks, exact duplicate images, mask value set after merging, size ranges, persons per image, foreground-fraction histogram, split balance.
* **Split:** 70/15/15, seed 42, stratified by (source × person-size tertile); indices stored in `results/splits.json`.
* **Known risks:** small size; similar scenes may fall in different splits (no group id); pedestrians are privacy-sensitive imagery.
