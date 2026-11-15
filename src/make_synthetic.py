"""Create a SYNTHETIC Penn-Fudan-shaped dataset (random 'pedestrian' silhouettes on textured backgrounds).
Used ONLY to smoke-test the pipeline when the real data is unavailable. Results on it mean nothing."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def make(n, out, seed=0):
    rng = np.random.default_rng(seed)
    out = Path(out)
    (out / "PNGImages").mkdir(parents=True, exist_ok=True)
    (out / "PedMasks").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        h, w = int(rng.integers(300, 480)), int(rng.integers(380, 560))
        bg = cv2.GaussianBlur(rng.integers(60, 200, (h // 8, w // 8, 3), dtype=np.uint8), (3, 3), 0)
        img = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img += rng.normal(0, 6, img.shape)
        inst = np.zeros((h, w), np.uint8)
        for k in range(1, int(rng.integers(1, 4)) + 1):
            ph = int(rng.uniform(0.35, 0.8) * h); pw = max(14, int(ph * rng.uniform(0.22, 0.32)))
            cx, top = int(rng.integers(pw, w - pw)), int(rng.integers(0, h - ph))
            m = np.zeros((h, w), np.uint8)
            cv2.ellipse(m, (cx, top + ph // 10), (pw // 3, ph // 10), 0, 0, 360, 1, -1)                      # head
            cv2.rectangle(m, (cx - pw // 2, top + ph // 5), (cx + pw // 2, top + int(ph * .6)), 1, -1)        # torso
            cv2.rectangle(m, (cx - pw // 2, top + int(ph * .6)), (cx - 2, top + ph), 1, -1)                   # legs
            cv2.rectangle(m, (cx + 2, top + int(ph * .6)), (cx + pw // 2, top + ph), 1, -1)
            col = rng.integers(0, 255, 3).astype(np.float32)
            img[m > 0] = 0.15 * img[m > 0] + 0.85 * col
            inst[m > 0] = k
        name = ("FudanPed" if i % 2 else "PennPed") + f"{i:05d}"
        cv2.imwrite(str(out / "PNGImages" / f"{name}.png"), np.clip(img, 0, 255).astype(np.uint8)[:, :, ::-1])
        pm = Image.fromarray(inst, mode="L").convert("P")
        pm.putpalette([c for k in range(256) for c in (k * 40 % 256, k * 90 % 256, k * 160 % 256)])
        Image.fromarray(inst).save(out / "PedMasks" / f"{name}_mask.png")
    print(f"wrote {n} synthetic images to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="data/synth/PennFudanPed")
    a = ap.parse_args()
    make(a.n, a.out)
