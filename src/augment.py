"""Joint image/mask augmentation. Geometry is applied identically to image and mask
(bilinear for the image, nearest-neighbour for the mask), so labels stay exactly binary."""
import cv2
import numpy as np


def apply_aug(img, mask, a, rng):
    """a: dict with optional keys
         flip   (bool)  horizontal flip with p=0.5
         zoom   (float) scale ~ U(1-zoom, 1+zoom) about the image centre   (>1 zooms in)
         rotate (float) rotation ~ U(-rotate, +rotate) degrees              [extension]
         shift  (float) translation up to +-shift*size                      [extension]
         color  (float) brightness/contrast/saturation jitter strength      [extension]
    """
    h, w = mask.shape
    if a.get("flip") and rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    z, r, t = a.get("zoom", 0.0), a.get("rotate", 0.0), a.get("shift", 0.0)
    if z > 0 or r > 0 or t > 0:
        s = rng.uniform(1 - z, 1 + z) if z > 0 else 1.0
        ang = rng.uniform(-r, r) if r > 0 else 0.0
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
        if t > 0:
            M[0, 2] += rng.uniform(-t, t) * w
            M[1, 2] += rng.uniform(-t, t) * h
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    c = a.get("color", 0.0)
    if c > 0:
        f = img.astype(np.float32)
        f *= rng.uniform(1 - c, 1 + c)                                    # brightness
        mean = f.mean()
        f = (f - mean) * rng.uniform(1 - c, 1 + c) + mean                 # contrast
        gray = f.mean(axis=2, keepdims=True)
        f = gray + (f - gray) * rng.uniform(1 - c, 1 + c)                 # saturation
        img = np.clip(f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)
