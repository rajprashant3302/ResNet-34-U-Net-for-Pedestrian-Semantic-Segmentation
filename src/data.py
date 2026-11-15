"""Penn-Fudan loading, binary-mask merging, resizing, leak-free stratified splitting, Dataset."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .augment import apply_aug

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_pairs(root):
    root = Path(root)
    img_dir, mask_dir = root / "PNGImages", root / "PedMasks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {img_dir} and {mask_dir}. Run `python -m src.download_data` first "
            f"(or point data_root in the config to the extracted PennFudanPed folder).")
    pairs = []
    for ip in sorted(img_dir.glob("*.png")):
        mp = mask_dir / f"{ip.stem}_mask.png"
        if not mp.exists():
            raise FileNotFoundError(f"Missing mask for {ip.name}")
        pairs.append((ip, mp))
    if not pairs:
        raise RuntimeError(f"No images found in {img_dir}")
    return pairs


def load_dataset(root, size, cache_dir):
    """Read every image once, merge all person instances into ONE binary mask (mask>0),
    resize to size x size and cache as uint8 arrays."""
    cache = Path(cache_dir) / f"pennfudan_{size}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    pairs = find_pairs(root)
    imgs, masks, names, orig_hw, n_inst = [], [], [], [], []
    for ip, mp in pairs:
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unreadable image {ip}")
        rgb = bgr[:, :, ::-1]
        m = np.array(Image.open(mp))             # palette indices: 0 = background, 1..K = person instances
        if m.ndim == 3:
            m = m[..., 0]
        n_inst.append(len(np.unique(m[m > 0])))
        binary = (m > 0).astype(np.float32)      # merge overlapping people -> one foreground class
        h, w = binary.shape
        orig_hw.append((h, w))
        imgs.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA))
        masks.append((cv2.resize(binary, (size, size), interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8))
        names.append(ip.stem)
    out = {
        "images": np.stack(imgs).astype(np.uint8),
        "masks": np.stack(masks).astype(np.uint8),
        "names": np.array(names),
        "orig_hw": np.array(orig_hw),
        "n_instances": np.array(n_inst),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def source_of(name):
    return "Fudan" if name.lower().startswith("fudan") else "Penn"


def make_split(data, cfg_split):
    """Train/val/test split fixed once. Stratified by (image source x person-size tertile) so every
    split has a similar mix of scenes and person sizes. Label size statistics are used only to
    *design the split*, never to fit anything."""
    from sklearn.model_selection import train_test_split
    n = len(data["names"])
    fg = data["masks"].reshape(n, -1).mean(1)
    tert = np.digitize(fg, np.quantile(fg, [1 / 3, 2 / 3]))
    strata = np.array([f"{source_of(nm)}_{t}" for nm, t in zip(data["names"], tert)])
    idx = np.arange(n)
    val_frac, test_frac = cfg_split["val"], cfg_split["test"]
    seed = cfg_split["seed"]

    def _split(items, y, frac):
        try:
            return train_test_split(items, test_size=frac, random_state=seed, stratify=y)
        except ValueError:                      # a stratum is too small -> fall back to a plain split
            return train_test_split(items, test_size=frac, random_state=seed)

    rest, test = _split(idx, strata, test_frac)
    train, val = _split(rest, strata[rest], val_frac / (1 - test_frac))
    split = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    assert len(set(split["train"]) & set(split["val"])) == 0
    assert len(set(split["train"]) & set(split["test"])) == 0
    assert len(set(split["val"]) & set(split["test"])) == 0
    assert sum(len(v) for v in split.values()) == n
    return split


def to_tensor_batch(images_uint8):
    """(N,H,W,3) uint8 -> normalised float tensor (N,3,H,W). Fixed ImageNet statistics: nothing is fitted on data."""
    import torch
    x = (images_uint8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


class SegDataset:
    def __init__(self, images, masks, indices, aug=None, seed=0):
        self.X, self.Y, self.idx, self.aug = images, masks, np.asarray(indices), aug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        import torch
        j = self.idx[i]
        img, m = self.X[j], self.Y[j]
        if self.aug:
            img, m = apply_aug(img, m, self.aug, self.rng)
        x = to_tensor_batch(img[None])[0]
        y = torch.from_numpy(m.astype(np.float32))[None]
        return x, y, int(j)
