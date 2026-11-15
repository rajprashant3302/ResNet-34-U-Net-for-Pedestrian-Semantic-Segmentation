import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.augment import apply_aug
from src.data import make_split
from src.losses import BCEDiceLoss
from src.metrics import evaluate_masks
from src.model import build_model


def _fake(n=60):
    rng = np.random.default_rng(0)
    return {"names": np.array([("FudanPed" if i % 2 else "PennPed") + f"{i:05d}" for i in range(n)]),
            "masks": (rng.random((n, 16, 16)) < rng.random((n, 1, 1))).astype(np.uint8)}


def test_split_is_disjoint_and_complete():
    s = make_split(_fake(), {"train": .7, "val": .15, "test": .15, "seed": 1})
    allidx = np.concatenate(list(s.values()))
    assert len(allidx) == len(set(allidx)) == 60
    assert abs(len(s["test"]) - 9) <= 1


def test_metrics_known_values():
    gt = np.zeros((1, 4, 4)); gt[0, :2] = 1           # 8 true pixels
    pr = np.zeros((1, 4, 4)); pr[0, 1:3] = 1          # 8 predicted, 4 overlap
    m = evaluate_masks(pr, gt)["mean"]
    assert m["precision"] == pytest.approx(0.5, abs=1e-5) and m["recall"] == pytest.approx(0.5, abs=1e-5)
    assert m["iou"] == pytest.approx(4 / 12, abs=1e-5) and m["f1"] == pytest.approx(0.5, abs=1e-5)
    assert evaluate_masks(gt, gt)["mean"]["iou"] == pytest.approx(1.0, abs=1e-5)


def test_augmentation_keeps_mask_binary_and_aligned():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    m = np.zeros((64, 64), np.uint8); m[10:40, 20:30] = 1
    for a in ({"flip": True}, {"zoom": .2}, {"flip": True, "zoom": .2, "rotate": 10, "shift": .1, "color": .2}):
        i2, m2 = apply_aug(img, m, a, rng)
        assert i2.shape == img.shape and m2.shape == m.shape and set(np.unique(m2)) <= {0, 1}
    # flip is exact: flipping the flipped output reproduces the input when flip fired
    rng2 = np.random.default_rng(3)
    i2, m2 = apply_aug(img, m, {"flip": True}, rng2)
    assert np.array_equal(m2, m) or np.array_equal(m2[:, ::-1], m)


@pytest.mark.parametrize("cfg", [{"arch": "resnet_unet", "encoder": "resnet18", "pretrained": False},
                                 {"arch": "vanilla_unet", "base_channels": 8}])
def test_model_shapes(cfg):
    out = build_model(cfg)(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_loss_low_for_perfect_prediction():
    y = (torch.rand(2, 1, 16, 16) > .5).float()
    good = BCEDiceLoss()((y * 2 - 1) * 12, y)
    bad = BCEDiceLoss()(-(y * 2 - 1) * 12, y)
    assert good < 0.05 < bad
