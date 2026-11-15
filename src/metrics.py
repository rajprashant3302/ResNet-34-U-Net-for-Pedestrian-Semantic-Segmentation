"""Precision / recall / F1 / IoU (+ boundary-F1) for binary masks, per image and pooled."""
import cv2
import numpy as np

EPS = 1e-7
KEYS = ("precision", "recall", "f1", "iou", "bf1")


def counts(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum((1, 2)).astype(np.float64)
    fp = (pred & ~gt).sum((1, 2)).astype(np.float64)
    fn = (~pred & gt).sum((1, 2)).astype(np.float64)
    return tp, fp, fn


def from_counts(tp, fp, fn):
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * tp / (2 * tp + fp + fn + EPS)
    union = tp + fp + fn
    iou = np.where(union == 0, 1.0, tp / (union + EPS))
    return precision, recall, f1, iou


def _edge(m):
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def boundary_f1(pred, gt, tol=2):
    """F1 between predicted and true outlines, with a +-tol pixel tolerance (sensitive to outline quality)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    out = []
    for p, g in zip(pred, gt):
        pe, ge = _edge(p), _edge(g)
        if pe.sum() == 0 and ge.sum() == 0:
            out.append(1.0)
        elif pe.sum() == 0 or ge.sum() == 0:
            out.append(0.0)
        else:
            gd = cv2.dilate(ge.astype(np.uint8), k) > 0
            pd = cv2.dilate(pe.astype(np.uint8), k) > 0
            pr, rc = (pe & gd).sum() / pe.sum(), (ge & pd).sum() / ge.sum()
            out.append(2 * pr * rc / (pr + rc + EPS))
    return np.array(out)


def evaluate_masks(probs, gt, thr=0.5):
    """probs (N,H,W) in [0,1], gt (N,H,W) binary.
    Returns {'per_image': {metric: list}, 'mean': {metric: float}, 'pooled': {metric: float}}"""
    pred = probs >= thr
    tp, fp, fn = counts(pred, gt)
    p, r, f, i = from_counts(tp, fp, fn)
    b = boundary_f1(pred, gt)
    per = {"precision": p, "recall": r, "f1": f, "iou": i, "bf1": b}
    pp, pr_, pf, pi = from_counts(tp.sum(), fp.sum(), fn.sum())
    return {"per_image": {k: v.tolist() for k, v in per.items()},
            "mean": {k: float(v.mean()) for k, v in per.items()},
            "pooled": {"precision": float(pp), "recall": float(pr_), "f1": float(pf), "iou": float(pi)}}
