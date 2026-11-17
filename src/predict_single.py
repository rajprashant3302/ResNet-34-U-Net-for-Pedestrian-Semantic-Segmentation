"""Run inference on a single image and create an overlay mask visualization.
Usage:
    .venv/bin/python -m src.predict_single --image <path_to_image> [--checkpoint <path>] [--output <path>]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from .data import to_tensor_batch
from .model import build_model
from .train import predict_probs


def run_inference(image_path, checkpoint_path=None, output_path=None, tta=True, thr=0.5):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load configuration
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    if checkpoint_path is None:
        checkpoint_path = Path("results/runs/baseline_seed0/best.pt")
        if not checkpoint_path.exists():
            checkpoint_path = Path("results/runs/proposed_seed0/best.pt")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {checkpoint_path} on {device}...")

    # Build and load model
    model = build_model(cfg["model"]).to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    # Read image
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image from {image_path}")
    orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_rgb.shape[:2]

    # Resize to 256x256 for model
    img_256 = cv2.resize(orig_rgb, (256, 256), interpolation=cv2.INTER_AREA)

    # Predict
    prob = predict_probs(model, img_256[None], device, tta=tta)[0]  # shape (256, 256)
    mask_256 = (prob >= thr).astype(np.uint8)

    # Upsample probability / mask back to original resolution
    prob_orig = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask_orig = (prob_orig >= thr).astype(np.uint8)

    # Create visual overlay: semi-transparent green mask over person + glowing contours
    overlay = orig_rgb.copy()
    color_mask = np.zeros_like(orig_rgb)
    color_mask[mask_orig > 0] = [0, 255, 128]  # vibrant emerald green

    alpha = 0.45
    overlay = np.where(mask_orig[..., None] > 0,
                       cv2.addWeighted(orig_rgb, 1 - alpha, color_mask, alpha, 0),
                       orig_rgb)

    # Draw sharp contours
    contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)  # white contour line

    # Create 3-panel figure: [Original Image, Predicted Binary Mask, Overlay]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(orig_rgb)
    axes[0].set_title(f"Original Image ({orig_w}x{orig_h})", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mask_orig, cmap="gray")
    axes[1].set_title("Predicted Mask (Binary)", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Mask Overlaid on Image", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    if output_path is None:
        out_dir = Path("results/test_predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{image_path.stem}_overlay.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Also save the raw standalone overlay image for direct viewing
    standalone_path = output_path.parent / f"{image_path.stem}_overlay_only.png"
    cv2.imwrite(str(standalone_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Visualization saved to: {output_path}")
    print(f"Direct overlay saved to: {standalone_path}")
    return output_path, standalone_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="test_image.jpg", help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--thr", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    run_inference(args.image, args.checkpoint, args.output, tta=not args.no_tta, thr=args.thr)
