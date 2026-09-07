"""
export_polypgen_external.py
===========================
Export a sampled PolypGen positive set into the NumPy format used by the prompt
quality experiments.

The exported pred_heatmap is an inference-only saliency/center prior derived
from image appearance and geometry, not from the ground-truth mask.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def center_gaussian(shape: tuple[int, int], sigma: float = 0.35) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist2 = ((yy - cy) / max(1, h)) ** 2 + ((xx - cx) / max(1, w)) ** 2
    return np.exp(-dist2 / (2 * sigma ** 2)).astype(np.float32)


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def saliency_heatmap(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    dog = cv2.absdiff(gray, blur)
    sat = lab[..., 1]
    grad_y, grad_x = np.gradient(gray / 255.0)
    grad = np.hypot(grad_y, grad_x)
    heatmap = 0.35 * normalize(dog) + 0.25 * normalize(sat) + 0.20 * normalize(grad) + 0.20 * center_gaussian(gray.shape)
    return normalize(cv2.GaussianBlur(heatmap, (0, 0), 2.0))


def center_name_group(name: str) -> str:
    match = re.match(r"(C\d+|seq\d+)", name)
    return match.group(1) if match else "unknown"


def main():
    parser = argparse.ArgumentParser(description="Export PolypGen subset for external prompt validation")
    parser.add_argument("--polypgen_root", default="/Users/ilaydadilek/Downloads/PolypGen2021_MultiCenterData_v3/positive")
    parser.add_argument("--output_dir", default="endoscopy_guidance/exports/polypgen_external_80")
    parser.add_argument("--max_samples", type=int, default=80)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--min_mask_pixels", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    root = Path(args.polypgen_root)
    image_dir = root / "images"
    mask_dir = root / "masks"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    pairs = [(p, mask_dir / p.name) for p in images if (mask_dir / p.name).exists()]
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {root}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))
    selected = [pairs[int(i)] for i in idx]

    rows = []
    skipped_empty = 0
    for image_path, mask_path in selected:
        if len(rows) >= args.max_samples:
            break
        count = len(rows)
        sid = f"polypgen_{count:04d}"
        image = np.asarray(Image.open(image_path).convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR))
        mask = np.asarray(Image.open(mask_path).convert("L").resize((args.image_size, args.image_size), Image.NEAREST))
        gt = (mask > 0).astype(np.uint8)
        if int(gt.sum()) < args.min_mask_pixels:
            skipped_empty += 1
            continue
        heatmap = saliency_heatmap(image)
        pred_mask = (heatmap >= np.quantile(heatmap, 0.85)).astype(np.uint8)

        np.save(out / f"image_{sid}.npy", image)
        np.save(out / f"gt_mask_{sid}.npy", gt)
        np.save(out / f"pred_heatmap_{sid}.npy", heatmap.astype(np.float32))
        np.save(out / f"pred_mask_{sid}.npy", pred_mask)
        rows.append({
            "sample_id": sid,
            "source_file": image_path.name,
            "source_mask": mask_path.name,
            "source_group": center_name_group(image_path.name),
            "mask_pixels": int(gt.sum()),
        })

    with (out / "export_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "source_file", "source_mask", "source_group", "mask_pixels"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out} samples={len(rows)} skipped_empty={skipped_empty}")


if __name__ == "__main__":
    main()
