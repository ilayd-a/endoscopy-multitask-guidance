"""
export_kvasir_external.py
=========================
Export a sampled Kvasir-SEG subset into the NumPy format used by the prompt
quality experiments.

The exported pred_heatmap is an inference-only saliency/center prior derived
from image appearance and geometry, not from the ground-truth mask.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from export_polypgen_external import saliency_heatmap


def find_kvasir_dirs(root: Path) -> tuple[Path, Path]:
    candidates = [
        (root / "images", root / "masks"),
        (root / "Kvasir-SEG" / "images", root / "Kvasir-SEG" / "masks"),
        (root / "kvasir-seg" / "images", root / "kvasir-seg" / "masks"),
    ]
    for image_dir, mask_dir in candidates:
        if image_dir.exists() and mask_dir.exists():
            return image_dir, mask_dir
    image_dirs = [p for p in root.rglob("images") if p.is_dir()]
    for image_dir in image_dirs:
        mask_dir = image_dir.parent / "masks"
        if mask_dir.exists():
            return image_dir, mask_dir
    raise FileNotFoundError(f"Could not find Kvasir images/masks folders under {root}")


def load_binary_mask(mask_path: Path, image_size: int) -> np.ndarray:
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
    values = np.asarray(mask)
    return (values > 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Export Kvasir-SEG subset for external prompt validation")
    parser.add_argument("--kvasir_root", default="/Users/ilaydadilek/Downloads/Kvasir-SEG")
    parser.add_argument("--output_dir", default="endoscopy_guidance/exports/kvasir_external_120")
    parser.add_argument("--max_samples", type=int, default=120)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--min_mask_pixels", type=int, default=25)
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()

    root = Path(args.kvasir_root)
    image_dir, mask_dir = find_kvasir_dirs(root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    pairs = []
    for image_path in images:
        same_name = mask_dir / image_path.name
        png_name = mask_dir / f"{image_path.stem}.png"
        jpg_name = mask_dir / f"{image_path.stem}.jpg"
        for mask_path in [same_name, png_name, jpg_name]:
            if mask_path.exists():
                pairs.append((image_path, mask_path))
                break
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {root}")

    rng = np.random.default_rng(args.seed)
    selected = [pairs[int(i)] for i in rng.permutation(len(pairs))]

    rows = []
    skipped_empty = 0
    for image_path, mask_path in selected:
        if len(rows) >= args.max_samples:
            break
        sid = f"kvasir_{len(rows):04d}"
        image = np.asarray(Image.open(image_path).convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR))
        gt = load_binary_mask(mask_path, args.image_size)
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
            "source_group": "kvasir",
            "mask_pixels": int(gt.sum()),
        })

    with (out / "export_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "source_file", "source_mask", "source_group", "mask_pixels"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out} samples={len(rows)} skipped_empty={skipped_empty}")


if __name__ == "__main__":
    main()
