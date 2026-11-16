"""Prepare PolypGen 2021 single-frame positives for external evaluation.

Reads the official PolypGen2021_MultiCenterData_v3 release (data_C1..data_C6,
each with images_Cn/ and masks_Cn/, masks named <image>_mask.jpg) and writes

    <out>/images/<name>.jpg
    <out>/masks/<name>.jpg      (same file name as the image)
    <out>/manifest.csv          (name, center, width, height, mask_pixels)

which is the layout expected by `eval/evaluate_outputs.py --dataset polypgen`.

Usage:
    python3 dataset/dataset_c/prepare_polypgen.py \
        --source ~/Downloads/PolypGen2021_MultiCenterData_v3 \
        --out dataset/PolypGen/positive
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CENTERS = range(1, 7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "dataset" / "PolypGen" / "positive")
    args = parser.parse_args()

    source = args.source.expanduser()
    out_images = args.out / "images"
    out_masks = args.out / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    rows = []
    seen: set[str] = set()
    skipped_missing = skipped_empty = 0
    for center in CENTERS:
        image_dir = source / f"data_C{center}" / f"images_C{center}"
        mask_dir = source / f"data_C{center}" / f"masks_C{center}"
        for image_path in sorted(image_dir.glob("*.jpg")):
            # Two files in the release have irregular names:
            #   "C3_EndoCV2021_00489].jpg"      -> mask "C3_EndoCV2021_00489_mask.jpg"
            #   "31_endocv2021_positive_329 2.jpg" -> mask "..._329 2_mask.jpg"
            # Look the mask up by the original stem first, then the cleaned one,
            # and write both files under a cleaned name.
            clean_name = re.sub(r"[^A-Za-z0-9_.-]", "", image_path.name.replace(" ", "_"))
            mask_path = next(
                (
                    candidate
                    for candidate in (
                        mask_dir / f"{image_path.stem}_mask.jpg",
                        mask_dir / f"{Path(clean_name).stem}_mask.jpg",
                        mask_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '', image_path.stem)}_mask.jpg",
                    )
                    if candidate.exists()
                ),
                mask_dir / f"{image_path.stem}_mask.jpg",
            )
            if not mask_path.exists():
                skipped_missing += 1
                continue
            if clean_name in seen:
                raise RuntimeError(f"Duplicate file name across centers: {clean_name}")
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise RuntimeError(f"Size mismatch: {image_path} {image.size} vs {mask.size}")
                mask_pixels = int((np.asarray(mask.convert("L")) > 127).sum())
                width, height = image.size
            if mask_pixels == 0:
                skipped_empty += 1
                continue
            seen.add(clean_name)
            shutil.copy2(image_path, out_images / clean_name)
            shutil.copy2(mask_path, out_masks / clean_name)
            rows.append((clean_name, f"C{center}", width, height, mask_pixels))

    with open(args.out / "manifest.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "center", "width", "height", "mask_pixels"])
        writer.writerows(rows)

    per_center = {}
    for _, center, *_ in rows:
        per_center[center] = per_center.get(center, 0) + 1
    print(f"Wrote {len(rows)} image/mask pairs to {args.out}")
    print(f"Per center: {per_center}")
    print(f"Skipped: {skipped_missing} without mask, {skipped_empty} with empty mask")


if __name__ == "__main__":
    main()
