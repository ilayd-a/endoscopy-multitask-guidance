"""
build_residual_patch_dataset.py
===============================
Build a patch-level residual-correction dataset from exported classical
segmentation baseline predictions.

Each row describes a local patch around a pixel sampled from the predicted
boundary, ground-truth boundary, uncertain regions, false positives, false
negatives, and easy-correct regions. The label is the residual class:

    0 = correct background
    1 = false positive, remove from mask
    2 = false negative, add to mask
    3 = correct foreground

This dataset is designed for classical-vs-quantum residual correction studies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def local_stats(values: np.ndarray, y: int, x: int, radius: int) -> list[float]:
    h, w = values.shape[:2]
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch = values[y0:y1, x0:x1].astype(np.float32)
    return [
        float(patch.mean()),
        float(patch.std()),
        float(patch.min()),
        float(patch.max()),
    ]


def image_patch_stats(image: np.ndarray, y: int, x: int, radius: int) -> list[float]:
    stats = []
    for channel in range(3):
        stats.extend(local_stats(image[..., channel], y, x, radius))
    return stats


def boundary(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    return dilated != eroded


def sample_indices(mask: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(ys) == 0 or count <= 0:
        return np.empty((0, 2), dtype=int)
    idx = rng.choice(len(ys), size=min(count, len(ys)), replace=False)
    return np.column_stack([ys[idx], xs[idx]])


def residual_label(pred: int, gt: int) -> int:
    if pred == 0 and gt == 0:
        return 0
    if pred == 1 and gt == 0:
        return 1
    if pred == 0 and gt == 1:
        return 2
    return 3


def build_features(image: np.ndarray, prob: np.ndarray, pred: np.ndarray, gt: np.ndarray, y: int, x: int, radius: int) -> list[float]:
    uncertainty = 1.0 - abs(float(prob[y, x]) - 0.5) * 2.0
    h, w = prob.shape
    yy, xx = np.mgrid[0:h, 0:w]
    gt_pixels = np.column_stack(np.where(gt > 0))
    if len(gt_pixels) > 0:
        cy, cx = gt_pixels.mean(axis=0)
        center_dist = float(np.hypot(y - cy, x - cx) / max(h, w))
    else:
        center_dist = 1.0
    features = [
        float(y / max(1, h - 1)),
        float(x / max(1, w - 1)),
        float(prob[y, x]),
        float(pred[y, x]),
        uncertainty,
        center_dist,
    ]
    features.extend(local_stats(prob, y, x, radius))
    features.extend(local_stats(pred, y, x, radius))
    features.extend(image_patch_stats(image / 255.0, y, x, radius))
    return features


def feature_names() -> list[str]:
    names = ["y_norm", "x_norm", "prob", "pred", "uncertainty", "gt_center_dist"]
    for prefix in ["prob", "pred"]:
        names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_min", f"{prefix}_max"])
    for channel in ["r", "g", "b"]:
        names.extend([f"{channel}_mean", f"{channel}_std", f"{channel}_min", f"{channel}_max"])
    return names


def main():
    parser = argparse.ArgumentParser(description="Build residual patch dataset from baseline predictions")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/classical_cvc_baseline_val_test_sweep")
    parser.add_argument("--output_npz", default="endoscopy_guidance/results/residual_patch_dataset_cvc_val_test.npz")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/residual_patch_dataset_cvc_val_test.csv")
    parser.add_argument("--patch_radius", type=int, default=5)
    parser.add_argument("--boundary_radius", type=int, default=3)
    parser.add_argument("--samples_per_frame", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    base = Path(args.baseline_dir)
    metrics = pd.read_csv(base / "baseline_metrics.csv")
    rng = np.random.default_rng(args.seed)
    rows = []
    X = []
    y = []

    per_region = max(1, args.samples_per_frame // 6)
    for record in metrics.itertuples(index=False):
        sid = record.sample_id
        image = np.load(base / "images" / f"{sid}.npy")
        gt = np.load(base / "gt_masks" / f"{sid}.npy").astype(np.uint8)
        prob = np.load(base / "prob_maps" / f"{sid}.npy").astype(np.float32)
        pred = np.load(base / "pred_masks" / f"{sid}.npy").astype(np.uint8)
        uncertainty = 1.0 - np.abs(prob - 0.5) * 2.0
        regions = {
            "false_positive": (pred == 1) & (gt == 0),
            "false_negative": (pred == 0) & (gt == 1),
            "pred_boundary": boundary(pred, args.boundary_radius),
            "gt_boundary": boundary(gt, args.boundary_radius),
            "uncertain": uncertainty >= np.quantile(uncertainty, 0.95),
            "correct": pred == gt,
        }
        coords = []
        for region_name, region_mask in regions.items():
            sampled = sample_indices(region_mask, per_region, rng)
            for yy, xx in sampled:
                coords.append((int(yy), int(xx), region_name))
        if not coords:
            continue
        seen = set()
        for yy, xx, region_name in coords:
            key = (yy, xx)
            if key in seen:
                continue
            seen.add(key)
            label = residual_label(int(pred[yy, xx]), int(gt[yy, xx]))
            X.append(build_features(image, prob, pred, gt, yy, xx, args.patch_radius))
            y.append(label)
            rows.append({
                "sample_id": sid,
                "split": record.split,
                "source_file": record.source_file,
                "sequence_id": int(record.sequence_id),
                "frame_dice": float(record.dice),
                "region": region_name,
                "pixel_y": yy,
                "pixel_x": xx,
                "label": label,
            })

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    row_df = pd.DataFrame(rows)
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, X=X_arr, y=y_arr, feature_names=np.asarray(feature_names(), dtype=object))
    row_df.to_csv(args.output_csv, index=False)
    print(f"[saved] {output_npz} X={X_arr.shape} y={y_arr.shape}")
    print(f"[saved] {args.output_csv}")
    print(row_df.groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
