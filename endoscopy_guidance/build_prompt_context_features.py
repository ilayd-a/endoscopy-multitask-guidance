"""
build_prompt_context_features.py
================================
Append inference-available candidate context features to a prompt feature cache.

These features describe where each candidate sits relative to other candidates
in the same frame: normalized scores, ranks, position, and simple geometry.
They do not use ground-truth masks or SAM Dice labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def minmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    lo = float(np.nanmin(values[finite]))
    hi = float(np.nanmax(values[finite]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    out = (values - lo) / (hi - lo)
    out[~finite] = 0.0
    return out.astype(np.float32)


def percentile_rank(values: np.ndarray, descending: bool) -> np.ndarray:
    values = values.astype(float)
    if not np.isfinite(values).any():
        return np.zeros(len(values), dtype=np.float32)
    fill = -np.inf if descending else np.inf
    ranked_values = np.where(np.isfinite(values), values, fill)
    order = np.argsort(-ranked_values if descending else ranked_values)
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    denom = max(1, len(values) - 1)
    return 1.0 - ranks / denom


def build_context(qdf: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(qdf), 12), dtype=np.float32)
    cursor = 0
    for _, group in qdf.groupby("sample_id", sort=False):
        n = len(group)
        heatmap = group["heatmap_score"].to_numpy(dtype=float)
        sam = group["sam_score"].to_numpy(dtype=float)
        center_dist = group["center_dist"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)
        x = group["x"].to_numpy(dtype=float)
        radius = group["radius"].to_numpy(dtype=float)

        heatmap_norm = minmax(heatmap)
        sam_norm = minmax(sam)
        center_norm = minmax(center_dist)
        inv_center = 1.0 - center_norm
        y_max = float(np.nanmax(y)) if np.isfinite(y).any() else 1.0
        x_max = float(np.nanmax(x)) if np.isfinite(x).any() else 1.0
        y_norm = (np.nan_to_num(y, nan=0.0) / max(1.0, y_max)).astype(np.float32)
        x_norm = (np.nan_to_num(x, nan=0.0) / max(1.0, x_max)).astype(np.float32)
        radius_norm = (radius / 128.0).astype(np.float32)
        heatmap_rank = percentile_rank(heatmap, descending=True)
        sam_rank = percentile_rank(sam, descending=True)
        center_rank = percentile_rank(center_dist, descending=False)
        combined_prior = (0.45 * heatmap_norm + 0.45 * sam_norm + 0.10 * inv_center).astype(np.float32)
        heatmap_sam_gap = (heatmap_norm - sam_norm).astype(np.float32)

        out[cursor:cursor + n] = np.column_stack([
            heatmap_norm,
            sam_norm,
            inv_center,
            y_norm,
            x_norm,
            radius_norm,
            heatmap_rank,
            sam_rank,
            center_rank,
            combined_prior,
            heatmap_sam_gap,
            np.full(n, n / 128.0, dtype=np.float32),
        ])
        cursor += n
    return out


def main():
    parser = argparse.ArgumentParser(description="Append candidate context features to prompt feature cache")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--input_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed.npy")
    parser.add_argument("--output_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed_context.npy")
    args = parser.parse_args()

    qdf = pd.read_csv(args.prompt_quality_csv)
    X = np.load(args.input_features).astype(np.float32)
    if len(qdf) != len(X):
        raise ValueError(f"Cache mismatch: {len(qdf)} rows but {len(X)} feature rows")
    context = build_context(qdf)
    out = np.concatenate([X, context], axis=1).astype(np.float32)
    output = Path(args.output_features)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, out)
    print(f"[saved] {output} shape={out.shape}")


if __name__ == "__main__":
    main()
