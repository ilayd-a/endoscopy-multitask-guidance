"""
build_sam_embedding_prompt_features.py
======================================
Augment cached prompt-quality features with local SAM image-embedding features.

The prompt-quality cache already stores candidate-level handcrafted features and
the downstream SAM Dice labels. This script samples the frozen SAM image
embedding at each candidate prompt location and appends compact local statistics
without rerunning SAM mask prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_embedding(cache_dir: Path, sample_id: str) -> tuple[np.ndarray, tuple[int, int]]:
    path = cache_dir / f"{sample_id}_sam_embedding.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing SAM embedding cache for {sample_id}: {path}")
    payload = torch.load(path, map_location="cpu")
    features = payload["features"].squeeze(0).numpy().astype(np.float32)
    original_size = tuple(int(v) for v in payload["original_size"])
    return features, original_size


def sample_descriptor(features: np.ndarray, original_size: tuple[int, int], y: int, x: int, pool_radius: int) -> np.ndarray:
    channels, eh, ew = features.shape
    oh, ow = original_size
    gy = int(np.clip(round((float(y) / max(1, oh - 1)) * (eh - 1)), 0, eh - 1))
    gx = int(np.clip(round((float(x) / max(1, ow - 1)) * (ew - 1)), 0, ew - 1))
    y0 = max(0, gy - pool_radius)
    y1 = min(eh, gy + pool_radius + 1)
    x0 = max(0, gx - pool_radius)
    x1 = min(ew, gx + pool_radius + 1)
    patch = features[:, y0:y1, x0:x1].reshape(channels, -1)
    center = features[:, gy, gx]
    mean = patch.mean(axis=1)
    std = patch.std(axis=1)
    return np.concatenate([center, mean, std]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build SAM-embedding augmented prompt feature cache")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--base_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_mps.npy")
    parser.add_argument("--embedding_cache_dir", default="endoscopy_guidance/results/sam_embedding_cache_vit_b")
    parser.add_argument("--output_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed.npy")
    parser.add_argument("--pool_radius", type=int, default=1)
    args = parser.parse_args()

    qdf = pd.read_csv(args.prompt_quality_csv)
    base = np.load(args.base_features).astype(np.float32)
    if len(qdf) != len(base):
        raise ValueError(f"Cache mismatch: {len(qdf)} rows but {len(base)} base feature rows")

    cache_dir = Path(args.embedding_cache_dir)
    descriptors = np.zeros((len(qdf), 768), dtype=np.float32)
    cursor = 0
    for sample_count, (sid, group) in enumerate(qdf.groupby("sample_id", sort=False), start=1):
        features, original_size = load_embedding(cache_dir, sid)
        desc = [
            sample_descriptor(features, original_size, int(row.y), int(row.x), args.pool_radius)
            for row in group.itertuples(index=False)
        ]
        rows = np.vstack(desc)
        descriptors[cursor:cursor + len(rows)] = rows
        cursor += len(rows)
        if sample_count % 50 == 0:
            print(f"[sam-embed] {sample_count} samples, {cursor} candidates", flush=True)

    out = np.concatenate([base, descriptors], axis=1).astype(np.float32)
    output = Path(args.output_features)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, out)
    print(f"[saved] {output} shape={out.shape}")


if __name__ == "__main__":
    main()
