"""
build_external_prompt_quality_cache.py
======================================
Build a prompt-quality cache for an external exported dataset.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from candidate_ranking_benchmark import build_candidates
from sam_prompt_quality_ranker import sam_prompt_masks_all
from sam_quantum_prompt_benchmark import choose_device, dice_iou, load_sam_predictor, set_cached_or_compute_image


def load_or_build_candidates(args):
    data_dir = Path(args.data_dir)
    cache_path = Path(args.candidate_cache) if args.candidate_cache else None
    cache_key = {
        "data_dir": str(data_dir.resolve()),
        "top_n": args.top_n,
        "grid_stride": args.grid_stride,
        "nms_dist": args.nms_dist,
        "patch_radius": args.patch_radius,
        "use_image_features": True,
    }
    if cache_path is not None and cache_path.exists():
        payload = pickle.loads(cache_path.read_bytes())
        if payload.get("cache_key") == cache_key:
            print(f"[cache] loaded candidates from {cache_path}")
            return payload["X"], payload["y"], payload["rows"]

    X, y, rows = build_candidates(
        data_dir=data_dir,
        top_n=args.top_n,
        grid_stride=args.grid_stride,
        nms_dist=args.nms_dist,
        radius=args.patch_radius,
        use_image_features=True,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps({"cache_key": cache_key, "X": X, "y": y, "rows": rows}))
        print(f"[cache] saved candidates to {cache_path}")
    return X, y, rows


def main():
    parser = argparse.ArgumentParser(description="Build external prompt-quality cache")
    parser.add_argument("--data_dir", default="endoscopy_guidance/exports/polypgen_external_80")
    parser.add_argument("--checkpoint", default="models/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--candidate_cache", default="endoscopy_guidance/results/polypgen_external_80_candidate_cache.pkl")
    parser.add_argument("--sam_embedding_cache_dir", default="endoscopy_guidance/results/sam_embedding_cache_polypgen_external_80")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/polypgen_external_80_prompt_quality.csv")
    parser.add_argument("--prompt_quality_features", default="endoscopy_guidance/results/polypgen_external_80_prompt_quality_features.npy")
    parser.add_argument("--grid_stride", type=int, default=32)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--nms_dist", type=int, default=24)
    parser.add_argument("--patch_radius", type=int, default=24)
    parser.add_argument("--radius", type=int, default=48, help="Single prompt box radius kept for backward compatibility.")
    parser.add_argument("--radii", type=int, nargs="+", default=[], help="One or more prompt box radii. Overrides --radius when provided.")
    parser.add_argument("--max_samples", type=int, default=0, help="If >0, build only the first N exported samples.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X, _, rows = load_or_build_candidates(args)
    df = pd.DataFrame(rows)
    summary = pd.read_csv(data_dir / "export_summary.csv")
    df = df.merge(summary, on="sample_id", how="left")
    df["split"] = "external"
    df["sequence_id"] = -1

    predictor = load_sam_predictor(Path(args.checkpoint), args.model_type, choose_device(args.device))
    embedding_cache = Path(args.sam_embedding_cache_dir) if args.sam_embedding_cache_dir else None

    out_rows = []
    out_features = []
    sample_ids = sorted(df["sample_id"].unique())
    if args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]
    radii = args.radii if args.radii else [args.radius]
    for sample_count, sid in enumerate(sample_ids, start=1):
        sample = df[df["sample_id"].eq(sid)].copy()
        image = np.load(data_dir / f"image_{sid}.npy")
        gt = np.load(data_dir / f"gt_mask_{sid}.npy")
        set_cached_or_compute_image(predictor, image, sid, embedding_cache)
        points = [(int(row.y), int(row.x)) for row in sample.itertuples(index=False)]
        for radius in radii:
            masks, sam_scores = sam_prompt_masks_all(predictor, points, image.shape[:2], radius)
            for row, mask, sam_score in zip(sample.itertuples(index=True), masks, sam_scores):
                dice, iou = dice_iou(mask, gt)
                out_rows.append({
                    "sample_id": sid,
                    "candidate_index": int(row.Index),
                    "source_file": row.source_file,
                    "source_mask": row.source_mask,
                    "source_group": row.source_group,
                    "sequence_id": -1,
                    "split": "external",
                    "y": int(row.y),
                    "x": int(row.x),
                    "radius": int(radius),
                    "heatmap_score": float(row.heatmap_score),
                    "point_hit": int(row.label > 0),
                    "center_dist": float(row.center_dist),
                    "sam_score": float(sam_score),
                    "sam_dice": float(dice),
                    "sam_iou": float(iou),
                })
                radius_feature = np.asarray([radius / 128.0, sam_score], dtype=float)
                out_features.append(np.concatenate([X[int(row.Index)], radius_feature]))
        if sample_count % 10 == 0 or sample_count == len(sample_ids):
            print(f"[external-quality] {sample_count}/{len(sample_ids)} samples", flush=True)

    qdf = pd.DataFrame(out_rows)
    Q = np.vstack(out_features).astype(np.float32)
    Path(args.prompt_quality_csv).parent.mkdir(parents=True, exist_ok=True)
    qdf.to_csv(args.prompt_quality_csv, index=False)
    np.save(args.prompt_quality_features, Q)
    print(f"[saved] {args.prompt_quality_csv}")
    print(f"[saved] {args.prompt_quality_features} shape={Q.shape}")


if __name__ == "__main__":
    main()
