"""
sam_quantum_prompt_benchmark.py
===============================
Evaluate quantum-kernel candidate ranking as a prompt-selection layer for SAM.

The experiment freezes SAM and asks a narrow question:

    If we can only pass one positive point prompt to a promptable segmentation
    model, does a projected quantum kernel pick better prompts than heatmap
    peaks, random candidates, or a classical ranker?

This is the missing bridge between the earlier PQK candidate-recovery result and
actual mask-quality metrics.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
EBTC_EXPERIMENTS = ROOT / "EndoscopicBladderTissue" / "experiments"
if str(EBTC_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EBTC_EXPERIMENTS))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from publication_benchmark_ebtc import ProjectedQuantumKernelSVC
from candidate_ranking_benchmark import build_candidates, fit_low_dim, limit_training_candidates, prediction_scores


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    denom = pred_bool.sum() + gt_bool.sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    dice = 1.0 if denom == 0 else float(2 * intersection / denom)
    iou = 1.0 if union == 0 else float(intersection / union)
    return dice, iou


def load_sequence_map(endoscopy_repo: Path) -> dict[str, int]:
    cvc_dir = endoscopy_repo / "dataset" / "CVC-ClinicDB"
    metadata = pd.read_csv(cvc_dir / "metadata.csv")
    return {
        Path(row.png_image_path).name: int(row.sequence_id)
        for row in metadata.itertuples(index=False)
    }


def row_dataframe(rows: list[dict], export_summary: Path, sequence_map: dict[str, int]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary = pd.read_csv(export_summary)[["sample_id", "source_file"]]
    df = df.merge(summary, on="sample_id", how="left")
    df["sequence_id"] = df["source_file"].map(sequence_map)
    df["split"] = np.where(df["sequence_id"] <= 23, "train", np.where(df["sequence_id"] <= 26, "val", "test"))
    return df


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sam_predictor(checkpoint: Path, model_type: str, device: str):
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def prompt_box(point_yx: tuple[int, int], shape: tuple[int, int], radius: int) -> np.ndarray:
    y, x = point_yx
    h, w = shape
    return np.asarray([
        max(0, x - radius),
        max(0, y - radius),
        min(w - 1, x + radius),
        min(h - 1, y + radius),
    ], dtype=np.float32)


def set_cached_or_compute_image(predictor, image: np.ndarray, sample_id: str, cache_dir: Path | None):
    cache_path = cache_dir / f"{sample_id}_sam_embedding.pt" if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location=predictor.device)
        predictor.reset_image()
        predictor.original_size = tuple(payload["original_size"])
        predictor.input_size = tuple(payload["input_size"])
        predictor.features = payload["features"].to(predictor.device)
        predictor.is_image_set = True
        return

    predictor.set_image(image)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "original_size": tuple(predictor.original_size),
            "input_size": tuple(predictor.input_size),
            "features": predictor.get_image_embedding().detach().cpu(),
        }, cache_path)


def sam_prompt_batch(
    predictor,
    point_yxs: list[tuple[int, int]],
    image_shape: tuple[int, int],
    prompt_mode: str,
    box_radius: int,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    device = predictor.device
    point_coords = None
    point_labels = None
    boxes = None
    if prompt_mode in {"point", "point_box"}:
        coords = np.asarray([[[x, y]] for y, x in point_yxs], dtype=np.float32)
        coords = predictor.transform.apply_coords(coords, image_shape)
        point_coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
        point_labels = torch.ones((len(point_yxs), 1), dtype=torch.int, device=device)
    if prompt_mode in {"box", "point_box"}:
        box_arr = np.asarray([prompt_box(point, image_shape, box_radius) for point in point_yxs], dtype=np.float32)
        box_arr = predictor.transform.apply_boxes(box_arr, image_shape)
        boxes = torch.as_tensor(box_arr, dtype=torch.float32, device=device)

    masks, scores, _ = predictor.predict_torch(
        point_coords=point_coords,
        point_labels=point_labels,
        boxes=boxes,
        multimask_output=True,
        return_logits=False,
    )
    flat = scores.reshape(-1)
    best_flat = int(torch.argmax(flat).item())
    prompt_idx = best_flat // scores.shape[1]
    mask_idx = best_flat % scores.shape[1]
    mask = masks[prompt_idx, mask_idx].detach().cpu().numpy().astype(bool)
    return mask, float(scores[prompt_idx, mask_idx].item()), point_yxs[prompt_idx]


def heatmap_baseline(data_dir: Path, sample_id: str, threshold: float) -> np.ndarray:
    heatmap = np.load(data_dir / f"pred_heatmap_{sample_id}.npy")
    return heatmap >= threshold


def ranked_points(sample_rows: pd.DataFrame, scores: np.ndarray, top_k: int) -> list[tuple[int, int]]:
    order = np.argsort(scores)[::-1]
    out = []
    for idx in order[: max(1, top_k)]:
        row = sample_rows.iloc[int(idx)]
        out.append((int(row.y), int(row.x)))
    return out


def normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def hybrid_scores(sample_rows: pd.DataFrame, model_scores: np.ndarray, alpha: float) -> np.ndarray:
    model_norm = normalize_scores(model_scores)
    heatmap_norm = normalize_scores(sample_rows["heatmap_score"].to_numpy(dtype=float))
    return alpha * model_norm + (1.0 - alpha) * heatmap_norm


def oracle_point(sample_rows: pd.DataFrame) -> tuple[int, int]:
    positives = sample_rows[sample_rows["label"] > 0]
    if positives.empty:
        row = sample_rows.sort_values("center_dist").iloc[0]
    else:
        row = positives.sort_values("center_dist").iloc[0]
    return int(row.y), int(row.x)


def evaluate_prompt_strategy(
    predictor,
    data_dir: Path,
    sample_ids: list[str],
    points_by_sample: dict[str, list[tuple[int, int]]],
    name: str,
    prompt_mode: str,
    box_radius: int,
    cache_dir: Path | None,
    quiet: bool,
):
    rows = []
    for sid in sample_ids:
        image = np.load(data_dir / f"image_{sid}.npy")
        gt = np.load(data_dir / f"gt_mask_{sid}.npy")
        t0 = time.time()
        candidates = points_by_sample[sid]
        set_cached_or_compute_image(predictor, image, sid, cache_dir)
        mask, sam_score, point_yx = sam_prompt_batch(
            predictor,
            candidates,
            image.shape[:2],
            prompt_mode=prompt_mode,
            box_radius=box_radius,
        )
        elapsed = time.time() - t0
        dice, iou = dice_iou(mask, gt)
        y, x = point_yx
        rows.append({
            "strategy": name,
            "sample_id": sid,
            "prompt_y": y,
            "prompt_x": x,
            "sam_score": sam_score,
            "dice": dice,
            "iou": iou,
            "prompt_hit": int(gt[y, x] > 0),
            "elapsed_sec": elapsed,
        })
        if not quiet:
            print(f"[sam] {name} {sid} dice={dice:.3f} hit={int(gt[y, x] > 0)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="SAM prompt benchmark with PQK-ranked endoscopy candidates")
    parser.add_argument("--data_dir", default="endoscopy_guidance/exports/cvc_all_rgb")
    parser.add_argument("--endoscopy_repo", default="/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance")
    parser.add_argument("--checkpoint", default="models/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/sam_quantum_prompt_benchmark.csv")
    parser.add_argument("--candidate_cache", default="endoscopy_guidance/results/sam_prompt_candidate_cache.pkl")
    parser.add_argument("--sam_embedding_cache_dir", default="endoscopy_guidance/results/sam_embedding_cache_vit_b")
    parser.add_argument("--grid_stride", type=int, default=32)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--nms_dist", type=int, default=24)
    parser.add_argument("--patch_radius", type=int, default=24)
    parser.add_argument("--n_components", type=int, default=6)
    parser.add_argument("--train_candidate_limit", type=int, default=600)
    parser.add_argument("--max_test_samples", type=int, default=12)
    parser.add_argument("--eval_split", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline_threshold", type=float, default=0.3)
    parser.add_argument("--prompt_mode", choices=["point", "box", "point_box"], default="point_box")
    parser.add_argument("--box_radius", type=int, default=48)
    parser.add_argument("--sam_top_k", type=int, default=1)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--pqk_c", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--rank_blend_alpha", type=float, default=1.0, help="1=model score only, 0=heatmap score only")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sequence_map = load_sequence_map(Path(args.endoscopy_repo))
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
            X, y, rows = payload["X"], payload["y"], payload["rows"]
            print(f"[cache] loaded candidates from {cache_path}")
        else:
            X, y, rows = None, None, None
    else:
        X, y, rows = None, None, None
    if X is None:
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
    df = row_dataframe(rows, data_dir / "export_summary.csv", sequence_map)
    if args.eval_split == "val":
        train_idx = df.index[df["sequence_id"] <= 23].to_numpy()
        eval_mask = df["split"].eq("val")
    else:
        train_idx = df.index[df["sequence_id"] <= 26].to_numpy()
        eval_mask = df["split"].eq("test")
    eval_sample_ids = sorted(df.loc[eval_mask, "sample_id"].unique())
    if args.max_test_samples > 0:
        eval_sample_ids = eval_sample_ids[: args.max_test_samples]
    test_idx = df.index[df["sample_id"].isin(eval_sample_ids)].to_numpy()
    train_idx = limit_training_candidates(train_idx, y, args.train_candidate_limit, args.seed)
    print(f"[data] candidates={len(y)} train={len(train_idx)} eval_candidates={len(test_idx)} eval_samples={len(eval_sample_ids)} eval_split={args.eval_split}")

    classical = ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=args.seed)
    classical.fit(X[train_idx], y[train_idx])
    classical_scores = prediction_scores(classical, X[test_idx])

    X_train_q, X_test_q, pca_info = fit_low_dim(X[train_idx], X[test_idx], args.n_components, args.seed)
    pqk = ProjectedQuantumKernelSVC(gamma="scale", reps=args.pqk_reps, C=args.pqk_c, class_weight="balanced")
    pqk.fit(X_train_q, y[train_idx])
    pqk_scores = prediction_scores(pqk, X_test_q)
    print(f"[pqk] pca_variance_retained={pca_info['pca_variance_retained']:.3f}")

    rng = np.random.default_rng(args.seed)
    test_df = df.loc[test_idx].copy().reset_index(drop=True)
    blend_suffix = "" if args.rank_blend_alpha >= 0.999 else f"_blend{args.rank_blend_alpha:g}"
    pqk_name = f"pqk_reps{args.pqk_reps}_C{args.pqk_c:g}{blend_suffix}"
    classical_name = f"classical_extratrees{blend_suffix}"
    point_maps = {"heatmap_peak": {}, "random_candidate": {}, classical_name: {}, pqk_name: {}, "oracle_candidate": {}}
    for sid in eval_sample_ids:
        sample_mask = test_df["sample_id"].eq(sid).to_numpy()
        sample_rows = test_df[sample_mask].reset_index(drop=True)
        heatmap = np.load(data_dir / f"pred_heatmap_{sid}.npy")
        hy, hx = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        point_maps["heatmap_peak"][sid] = [(int(hy), int(hx))]
        random_rows = sample_rows.iloc[rng.choice(len(sample_rows), size=max(1, args.sam_top_k), replace=False)]
        point_maps["random_candidate"][sid] = [(int(row.y), int(row.x)) for row in random_rows.itertuples(index=False)]
        point_maps["oracle_candidate"][sid] = [oracle_point(sample_rows)]
        classical_rank_scores = hybrid_scores(sample_rows, classical_scores[sample_mask], args.rank_blend_alpha)
        pqk_rank_scores = hybrid_scores(sample_rows, pqk_scores[sample_mask], args.rank_blend_alpha)
        point_maps[classical_name][sid] = ranked_points(sample_rows, classical_rank_scores, args.sam_top_k)
        point_maps[pqk_name][sid] = ranked_points(sample_rows, pqk_rank_scores, args.sam_top_k)

    device = choose_device(args.device)
    print(f"[sam] loading {args.model_type} on {device}")
    predictor = load_sam_predictor(Path(args.checkpoint), args.model_type, device)
    embedding_cache = Path(args.sam_embedding_cache_dir) if args.sam_embedding_cache_dir else None

    out_rows = []
    for sid in eval_sample_ids:
        gt = np.load(data_dir / f"gt_mask_{sid}.npy")
        base = heatmap_baseline(data_dir, sid, args.baseline_threshold)
        dice, iou = dice_iou(base, gt)
        out_rows.append({
            "strategy": f"base_heatmap_thr{args.baseline_threshold:g}",
            "sample_id": sid,
            "prompt_y": np.nan,
            "prompt_x": np.nan,
            "sam_score": np.nan,
            "dice": dice,
            "iou": iou,
            "prompt_hit": np.nan,
            "elapsed_sec": 0.0,
        })

    for name, points in point_maps.items():
        out_rows.extend(evaluate_prompt_strategy(
            predictor,
            data_dir,
            eval_sample_ids,
            points,
            name,
            prompt_mode=args.prompt_mode,
            box_radius=args.box_radius,
            cache_dir=embedding_cache,
            quiet=args.quiet,
        ))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        fieldnames = ["strategy", "sample_id", "prompt_y", "prompt_x", "prompt_hit", "sam_score", "dice", "iou", "elapsed_sec"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    summary = pd.DataFrame(out_rows).groupby("strategy").agg(
        samples=("sample_id", "count"),
        prompt_hit=("prompt_hit", "mean"),
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        iou_mean=("iou", "mean"),
        elapsed_sec=("elapsed_sec", "mean"),
    ).sort_values("dice_mean", ascending=False)
    print(f"[saved] {output_csv}")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
