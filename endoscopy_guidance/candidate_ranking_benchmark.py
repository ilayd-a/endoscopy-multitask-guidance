"""
candidate_ranking_benchmark.py
==============================
Candidate-ranking benchmark for endoscopic image guidance.

This experiment treats a segmentation/heatmap model as the clinical backbone
and evaluates whether a compact projected quantum kernel can rerank candidate
target locations under low-label conditions.

Inputs are NumPy arrays exported by the endoscopy-multitask-guidance repo:

  gt_mask_*.npy        binary target masks
  pred_mask_*.npy      predicted masks from the classical guidance model
  pred_heatmap_*.npy   predicted guidance heatmaps

The script builds candidate points from heatmap peaks and a sparse image grid,
extracts local heatmap/mask/geometric descriptors, and evaluates classical
models against a projected quantum kernel SVM using leave-one-frame-out splits.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "EndoscopicBladderTissue" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from publication_benchmark_ebtc import ProjectedQuantumKernelSVC, model_kernel_diagnostics


def discover_sample_ids(data_dir: Path) -> list[str]:
    ids = []
    for path in sorted(data_dir.glob("gt_mask_*.npy")):
        match = re.search(r"gt_mask_(.+)\.npy$", path.name)
        if match:
            sid = match.group(1)
            if (data_dir / f"pred_mask_{sid}.npy").exists() and (data_dir / f"pred_heatmap_{sid}.npy").exists():
                ids.append(sid)
    return ids


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return float("nan"), float("nan")
    return float(ys.mean()), float(xs.mean())


def patch(arr: np.ndarray, y: int, x: int, radius: int) -> np.ndarray:
    y0, y1 = max(0, y - radius), min(arr.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(arr.shape[1], x + radius + 1)
    return arr[y0:y1, x0:x1]


def local_stats(values: np.ndarray) -> list[float]:
    flat = values.astype(float).ravel()
    if flat.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(flat.mean()),
        float(flat.std()),
        float(flat.min()),
        float(flat.max()),
        float(np.percentile(flat, 90)),
    ]


def image_patch_features(image: np.ndarray | None, y: int, x: int, radius: int) -> list[float]:
    if image is None:
        return []

    image_float = image.astype(float) / 255.0
    rgb_patch = patch(image_float, y, x, radius)
    if rgb_patch.size == 0:
        return [0.0] * 30

    features = []
    whole_mean = image_float.reshape(-1, image_float.shape[-1]).mean(axis=0)
    whole_std = image_float.reshape(-1, image_float.shape[-1]).std(axis=0)
    patch_pixels = rgb_patch.reshape(-1, rgb_patch.shape[-1])
    patch_mean = patch_pixels.mean(axis=0)
    patch_std = patch_pixels.std(axis=0)
    patch_min = patch_pixels.min(axis=0)
    patch_max = patch_pixels.max(axis=0)
    features.extend(patch_mean.tolist())
    features.extend(patch_std.tolist())
    features.extend(patch_min.tolist())
    features.extend(patch_max.tolist())
    features.extend((patch_mean - whole_mean).tolist())
    features.extend((patch_std - whole_std).tolist())

    gray = (
        0.2989 * rgb_patch[..., 0]
        + 0.5870 * rgb_patch[..., 1]
        + 0.1140 * rgb_patch[..., 2]
    )
    gy, gx = np.gradient(gray)
    grad_mag = np.hypot(gy, gx)
    features.extend(local_stats(gray))
    features.extend(local_stats(grad_mag))
    return [float(v) for v in features]


def nms_top_points(heatmap: np.ndarray, top_n: int, min_dist: int) -> list[tuple[int, int]]:
    order = np.argsort(heatmap.ravel())[::-1]
    selected: list[tuple[int, int]] = []
    for idx in order:
        y, x = np.unravel_index(int(idx), heatmap.shape)
        if all((y - yy) ** 2 + (x - xx) ** 2 >= min_dist ** 2 for yy, xx in selected):
            selected.append((int(y), int(x)))
            if len(selected) >= top_n:
                break
    return selected


def grid_points(shape: tuple[int, int], stride: int) -> list[tuple[int, int]]:
    h, w = shape
    ys = list(range(stride // 2, h, stride))
    xs = list(range(stride // 2, w, stride))
    return [(int(y), int(x)) for y in ys for x in xs]


def candidate_features(
    y: int,
    x: int,
    heatmap: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    image: np.ndarray | None,
    radius: int,
    use_image_features: bool,
) -> tuple[list[float], dict]:
    h, w = heatmap.shape
    hm_patch = patch(heatmap, y, x, radius)
    pm_patch = patch(pred_mask, y, x, radius)
    cy, cx = mask_centroid(gt_mask)
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    gy, gx = np.gradient(heatmap.astype(float))

    features = [
        float(y / max(h - 1, 1)),
        float(x / max(w - 1, 1)),
        float(heatmap[y, x]),
        float(pred_mask[y, x] > 0),
        float(np.hypot(y - peak_y, x - peak_x) / np.hypot(h, w)),
        float(np.hypot(y - h / 2, x - w / 2) / np.hypot(h, w)),
        float(gy[y, x]),
        float(gx[y, x]),
        float(np.hypot(gy[y, x], gx[y, x])),
    ]
    features.extend(local_stats(hm_patch))
    features.extend(local_stats(pm_patch))
    if use_image_features:
        features.extend(image_patch_features(image, y, x, radius))

    label = int(gt_mask[y, x] > 0)
    center_dist = float(np.hypot(y - cy, x - cx)) if not math.isnan(cy) else float("nan")
    meta = {
        "label": label,
        "heatmap_score": float(heatmap[y, x]),
        "center_dist": center_dist,
    }
    return features, meta


def build_candidates(
    data_dir: Path,
    top_n: int,
    grid_stride: int,
    nms_dist: int,
    radius: int,
    use_image_features: bool,
):
    rows = []
    features = []
    labels = []
    sample_ids = discover_sample_ids(data_dir)
    if not sample_ids:
        raise FileNotFoundError(f"No complete gt/pred/heatmap sample triplets found in {data_dir}")

    for sid in sample_ids:
        gt = np.load(data_dir / f"gt_mask_{sid}.npy")
        pred = np.load(data_dir / f"pred_mask_{sid}.npy")
        heatmap = np.load(data_dir / f"pred_heatmap_{sid}.npy")
        image_path = data_dir / f"image_{sid}.npy"
        image = np.load(image_path) if use_image_features and image_path.exists() else None
        points = nms_top_points(heatmap, top_n=top_n, min_dist=nms_dist)
        points.extend(grid_points(heatmap.shape, stride=grid_stride))

        seen = set()
        for source_index, (y, x) in enumerate(points):
            if (y, x) in seen:
                continue
            seen.add((y, x))
            feat, meta = candidate_features(
                y,
                x,
                heatmap,
                pred,
                gt,
                image=image,
                radius=radius,
                use_image_features=use_image_features,
            )
            features.append(feat)
            labels.append(meta["label"])
            rows.append({
                "sample_id": sid,
                "y": y,
                "x": x,
                "source_index": source_index,
                **meta,
            })
    return np.asarray(features, dtype=float), np.asarray(labels, dtype=int), rows


def fit_low_dim(X_train_raw, X_test_raw, n_components: int, seed: int):
    max_components = max(1, min(n_components, X_train_raw.shape[1], X_train_raw.shape[0] - 1))
    scaler = StandardScaler()
    pca = PCA(n_components=max_components, random_state=seed)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_std = scaler.fit_transform(X_train_raw)
    X_train_pca = pca.fit_transform(X_train_std)
    X_train = angle_scaler.fit_transform(X_train_pca)
    X_test = angle_scaler.transform(pca.transform(scaler.transform(X_test_raw)))
    return X_train, X_test, {
        "pca_components": max_components,
        "pca_variance_retained": float(pca.explained_variance_ratio_.sum()),
    }


def prediction_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def rank_metrics(test_rows: list[dict], scores: np.ndarray, prefix: str) -> dict:
    by_sample = defaultdict(list)
    for row, score in zip(test_rows, scores):
        by_sample[row["sample_id"]].append((row, float(score)))

    top1_hits = []
    top3_hits = []
    top5_hits = []
    top1_distances = []
    positive_ranks = []
    for sample_rows in by_sample.values():
        ranked = sorted(sample_rows, key=lambda item: item[1], reverse=True)
        top1_hits.append(int(ranked[0][0]["label"] > 0))
        top3_hits.append(int(any(row["label"] > 0 for row, _ in ranked[:3])))
        top5_hits.append(int(any(row["label"] > 0 for row, _ in ranked[:5])))
        top1_distances.append(float(ranked[0][0]["center_dist"]))
        ranks = [idx + 1 for idx, (row, _) in enumerate(ranked) if row["label"] > 0]
        positive_ranks.append(float(min(ranks)) if ranks else float("nan"))

    return {
        f"{prefix}_top1_hit": float(np.mean(top1_hits)),
        f"{prefix}_top3_hit": float(np.mean(top3_hits)),
        f"{prefix}_top5_hit": float(np.mean(top5_hits)),
        f"{prefix}_top1_center_dist": float(np.nanmean(top1_distances)),
        f"{prefix}_best_positive_rank": float(np.nanmean(positive_ranks)),
    }


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def heatmap_metrics(heatmap: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> dict:
    heatmap_norm = normalize01(heatmap)
    peak = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)
    cy, cx = mask_centroid(gt_mask)
    pred_mask = heatmap_norm >= threshold
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt.sum()
    union = np.logical_or(pred_mask, gt).sum()
    dice = 1.0 if pred_sum + gt_sum == 0 else float(2 * intersection / (pred_sum + gt_sum))
    iou = 1.0 if union == 0 else float(intersection / union)
    center_dist = float(np.hypot(peak[0] - cy, peak[1] - cx)) if not math.isnan(cy) else float("nan")
    return {
        "pointing": float(gt_mask[peak[0], peak[1]] > 0),
        "peak_center_dist": center_dist,
        "dice": dice,
        "iou": iou,
    }


def gaussian_candidate_map(
    shape: tuple[int, int],
    sample_rows: list[dict],
    scores: np.ndarray,
    sigma: float,
    top_k: int,
) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros(shape, dtype=float)
    if len(sample_rows) == 0:
        return out

    order = np.argsort(scores)[::-1]
    if top_k > 0:
        order = order[:top_k]
    score_norm = normalize01(scores)
    for idx in order:
        row = sample_rows[int(idx)]
        weight = float(score_norm[int(idx)])
        if weight <= 0:
            weight = 1e-6
        dist2 = (yy - int(row["y"])) ** 2 + (xx - int(row["x"])) ** 2
        out += weight * np.exp(-dist2 / (2 * sigma ** 2))
    return normalize01(out)


def refinement_metrics(
    data_dir: Path,
    test_rows: list[dict],
    scores: np.ndarray,
    alpha: float,
    sigma: float,
    top_k: int,
    threshold: float,
) -> dict:
    by_sample = defaultdict(list)
    by_score = defaultdict(list)
    for row, score in zip(test_rows, scores):
        by_sample[row["sample_id"]].append(row)
        by_score[row["sample_id"]].append(float(score))

    base_records = []
    refined_records = []
    for sid, sample_rows in by_sample.items():
        gt = np.load(data_dir / f"gt_mask_{sid}.npy")
        heatmap = np.load(data_dir / f"pred_heatmap_{sid}.npy")
        base = normalize01(heatmap)
        candidate_map = gaussian_candidate_map(
            base.shape,
            sample_rows,
            np.asarray(by_score[sid], dtype=float),
            sigma=sigma,
            top_k=top_k,
        )
        refined = normalize01(alpha * base + (1.0 - alpha) * candidate_map)
        base_records.append(heatmap_metrics(base, gt, threshold=threshold))
        refined_records.append(heatmap_metrics(refined, gt, threshold=threshold))

    out = {}
    for key in ["pointing", "peak_center_dist", "dice", "iou"]:
        base_values = np.asarray([record[key] for record in base_records], dtype=float)
        refined_values = np.asarray([record[key] for record in refined_records], dtype=float)
        out[f"base_{key}"] = float(np.nanmean(base_values))
        out[f"refined_{key}"] = float(np.nanmean(refined_values))
        direction = -1.0 if key == "peak_center_dist" else 1.0
        out[f"refined_{key}_delta"] = float(direction * np.nanmean(refined_values - base_values))
    return out


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    test_rows,
    data_dir: Path,
    refine_alpha: float,
    refine_sigma: float,
    refine_top_k: int,
    refine_threshold: float,
):
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    pred = model.predict(X_test)
    scores = prediction_scores(model, X_test)
    row = {
        "model": name,
        "candidate_accuracy": accuracy_score(y_test, pred),
        "candidate_balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "candidate_f1": f1_score(y_test, pred, zero_division=0),
        "candidate_roc_auc": roc_auc_score(y_test, scores) if len(np.unique(y_test)) > 1 else float("nan"),
        "train_time_sec": elapsed,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    row.update(rank_metrics(test_rows, scores, "model"))
    row.update(refinement_metrics(
        data_dir,
        test_rows,
        scores,
        alpha=refine_alpha,
        sigma=refine_sigma,
        top_k=refine_top_k,
        threshold=refine_threshold,
    ))
    row.update(model_kernel_diagnostics(model, y_train))
    return row


def models(seed: int):
    return {
        "Classical_LogReg_C1": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed),
        "Classical_LinearSVM_C1": SVC(C=1.0, kernel="linear", probability=True, class_weight="balanced", random_state=seed),
        "Classical_RBFSVM_C1_gammaScale": SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=seed),
        "Classical_RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed),
        "Classical_ExtraTrees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed),
        "Classical_HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=seed,
        ),
        "QML_PQK_reps1_C1_balanced": ProjectedQuantumKernelSVC(
            gamma="scale", reps=1, C=1.0, class_weight="balanced"
        ),
        "QML_PQK_reps2_C1_balanced": ProjectedQuantumKernelSVC(
            gamma="scale", reps=2, C=1.0, class_weight="balanced"
        ),
        "QML_PQK_reps3_C1_balanced": ProjectedQuantumKernelSVC(
            gamma="scale", reps=3, C=1.0, class_weight="balanced"
        ),
        "QML_PQK_reps3_C10_balanced": ProjectedQuantumKernelSVC(
            gamma="scale", reps=3, C=10.0, class_weight="balanced"
        ),
    }


def sample_folds(sample_ids: np.ndarray, n_folds: int):
    unique_ids = np.asarray(sorted(np.unique(sample_ids)))
    if n_folds <= 0 or n_folds >= len(unique_ids):
        for sid in unique_ids:
            yield str(sid), np.asarray([sid])
        return

    for fold_idx in range(n_folds):
        held_out = unique_ids[fold_idx::n_folds]
        yield f"fold_{fold_idx + 1:02d}", held_out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metric_rows(rows: list[dict]) -> list[dict]:
    numeric_fields = [
        "candidate_accuracy",
        "candidate_balanced_accuracy",
        "candidate_f1",
        "candidate_roc_auc",
        "model_top1_hit",
        "model_top3_hit",
        "model_top5_hit",
        "model_top1_center_dist",
        "model_best_positive_rank",
        "base_pointing",
        "refined_pointing",
        "refined_pointing_delta",
        "base_peak_center_dist",
        "refined_peak_center_dist",
        "refined_peak_center_dist_delta",
        "base_dice",
        "refined_dice",
        "refined_dice_delta",
        "base_iou",
        "refined_iou",
        "refined_iou_delta",
        "train_time_sec",
        "pca_variance_retained",
        "kernel_target_alignment",
        "kernel_offdiag_mean",
        "kernel_offdiag_std",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    aggregate_rows = []
    for model_name, model_rows in sorted(grouped.items()):
        out = {"model": model_name, "folds": len(model_rows)}
        for field in numeric_fields:
            values = []
            for row in model_rows:
                try:
                    value = float(row.get(field, "nan"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            out[f"{field}_mean"] = float(np.mean(values)) if values else float("nan")
            out[f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        aggregate_rows.append(out)
    return aggregate_rows


def main():
    parser = argparse.ArgumentParser(description="Endoscopy guidance candidate-ranking benchmark")
    parser.add_argument(
        "--data_dir",
        default="/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance/eval/sample_data",
        help="Folder containing gt_mask_*.npy, pred_mask_*.npy, and pred_heatmap_*.npy",
    )
    parser.add_argument("--results_csv", default="endoscopy_guidance/results/candidate_ranking_metrics.csv")
    parser.add_argument("--aggregate_csv", default="endoscopy_guidance/results/candidate_ranking_aggregate.csv")
    parser.add_argument("--candidates_csv", default="endoscopy_guidance/results/candidate_table.csv")
    parser.add_argument("--top_n", type=int, default=12)
    parser.add_argument("--grid_stride", type=int, default=32)
    parser.add_argument("--nms_dist", type=int, default=18)
    parser.add_argument("--patch_radius", type=int, default=12)
    parser.add_argument("--n_components", type=int, default=6)
    parser.add_argument(
        "--image_features",
        action="store_true",
        help="Use image_*.npy RGB patch features when exported frames are available.",
    )
    parser.add_argument(
        "--sample_folds",
        type=int,
        default=0,
        help="Use grouped k-fold over sample IDs. Default 0 means leave-one-sample-out.",
    )
    parser.add_argument("--refine_alpha", type=float, default=0.35, help="Blend weight for original heatmap.")
    parser.add_argument("--refine_sigma", type=float, default=12.0, help="Gaussian sigma for candidate score blobs.")
    parser.add_argument("--refine_top_k", type=int, default=5, help="Number of ranked candidates used for refinement.")
    parser.add_argument("--refine_threshold", type=float, default=0.5, help="Threshold for refined heatmap Dice/IoU.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    X_raw, y, candidate_rows = build_candidates(
        data_dir,
        top_n=args.top_n,
        grid_stride=args.grid_stride,
        nms_dist=args.nms_dist,
        radius=args.patch_radius,
        use_image_features=args.image_features,
    )
    sample_ids = np.asarray([row["sample_id"] for row in candidate_rows])
    print(
        f"[data] candidates={len(y)} positives={int(y.sum())} "
        f"samples={len(np.unique(sample_ids))} features={X_raw.shape[1]}"
    )

    baseline_rows = []
    metric_rows = []
    for fold_name, held_out_ids in sample_folds(sample_ids, args.sample_folds):
        test_mask = np.isin(sample_ids, held_out_ids)
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_rows = [candidate_rows[i] for i in test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"[skip] {fold_name}: train/test split lacks both classes")
            continue

        X_train, X_test, prep = fit_low_dim(X_train_raw, X_test_raw, args.n_components, args.seed)
        heatmap_scores = np.asarray([row["heatmap_score"] for row in test_rows], dtype=float)
        baseline = {
            "held_out_sample": fold_name,
            "model": "Baseline_HeatmapScore",
            "candidate_accuracy": float("nan"),
            "candidate_balanced_accuracy": float("nan"),
            "candidate_f1": float("nan"),
            "candidate_roc_auc": roc_auc_score(y_test, heatmap_scores),
            "train_time_sec": 0.0,
            "confusion_matrix": None,
            **rank_metrics(test_rows, heatmap_scores, "model"),
            **prep,
            "train_count": len(y_train),
            "test_count": len(y_test),
            "train_positive": int(y_train.sum()),
            "test_positive": int(y_test.sum()),
        }
        baseline.update(refinement_metrics(
            data_dir,
            test_rows,
            heatmap_scores,
            alpha=args.refine_alpha,
            sigma=args.refine_sigma,
            top_k=args.refine_top_k,
            threshold=args.refine_threshold,
        ))
        baseline_rows.append(baseline)
        metric_rows.append(baseline)

        for name, estimator in models(args.seed).items():
            print(f"[run] held_out={fold_name} model={name}")
            row = evaluate_model(
                name,
                estimator,
                X_train,
                y_train,
                X_test,
                y_test,
                test_rows,
                data_dir=data_dir,
                refine_alpha=args.refine_alpha,
                refine_sigma=args.refine_sigma,
                refine_top_k=args.refine_top_k,
                refine_threshold=args.refine_threshold,
            )
            row.update({
                "held_out_sample": fold_name,
                **prep,
                "train_count": len(y_train),
                "test_count": len(y_test),
                "train_positive": int(y_train.sum()),
                "test_positive": int(y_test.sum()),
            })
            metric_rows.append(row)

    fields = [
        "held_out_sample",
        "model",
        "candidate_accuracy",
        "candidate_balanced_accuracy",
        "candidate_f1",
        "candidate_roc_auc",
        "model_top1_hit",
        "model_top3_hit",
        "model_top5_hit",
        "model_top1_center_dist",
        "model_best_positive_rank",
        "base_pointing",
        "refined_pointing",
        "refined_pointing_delta",
        "base_peak_center_dist",
        "refined_peak_center_dist",
        "refined_peak_center_dist_delta",
        "base_dice",
        "refined_dice",
        "refined_dice_delta",
        "base_iou",
        "refined_iou",
        "refined_iou_delta",
        "train_time_sec",
        "confusion_matrix",
        "pca_components",
        "pca_variance_retained",
        "train_count",
        "test_count",
        "train_positive",
        "test_positive",
        "kernel_target_alignment",
        "kernel_diag_mean",
        "kernel_offdiag_mean",
        "kernel_offdiag_std",
    ]
    write_csv(Path(args.results_csv), metric_rows, fields)
    aggregate_rows = aggregate_metric_rows(metric_rows)
    aggregate_fields = ["model", "folds"]
    for field in [
        "candidate_accuracy",
        "candidate_balanced_accuracy",
        "candidate_f1",
        "candidate_roc_auc",
        "model_top1_hit",
        "model_top3_hit",
        "model_top5_hit",
        "model_top1_center_dist",
        "model_best_positive_rank",
        "base_pointing",
        "refined_pointing",
        "refined_pointing_delta",
        "base_peak_center_dist",
        "refined_peak_center_dist",
        "refined_peak_center_dist_delta",
        "base_dice",
        "refined_dice",
        "refined_dice_delta",
        "base_iou",
        "refined_iou",
        "refined_iou_delta",
        "train_time_sec",
        "pca_variance_retained",
        "kernel_target_alignment",
        "kernel_offdiag_mean",
        "kernel_offdiag_std",
    ]:
        aggregate_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(Path(args.aggregate_csv), aggregate_rows, aggregate_fields)
    write_csv(
        Path(args.candidates_csv),
        candidate_rows,
        ["sample_id", "y", "x", "source_index", "label", "heatmap_score", "center_dist"],
    )
    print(f"[saved] {args.results_csv}")
    print(f"[saved] {args.aggregate_csv}")
    print(f"[saved] {args.candidates_csv}")


if __name__ == "__main__":
    main()
