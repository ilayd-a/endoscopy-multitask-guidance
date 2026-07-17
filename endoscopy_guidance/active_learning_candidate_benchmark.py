"""
active_learning_candidate_benchmark.py
======================================
Active-learning benchmark for endoscopic candidate guidance.

This experiment treats candidate labels as expensive annotations. It simulates
annotation rounds on sequence-held-out candidate pools and compares whether a
projected quantum-kernel uncertainty policy selects more useful annotations
than random or classical uncertainty sampling.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from candidate_ranking_benchmark import (
    ProjectedQuantumKernelSVC,
    build_candidates,
    model_kernel_diagnostics,
    prediction_scores,
    rank_metrics,
    refinement_metrics,
    sample_folds,
)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def balanced_seed_indices(pool_idx: np.ndarray, y: np.ndarray, n_seed: int, rng: np.random.Generator) -> np.ndarray:
    positives = pool_idx[y[pool_idx] == 1]
    negatives = pool_idx[y[pool_idx] == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return rng.choice(pool_idx, size=min(n_seed, len(pool_idx)), replace=False)

    pos_n = min(n_seed // 2, len(positives))
    neg_n = min(n_seed - pos_n, len(negatives))
    selected = [
        rng.choice(positives, size=pos_n, replace=False),
        rng.choice(negatives, size=neg_n, replace=False),
    ]
    selected = np.concatenate(selected)
    if len(selected) < n_seed:
        remaining = np.setdiff1d(pool_idx, selected, assume_unique=False)
        extra_n = min(n_seed - len(selected), len(remaining))
        if extra_n:
            selected = np.concatenate([selected, rng.choice(remaining, size=extra_n, replace=False)])
    rng.shuffle(selected)
    return selected


class LowDimPreprocessor:
    def __init__(self, n_components: int, seed: int):
        self.n_components = n_components
        self.seed = seed

    def fit(self, X_raw: np.ndarray):
        max_components = max(1, min(self.n_components, X_raw.shape[1], X_raw.shape[0] - 1))
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=max_components, random_state=self.seed)
        self.angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        X_std = self.scaler.fit_transform(X_raw)
        X_pca = self.pca.fit_transform(X_std)
        self.angle_scaler.fit(X_pca)
        self.pca_components_ = max_components
        self.pca_variance_retained_ = float(self.pca.explained_variance_ratio_.sum())
        return self

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        return self.angle_scaler.transform(self.pca.transform(self.scaler.transform(X_raw)))


def make_acquisition_model(strategy: str, seed: int):
    if strategy == "classical_uncertainty":
        return LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed)
    if strategy in {"pqk_uncertainty", "pqk_diversity", "pqk_hybrid"}:
        return ProjectedQuantumKernelSVC(gamma="scale", reps=2, C=1.0, class_weight="balanced")
    raise ValueError(f"Unsupported acquisition strategy: {strategy}")


def make_eval_models(seed: int):
    return {
        "Classical_LogReg_C1": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed),
        "Classical_RBFSVM_C1_gammaScale": SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=seed
        ),
        "QML_PQK_reps2_C1_balanced": ProjectedQuantumKernelSVC(
            gamma="scale", reps=2, C=1.0, class_weight="balanced"
        ),
    }


def uncertainty_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if np.nanmin(scores) < 0.0 or np.nanmax(scores) > 1.0:
        scores = (scores - np.nanmin(scores)) / max(np.nanmax(scores) - np.nanmin(scores), 1e-12)
    return np.abs(scores - 0.5)


def acquire_batch(
    strategy: str,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    unlabeled_idx: np.ndarray,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(unlabeled_idx) <= batch_size:
        return unlabeled_idx
    if strategy == "random":
        return rng.choice(unlabeled_idx, size=batch_size, replace=False)

    model = make_acquisition_model(strategy, seed)
    model.fit(X_labeled, y_labeled)
    scores = prediction_scores(model, X_unlabeled)
    uncertainty = uncertainty_from_scores(scores)

    if strategy != "pqk_diversity":
        if strategy == "pqk_hybrid":
            uncertain_n = max(1, batch_size // 2)
            uncertain_local = np.argsort(uncertainty)[:uncertain_n]
            remaining_local = np.setdiff1d(np.arange(len(unlabeled_idx)), uncertain_local, assume_unique=False)
            if len(remaining_local) == 0:
                return unlabeled_idx[uncertain_local]
            random_n = min(batch_size - len(uncertain_local), len(remaining_local))
            random_local = rng.choice(remaining_local, size=random_n, replace=False)
            return unlabeled_idx[np.concatenate([uncertain_local, random_local])]
        chosen_local = np.argsort(uncertainty)[:batch_size]
        return unlabeled_idx[chosen_local]

    candidate_n = min(len(unlabeled_idx), max(batch_size * 6, batch_size))
    candidate_local = np.argsort(uncertainty)[:candidate_n]
    project = getattr(model, "_project", None)
    if project is None:
        return unlabeled_idx[candidate_local[:batch_size]]

    selected_local = []
    candidate_features = project(X_unlabeled[candidate_local])
    labeled_features = project(X_labeled)
    distances_to_labeled = np.min(
        np.sum((candidate_features[:, None, :] - labeled_features[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    first = int(np.argmax(distances_to_labeled))
    selected_local.append(first)
    while len(selected_local) < min(batch_size, len(candidate_local)):
        selected_features = candidate_features[selected_local]
        distances_to_selected = np.min(
            np.sum((candidate_features[:, None, :] - selected_features[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        combined_distance = np.minimum(distances_to_labeled, distances_to_selected)
        combined_distance[selected_local] = -np.inf
        selected_local.append(int(np.argmax(combined_distance)))
    return unlabeled_idx[candidate_local[selected_local]]


def evaluate_model(
    model_name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_rows: list[dict],
    data_dir: Path,
    refine_alpha: float,
    refine_sigma: float,
    refine_top_k: int,
    refine_threshold: float,
) -> dict:
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    y_pred = model.predict(X_test)
    scores = prediction_scores(model, X_test)
    row = {
        "eval_model": model_name,
        "candidate_accuracy": accuracy_score(y_test, y_pred),
        "candidate_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "candidate_f1": f1_score(y_test, y_pred, zero_division=0),
        "candidate_roc_auc": roc_auc_score(y_test, scores) if len(np.unique(y_test)) > 1 else float("nan"),
        "train_time_sec": elapsed,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
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


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"], row["eval_model"], row["labeled_count"])].append(row)

    metrics = [
        "candidate_balanced_accuracy",
        "candidate_f1",
        "candidate_roc_auc",
        "model_top1_hit",
        "model_top3_hit",
        "model_top5_hit",
        "model_best_positive_rank",
        "refined_pointing",
        "refined_dice",
        "refined_iou",
        "refined_peak_center_dist",
        "selected_positive_rate",
        "train_time_sec",
        "pca_variance_retained",
        "kernel_target_alignment",
        "kernel_offdiag_mean",
        "kernel_offdiag_std",
    ]
    out = []
    for (strategy, eval_model, labeled_count), group_rows in sorted(grouped.items()):
        row = {
            "strategy": strategy,
            "eval_model": eval_model,
            "labeled_count": labeled_count,
            "runs": len(group_rows),
        }
        for metric in metrics:
            values = []
            for source in group_rows:
                try:
                    value = float(source.get(metric, "nan"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            row[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description="Active learning for endoscopic candidate guidance")
    parser.add_argument("--data_dir", default="endoscopy_guidance/exports/cvc_test_rgb")
    parser.add_argument("--results_csv", default="endoscopy_guidance/results/cvc_test_rgb_active_learning_metrics.csv")
    parser.add_argument("--aggregate_csv", default="endoscopy_guidance/results/cvc_test_rgb_active_learning_aggregate.csv")
    parser.add_argument("--top_n", type=int, default=8)
    parser.add_argument("--grid_stride", type=int, default=48)
    parser.add_argument("--nms_dist", type=int, default=20)
    parser.add_argument("--patch_radius", type=int, default=14)
    parser.add_argument("--n_components", type=int, default=6)
    parser.add_argument("--sample_folds", type=int, default=5)
    parser.add_argument("--initial_labels", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "classical_uncertainty", "pqk_uncertainty", "pqk_diversity", "pqk_hybrid"],
    )
    parser.add_argument("--image_features", action="store_true")
    parser.add_argument("--refine_alpha", type=float, default=0.0)
    parser.add_argument("--refine_sigma", type=float, default=20.0)
    parser.add_argument("--refine_top_k", type=int, default=5)
    parser.add_argument("--refine_threshold", type=float, default=0.5)
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

    all_rows = []
    for fold_name, held_out_ids in sample_folds(sample_ids, args.sample_folds):
        test_mask = np.isin(sample_ids, held_out_ids)
        pool_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        test_rows = [candidate_rows[i] for i in test_idx]
        if len(np.unique(y[pool_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            print(f"[skip] {fold_name}: split lacks both classes")
            continue

        for repeat_idx in range(args.repeats):
            repeat_seed = args.seed + 1009 * repeat_idx
            seed_idx = balanced_seed_indices(pool_idx, y, args.initial_labels, np.random.default_rng(repeat_seed))
            for strategy in args.strategies:
                labeled_idx = seed_idx.copy()
                newly_selected = seed_idx.copy()
                for round_idx in range(args.rounds + 1):
                    prep = LowDimPreprocessor(args.n_components, seed=repeat_seed + round_idx).fit(X_raw[labeled_idx])
                    X_labeled = prep.transform(X_raw[labeled_idx])
                    X_test = prep.transform(X_raw[test_idx])
                    y_labeled = y[labeled_idx]
                    y_test = y[test_idx]

                    if len(np.unique(y_labeled)) < 2:
                        continue

                    for model_name, model in make_eval_models(repeat_seed + round_idx).items():
                        print(
                            f"[run] fold={fold_name} repeat={repeat_idx} strategy={strategy} "
                            f"round={round_idx} labels={len(labeled_idx)} eval={model_name}"
                        )
                        row = evaluate_model(
                            model_name,
                            model,
                            X_labeled,
                            y_labeled,
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
                            "repeat": repeat_idx,
                            "strategy": strategy,
                            "round": round_idx,
                            "labeled_count": len(labeled_idx),
                            "labeled_positive": int(y_labeled.sum()),
                            "selected_count": len(newly_selected),
                            "selected_positive": int(y[newly_selected].sum()),
                            "selected_positive_rate": float(y[newly_selected].mean()) if len(newly_selected) else float("nan"),
                            "test_count": len(test_idx),
                            "test_positive": int(y_test.sum()),
                            "pca_components": prep.pca_components_,
                            "pca_variance_retained": prep.pca_variance_retained_,
                        })
                        all_rows.append(row)

                    if round_idx == args.rounds:
                        break
                    unlabeled_idx = np.setdiff1d(pool_idx, labeled_idx, assume_unique=False)
                    if len(unlabeled_idx) == 0:
                        break
                    X_unlabeled = prep.transform(X_raw[unlabeled_idx])
                    newly_selected = acquire_batch(
                        strategy,
                        X_labeled,
                        y_labeled,
                        X_unlabeled,
                        unlabeled_idx,
                        batch_size=args.batch_size,
                        seed=repeat_seed + 7919 * round_idx,
                    )
                    labeled_idx = np.concatenate([labeled_idx, newly_selected])

    fields = [
        "held_out_sample",
        "repeat",
        "strategy",
        "round",
        "labeled_count",
        "labeled_positive",
        "selected_count",
        "selected_positive",
        "selected_positive_rate",
        "eval_model",
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
        "base_peak_center_dist",
        "refined_peak_center_dist",
        "base_dice",
        "refined_dice",
        "base_iou",
        "refined_iou",
        "train_time_sec",
        "confusion_matrix",
        "pca_components",
        "pca_variance_retained",
        "test_count",
        "test_positive",
        "kernel_target_alignment",
        "kernel_diag_mean",
        "kernel_offdiag_mean",
        "kernel_offdiag_std",
    ]
    write_csv(Path(args.results_csv), all_rows, fields)
    aggregate = aggregate_rows(all_rows)
    aggregate_fields = ["strategy", "eval_model", "labeled_count", "runs"]
    for field in [
        "candidate_balanced_accuracy",
        "candidate_f1",
        "candidate_roc_auc",
        "model_top1_hit",
        "model_top3_hit",
        "model_top5_hit",
        "model_best_positive_rank",
        "refined_pointing",
        "refined_dice",
        "refined_iou",
        "refined_peak_center_dist",
        "selected_positive_rate",
        "train_time_sec",
        "pca_variance_retained",
        "kernel_target_alignment",
        "kernel_offdiag_mean",
        "kernel_offdiag_std",
    ]:
        aggregate_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(Path(args.aggregate_csv), aggregate, aggregate_fields)
    print(f"[saved] {args.results_csv}")
    print(f"[saved] {args.aggregate_csv}")


if __name__ == "__main__":
    main()
