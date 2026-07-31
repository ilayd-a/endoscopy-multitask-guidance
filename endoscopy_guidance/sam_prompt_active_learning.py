"""
sam_prompt_active_learning.py
=============================
Active learning for SAM prompt-quality supervision.

This experiment treats SAM prompt masks as candidate interventions and assumes
their true Dice scores are expensive labels. Acquisition policies choose which
candidate prompts to label next; a final prompt-quality regressor is trained on
only those labels and evaluated on held-out frames.

The quantum role is deliberately upstream: a projected quantum kernel selects
informative prompt labels under a small annotation budget. The final clinical
output remains a ranked prompt/mask choice per frame.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
EBTC_EXPERIMENTS = ROOT / "EndoscopicBladderTissue" / "experiments"
if str(EBTC_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EBTC_EXPERIMENTS))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from publication_benchmark_ebtc import ProjectedQuantumKernelSVC
from sam_prompt_quality_ranker import (
    blended_sample_scores,
    prefilter_eval_candidates,
    projected_quantum_features,
    sample_level_eval,
)


def stable_name_seed(text: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % 1009


class QuantumAcquisitionModel:
    def __init__(self, components: int, reps: int, seed: int):
        self.components = components
        self.reps = reps
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca = None
        self.angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        self.model = ProjectedQuantumKernelSVC(gamma="scale", reps=reps, C=1.0, class_weight="balanced")
        self.variance_retained = np.nan

    def _fit_transform(self, X: np.ndarray) -> np.ndarray:
        max_components = max(1, min(self.components, X.shape[1], X.shape[0] - 1))
        self.pca = PCA(n_components=max_components, random_state=self.seed)
        X_std = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_std)
        self.variance_retained = float(self.pca.explained_variance_ratio_.sum())
        return self.angle_scaler.fit_transform(X_pca)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self.angle_scaler.transform(self.pca.transform(self.scaler.transform(X)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self._fit_transform(X), y)
        self.classes_ = self.model.model.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._transform(X))


class QuantumFeatureRidgeRegressor:
    def __init__(self, components: int, reps: int, seed: int):
        self.components = components
        self.reps = reps
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca = None
        self.angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        self.variance_retained = np.nan

    def _quantum_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            max_components = max(1, min(self.components, X.shape[1], X.shape[0] - 1))
            self.pca = PCA(n_components=max_components, random_state=self.seed)
            X_std = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_std)
            self.variance_retained = float(self.pca.explained_variance_ratio_.sum())
            X_angle = self.angle_scaler.fit_transform(X_pca)
        else:
            X_angle = self.angle_scaler.transform(self.pca.transform(self.scaler.transform(X)))
        return projected_quantum_features(X_angle, self.reps)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self._quantum_features(X, fit=True), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._quantum_features(X))


def load_cache(csv_path: Path, features_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    qdf = pd.read_csv(csv_path)
    Q = np.load(features_path).astype(np.float32)
    if len(qdf) != len(Q):
        raise ValueError(f"Cache mismatch: {len(qdf)} rows but {len(Q)} feature rows")
    return qdf, Q


def split_masks(qdf: pd.DataFrame, eval_split: str) -> tuple[np.ndarray, np.ndarray]:
    if eval_split == "val":
        return qdf["sequence_id"].le(23).to_numpy(), qdf["split"].eq("val").to_numpy()
    if eval_split == "test":
        return qdf["sequence_id"].le(26).to_numpy(), qdf["split"].eq("test").to_numpy()
    raise ValueError(f"Unknown eval_split={eval_split}")


def balanced_seed_indices(pool_idx: np.ndarray, labels: np.ndarray, n_seed: int, rng: np.random.Generator) -> np.ndarray:
    positives = pool_idx[labels[pool_idx] == 1]
    negatives = pool_idx[labels[pool_idx] == 0]
    selected = []
    if len(positives):
        selected.append(rng.choice(positives, size=min(n_seed // 2, len(positives)), replace=False))
    selected_count = sum(len(part) for part in selected)
    if len(negatives) and selected_count < n_seed:
        selected.append(rng.choice(negatives, size=min(n_seed - selected_count, len(negatives)), replace=False))
    if selected:
        chosen = np.concatenate(selected)
    else:
        chosen = np.asarray([], dtype=int)
    if len(chosen) < min(n_seed, len(pool_idx)):
        remaining = np.setdiff1d(pool_idx, chosen, assume_unique=False)
        chosen = np.concatenate([chosen, rng.choice(remaining, size=min(n_seed - len(chosen), len(remaining)), replace=False)])
    rng.shuffle(chosen)
    return chosen.astype(int)


def fit_acquisition(strategy: str, X: np.ndarray, y: np.ndarray, components: int, reps: int, seed: int):
    if strategy in {"classical_uncertainty", "classical_positive", "classical_positive_hybrid"}:
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)).fit(X, y)
    if strategy in {"pqk_uncertainty", "pqk_hybrid", "pqk_positive", "pqk_positive_hybrid"}:
        return QuantumAcquisitionModel(components, reps, seed).fit(X, y)
    raise ValueError(f"Unsupported acquisition strategy={strategy}")


def positive_scores(model, X: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(X)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    return probs[:, classes.index(1)]


def acquire(
    strategy: str,
    X: np.ndarray,
    labels: np.ndarray,
    labeled_idx: np.ndarray,
    unlabeled_idx: np.ndarray,
    batch_size: int,
    components: int,
    reps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(unlabeled_idx) <= batch_size:
        return unlabeled_idx
    if strategy == "random":
        return rng.choice(unlabeled_idx, size=batch_size, replace=False)
    model = fit_acquisition(strategy, X[labeled_idx], labels[labeled_idx], components, reps, seed)
    scores = positive_scores(model, X[unlabeled_idx])
    uncertainty = np.abs(scores - 0.5)
    if strategy in {"classical_positive", "pqk_positive"}:
        return unlabeled_idx[np.argsort(scores)[::-1][:batch_size]]
    if strategy in {"classical_positive_hybrid", "pqk_positive_hybrid"}:
        positive_n = max(1, batch_size // 2)
        positive_local = np.argsort(scores)[::-1][:positive_n]
        remaining = np.setdiff1d(np.arange(len(unlabeled_idx)), positive_local, assume_unique=False)
        random_n = min(batch_size - len(positive_local), len(remaining))
        random_local = rng.choice(remaining, size=random_n, replace=False) if random_n else np.asarray([], dtype=int)
        return unlabeled_idx[np.concatenate([positive_local, random_local])]
    if strategy == "pqk_hybrid":
        uncertain_n = max(1, batch_size // 2)
        uncertain_local = np.argsort(uncertainty)[:uncertain_n]
        remaining = np.setdiff1d(np.arange(len(unlabeled_idx)), uncertain_local, assume_unique=False)
        random_n = min(batch_size - len(uncertain_local), len(remaining))
        random_local = rng.choice(remaining, size=random_n, replace=False) if random_n else np.asarray([], dtype=int)
        return unlabeled_idx[np.concatenate([uncertain_local, random_local])]
    return unlabeled_idx[np.argsort(uncertainty)[:batch_size]]


def make_final_model(name: str, components: int, reps: int, seed: int):
    if name == "classical_histgb":
        return HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=seed)
    if name == "classical_ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "quantum_feature_ridge":
        return QuantumFeatureRidgeRegressor(components, reps, seed)
    raise ValueError(f"Unknown final_model={name}")


def evaluate_final_model(
    strategy: str,
    final_model_name: str,
    model,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    labeled_count: int,
    repeat: int,
    seed_count: int,
    acquired_positive_rate: float,
) -> dict:
    model.fit(X_train, y_train)
    scores = model.predict(X_eval)
    row = sample_level_eval(eval_df, scores, f"{strategy}:{final_model_name}")
    row.update({
        "strategy": strategy,
        "final_model": final_model_name,
        "labeled_count": int(labeled_count),
        "repeat": int(repeat),
        "seed_labels": int(seed_count),
        "selected_positive_rate": float(acquired_positive_rate),
        "train_label_dice_mean": float(train_df["sam_dice"].mean()),
    })
    if final_model_name != "quantum_feature_ridge":
        for model_w, sam_w, heatmap_w in [(0.7, 0.2, 0.1), (0.5, 0.3, 0.2)]:
            blend = blended_sample_scores(eval_df, scores, model_w, sam_w, heatmap_w)
            blend_row = sample_level_eval(eval_df, blend, f"{strategy}:{final_model_name}:blend")
            row[f"blend_{model_w:g}_{sam_w:g}_{heatmap_w:g}_dice"] = blend_row["dice_mean"]
    return row


def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metrics = ["dice_mean", "iou_mean", "prompt_hit", "selected_positive_rate", "train_label_dice_mean"]
    grouped = df.groupby(["strategy", "final_model", "labeled_count"], as_index=False)
    out = grouped.agg(
        runs=("repeat", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_std": (metric, "std") for metric in metrics},
    )
    return out.sort_values(["labeled_count", "dice_mean_mean"], ascending=[True, False])


def paired_differences(rows: list[dict], metric: str = "dice_mean") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keys = ["repeat", "final_model", "labeled_count"]
    out = []
    baselines = ["random", "classical_uncertainty"]
    for strategy in ["pqk_uncertainty", "pqk_hybrid"]:
        for baseline in baselines:
            left = df[df["strategy"].eq(strategy)][keys + [metric]].rename(columns={metric: "strategy_value"})
            right = df[df["strategy"].eq(baseline)][keys + [metric]].rename(columns={metric: "baseline_value"})
            paired = left.merge(right, on=keys, how="inner")
            for (final_model, labeled_count), group in paired.groupby(["final_model", "labeled_count"]):
                diff = group["strategy_value"].to_numpy(dtype=float) - group["baseline_value"].to_numpy(dtype=float)
                if len(diff) > 1:
                    rng = np.random.default_rng(17 + int(labeled_count))
                    boot = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
                    ci_low = float(np.percentile(boot, 2.5))
                    ci_high = float(np.percentile(boot, 97.5))
                    signs = rng.choice([-1.0, 1.0], size=(10000, len(diff)), replace=True)
                    observed = abs(float(diff.mean()))
                    permuted = np.abs((signs * diff).mean(axis=1))
                    p_value = float((np.sum(permuted >= observed) + 1) / (len(permuted) + 1))
                else:
                    ci_low = ci_high = p_value = np.nan
                out.append({
                    "strategy": strategy,
                    "baseline": baseline,
                    "final_model": final_model,
                    "labeled_count": int(labeled_count),
                    "n_pairs": int(len(diff)),
                    "strategy_mean": float(group["strategy_value"].mean()),
                    "baseline_mean": float(group["baseline_value"].mean()),
                    "diff_mean": float(diff.mean()),
                    "diff_std": float(diff.std(ddof=1)) if len(diff) > 1 else np.nan,
                    "diff_ci_low": ci_low,
                    "diff_ci_high": ci_high,
                    "signflip_p": p_value,
                })
    return pd.DataFrame(out).sort_values(["final_model", "labeled_count", "strategy", "baseline"])


def main():
    parser = argparse.ArgumentParser(description="Active learning for SAM prompt-quality labels")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--prompt_quality_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed_context.npy")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/sam_prompt_active_learning_metrics.csv")
    parser.add_argument("--aggregate_csv", default="endoscopy_guidance/results/sam_prompt_active_learning_aggregate.csv")
    parser.add_argument("--paired_csv", default="endoscopy_guidance/results/sam_prompt_active_learning_paired.csv")
    parser.add_argument("--eval_split", choices=["val", "test"], default="test")
    parser.add_argument("--eval_candidate_cap", type=int, default=60)
    parser.add_argument("--initial_labels", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--good_threshold", type=float, default=0.5)
    parser.add_argument("--strategies", nargs="+", default=["random", "classical_uncertainty", "pqk_uncertainty", "pqk_hybrid"])
    parser.add_argument("--final_models", nargs="+", default=["classical_histgb", "quantum_feature_ridge"])
    parser.add_argument("--pqk_components", type=int, default=12)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    qdf, Q = load_cache(Path(args.prompt_quality_csv), Path(args.prompt_quality_features))
    train_mask, eval_mask = split_masks(qdf, args.eval_split)
    train_df_all = qdf.loc[train_mask].copy().reset_index(drop=True)
    eval_df_all = qdf.loc[eval_mask].copy().reset_index(drop=True)
    X_train_all = Q[train_mask]
    X_eval_all = Q[eval_mask]
    eval_df, X_eval = prefilter_eval_candidates(eval_df_all, X_eval_all, args.eval_candidate_cap)
    eval_df = eval_df.reset_index(drop=True)

    y_dice = train_df_all["sam_dice"].to_numpy(dtype=float)
    good_threshold = args.good_threshold if args.good_threshold > 0 else float(np.quantile(y_dice, 0.70))
    y_good = (y_dice >= good_threshold).astype(np.int64)
    pool_idx = np.arange(len(train_df_all))
    rows = []

    for repeat in range(args.repeats):
        for strategy in args.strategies:
            rng = np.random.default_rng(args.seed + 1000 * repeat + stable_name_seed(strategy))
            labeled_idx = balanced_seed_indices(pool_idx, y_good, args.initial_labels, rng)
            acquired_positive_rates = []
            for round_idx in range(args.rounds + 1):
                labeled_df = train_df_all.iloc[labeled_idx].copy()
                for final_model_name in args.final_models:
                    model = make_final_model(final_model_name, args.pqk_components, args.pqk_reps, args.seed + repeat + round_idx)
                    rows.append(evaluate_final_model(
                        strategy,
                        final_model_name,
                        model,
                        labeled_df,
                        eval_df,
                        X_train_all[labeled_idx],
                        y_dice[labeled_idx],
                        X_eval,
                        len(labeled_idx),
                        repeat,
                        args.initial_labels,
                        float(np.mean(acquired_positive_rates)) if acquired_positive_rates else float(y_good[labeled_idx].mean()),
                    ))
                if round_idx == args.rounds:
                    break
                unlabeled_idx = np.setdiff1d(pool_idx, labeled_idx, assume_unique=False)
                batch = acquire(
                    strategy,
                    X_train_all,
                    y_good,
                    labeled_idx,
                    unlabeled_idx,
                    args.batch_size,
                    args.pqk_components,
                    args.pqk_reps,
                    args.seed + 1000 * repeat + 31 * round_idx,
                )
                acquired_positive_rates.append(float(y_good[batch].mean()) if len(batch) else np.nan)
                labeled_idx = np.concatenate([labeled_idx, batch])
            print(f"[repeat {repeat + 1}/{args.repeats}] {strategy} labeled={len(labeled_idx)}")

    output_csv = Path(args.output_csv)
    aggregate_csv = Path(args.aggregate_csv)
    paired_csv = Path(args.paired_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    aggregate(rows).to_csv(aggregate_csv, index=False)
    paired_differences(rows).to_csv(paired_csv, index=False)
    print(f"[saved] {output_csv}")
    print(f"[saved] {aggregate_csv}")
    print(f"[saved] {paired_csv}")
    print(aggregate(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
