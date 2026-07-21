"""
label_efficiency_prompt_selector.py
===================================
Small, cached label-efficiency study for prompt-quality selectors.

The full-context selector is strong, but a publishable QML angle may be stronger
under limited prompt-quality labels. This script trains on increasing numbers of
training frames and evaluates the same held-out split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from candidate_ranking_benchmark import fit_low_dim
from sam_prompt_quality_ranker import projected_quantum_features, sample_level_eval


def split_masks(qdf: pd.DataFrame, eval_split: str) -> tuple[np.ndarray, np.ndarray]:
    if eval_split == "val":
        train_ids = sorted(qdf.loc[qdf["sequence_id"].le(23), "sample_id"].unique())
        eval_mask = qdf["split"].eq("val").to_numpy()
    else:
        train_ids = sorted(qdf.loc[qdf["sequence_id"].le(26), "sample_id"].unique())
        eval_mask = qdf["split"].eq("test").to_numpy()
    return np.asarray(train_ids), eval_mask


def choose_train_ids(train_ids: np.ndarray, n_frames: int, seed: int) -> np.ndarray:
    if n_frames <= 0 or n_frames >= len(train_ids):
        return train_ids
    rng = np.random.default_rng(seed)
    chosen = rng.choice(train_ids, size=n_frames, replace=False)
    return np.sort(chosen)


def main():
    parser = argparse.ArgumentParser(description="Label-efficiency benchmark for prompt-quality selectors")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--semantic_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed.npy")
    parser.add_argument("--context_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed_context.npy")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/label_efficiency_prompt_selector.csv")
    parser.add_argument("--eval_split", choices=["val", "test"], default="test")
    parser.add_argument("--train_frames", type=int, nargs="+", default=[25, 50, 100, 200, 0])
    parser.add_argument("--pqk_components", type=int, default=12)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    qdf = pd.read_csv(args.prompt_quality_csv)
    X_semantic = np.load(args.semantic_features).astype(np.float32)
    X_context = np.load(args.context_features).astype(np.float32)
    train_ids, eval_mask = split_masks(qdf, args.eval_split)
    eval_df = qdf.loc[eval_mask].copy()
    y_all = qdf["sam_dice"].to_numpy(dtype=float)

    rows = []
    rows.append(sample_level_eval(eval_df, eval_df["sam_dice"].to_numpy(dtype=float), "oracle_prompt_quality"))
    rows[-1]["train_frames"] = np.nan
    rows.append(sample_level_eval(eval_df, eval_df["heatmap_score"].to_numpy(dtype=float), "heatmap_score"))
    rows[-1]["train_frames"] = np.nan

    for n_frames in args.train_frames:
        chosen_ids = choose_train_ids(train_ids, n_frames, args.seed)
        train_mask = qdf["sample_id"].isin(chosen_ids).to_numpy()
        y_train = y_all[train_mask]

        classical = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=args.seed)
        classical.fit(X_context[train_mask], y_train)
        row = sample_level_eval(eval_df, classical.predict(X_context[eval_mask]), "Classical_context_HistGB")
        row["train_frames"] = len(chosen_ids)
        rows.append(row)

        X_train_q, X_eval_q, pca_info = fit_low_dim(
            X_semantic[train_mask],
            X_semantic[eval_mask],
            args.pqk_components,
            args.seed,
        )
        Z_train = projected_quantum_features(X_train_q, args.pqk_reps)
        Z_eval = projected_quantum_features(X_eval_q, args.pqk_reps)
        qml = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=args.seed)
        qml.fit(Z_train, y_train)
        row = sample_level_eval(eval_df, qml.predict(Z_eval), f"QML_PQF_semantic_HistGB_{args.pqk_components}pc")
        row["train_frames"] = len(chosen_ids)
        row["pca_variance"] = pca_info["pca_variance_retained"]
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["train_frames", "dice_mean"], na_position="first")
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
