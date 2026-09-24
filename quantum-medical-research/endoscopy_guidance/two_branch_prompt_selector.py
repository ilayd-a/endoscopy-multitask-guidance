"""
two_branch_prompt_selector.py
=============================
Two-branch prompt-quality selector for cached SAM prompt candidates.

Branch A learns a strong classical prior from context-augmented prompt features.
Branch B learns a projected-quantum-feature semantic score from SAM-embedding
features only. The branches are blended per frame after score normalization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
EBTC_EXPERIMENTS = ROOT / "EndoscopicBladderTissue" / "experiments"
import sys

if str(EBTC_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EBTC_EXPERIMENTS))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from candidate_ranking_benchmark import fit_low_dim
from sam_prompt_quality_ranker import per_sample_normalize, projected_quantum_features, sample_level_eval


def split_masks(qdf: pd.DataFrame, eval_split: str) -> tuple[np.ndarray, np.ndarray]:
    if eval_split == "val":
        train_mask = qdf["sequence_id"].le(23).to_numpy()
        eval_mask = qdf["split"].eq("val").to_numpy()
    else:
        train_mask = qdf["sequence_id"].le(26).to_numpy()
        eval_mask = qdf["split"].eq("test").to_numpy()
    return train_mask, eval_mask


def blend_scores(qdf: pd.DataFrame, prior_scores: np.ndarray, semantic_scores: np.ndarray, prior_weight: float) -> np.ndarray:
    sample_ids = qdf["sample_id"].to_numpy()
    prior = per_sample_normalize(prior_scores, sample_ids)
    semantic = per_sample_normalize(semantic_scores, sample_ids)
    return prior_weight * prior + (1.0 - prior_weight) * semantic


def residual_scores(prior_scores: np.ndarray, residual_predictions: np.ndarray, residual_weight: float) -> np.ndarray:
    return prior_scores + residual_weight * residual_predictions


def shortlist_rerank_scores(
    qdf: pd.DataFrame,
    primary_scores: np.ndarray,
    top_k: int,
    rerank_mode: str,
) -> np.ndarray:
    out = np.full(len(qdf), -1e9, dtype=float)
    sample_ids = qdf["sample_id"].to_numpy()
    sam_norm = per_sample_normalize(qdf["sam_score"].to_numpy(dtype=float), sample_ids)
    heatmap_norm = per_sample_normalize(qdf["heatmap_score"].to_numpy(dtype=float), sample_ids)
    center_norm = per_sample_normalize(qdf["center_dist"].to_numpy(dtype=float), sample_ids)
    if rerank_mode == "sam":
        rerank = sam_norm
    elif rerank_mode == "heatmap":
        rerank = heatmap_norm
    elif rerank_mode == "combined":
        rerank = 0.45 * sam_norm + 0.45 * heatmap_norm + 0.10 * (1.0 - center_norm)
    else:
        raise ValueError(f"Unknown rerank_mode={rerank_mode}")
    rows = qdf.reset_index(drop=True)
    for _, group in rows.groupby("sample_id", sort=False):
        idx = group.index.to_numpy()
        order = idx[np.argsort(primary_scores[idx])[::-1][:top_k]]
        out[order] = rerank[order]
    return out


def main():
    parser = argparse.ArgumentParser(description="Two-branch classical-prior + QML-semantic prompt selector")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--semantic_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed.npy")
    parser.add_argument("--prior_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed_context.npy")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/two_branch_prompt_selector.csv")
    parser.add_argument("--eval_split", choices=["val", "test"], default="val")
    parser.add_argument("--pqk_components", type=int, default=12)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--prior_weights", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--residual_weights", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--shortlist_top_k", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--histgb_max_iter", type=int, default=180)
    parser.add_argument("--histgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    qdf = pd.read_csv(args.prompt_quality_csv)
    X_semantic = np.load(args.semantic_features).astype(np.float32)
    X_prior = np.load(args.prior_features).astype(np.float32)
    if len(qdf) != len(X_semantic) or len(qdf) != len(X_prior):
        raise ValueError("Prompt-quality rows and feature rows do not match.")

    train_mask, eval_mask = split_masks(qdf, args.eval_split)
    train_df = qdf.loc[train_mask].copy()
    eval_df = qdf.loc[eval_mask].copy()
    y_train = train_df["sam_dice"].to_numpy(dtype=float)

    prior_model = HistGradientBoostingRegressor(
        max_iter=args.histgb_max_iter,
        learning_rate=args.histgb_learning_rate,
        random_state=args.seed,
    )
    prior_model.fit(X_prior[train_mask], y_train)
    train_prior_scores = prior_model.predict(X_prior[train_mask])
    prior_scores = prior_model.predict(X_prior[eval_mask])

    X_train_q, X_eval_q, pca_info = fit_low_dim(
        X_semantic[train_mask],
        X_semantic[eval_mask],
        args.pqk_components,
        args.seed,
    )
    Z_train = projected_quantum_features(X_train_q, args.pqk_reps)
    Z_eval = projected_quantum_features(X_eval_q, args.pqk_reps)
    semantic_model = HistGradientBoostingRegressor(
        max_iter=args.histgb_max_iter,
        learning_rate=args.histgb_learning_rate,
        random_state=args.seed,
    )
    semantic_model.fit(Z_train, y_train)
    semantic_scores = semantic_model.predict(Z_eval)

    residual_model = HistGradientBoostingRegressor(
        max_iter=args.histgb_max_iter,
        learning_rate=args.histgb_learning_rate,
        random_state=args.seed,
    )
    residual_model.fit(Z_train, y_train - train_prior_scores)
    residual_predictions = residual_model.predict(Z_eval)

    results = [
        sample_level_eval(eval_df, eval_df["sam_dice"].to_numpy(dtype=float), "oracle_prompt_quality"),
        sample_level_eval(eval_df, prior_scores, "Classical_prior_context_augmented"),
        sample_level_eval(eval_df, semantic_scores, f"QML_semantic_PQF_HistGB_{args.pqk_components}pc"),
        sample_level_eval(eval_df, residual_scores(prior_scores, residual_predictions, 1.0), "ResidualQML_prior_plus_semantic"),
        sample_level_eval(eval_df, eval_df["heatmap_score"].to_numpy(dtype=float), "heatmap_score"),
    ]
    for residual_weight in args.residual_weights:
        row = sample_level_eval(
            eval_df,
            residual_scores(prior_scores, residual_predictions, residual_weight),
            f"ResidualQML_weight{residual_weight:g}",
        )
        row["pca_variance"] = pca_info["pca_variance_retained"]
        results.append(row)
    for top_k in args.shortlist_top_k:
        for mode in ["sam", "combined"]:
            results.append(sample_level_eval(
                eval_df,
                shortlist_rerank_scores(eval_df, prior_scores, top_k, mode),
                f"Classical_prior_top{top_k}_rerank_{mode}",
            ))
    for prior_weight in args.prior_weights:
        blended = blend_scores(eval_df, prior_scores, semantic_scores, prior_weight)
        row = sample_level_eval(
            eval_df,
            blended,
            f"TwoBranch_prior{prior_weight:g}_semantic{1.0 - prior_weight:g}",
        )
        row["pca_variance"] = pca_info["pca_variance_retained"]
        results.append(row)
        for top_k in args.shortlist_top_k:
            for mode in ["sam", "combined"]:
                rerank_row = sample_level_eval(
                    eval_df,
                    shortlist_rerank_scores(eval_df, blended, top_k, mode),
                    f"TwoBranch_prior{prior_weight:g}_top{top_k}_rerank_{mode}",
                )
                rerank_row["pca_variance"] = pca_info["pca_variance_retained"]
                results.append(rerank_row)

    summary = pd.DataFrame(results).sort_values("dice_mean", ascending=False)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
