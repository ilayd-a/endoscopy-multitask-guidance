"""
sam_listwise_prompt_ranker.py
=============================
Memory-capped listwise prompt ranking from cached SAM prompt-quality labels.

For each frame, the top Dice prompts are treated as image-local winners. This
keeps the target aligned with within-image prompt selection while avoiding the
large pairwise kernel matrices created by all prompt-pair comparisons.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
EBTC_EXPERIMENTS = ROOT / "EndoscopicBladderTissue" / "experiments"
if str(EBTC_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EBTC_EXPERIMENTS))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from publication_benchmark_ebtc import ProjectedQuantumKernelSVC
from candidate_ranking_benchmark import fit_low_dim, prediction_scores
from sam_pairwise_prompt_ranker import load_cache, prefilter_eval_rows, split_masks
from sam_prompt_quality_ranker import sample_level_eval


def build_listwise_training_indices(
    qdf: pd.DataFrame,
    train_mask: np.ndarray,
    top_k: int,
    negatives_per_positive: int,
    min_positive_dice: float,
    max_train_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected_idx = []
    labels = []
    train_df = qdf.loc[train_mask].copy()
    for _, group in train_df.groupby("sample_id", sort=False):
        ordered = group.sort_values("sam_dice", ascending=False)
        positives = ordered[ordered["sam_dice"].ge(min_positive_dice)].head(top_k)
        if positives.empty:
            positives = ordered.head(1)
        positive_indices = set(int(i) for i in positives.index)
        hard_negatives = ordered.loc[~ordered.index.isin(positive_indices)]
        hard_negatives = hard_negatives.sort_values(["heatmap_score", "sam_score"], ascending=False)
        hard_negatives = hard_negatives.head(max(1, negatives_per_positive * len(positives)))
        for idx in positives.index:
            selected_idx.append(int(idx))
            labels.append(1)
        for idx in hard_negatives.index:
            selected_idx.append(int(idx))
            labels.append(0)

    selected_idx = np.asarray(selected_idx, dtype=int)
    labels = np.asarray(labels, dtype=np.int8)
    if max_train_rows > 0 and len(labels) > max_train_rows:
        pos_idx = np.flatnonzero(labels == 1)
        neg_idx = np.flatnonzero(labels == 0)
        pos_take = min(len(pos_idx), max_train_rows // 2)
        neg_take = min(len(neg_idx), max_train_rows - pos_take)
        chosen = np.concatenate([
            rng.choice(pos_idx, size=pos_take, replace=False),
            rng.choice(neg_idx, size=neg_take, replace=False),
        ])
        rng.shuffle(chosen)
        selected_idx = selected_idx[chosen]
        labels = labels[chosen]
    return selected_idx, labels


def main():
    parser = argparse.ArgumentParser(description="Listwise prompt ranker from cached SAM prompt quality labels")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--prompt_quality_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_mps.npy")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/sam_listwise_prompt_ranker.csv")
    parser.add_argument("--eval_split", choices=["val", "test"], default="test")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--negatives_per_positive", type=int, default=4)
    parser.add_argument("--min_positive_dice", type=float, default=0.55)
    parser.add_argument("--max_train_rows", type=int, default=2400)
    parser.add_argument("--eval_candidate_cap", type=int, default=0)
    parser.add_argument("--pqk_components", type=int, default=12)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--pqk_c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    qdf, Q = load_cache(Path(args.prompt_quality_csv), Path(args.prompt_quality_features))
    train_mask, eval_mask = split_masks(qdf, args.eval_split)
    train_idx, y_train = build_listwise_training_indices(
        qdf=qdf,
        train_mask=train_mask,
        top_k=args.top_k,
        negatives_per_positive=args.negatives_per_positive,
        min_positive_dice=args.min_positive_dice,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
    )
    eval_df_full = qdf.loc[eval_mask].copy().reset_index(drop=True)
    eval_Q_full = Q[eval_mask]
    eval_df, eval_Q = prefilter_eval_rows(eval_df_full, eval_Q_full, args.eval_candidate_cap)
    eval_df = eval_df.reset_index(drop=True)
    X_train = Q[train_idx]

    print(
        f"[listwise] train_rows={len(train_idx)} positives={int(y_train.sum())} "
        f"eval_samples={eval_df['sample_id'].nunique()} eval_candidates={len(eval_df)}"
    )

    results = [
        sample_level_eval(eval_df, eval_df["sam_dice"].to_numpy(dtype=float), "oracle_prompt_quality"),
        sample_level_eval(eval_df, eval_df["heatmap_score"].to_numpy(dtype=float), "heatmap_score"),
        sample_level_eval(eval_df, eval_df["sam_score"].to_numpy(dtype=float), "sam_score"),
    ]

    classical = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=500, random_state=args.seed),
    )
    classical.fit(X_train, y_train)
    results.append(sample_level_eval(
        eval_df,
        prediction_scores(classical, eval_Q),
        "Classical_LogReg_listwise",
    ))

    X_train_q, X_eval_q, pca_info = fit_low_dim(X_train, eval_Q, args.pqk_components, args.seed)
    pqk = ProjectedQuantumKernelSVC(gamma="scale", reps=args.pqk_reps, C=args.pqk_c, class_weight="balanced")
    pqk.fit(X_train_q, y_train)
    row = sample_level_eval(
        eval_df,
        prediction_scores(pqk, X_eval_q),
        f"QML_PQK_listwise_{args.pqk_components}pc",
    )
    row["pca_variance"] = pca_info["pca_variance_retained"]
    results.append(row)

    summary = pd.DataFrame(results).sort_values("dice_mean", ascending=False)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[saved] {output_csv}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
