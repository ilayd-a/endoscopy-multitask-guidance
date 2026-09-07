"""
sam_pairwise_prompt_ranker.py
=============================
Memory-capped pairwise prompt ranking from a cached SAM prompt-quality table.

Instead of predicting absolute SAM Dice for each candidate prompt, this script
learns whether prompt A should rank above prompt B within the same frame. That
matches the downstream decision: choose the best prompt for each image.
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
EBTC_EXPERIMENTS = ROOT / "EndoscopicBladderTissue" / "experiments"
if str(EBTC_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EBTC_EXPERIMENTS))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from publication_benchmark_ebtc import ProjectedQuantumKernelSVC
from sam_prompt_quality_ranker import per_sample_normalize, sample_level_eval


def load_cache(quality_csv: Path, quality_features: Path) -> tuple[pd.DataFrame, np.ndarray]:
    qdf = pd.read_csv(quality_csv)
    Q = np.load(quality_features).astype(np.float32)
    if len(qdf) != len(Q):
        raise ValueError(f"Cache mismatch: {len(qdf)} rows but {len(Q)} feature rows")
    return qdf, Q


def split_masks(qdf: pd.DataFrame, eval_split: str) -> tuple[np.ndarray, np.ndarray]:
    if eval_split == "val":
        train_mask = qdf["sequence_id"].le(23).to_numpy()
        eval_mask = qdf["split"].eq("val").to_numpy()
    else:
        train_mask = qdf["sequence_id"].le(26).to_numpy()
        eval_mask = qdf["split"].eq("test").to_numpy()
    return train_mask, eval_mask


def prefilter_eval_rows(qdf: pd.DataFrame, Q: np.ndarray, cap: int) -> tuple[pd.DataFrame, np.ndarray]:
    if cap <= 0:
        return qdf.copy(), Q
    rows = qdf.copy()
    sample_ids = rows["sample_id"].to_numpy()
    heatmap = per_sample_normalize(rows["heatmap_score"].to_numpy(dtype=float), sample_ids)
    sam = per_sample_normalize(rows["sam_score"].to_numpy(dtype=float), sample_ids)
    center = -per_sample_normalize(rows["center_dist"].to_numpy(dtype=float), sample_ids)
    rows["_prefilter"] = 0.45 * heatmap + 0.45 * sam + 0.10 * center
    keep_idx = rows.sort_values("_prefilter", ascending=False).groupby("sample_id").head(cap).index.to_numpy()
    keep_idx = np.sort(keep_idx)
    return rows.loc[keep_idx].drop(columns=["_prefilter"]), Q[keep_idx]


def build_pairwise_training_set(
    qdf: pd.DataFrame,
    Q: np.ndarray,
    train_mask: np.ndarray,
    top_k: int,
    negatives_per_positive: int,
    min_dice_gap: float,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_df = qdf.loc[train_mask].copy()
    pairs = []
    labels = []
    for _, group in train_df.groupby("sample_id", sort=False):
        ordered = group.sort_values("sam_dice", ascending=False)
        positives = ordered.head(top_k)
        hard_pool = ordered.iloc[top_k:]
        if hard_pool.empty:
            continue
        for pos in positives.itertuples():
            eligible = hard_pool[hard_pool["sam_dice"].le(float(pos.sam_dice) - min_dice_gap)]
            if eligible.empty:
                continue
            hard = eligible.sort_values(["heatmap_score", "sam_score"], ascending=False).head(negatives_per_positive)
            for neg in hard.itertuples():
                pos_idx = int(pos.Index)
                neg_idx = int(neg.Index)
                pairs.append(Q[pos_idx] - Q[neg_idx])
                labels.append(1)
                pairs.append(Q[neg_idx] - Q[pos_idx])
                labels.append(0)

    if not pairs:
        raise ValueError("No pairwise training pairs were generated; relax min_dice_gap or pair settings.")
    X_pair = np.vstack(pairs).astype(np.float32)
    y_pair = np.asarray(labels, dtype=np.int8)
    if max_pairs > 0 and len(y_pair) > max_pairs:
        pos_idx = np.flatnonzero(y_pair == 1)
        neg_idx = np.flatnonzero(y_pair == 0)
        half = max_pairs // 2
        selected = np.concatenate([
            rng.choice(pos_idx, size=min(half, len(pos_idx)), replace=False),
            rng.choice(neg_idx, size=min(max_pairs - min(half, len(pos_idx)), len(neg_idx)), replace=False),
        ])
        rng.shuffle(selected)
        X_pair = X_pair[selected]
        y_pair = y_pair[selected]
    return X_pair, y_pair


def pairwise_win_scores(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    n = len(X)
    scores = np.zeros(n, dtype=float)
    pair_i = []
    pair_j = []
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_i.append(i)
            pair_j.append(j)
            diffs.append(X[i] - X[j])
    if not diffs:
        return scores
    diffs = np.vstack(diffs).astype(np.float32)
    probs = []
    for start in range(0, len(diffs), batch_size):
        p = model.predict_proba(diffs[start:start + batch_size])[:, 1]
        probs.append(p)
    probs = np.concatenate(probs)
    for i, j, p in zip(pair_i, pair_j, probs):
        scores[i] += float(p)
        scores[j] += float(1.0 - p)
    return scores / max(1, n - 1)


class LowDimAngleTransform:
    def __init__(self, n_components: int, seed: int):
        self.n_components = n_components
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca = None
        self.angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        self.variance_retained = np.nan

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        max_components = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.pca = PCA(n_components=max_components, random_state=self.seed)
        X_std = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_std)
        self.variance_retained = float(self.pca.explained_variance_ratio_.sum())
        return self.angle_scaler.fit_transform(X_pca)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("LowDimAngleTransform must be fitted before transform.")
        return self.angle_scaler.transform(self.pca.transform(self.scaler.transform(X)))


def evaluate_pairwise_model(qdf: pd.DataFrame, Q: np.ndarray, model, name: str) -> dict:
    scores = np.zeros(len(qdf), dtype=float)
    for _, group in qdf.groupby("sample_id", sort=False):
        local_positions = qdf.index.get_indexer(group.index)
        local_scores = pairwise_win_scores(model, Q[local_positions])
        scores[local_positions] = local_scores
    return sample_level_eval(qdf, scores, name)


def main():
    parser = argparse.ArgumentParser(description="Pairwise prompt ranker from cached SAM prompt quality labels")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--prompt_quality_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_mps.npy")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/sam_pairwise_prompt_ranker.csv")
    parser.add_argument("--eval_split", choices=["val", "test"], default="test")
    parser.add_argument("--pair_top_k", type=int, default=2)
    parser.add_argument("--negatives_per_positive", type=int, default=3)
    parser.add_argument("--min_dice_gap", type=float, default=0.25)
    parser.add_argument("--max_pairs", type=int, default=1600)
    parser.add_argument("--eval_candidate_cap", type=int, default=60)
    parser.add_argument("--pqk_components", type=int, default=12)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--pqk_c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    qdf, Q = load_cache(Path(args.prompt_quality_csv), Path(args.prompt_quality_features))
    train_mask, eval_mask = split_masks(qdf, args.eval_split)
    eval_df_full = qdf.loc[eval_mask].copy()
    eval_Q_full = Q[eval_mask]
    eval_df, eval_Q = prefilter_eval_rows(eval_df_full.reset_index(drop=True), eval_Q_full, args.eval_candidate_cap)
    eval_df = eval_df.reset_index(drop=True)

    X_pair, y_pair = build_pairwise_training_set(
        qdf=qdf,
        Q=Q,
        train_mask=train_mask,
        top_k=args.pair_top_k,
        negatives_per_positive=args.negatives_per_positive,
        min_dice_gap=args.min_dice_gap,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    print(
        f"[pairs] train_pairs={len(y_pair)} positives={int(y_pair.sum())} "
        f"eval_samples={eval_df['sample_id'].nunique()} eval_candidates={len(eval_df)}"
    )

    results = [
        sample_level_eval(eval_df, eval_df["sam_dice"].to_numpy(dtype=float), "oracle_prompt_quality_prefiltered"),
        sample_level_eval(eval_df, eval_df["heatmap_score"].to_numpy(dtype=float), "heatmap_score_prefiltered"),
        sample_level_eval(eval_df, eval_df["sam_score"].to_numpy(dtype=float), "sam_score_prefiltered"),
    ]

    classical = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=500, random_state=args.seed),
    )
    classical.fit(X_pair, y_pair)
    results.append(evaluate_pairwise_model(eval_df, eval_Q, classical, "Classical_LogReg_pairwise"))

    angle_transform = LowDimAngleTransform(args.pqk_components, args.seed)
    X_pair_q = angle_transform.fit_transform(X_pair)
    pqk = ProjectedQuantumKernelSVC(gamma="scale", reps=args.pqk_reps, C=args.pqk_c, class_weight="balanced")
    pqk.fit(X_pair_q, y_pair)

    class PQKPairwiseWrapper:
        def __init__(self, pqk_model, transform):
            self.pqk_model = pqk_model
            self.transform = transform

        def predict_proba(self, X):
            X_q = self.transform.transform(X)
            return self.pqk_model.predict_proba(X_q)

    results.append(evaluate_pairwise_model(
        eval_df,
        eval_Q,
        PQKPairwiseWrapper(pqk, angle_transform),
        f"QML_PQK_pairwise_{args.pqk_components}pc",
    ))
    results[-1]["pca_variance"] = angle_transform.variance_retained

    summary = pd.DataFrame(results).sort_values("dice_mean", ascending=False)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[saved] {output_csv}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
