"""
prompt_selector_confidence_analysis.py
======================================
Confidence/abstention analysis for the best cached prompt selector.

Medical deployment rarely needs every frame to be fully automatic. This script
estimates whether the selector can identify high-confidence cases whose selected
SAM prompt masks are reliable, while routing uncertain cases to review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from sam_prompt_quality_ranker import sample_level_eval


def split_masks(qdf: pd.DataFrame, eval_split: str) -> tuple[np.ndarray, np.ndarray]:
    if eval_split == "val":
        train_mask = qdf["sequence_id"].le(23).to_numpy()
        eval_mask = qdf["split"].eq("val").to_numpy()
    else:
        train_mask = qdf["sequence_id"].le(26).to_numpy()
        eval_mask = qdf["split"].eq("test").to_numpy()
    return train_mask, eval_mask


def selected_with_confidence(eval_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    rows = eval_df.copy().reset_index(drop=True)
    rows["_score"] = scores
    out = []
    for sid, group in rows.groupby("sample_id", sort=False):
        ranked = group.sort_values("_score", ascending=False)
        top = ranked.iloc[0].copy()
        top_score = float(top["_score"])
        second_score = float(ranked.iloc[1]["_score"]) if len(ranked) > 1 else top_score
        score_values = group["_score"].to_numpy(dtype=float)
        spread = float(np.nanmax(score_values) - np.nanmin(score_values))
        top["score_margin"] = top_score - second_score
        top["score_margin_norm"] = 0.0 if spread <= 0 else (top_score - second_score) / spread
        top["predicted_quality"] = top_score
        top["sample_id"] = sid
        out.append(top)
    return pd.DataFrame(out)


def coverage_rows(chosen: pd.DataFrame, coverages: list[float], confidence_col: str) -> list[dict]:
    out = []
    ranked = chosen.sort_values(confidence_col, ascending=False).reset_index(drop=True)
    for coverage in coverages:
        n = max(1, int(round(len(ranked) * coverage)))
        subset = ranked.head(n)
        out.append({
            "confidence": confidence_col,
            "coverage": coverage,
            "accepted_samples": n,
            "review_samples": int(len(ranked) - n),
            "dice_mean": float(subset["sam_dice"].mean()),
            "iou_mean": float(subset["sam_iou"].mean()),
            "prompt_hit": float(subset["point_hit"].mean()),
            "dice_ge_050": float((subset["sam_dice"] >= 0.50).mean()),
            "dice_ge_070": float((subset["sam_dice"] >= 0.70).mean()),
            "mean_margin_norm": float(subset["score_margin_norm"].mean()),
            "mean_predicted_quality": float(subset["predicted_quality"].mean()),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Confidence analysis for cached prompt selector")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed_context.npy")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/prompt_selector_confidence_analysis.csv")
    parser.add_argument("--eval_split", choices=["val", "test"], default="test")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00])
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    qdf = pd.read_csv(args.prompt_quality_csv)
    X = np.load(args.features).astype(np.float32)
    train_mask, eval_mask = split_masks(qdf, args.eval_split)
    model = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=args.seed)
    model.fit(X[train_mask], qdf.loc[train_mask, "sam_dice"].to_numpy(dtype=float))
    eval_df = qdf.loc[eval_mask].copy().reset_index(drop=True)
    scores = model.predict(X[eval_mask])

    chosen = selected_with_confidence(eval_df, scores)
    baseline = sample_level_eval(eval_df, scores, "all_auto")
    rows = [{
        "confidence": "all",
        "coverage": 1.0,
        "accepted_samples": int(baseline["samples"]),
        "review_samples": 0,
        "dice_mean": baseline["dice_mean"],
        "iou_mean": baseline["iou_mean"],
        "prompt_hit": baseline["prompt_hit"],
        "dice_ge_050": float((chosen["sam_dice"] >= 0.50).mean()),
        "dice_ge_070": float((chosen["sam_dice"] >= 0.70).mean()),
        "mean_margin_norm": float(chosen["score_margin_norm"].mean()),
        "mean_predicted_quality": float(chosen["predicted_quality"].mean()),
    }]
    rows.extend(coverage_rows(chosen, args.coverages, "score_margin_norm"))
    rows.extend(coverage_rows(chosen, args.coverages, "predicted_quality"))
    summary = pd.DataFrame(rows).drop_duplicates(subset=["confidence", "coverage", "accepted_samples"]).sort_values(["confidence", "coverage"])
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
