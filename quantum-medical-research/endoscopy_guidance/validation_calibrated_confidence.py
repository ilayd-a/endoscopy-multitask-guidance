"""
validation_calibrated_confidence.py
===================================
Validation-calibrated confidence gating for prompt selection.

This script trains the context-augmented prompt-quality selector on training
sequences only, chooses confidence thresholds on the validation split, then
applies those frozen thresholds to the held-out test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from prompt_selector_confidence_analysis import selected_with_confidence
from sam_prompt_quality_ranker import sample_level_eval


def split_masks(qdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = qdf["sequence_id"].le(23).to_numpy()
    val_mask = qdf["split"].eq("val").to_numpy()
    test_mask = qdf["split"].eq("test").to_numpy()
    return train_mask, val_mask, test_mask


def fit_selector(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray, seed: int):
    model = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=seed)
    model.fit(X[train_mask], y[train_mask])
    return model


def threshold_metrics(chosen: pd.DataFrame, threshold: float) -> dict:
    accepted = chosen[chosen["predicted_quality"] >= threshold]
    if accepted.empty:
        return {
            "accepted_samples": 0,
            "coverage": 0.0,
            "dice_mean": np.nan,
            "iou_mean": np.nan,
            "dice_ge_050": np.nan,
            "dice_ge_070": np.nan,
            "prompt_hit": np.nan,
        }
    return {
        "accepted_samples": int(len(accepted)),
        "coverage": float(len(accepted) / len(chosen)),
        "dice_mean": float(accepted["sam_dice"].mean()),
        "iou_mean": float(accepted["sam_iou"].mean()),
        "dice_ge_050": float((accepted["sam_dice"] >= 0.50).mean()),
        "dice_ge_070": float((accepted["sam_dice"] >= 0.70).mean()),
        "prompt_hit": float(accepted["point_hit"].mean()),
    }


def choose_thresholds(chosen_val: pd.DataFrame, target_dice_values: list[float], min_coverage: float) -> list[dict]:
    thresholds = np.unique(np.quantile(chosen_val["predicted_quality"], np.linspace(0, 1, 101)))
    candidates = []
    for threshold in thresholds:
        metrics = threshold_metrics(chosen_val, float(threshold))
        metrics["threshold"] = float(threshold)
        candidates.append(metrics)

    selections = []
    for target_dice in target_dice_values:
        feasible = [
            row for row in candidates
            if row["coverage"] >= min_coverage and np.isfinite(row["dice_mean"]) and row["dice_mean"] >= target_dice
        ]
        if feasible:
            selected = max(feasible, key=lambda row: (row["coverage"], row["dice_mean"]))
        else:
            selected = max(candidates, key=lambda row: (-abs(row["dice_mean"] - target_dice) if np.isfinite(row["dice_mean"]) else -999, row["coverage"]))
        selected = dict(selected)
        selected["target_dice"] = target_dice
        selections.append(selected)
    return selections


def main():
    parser = argparse.ArgumentParser(description="Validation-calibrated confidence gating")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed_context.npy")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/validation_calibrated_confidence.csv")
    parser.add_argument("--target_dice", type=float, nargs="+", default=[0.60, 0.65, 0.70])
    parser.add_argument("--min_coverage", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    qdf = pd.read_csv(args.prompt_quality_csv)
    X = np.load(args.features).astype(np.float32)
    y = qdf["sam_dice"].to_numpy(dtype=float)
    train_mask, val_mask, test_mask = split_masks(qdf)

    model = fit_selector(X, y, train_mask, args.seed)
    val_df = qdf.loc[val_mask].copy().reset_index(drop=True)
    test_df = qdf.loc[test_mask].copy().reset_index(drop=True)
    val_scores = model.predict(X[val_mask])
    test_scores = model.predict(X[test_mask])
    chosen_val = selected_with_confidence(val_df, val_scores)
    chosen_test = selected_with_confidence(test_df, test_scores)

    rows = []
    val_all = sample_level_eval(val_df, val_scores, "val_all_auto")
    test_all = sample_level_eval(test_df, test_scores, "test_all_auto")
    for split, baseline in [("val", val_all), ("test", test_all)]:
        rows.append({
            "split": split,
            "policy": "all_auto",
            "target_dice": np.nan,
            "threshold": -np.inf,
            "accepted_samples": int(baseline["samples"]),
            "coverage": 1.0,
            "dice_mean": baseline["dice_mean"],
            "iou_mean": baseline["iou_mean"],
            "dice_ge_050": np.nan,
            "dice_ge_070": np.nan,
            "prompt_hit": baseline["prompt_hit"],
        })

    selections = choose_thresholds(chosen_val, args.target_dice, args.min_coverage)
    for selected in selections:
        threshold = selected["threshold"]
        for split, chosen in [("val", chosen_val), ("test", chosen_test)]:
            metrics = threshold_metrics(chosen, threshold)
            rows.append({
                "split": split,
                "policy": f"val_calibrated_target{selected['target_dice']:.2f}",
                "target_dice": selected["target_dice"],
                "threshold": threshold,
                **metrics,
            })

    summary = pd.DataFrame(rows)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
