"""
quantum_agreement_confidence.py
===============================
Quantum-assisted confidence gating for prompt selection.

The strongest selector is classical/context-driven. This script tests a more
realistic quantum role: use a projected-quantum semantic branch to estimate
agreement with the classical selector, then calibrate auto-accept thresholds on
validation and evaluate them once on held-out test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from candidate_ranking_benchmark import fit_low_dim
from sam_prompt_quality_ranker import per_sample_normalize, projected_quantum_features, sample_level_eval


def split_masks(qdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        qdf["sequence_id"].le(23).to_numpy(),
        qdf["split"].eq("val").to_numpy(),
        qdf["split"].eq("test").to_numpy(),
    )


def selected_rows(eval_df: pd.DataFrame, primary_scores: np.ndarray, semantic_scores: np.ndarray) -> pd.DataFrame:
    rows = eval_df.copy().reset_index(drop=True)
    sample_ids = rows["sample_id"].to_numpy()
    prior_norm = per_sample_normalize(primary_scores, sample_ids)
    semantic_norm = per_sample_normalize(semantic_scores, sample_ids)
    rows["_primary"] = primary_scores
    rows["_prior_norm"] = prior_norm
    rows["_semantic_norm"] = semantic_norm
    out = []
    for sid, group in rows.groupby("sample_id", sort=False):
        ranked = group.sort_values("_primary", ascending=False)
        top = ranked.iloc[0].copy()
        top["predicted_quality"] = float(top["_primary"])
        top["quantum_agreement"] = float(1.0 - abs(top["_prior_norm"] - top["_semantic_norm"]))
        top["quantum_semantic_score"] = float(top["_semantic_norm"])
        top["sample_id"] = sid
        out.append(top)
    return pd.DataFrame(out)


def confidence_score(chosen: pd.DataFrame, agreement_weight: float) -> np.ndarray:
    quality = chosen["predicted_quality"].to_numpy(dtype=float)
    agreement = chosen["quantum_agreement"].to_numpy(dtype=float)
    return (1.0 - agreement_weight) * quality + agreement_weight * agreement


def choose_thresholds(chosen_val: pd.DataFrame, conf_val: np.ndarray, target_dice_values: list[float], min_coverage: float) -> list[dict]:
    tmp = chosen_val.copy()
    tmp["confidence"] = conf_val
    thresholds = np.unique(np.quantile(conf_val, np.linspace(0, 1, 101)))
    candidates = []
    for threshold in thresholds:
        accepted = tmp[tmp["confidence"] >= threshold]
        if accepted.empty:
            continue
        row = {
            "threshold": float(threshold),
            "coverage": float(len(accepted) / len(tmp)),
            "dice_mean": float(accepted["sam_dice"].mean()),
            "accepted_samples": int(len(accepted)),
        }
        candidates.append(row)
    selections = []
    for target_dice in target_dice_values:
        feasible = [
            row for row in candidates
            if row["coverage"] >= min_coverage and row["dice_mean"] >= target_dice
        ]
        selected = max(feasible, key=lambda row: (row["coverage"], row["dice_mean"])) if feasible else max(candidates, key=lambda row: row["dice_mean"])
        selected = dict(selected)
        selected["target_dice"] = target_dice
        selections.append(selected)
    return selections


def apply_threshold(chosen: pd.DataFrame, confidence: np.ndarray, threshold: float) -> dict:
    tmp = chosen.copy()
    tmp["confidence"] = confidence
    accepted = tmp[tmp["confidence"] >= threshold]
    if accepted.empty:
        return {
            "accepted_samples": 0,
            "coverage": 0.0,
            "dice_mean": np.nan,
            "iou_mean": np.nan,
            "dice_ge_050": np.nan,
            "dice_ge_070": np.nan,
            "prompt_hit": np.nan,
            "quantum_agreement": np.nan,
        }
    return {
        "accepted_samples": int(len(accepted)),
        "coverage": float(len(accepted) / len(tmp)),
        "dice_mean": float(accepted["sam_dice"].mean()),
        "iou_mean": float(accepted["sam_iou"].mean()),
        "dice_ge_050": float((accepted["sam_dice"] >= 0.50).mean()),
        "dice_ge_070": float((accepted["sam_dice"] >= 0.70).mean()),
        "prompt_hit": float(accepted["point_hit"].mean()),
        "quantum_agreement": float(accepted["quantum_agreement"].mean()),
    }


def all_auto_metrics(chosen: pd.DataFrame) -> dict:
    return {
        "accepted_samples": int(len(chosen)),
        "coverage": 1.0,
        "dice_mean": float(chosen["sam_dice"].mean()),
        "iou_mean": float(chosen["sam_iou"].mean()),
        "dice_ge_050": float((chosen["sam_dice"] >= 0.50).mean()),
        "dice_ge_070": float((chosen["sam_dice"] >= 0.70).mean()),
        "prompt_hit": float(chosen["point_hit"].mean()),
        "quantum_agreement": float(chosen["quantum_agreement"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Quantum-agreement validation-calibrated confidence gating")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--semantic_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed.npy")
    parser.add_argument("--context_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed_context.npy")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/quantum_agreement_confidence.csv")
    parser.add_argument("--pqk_components", type=int, default=12)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--agreement_weights", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--target_dice", type=float, nargs="+", default=[0.82, 0.84, 0.86, 0.88])
    parser.add_argument("--min_coverage", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    qdf = pd.read_csv(args.prompt_quality_csv)
    X_semantic = np.load(args.semantic_features).astype(np.float32)
    X_context = np.load(args.context_features).astype(np.float32)
    y = qdf["sam_dice"].to_numpy(dtype=float)
    train_mask, val_mask, test_mask = split_masks(qdf)

    prior = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=args.seed)
    prior.fit(X_context[train_mask], y[train_mask])
    val_prior = prior.predict(X_context[val_mask])
    test_prior = prior.predict(X_context[test_mask])

    X_train_q, X_val_q, pca_info = fit_low_dim(X_semantic[train_mask], X_semantic[val_mask], args.pqk_components, args.seed)
    _, X_test_q, _ = fit_low_dim(X_semantic[train_mask], X_semantic[test_mask], args.pqk_components, args.seed)
    Z_train = projected_quantum_features(X_train_q, args.pqk_reps)
    Z_val = projected_quantum_features(X_val_q, args.pqk_reps)
    Z_test = projected_quantum_features(X_test_q, args.pqk_reps)
    qml = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=args.seed)
    qml.fit(Z_train, y[train_mask])
    val_q = qml.predict(Z_val)
    test_q = qml.predict(Z_test)

    val_df = qdf.loc[val_mask].copy().reset_index(drop=True)
    test_df = qdf.loc[test_mask].copy().reset_index(drop=True)
    chosen_val = selected_rows(val_df, val_prior, val_q)
    chosen_test = selected_rows(test_df, test_prior, test_q)

    rows = []
    rows.append({"policy": "all_auto_prior", "split": "test", **all_auto_metrics(chosen_test), "agreement_weight": np.nan, "target_dice": np.nan, "threshold": -np.inf})
    for agreement_weight in args.agreement_weights:
        conf_val = confidence_score(chosen_val, agreement_weight)
        conf_test = confidence_score(chosen_test, agreement_weight)
        selections = choose_thresholds(chosen_val, conf_val, args.target_dice, args.min_coverage)
        for selected in selections:
            for split, chosen, confidence in [("val", chosen_val, conf_val), ("test", chosen_test, conf_test)]:
                rows.append({
                    "policy": "quantum_agreement_confidence",
                    "split": split,
                    "agreement_weight": agreement_weight,
                    "target_dice": selected["target_dice"],
                    "threshold": selected["threshold"],
                    "pca_variance": pca_info["pca_variance_retained"],
                    **apply_threshold(chosen, confidence, selected["threshold"]),
                })

    summary = pd.DataFrame(rows).sort_values(["split", "target_dice", "agreement_weight"], na_position="first")
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
