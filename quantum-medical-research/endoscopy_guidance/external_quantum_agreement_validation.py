"""
external_quantum_agreement_validation.py
=======================================
Train and calibrate on the internal CVC prompt-quality benchmark, then evaluate
the frozen selector and confidence gates on an external prompt-quality cache.

This script keeps external data out of model fitting and threshold selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from candidate_ranking_benchmark import fit_low_dim
from quantum_agreement_confidence import (
    all_auto_metrics,
    apply_threshold,
    choose_thresholds,
    confidence_score,
    selected_rows,
)
from sam_prompt_quality_ranker import projected_quantum_features


def internal_masks(qdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return qdf["sequence_id"].le(23).to_numpy(), qdf["split"].eq("val").to_numpy()


def oracle_metrics(qdf: pd.DataFrame, split: str) -> dict:
    chosen = qdf.sort_values("sam_dice", ascending=False).groupby("sample_id", as_index=False).head(1)
    return {
        "policy": "oracle_best_candidate",
        "split": split,
        "samples": int(chosen["sample_id"].nunique()),
        "accepted_samples": int(len(chosen)),
        "coverage": 1.0,
        "dice_mean": float(chosen["sam_dice"].mean()),
        "iou_mean": float(chosen["sam_iou"].mean()),
        "dice_ge_050": float((chosen["sam_dice"] >= 0.50).mean()),
        "dice_ge_070": float((chosen["sam_dice"] >= 0.70).mean()),
        "prompt_hit": float(chosen["point_hit"].mean()),
        "quantum_agreement": np.nan,
        "agreement_weight": np.nan,
        "target_dice": np.nan,
        "threshold": np.nan,
        "pca_variance": np.nan,
    }


def candidate_set_metrics(qdf: pd.DataFrame, split: str) -> dict:
    per_sample = qdf.groupby("sample_id").agg(
        candidates=("sample_id", "size"),
        any_prompt_hit=("point_hit", "max"),
        best_dice=("sam_dice", "max"),
    )
    return {
        "policy": "candidate_set",
        "split": split,
        "samples": int(len(per_sample)),
        "accepted_samples": int(len(per_sample)),
        "coverage": 1.0,
        "dice_mean": float(per_sample["best_dice"].mean()),
        "iou_mean": np.nan,
        "dice_ge_050": float((per_sample["best_dice"] >= 0.50).mean()),
        "dice_ge_070": float((per_sample["best_dice"] >= 0.70).mean()),
        "prompt_hit": float(per_sample["any_prompt_hit"].mean()),
        "quantum_agreement": np.nan,
        "agreement_weight": np.nan,
        "target_dice": np.nan,
        "threshold": np.nan,
        "pca_variance": np.nan,
        "candidates_per_sample_mean": float(per_sample["candidates"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="External validation for quantum-agreement confidence gating")
    parser.add_argument("--internal_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--internal_semantic_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed.npy")
    parser.add_argument("--internal_context_features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed_context.npy")
    parser.add_argument("--external_csv", default="endoscopy_guidance/results/polypgen_external_80_v2_prompt_quality.csv")
    parser.add_argument("--external_semantic_features", default="endoscopy_guidance/results/polypgen_external_80_v2_prompt_quality_features_samembed.npy")
    parser.add_argument("--external_context_features", default="endoscopy_guidance/results/polypgen_external_80_v2_prompt_quality_features_samembed_context.npy")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/polypgen_external_quantum_agreement_validation.csv")
    parser.add_argument("--per_sample_csv", default="endoscopy_guidance/results/polypgen_external_quantum_agreement_per_sample.csv")
    parser.add_argument("--external_name", default="external_polypgen")
    parser.add_argument("--pqk_components", type=int, default=12)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--agreement_weights", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--target_dice", type=float, nargs="+", default=[0.75, 0.82, 0.84, 0.86, 0.88])
    parser.add_argument("--min_coverage", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    internal = pd.read_csv(args.internal_csv)
    external = pd.read_csv(args.external_csv)
    X_int_sem = np.load(args.internal_semantic_features).astype(np.float32)
    X_int_ctx = np.load(args.internal_context_features).astype(np.float32)
    X_ext_sem = np.load(args.external_semantic_features).astype(np.float32)
    X_ext_ctx = np.load(args.external_context_features).astype(np.float32)

    if len(internal) != len(X_int_sem) or len(internal) != len(X_int_ctx):
        raise ValueError("Internal CSV/features row mismatch")
    if len(external) != len(X_ext_sem) or len(external) != len(X_ext_ctx):
        raise ValueError("External CSV/features row mismatch")

    train_mask, val_mask = internal_masks(internal)
    y_train = internal.loc[train_mask, "sam_dice"].to_numpy(dtype=float)

    prior = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=args.seed)
    prior.fit(X_int_ctx[train_mask], y_train)
    val_prior = prior.predict(X_int_ctx[val_mask])
    ext_prior = prior.predict(X_ext_ctx)

    X_train_q, X_val_q, pca_info = fit_low_dim(X_int_sem[train_mask], X_int_sem[val_mask], args.pqk_components, args.seed)
    _, X_ext_q, _ = fit_low_dim(X_int_sem[train_mask], X_ext_sem, args.pqk_components, args.seed)
    Z_train = projected_quantum_features(X_train_q, args.pqk_reps)
    Z_val = projected_quantum_features(X_val_q, args.pqk_reps)
    Z_ext = projected_quantum_features(X_ext_q, args.pqk_reps)
    qml = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=args.seed)
    qml.fit(Z_train, y_train)
    val_q = qml.predict(Z_val)
    ext_q = qml.predict(Z_ext)

    val_df = internal.loc[val_mask].copy().reset_index(drop=True)
    ext_df = external.copy().reset_index(drop=True)
    chosen_val = selected_rows(val_df, val_prior, val_q)
    chosen_ext = selected_rows(ext_df, ext_prior, ext_q)

    rows = [
        candidate_set_metrics(ext_df, args.external_name),
        oracle_metrics(ext_df, args.external_name),
        {
            "policy": "all_auto_prior",
            "split": args.external_name,
            "samples": int(chosen_ext["sample_id"].nunique()),
            **all_auto_metrics(chosen_ext),
            "agreement_weight": np.nan,
            "target_dice": np.nan,
            "threshold": -np.inf,
            "pca_variance": pca_info["pca_variance_retained"],
        },
    ]
    per_sample_rows = []
    for agreement_weight in args.agreement_weights:
        conf_val = confidence_score(chosen_val, agreement_weight)
        conf_ext = confidence_score(chosen_ext, agreement_weight)
        selections = choose_thresholds(chosen_val, conf_val, args.target_dice, args.min_coverage)
        for selected in selections:
            accepted = conf_ext >= selected["threshold"]
            for row, conf, accept in zip(chosen_ext.itertuples(index=False), conf_ext, accepted):
                per_sample_rows.append({
                    "split": args.external_name,
                    "sample_id": row.sample_id,
                    "source_file": getattr(row, "source_file", ""),
                    "source_group": getattr(row, "source_group", ""),
                    "agreement_weight": agreement_weight,
                    "target_dice": selected["target_dice"],
                    "threshold": selected["threshold"],
                    "confidence": float(conf),
                    "accepted": int(accept),
                    "sam_dice": float(row.sam_dice),
                    "sam_iou": float(row.sam_iou),
                    "point_hit": int(row.point_hit),
                    "predicted_quality": float(row.predicted_quality),
                    "quantum_agreement": float(row.quantum_agreement),
                })
            rows.append({
                "policy": "quantum_agreement_confidence",
                "split": args.external_name,
                "samples": int(chosen_ext["sample_id"].nunique()),
                "agreement_weight": agreement_weight,
                "target_dice": selected["target_dice"],
                "threshold": selected["threshold"],
                "pca_variance": pca_info["pca_variance_retained"],
                **apply_threshold(chosen_ext, conf_ext, selected["threshold"]),
            })

    summary = pd.DataFrame(rows)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    pd.DataFrame(per_sample_rows).to_csv(args.per_sample_csv, index=False)
    print(f"[saved] {output}")
    print(f"[saved] {args.per_sample_csv}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
