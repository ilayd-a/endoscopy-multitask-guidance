"""
hard_case_triage_ebtc.py
========================
Evaluate PQK as a second-reader only on classically uncertain EBTC cases.

This tests a more plausible clinical role than whole-dataset replacement:

    classical model handles confident cases
    PQK reranks / overrides only the least-confident classical cases

The input is `ebtc_publication_predictions.csv` from the publication benchmark.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "publication_benchmark"


def binary_metrics(y_true, y_pred, scores=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else np.nan,
        "false_positive_rate": fp / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if scores is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, scores)
        except Exception:
            out["roc_auc"] = np.nan
    return out


def align_pair(base, second):
    cols = ["sample_id", "y_true", "y_pred", "score"]
    merged = base[cols].merge(
        second[cols],
        on=["sample_id", "y_true"],
        suffixes=("_base", "_second"),
        validate="one_to_one",
    )
    return merged.sort_values("sample_id").reset_index(drop=True)


def triage_pair(base, second, coverage_values, second_min_margins):
    merged = align_pair(base, second)
    y_true = merged["y_true"].to_numpy(dtype=int)
    base_pred = merged["y_pred_base"].to_numpy(dtype=int)
    second_pred = merged["y_pred_second"].to_numpy(dtype=int)
    base_score = merged["score_base"].to_numpy(dtype=float)
    second_score = merged["score_second"].to_numpy(dtype=float)

    uncertainty = np.abs(base_score - 0.5)
    order = np.argsort(uncertainty)
    n = len(merged)

    base_metrics = binary_metrics(y_true, base_pred, base_score)
    second_metrics = binary_metrics(y_true, second_pred, second_score)

    rows = []
    second_margin = np.abs(second_score - 0.5)
    for coverage in coverage_values:
        k = int(round(n * coverage))
        k = min(max(k, 0), n)
        candidate = np.zeros(n, dtype=bool)
        if k:
            candidate[order[:k]] = True

        for second_min_margin in second_min_margins:
            switched = candidate & (second_margin >= second_min_margin)
            hybrid_pred = base_pred.copy()
            hybrid_pred[switched] = second_pred[switched]
            hybrid_score = base_score.copy()
            hybrid_score[switched] = second_score[switched]
            hybrid_metrics = binary_metrics(y_true, hybrid_pred, hybrid_score)

            # This upper bound tells us whether the selected cases contain any
            # recoverable signal at all, without presenting it as deployable.
            oracle_pred = base_pred.copy()
            oracle_mask = switched & (second_pred == y_true) & (base_pred != y_true)
            oracle_pred[oracle_mask] = second_pred[oracle_mask]
            oracle_metrics = binary_metrics(y_true, oracle_pred)

            row = {
                "coverage": coverage,
                "second_min_margin": second_min_margin,
                "n_test": n,
                "n_switched": int(switched.sum()),
                "base_accuracy": base_metrics["accuracy"],
                "base_balanced_accuracy": base_metrics["balanced_accuracy"],
                "base_f1": base_metrics["f1"],
                "base_roc_auc": base_metrics["roc_auc"],
                "base_sensitivity": base_metrics["sensitivity"],
                "base_specificity": base_metrics["specificity"],
                "base_false_negative_rate": base_metrics["false_negative_rate"],
                "base_false_positive_rate": base_metrics["false_positive_rate"],
                "second_accuracy": second_metrics["accuracy"],
                "second_balanced_accuracy": second_metrics["balanced_accuracy"],
                "second_f1": second_metrics["f1"],
                "second_roc_auc": second_metrics["roc_auc"],
                "second_sensitivity": second_metrics["sensitivity"],
                "second_specificity": second_metrics["specificity"],
                "second_false_negative_rate": second_metrics["false_negative_rate"],
                "second_false_positive_rate": second_metrics["false_positive_rate"],
                "hybrid_accuracy": hybrid_metrics["accuracy"],
                "hybrid_balanced_accuracy": hybrid_metrics["balanced_accuracy"],
                "hybrid_f1": hybrid_metrics["f1"],
                "hybrid_roc_auc": hybrid_metrics["roc_auc"],
                "hybrid_sensitivity": hybrid_metrics["sensitivity"],
                "hybrid_specificity": hybrid_metrics["specificity"],
                "hybrid_false_negative_rate": hybrid_metrics["false_negative_rate"],
                "hybrid_false_positive_rate": hybrid_metrics["false_positive_rate"],
                "oracle_accuracy": oracle_metrics["accuracy"],
                "oracle_balanced_accuracy": oracle_metrics["balanced_accuracy"],
                "oracle_sensitivity": oracle_metrics["sensitivity"],
                "oracle_specificity": oracle_metrics["specificity"],
                "delta_accuracy": hybrid_metrics["accuracy"] - base_metrics["accuracy"],
                "delta_balanced_accuracy": hybrid_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"],
                "delta_f1": hybrid_metrics["f1"] - base_metrics["f1"],
                "delta_sensitivity": hybrid_metrics["sensitivity"] - base_metrics["sensitivity"],
                "delta_specificity": hybrid_metrics["specificity"] - base_metrics["specificity"],
                "delta_false_negative_rate": hybrid_metrics["false_negative_rate"] - base_metrics["false_negative_rate"],
                "delta_false_positive_rate": hybrid_metrics["false_positive_rate"] - base_metrics["false_positive_rate"],
                "oracle_delta_balanced_accuracy": oracle_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"],
            }
            rows.append(row)
    return rows


def aggregate(rows):
    df = pd.DataFrame(rows)
    group_cols = ["train_limit", "base_model", "second_model", "coverage", "second_min_margin"]
    metric_cols = [c for c in df.columns if c not in group_cols + ["repeat", "seed"]]
    out = []
    for key, group in df.groupby(group_cols, sort=True):
        record = dict(zip(group_cols, key))
        record["n_runs"] = int(len(group))
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            record[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            record[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) else np.nan
        out.append(record)
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="PQK second-reader analysis for classically uncertain EBTC cases")
    parser.add_argument(
        "--predictions_csv",
        default=str(RESULTS_DIR / "ebtc_publication_predictions.csv"),
    )
    parser.add_argument(
        "--output_csv",
        default=str(RESULTS_DIR / "ebtc_hard_case_triage_metrics.csv"),
    )
    parser.add_argument(
        "--aggregate_csv",
        default=str(RESULTS_DIR / "ebtc_hard_case_triage_aggregate.csv"),
    )
    parser.add_argument("--coverage", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--second_min_margin", nargs="*", type=float, default=[0.0, 0.05, 0.1, 0.2])
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)
    required = {"model", "repeat", "seed", "train_limit", "sample_id", "y_true", "y_pred", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {sorted(missing)}")

    rows = []
    for (train_limit, repeat), group in df.groupby(["train_limit", "repeat"], sort=True):
        classical_models = sorted([m for m in group["model"].unique() if not m.startswith("QML")])
        qml_models = sorted([m for m in group["model"].unique() if m.startswith("QML_PQK")])
        for base_model in classical_models:
            base = group[group["model"] == base_model]
            for second_model in qml_models:
                second = group[group["model"] == second_model]
                for row in triage_pair(base, second, args.coverage, args.second_min_margin):
                    row.update({
                        "train_limit": int(train_limit),
                        "repeat": int(repeat),
                        "seed": int(group["seed"].iloc[0]),
                        "base_model": base_model,
                        "second_model": second_model,
                    })
                    rows.append(row)

    aggregate_rows = aggregate(rows)
    write_csv(Path(args.output_csv), rows)
    write_csv(Path(args.aggregate_csv), aggregate_rows)
    print(f"[results] triage metrics: {args.output_csv}")
    print(f"[results] triage aggregate: {args.aggregate_csv}")

    best = pd.DataFrame(aggregate_rows).sort_values("delta_balanced_accuracy_mean", ascending=False).head(12)
    for _, row in best.iterrows():
        print({
            "train_limit": int(row["train_limit"]),
            "coverage": float(row["coverage"]),
            "second_min_margin": float(row["second_min_margin"]),
            "base_model": row["base_model"],
            "second_model": row["second_model"],
            "delta_balanced_accuracy": round(row["delta_balanced_accuracy_mean"], 4),
            "delta_sensitivity": round(row["delta_sensitivity_mean"], 4),
            "delta_specificity": round(row["delta_specificity_mean"], 4),
            "hybrid_balanced_accuracy": round(row["hybrid_balanced_accuracy_mean"], 4),
            "base_balanced_accuracy": round(row["base_balanced_accuracy_mean"], 4),
            "oracle_delta_balanced_accuracy": round(row["oracle_delta_balanced_accuracy_mean"], 4),
        })


if __name__ == "__main__":
    main()
