"""
threshold_analysis_ebtc.py
==========================
Analyze EBTC model scores at clinically constrained thresholds.

The publication benchmark writes per-sample test scores to
`ebtc_publication_predictions.csv`. This script sweeps decision thresholds and
reports model behavior at fixed specificity or fixed sensitivity targets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "publication_benchmark"


def metrics_at_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else np.nan,
        "false_positive_rate": fp / (tn + fp) if (tn + fp) else np.nan,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def threshold_grid(scores):
    scores = np.asarray(scores, dtype=float)
    eps = 1e-9
    grid = np.unique(np.concatenate([
        [scores.min() - eps, scores.max() + eps],
        scores,
        (scores[:-1] + scores[1:]) / 2 if len(scores) > 1 else scores,
    ]))
    return np.sort(grid)


def best_at_constraint(rows, constraint, target):
    if constraint == "specificity":
        eligible = [row for row in rows if row["specificity"] >= target]
        sort_key = lambda row: (row["sensitivity"], row["balanced_accuracy"], row["specificity"])
    elif constraint == "sensitivity":
        eligible = [row for row in rows if row["sensitivity"] >= target]
        sort_key = lambda row: (row["specificity"], row["balanced_accuracy"], row["sensitivity"])
    else:
        raise ValueError(f"Unknown constraint: {constraint}")

    if not eligible:
        return None
    return max(eligible, key=sort_key)


def analyze_group(group, specificity_targets, sensitivity_targets):
    y_true = group["y_true"].to_numpy(dtype=int)
    scores = group["score"].to_numpy(dtype=float)
    rows = [metrics_at_threshold(y_true, scores, t) for t in threshold_grid(np.sort(scores))]

    out = []
    base = {
        "train_limit": int(group["train_limit"].iloc[0]),
        "repeat": int(group["repeat"].iloc[0]),
        "seed": int(group["seed"].iloc[0]),
        "model": group["model"].iloc[0],
        "n_test": int(len(group)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }

    for target in specificity_targets:
        selected = best_at_constraint(rows, "specificity", target)
        record = dict(base)
        record.update({
            "constraint": "specificity",
            "target": target,
            "feasible": selected is not None,
        })
        if selected is not None:
            record.update(selected)
        out.append(record)

    for target in sensitivity_targets:
        selected = best_at_constraint(rows, "sensitivity", target)
        record = dict(base)
        record.update({
            "constraint": "sensitivity",
            "target": target,
            "feasible": selected is not None,
        })
        if selected is not None:
            record.update(selected)
        out.append(record)

    return out


def aggregate(rows):
    df = pd.DataFrame(rows)
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "false_negative_rate",
        "false_positive_rate",
        "threshold",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    group_cols = ["train_limit", "model", "constraint", "target"]
    out = []
    for key, group in df.groupby(group_cols, dropna=False):
        record = dict(zip(group_cols, key))
        record["n_runs"] = int(len(group))
        record["feasible_runs"] = int(group["feasible"].sum())
        feasible = group[group["feasible"]]
        for col in metric_cols:
            vals = pd.to_numeric(feasible[col], errors="coerce").dropna()
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
    parser = argparse.ArgumentParser(description="Threshold-aware EBTC score analysis")
    parser.add_argument(
        "--predictions_csv",
        default=str(RESULTS_DIR / "ebtc_publication_predictions.csv"),
    )
    parser.add_argument(
        "--output_csv",
        default=str(RESULTS_DIR / "ebtc_threshold_metrics.csv"),
    )
    parser.add_argument(
        "--aggregate_csv",
        default=str(RESULTS_DIR / "ebtc_threshold_aggregate.csv"),
    )
    parser.add_argument("--specificity_targets", nargs="*", type=float, default=[0.8, 0.9, 0.95])
    parser.add_argument("--sensitivity_targets", nargs="*", type=float, default=[0.8, 0.9, 0.95])
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)
    required = {"model", "repeat", "seed", "train_limit", "y_true", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {sorted(missing)}")

    rows = []
    for _, group in df.groupby(["train_limit", "repeat", "model"], sort=True):
        rows.extend(analyze_group(group, args.specificity_targets, args.sensitivity_targets))

    aggregate_rows = aggregate(rows)
    write_csv(Path(args.output_csv), rows)
    write_csv(Path(args.aggregate_csv), aggregate_rows)

    print(f"[results] threshold metrics: {args.output_csv}")
    print(f"[results] threshold aggregate: {args.aggregate_csv}")
    for row in aggregate_rows:
        if row["target"] in {0.9, 0.95}:
            print(row)


if __name__ == "__main__":
    main()
