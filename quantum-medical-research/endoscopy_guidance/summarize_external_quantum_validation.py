"""
summarize_external_quantum_validation.py
=======================================
Summarize pooled external validation for pre-specified quantum-forward
confidence gates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_rows(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_csv"] = Path(path).name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def weighted_dice(rows: pd.DataFrame) -> float:
    accepted = rows["accepted_samples"].to_numpy(dtype=float)
    dice = rows["dice_mean"].to_numpy(dtype=float)
    total = accepted.sum()
    if total <= 0:
        return np.nan
    return float(np.sum(accepted * dice) / total)


def summarize(results: pd.DataFrame, targets: list[float], quantum_weights: list[float]) -> pd.DataFrame:
    q = results[results["policy"].eq("quantum_agreement_confidence")].copy()
    total_samples = int(results.groupby("split")["samples"].max().sum())
    rows = []
    for target, quantum_weight in zip(targets, quantum_weights):
        quantum = q[np.isclose(q["target_dice"], target) & np.isclose(q["agreement_weight"], quantum_weight)]
        confidence = q[np.isclose(q["target_dice"], target) & np.isclose(q["agreement_weight"], 0.0)]
        if quantum.empty or confidence.empty:
            continue
        q_accepted = int(quantum["accepted_samples"].sum())
        c_accepted = int(confidence["accepted_samples"].sum())
        q_dice = weighted_dice(quantum)
        c_dice = weighted_dice(confidence)
        rows.append({
            "target_dice": target,
            "quantum_agreement_weight": quantum_weight,
            "external_samples": total_samples,
            "quantum_accepted": q_accepted,
            "quantum_coverage": q_accepted / total_samples,
            "quantum_dice": q_dice,
            "confidence_accepted": c_accepted,
            "confidence_coverage": c_accepted / total_samples,
            "confidence_dice": c_dice,
            "dice_delta": q_dice - c_dice,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize pooled external quantum validation")
    parser.add_argument(
        "--result_csvs",
        nargs="+",
        default=[
            "endoscopy_guidance/results/polypgen_external_quantum_agreement_validation.csv",
            "endoscopy_guidance/results/kvasir_external_quantum_agreement_validation.csv",
        ],
    )
    parser.add_argument("--targets", type=float, nargs="+", default=[0.82, 0.86, 0.88])
    parser.add_argument("--quantum_weights", type=float, nargs="+", default=[0.4, 0.2, 0.4])
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/external_quantum_forward_summary.csv")
    args = parser.parse_args()

    if len(args.targets) != len(args.quantum_weights):
        raise ValueError("--targets and --quantum_weights must have the same length")

    results = load_rows(args.result_csvs)
    summary = summarize(results, args.targets, args.quantum_weights)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
