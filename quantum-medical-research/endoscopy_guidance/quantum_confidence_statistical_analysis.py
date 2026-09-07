"""
quantum_confidence_statistical_analysis.py
==========================================
Paired held-out statistical tests for quantum-agreement confidence gating.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def accepted_dice_by_sample(df: pd.DataFrame) -> pd.Series:
    rows = df.copy()
    rows["accepted_dice"] = np.where(rows["accepted"].astype(bool), rows["sam_dice"], np.nan)
    return rows.set_index("sample_id")["accepted_dice"]


def nanmean_delta(base: np.ndarray, quantum: np.ndarray) -> float:
    return float(np.nanmean(quantum) - np.nanmean(base))


def bootstrap_ci(base: np.ndarray, quantum: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(base)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        deltas.append(nanmean_delta(base[idx], quantum[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def permutation_p(base: np.ndarray, quantum: np.ndarray, n_permutations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(nanmean_delta(base, quantum))
    count = 0
    for _ in range(n_permutations):
        swap = rng.random(len(base)) < 0.5
        b = base.copy()
        q = quantum.copy()
        b[swap], q[swap] = q[swap], b[swap]
        if abs(nanmean_delta(b, q)) >= observed:
            count += 1
    return float((count + 1) / (n_permutations + 1))


def summarize_policy(df: pd.DataFrame) -> dict:
    accepted = df[df["accepted"].astype(bool)]
    return {
        "coverage": float(len(accepted) / df["sample_id"].nunique()),
        "accepted_samples": int(len(accepted)),
        "dice_mean": float(accepted["sam_dice"].mean()),
        "iou_mean": float(accepted["sam_iou"].mean()),
        "dice_ge_050": float((accepted["sam_dice"] >= 0.50).mean()),
        "dice_ge_070": float((accepted["sam_dice"] >= 0.70).mean()),
        "prompt_hit": float(accepted["point_hit"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Paired stats for quantum confidence gating")
    parser.add_argument("--per_sample_csv", default="endoscopy_guidance/results/quantum_agreement_confidence_per_sample.csv")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/quantum_confidence_statistical_analysis.csv")
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--n_permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    df = pd.read_csv(args.per_sample_csv)
    test = df[df["split"].eq("test")].copy()
    rows = []
    for target in sorted(test["target_dice"].unique()):
        base = test[(test["target_dice"].eq(target)) & (test["agreement_weight"].eq(0.0))]
        if base.empty:
            continue
        base_series = accepted_dice_by_sample(base)
        for weight in sorted(w for w in test["agreement_weight"].unique() if w > 0):
            q = test[(test["target_dice"].eq(target)) & (test["agreement_weight"].eq(weight))]
            q_series = accepted_dice_by_sample(q)
            aligned = pd.concat([base_series.rename("base"), q_series.rename("quantum")], axis=1).sort_index()
            base_arr = aligned["base"].to_numpy(dtype=float)
            q_arr = aligned["quantum"].to_numpy(dtype=float)
            observed = nanmean_delta(base_arr, q_arr)
            lo, hi = bootstrap_ci(base_arr, q_arr, args.n_bootstrap, args.seed)
            p = permutation_p(base_arr, q_arr, args.n_permutations, args.seed)
            base_summary = summarize_policy(base)
            q_summary = summarize_policy(q)
            rows.append({
                "target_dice": float(target),
                "agreement_weight": float(weight),
                "baseline_coverage": base_summary["coverage"],
                "quantum_coverage": q_summary["coverage"],
                "baseline_dice": base_summary["dice_mean"],
                "quantum_dice": q_summary["dice_mean"],
                "dice_delta": observed,
                "dice_delta_ci95_low": lo,
                "dice_delta_ci95_high": hi,
                "permutation_p": p,
                "baseline_dice_ge_050": base_summary["dice_ge_050"],
                "quantum_dice_ge_050": q_summary["dice_ge_050"],
                "baseline_dice_ge_070": base_summary["dice_ge_070"],
                "quantum_dice_ge_070": q_summary["dice_ge_070"],
            })

    out = pd.DataFrame(rows).sort_values(["target_dice", "dice_delta"], ascending=[True, False])
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
