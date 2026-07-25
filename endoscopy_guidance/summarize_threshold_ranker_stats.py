"""
summarize_threshold_ranker_stats.py
===================================
Paired statistics for threshold-hypothesis selector per-frame outputs.

The input is the per-frame CSV produced by
`quantum_threshold_pairwise_ranker.py`. The script summarizes model-vs-baseline
and model-vs-model Dice deltas with bootstrap confidence intervals,
sign-flip permutation p-values, and Wilcoxon signed-rank p-values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def signflip_p(delta: np.ndarray, rng: np.random.Generator, n_iter: int) -> float:
    delta = np.asarray(delta, dtype=float)
    obs = abs(float(delta.mean()))
    count = 0
    batch = 5000
    done = 0
    while done < n_iter:
        size = min(batch, n_iter - done)
        signs = rng.choice((-1.0, 1.0), size=(size, len(delta)))
        sims = np.abs((signs * delta).mean(axis=1))
        count += int((sims >= obs).sum())
        done += size
    return float((count + 1) / (n_iter + 1))


def bootstrap_ci(delta: np.ndarray, rng: np.random.Generator, n_iter: int) -> tuple[float, float, float]:
    delta = np.asarray(delta, dtype=float)
    idx = rng.integers(0, len(delta), size=(n_iter, len(delta)))
    means = delta[idx].mean(axis=1)
    low, median, high = np.quantile(means, [0.025, 0.50, 0.975])
    return float(low), float(median), float(high)


def stats_row(label: str, subset: str, delta: np.ndarray, rng: np.random.Generator, n_iter: int) -> dict:
    delta = np.asarray(delta, dtype=float)
    low, median, high = bootstrap_ci(delta, rng, n_iter)
    try:
        p_wilcoxon = float(wilcoxon(delta, zero_method="zsplit").pvalue)
    except ValueError:
        p_wilcoxon = np.nan
    return {
        "comparison": label,
        "subset": subset,
        "n": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci_low": low,
        "ci_median": median,
        "ci_high": high,
        "signflip_p": signflip_p(delta, rng, n_iter),
        "wilcoxon_p": p_wilcoxon,
        "improved": int((delta > 1e-9).sum()),
        "worsened": int((delta < -1e-9).sum()),
        "unchanged": int(np.isclose(delta, 0.0, atol=1e-9).sum()),
    }


def subset_masks(frame: pd.DataFrame, hard_dice_threshold: float) -> dict[str, np.ndarray]:
    baseline = frame["baseline_dice"].to_numpy(dtype=float)
    changed = ~np.isclose(frame["selected_threshold"].to_numpy(dtype=float), 0.5)
    return {
        "all": np.ones(len(frame), dtype=bool),
        "hard": baseline < hard_dice_threshold,
        "changed": changed,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize threshold selector paired statistics")
    parser.add_argument("--per_frame_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--hard_dice_threshold", type=float, default=0.80)
    parser.add_argument("--bootstrap_iter", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    frame = pd.read_csv(args.per_frame_csv)
    selected_models = args.models or sorted(frame["model"].unique())
    frame = frame.loc[frame["model"].isin(selected_models)].copy()
    by_model = {
        model: group.sort_values("sample_id").reset_index(drop=True)
        for model, group in frame.groupby("model")
    }

    rows = []
    for model, group in by_model.items():
        masks = subset_masks(group, args.hard_dice_threshold)
        delta = group["delta_dice"].to_numpy(dtype=float)
        for subset, mask in masks.items():
            if mask.sum() == 0:
                continue
            rows.append(stats_row(f"{model} vs fixed 0.50", subset, delta[mask], rng, args.bootstrap_iter))

    models = sorted(by_model)
    for idx, model_a in enumerate(models):
        for model_b in models[idx + 1 :]:
            a = by_model[model_a]
            b = by_model[model_b]
            if not a["sample_id"].equals(b["sample_id"]):
                raise ValueError(f"Sample mismatch: {model_a} vs {model_b}")
            diff = a["delta_dice"].to_numpy(dtype=float) - b["delta_dice"].to_numpy(dtype=float)
            masks = subset_masks(a, args.hard_dice_threshold)
            for subset, mask in masks.items():
                if mask.sum() == 0:
                    continue
                rows.append(stats_row(f"{model_a} - {model_b}", subset, diff[mask], rng, args.bootstrap_iter))

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
