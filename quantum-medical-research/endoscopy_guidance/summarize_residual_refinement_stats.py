"""
summarize_residual_refinement_stats.py
======================================
Summarize per-frame residual refinement runs with paired statistics.

The script expects one or more model result directories produced by
`apply_residual_mask_refinement.py`, each containing `per_frame.csv` and
`summary.csv`. It reports paired bootstrap confidence intervals and sign-flip
permutation p-values for baseline-vs-refined and model-vs-model deltas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load_model_frame(path: Path, model_name: str, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path / "per_frame.csv")
    summary = pd.read_csv(path / "summary.csv")
    frame["model"] = model_name
    frame["seed"] = seed
    if "triggered_frames" in summary.columns:
        frame["triggered_frames_summary"] = int(summary.loc[summary["split"].eq("test"), "triggered_frames"].iloc[0])
    return frame


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
    low, med, high = np.quantile(means, [0.025, 0.50, 0.975])
    return float(low), float(med), float(high)


def stats_row(label: str, subset: str, delta: np.ndarray, rng: np.random.Generator, n_iter: int) -> dict:
    delta = np.asarray(delta, dtype=float)
    ci_low, ci_med, ci_high = bootstrap_ci(delta, rng, n_iter)
    try:
        p_wilcoxon = float(wilcoxon(delta, zero_method="zsplit").pvalue)
    except ValueError:
        p_wilcoxon = np.nan
    return {
        "comparison": label,
        "subset": subset,
        "n": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci_low": ci_low,
        "ci_median": ci_med,
        "ci_high": ci_high,
        "signflip_p": signflip_p(delta, rng, n_iter),
        "wilcoxon_p": p_wilcoxon,
        "improved": int((delta > 1e-9).sum()),
        "worsened": int((delta < -1e-9).sum()),
        "unchanged": int(np.isclose(delta, 0.0, atol=1e-9).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize residual refinement paired statistics")
    parser.add_argument("--run", action="append", nargs=3, metavar=("MODEL", "SEED", "DIR"), required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--hard_dice_threshold", type=float, default=0.80)
    parser.add_argument("--bootstrap_iter", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    frames = []
    for model, seed_text, directory in args.run:
        frames.append(load_model_frame(Path(directory), model, int(seed_text)))
    all_frames = pd.concat(frames, ignore_index=True)
    test = all_frames.loc[all_frames["split"].eq("test")].copy()

    rows = []
    for (model, seed), group in test.groupby(["model", "seed"]):
        masks = {
            "all": np.ones(len(group), dtype=bool),
            "hard": group["baseline_dice"].to_numpy() < args.hard_dice_threshold,
            "triggered": group["triggered"].to_numpy(dtype=bool) if "triggered" in group.columns else np.ones(len(group), dtype=bool),
        }
        for subset, mask in masks.items():
            if mask.sum() == 0:
                continue
            rows.append(stats_row(f"{model} seed {seed} vs baseline", subset, group.loc[mask, "delta_dice"].to_numpy(), rng, args.bootstrap_iter))

    pivot = {
        (model, seed): group.sort_values("sample_id").reset_index(drop=True)
        for (model, seed), group in test.groupby(["model", "seed"])
    }
    keys = sorted(pivot)
    for i, (model_a, seed_a) in enumerate(keys):
        for model_b, seed_b in keys[i + 1 :]:
            if seed_a != seed_b:
                continue
            a = pivot[(model_a, seed_a)]
            b = pivot[(model_b, seed_b)]
            if not a["sample_id"].equals(b["sample_id"]):
                raise ValueError(f"Sample mismatch for seed {seed_a}: {model_a} vs {model_b}")
            masks = {
                "all": np.ones(len(a), dtype=bool),
                "hard": a["baseline_dice"].to_numpy() < args.hard_dice_threshold,
                "triggered": a["triggered"].to_numpy(dtype=bool) | b["triggered"].to_numpy(dtype=bool),
            }
            diff = a["delta_dice"].to_numpy() - b["delta_dice"].to_numpy()
            for subset, mask in masks.items():
                if mask.sum() == 0:
                    continue
                rows.append(stats_row(f"{model_a} - {model_b} seed {seed_a}", subset, diff[mask], rng, args.bootstrap_iter))

    summary = pd.DataFrame(rows)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
