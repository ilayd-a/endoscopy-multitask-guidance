"""
paired_switch_statistical_analysis.py
=====================================
Paired bootstrap confidence intervals and sign-flip permutation tests for
UNet-vs-switch Dice deltas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample = rng.integers(0, len(values), size=len(values))
        boots[idx] = float(values[sample].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def sign_flip_p(values: np.ndarray, n_permutations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    count = 0
    for _ in range(n_permutations):
        signs = np.where(rng.random(len(values)) < 0.5, -1.0, 1.0)
        if abs(float((values * signs).mean())) >= observed:
            count += 1
    return float((count + 1) / (n_permutations + 1))


def summarize(group: pd.DataFrame, n_bootstrap: int, n_permutations: int, seed: int) -> dict:
    delta = group["delta_vs_unet"].to_numpy(dtype=float)
    hard = group[group["hard_unet"].astype(bool)]
    hard_delta = hard["delta_vs_unet"].to_numpy(dtype=float)
    lo, hi = bootstrap_ci(delta, n_bootstrap, seed)
    p = sign_flip_p(delta, n_permutations, seed)
    if len(hard_delta):
        hlo, hhi = bootstrap_ci(hard_delta, n_bootstrap, seed)
        hp = sign_flip_p(hard_delta, n_permutations, seed)
    else:
        hlo = hhi = hp = np.nan
    return {
        "frames": int(len(group)),
        "hard_frames": int(len(hard)),
        "selected_dice": float(group["selected_dice"].mean()),
        "unet_dice": float(group["unet_dice"].mean()),
        "delta_vs_unet": float(delta.mean()),
        "delta_ci95_low": lo,
        "delta_ci95_high": hi,
        "delta_permutation_p": p,
        "sam_rate": float(group["use_sam"].mean()),
        "hard_selected_dice": float(hard["selected_dice"].mean()) if len(hard) else np.nan,
        "hard_unet_dice": float(hard["unet_dice"].mean()) if len(hard) else np.nan,
        "hard_delta_vs_unet": float(hard_delta.mean()) if len(hard_delta) else np.nan,
        "hard_delta_ci95_low": hlo,
        "hard_delta_ci95_high": hhi,
        "hard_delta_permutation_p": hp,
    }


def main():
    parser = argparse.ArgumentParser(description="Paired statistics for UNet-vs-switch per-frame outputs")
    parser.add_argument("--per_sample_glob", default="endoscopy_guidance/results/residual_gain_kvasir_hard_enriched_300_seed*_per_sample.csv")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/residual_gain_kvasir_hard_enriched_300_paired_stats.csv")
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--n_permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    files = sorted(Path().glob(args.per_sample_glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.per_sample_glob}")
    frames = []
    for path in files:
        seed_text = path.stem.split("seed")[-1].split("_")[0]
        frames.append(pd.read_csv(path).assign(seed=int(seed_text)))
    df = pd.concat(frames, ignore_index=True)
    if "policy" not in df.columns:
        raise ValueError("Per-sample files do not contain policy rows. Rerun residual_gain_prompt_fusion.py after the policy-output update.")

    rows = []
    for policy, group in df.groupby("policy", sort=False):
        row = {"policy": policy}
        row.update(summarize(group, args.n_bootstrap, args.n_permutations, args.seed))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["selected_dice", "hard_selected_dice"], ascending=False)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(out.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
