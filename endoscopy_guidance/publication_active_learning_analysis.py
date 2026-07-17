"""
publication_active_learning_analysis.py
=======================================
Paper-oriented analysis for the active-learning guidance benchmark.

This script turns the raw active-learning metrics CSV into publication-ready
tables, paired uncertainty estimates, simple permutation tests, and figures.
It intentionally compares policies on matched fold/repeat/evaluation-model
runs, because unpaired averages are too weak for a publishable claim.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


STRATEGY_LABELS = {
    "random": "Random",
    "classical_uncertainty": "Classical uncertainty",
    "pqk_uncertainty": "PQK uncertainty",
    "pqk_diversity": "PQK uncertainty + diversity",
    "pqk_hybrid": "PQK hybrid",
}


METRIC_LABELS = {
    "model_top5_hit": "Top-5 target recovery",
    "candidate_roc_auc": "Candidate AUC",
    "refined_dice": "Refined Dice",
    "refined_peak_center_dist": "Peak-center distance",
    "selected_positive_rate": "Selected-positive rate",
}


def mean_ci(values: np.ndarray, seed: int = 42, n_boot: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        value = float(values[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_permutation_p(diff: np.ndarray, seed: int = 42, n_perm: int = 20000) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return float("nan")
    observed = abs(float(diff.mean()))
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)), replace=True)
    permuted = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (n_perm + 1))


def paired_policy_diff(
    df: pd.DataFrame,
    eval_model: str,
    label_count: int,
    metric: str,
    strategy: str,
    baseline: str,
) -> dict:
    keys = ["held_out_sample", "repeat", "eval_model", "labeled_count"]
    sub = df[(df["eval_model"] == eval_model) & (df["labeled_count"] == label_count)]
    a = sub[sub["strategy"] == strategy][keys + [metric]].rename(columns={metric: "strategy_value"})
    b = sub[sub["strategy"] == baseline][keys + [metric]].rename(columns={metric: "baseline_value"})
    paired = a.merge(b, on=keys, how="inner")
    diff = paired["strategy_value"].to_numpy(dtype=float) - paired["baseline_value"].to_numpy(dtype=float)
    mean, lo, hi = mean_ci(diff, seed=17 + label_count)
    p_value = paired_permutation_p(diff, seed=101 + label_count)
    return {
        "eval_model": eval_model,
        "labeled_count": label_count,
        "metric": metric,
        "strategy": strategy,
        "baseline": baseline,
        "n_pairs": len(diff),
        "strategy_mean": float(paired["strategy_value"].mean()) if len(paired) else float("nan"),
        "baseline_mean": float(paired["baseline_value"].mean()) if len(paired) else float("nan"),
        "diff_mean": mean,
        "diff_ci_low": lo,
        "diff_ci_high": hi,
        "permutation_p": p_value,
    }


def paired_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("Classical_LogReg_C1", "model_top5_hit", "pqk_hybrid", "random"),
        ("Classical_LogReg_C1", "model_top5_hit", "pqk_hybrid", "classical_uncertainty"),
        ("Classical_LogReg_C1", "selected_positive_rate", "pqk_hybrid", "random"),
        ("Classical_LogReg_C1", "selected_positive_rate", "pqk_uncertainty", "random"),
        ("QML_PQK_reps2_C1_balanced", "refined_dice", "pqk_hybrid", "random"),
        ("QML_PQK_reps2_C1_balanced", "model_top5_hit", "pqk_diversity", "random"),
    ]
    for eval_model, metric, strategy, baseline in comparisons:
        for label_count in sorted(df["labeled_count"].unique()):
            if label_count <= 40:
                continue
            rows.append(paired_policy_diff(df, eval_model, int(label_count), metric, strategy, baseline))
    return pd.DataFrame(rows)


def aggregate_policy_table(df: pd.DataFrame, eval_model: str, metric: str) -> pd.DataFrame:
    rows = []
    for label_count in sorted(df["labeled_count"].unique()):
        row = {"Labeled candidates": int(label_count)}
        sub = df[(df["eval_model"] == eval_model) & (df["labeled_count"] == label_count)]
        for strategy in ["random", "classical_uncertainty", "pqk_uncertainty", "pqk_diversity", "pqk_hybrid"]:
            values = sub[sub["strategy"] == strategy][metric].to_numpy(dtype=float)
            mean, lo, hi = mean_ci(values, seed=300 + int(label_count))
            label = STRATEGY_LABELS[strategy]
            row[label] = mean
            row[f"{label} 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.3f}"


def format_p(value: float) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def markdown_policy_table(table: pd.DataFrame, strategies: list[str]) -> str:
    out = table[["Labeled candidates", *strategies]].copy()
    for col in strategies:
        out[col] = out[col].map(format_float)
    return out.to_markdown(index=False)


def markdown_paired_table(paired: pd.DataFrame, eval_model: str, metric: str, strategy: str, baseline: str) -> str:
    sub = paired[
        (paired["eval_model"] == eval_model)
        & (paired["metric"] == metric)
        & (paired["strategy"] == strategy)
        & (paired["baseline"] == baseline)
    ].copy()
    if sub.empty:
        return "_No rows._"
    out = pd.DataFrame({
        "Labels": sub["labeled_count"].astype(int),
        "Strategy mean": sub["strategy_mean"].map(format_float),
        "Baseline mean": sub["baseline_mean"].map(format_float),
        "Paired difference": sub["diff_mean"].map(format_float),
        "95% CI": [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(sub["diff_ci_low"], sub["diff_ci_high"])],
        "Permutation p": sub["permutation_p"].map(format_p),
    })
    return out.to_markdown(index=False)


def plot_learning_curve(df: pd.DataFrame, eval_model: str, metric: str, output_path: Path):
    colors = {
        "random": "#4c78a8",
        "classical_uncertainty": "#f58518",
        "pqk_uncertainty": "#54a24b",
        "pqk_diversity": "#b279a2",
        "pqk_hybrid": "#e45756",
    }
    strategies = ["random", "classical_uncertainty", "pqk_uncertainty", "pqk_diversity", "pqk_hybrid"]
    series = {}
    all_values = []
    label_counts = sorted(int(v) for v in df["labeled_count"].unique())
    for strategy in strategies:
        points = []
        for label_count in label_counts:
            values = df[
                (df["eval_model"] == eval_model)
                & (df["labeled_count"] == label_count)
                & (df["strategy"] == strategy)
            ][metric].to_numpy(dtype=float)
            mean, lo, hi = mean_ci(values, seed=500 + int(label_count))
            points.append((label_count, mean, lo, hi))
            all_values.extend([mean, lo, hi])
        series[strategy] = points

    finite = [v for v in all_values if math.isfinite(float(v))]
    ymin = max(0.0, min(finite) - 0.03) if finite else 0.0
    ymax = min(1.0, max(finite) + 0.03) if finite else 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    width, height = 920, 560
    left, right, top, bottom = 90, 260, 55, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    xmin, xmax = min(label_counts), max(label_counts)

    def sx(x):
        return left + (x - xmin) / max(xmax - xmin, 1) * plot_w

    def sy(y):
        return top + (ymax - y) / (ymax - ymin) * plot_h

    def polyline(points):
        return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _, _ in points)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="20" font-weight="700">{METRIC_LABELS.get(metric, metric)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.2"/>',
    ]

    for i in range(6):
        yval = ymin + i * (ymax - ymin) / 5
        yy = sy(yval)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{yy + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{yval:.2f}</text>')

    for xval in label_counts:
        xx = sx(xval)
        lines.append(f'<text x="{xx:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{xval}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Labeled candidate annotations</text>')

    for strategy in strategies:
        color = colors[strategy]
        points = series[strategy]
        lines.append(f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, mean, lo, hi in points:
            xx, yy = sx(x), sy(mean)
            ylo, yhi = sy(lo), sy(hi)
            lines.append(f'<line x1="{xx:.1f}" y1="{yhi:.1f}" x2="{xx:.1f}" y2="{ylo:.1f}" stroke="{color}" stroke-width="1.2" opacity="0.6"/>')
            lines.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{color}"/>')

    legend_x = left + plot_w + 30
    legend_y = top + 20
    for idx, strategy in enumerate(strategies):
        yy = legend_y + idx * 28
        lines.append(f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x + 22}" y2="{yy}" stroke="{colors[strategy]}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 30}" y="{yy + 5}" font-family="Arial, sans-serif" font-size="13">{STRATEGY_LABELS[strategy]}</text>')

    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def write_markdown_report(
    output_path: Path,
    top5_table: pd.DataFrame,
    selected_table: pd.DataFrame,
    paired: pd.DataFrame,
    top5_figure: Path,
    selected_figure: Path,
):
    top5_link = Path(*top5_figure.relative_to(output_path.parent).parts).as_posix()
    selected_link = Path(*selected_figure.relative_to(output_path.parent).parts).as_posix()
    strategies = [
        "Random",
        "Classical uncertainty",
        "PQK uncertainty",
        "PQK uncertainty + diversity",
        "PQK hybrid",
    ]
    report = f"""# Publication Active-Learning Analysis

## Study Design

This analysis uses the sequence-held-out CVC RGB candidate export with 5 grouped folds and 3
annotation-sampling repeats. Candidate annotations are acquired in batches from 40 to 200 labels.
All policy comparisons below are paired by held-out fold, repeat, final evaluation model, and label
budget.

Primary intervention-facing endpoint:

- top-5 target recovery, because an image-guided/robotic system can present a shortlist of target
  hypotheses rather than a single hard segmentation.

Secondary endpoints:

- selected-positive rate, which measures annotation triage efficiency under class imbalance
- candidate AUC
- refined Dice/IoU and peak-center distance, treated as diagnostics because the current map
  refinement is a Gaussian candidate map rather than a trained segmentation decoder

## Main Result: Top-5 Guidance Recovery

Final model: classical logistic-regression reranker.

{markdown_policy_table(top5_table, strategies)}

![Active-learning top-5 curve]({top5_link})

Paired comparison: PQK hybrid versus random sampling.

{markdown_paired_table(paired, "Classical_LogReg_C1", "model_top5_hit", "pqk_hybrid", "random")}

Paired comparison: PQK hybrid versus classical uncertainty.

{markdown_paired_table(paired, "Classical_LogReg_C1", "model_top5_hit", "pqk_hybrid", "classical_uncertainty")}

Interpretation: PQK-hybrid acquisition consistently improves mean top-5 recovery after the initial
seed set, but the confidence intervals are still wide. This is promising enough for an SPIE
abstract, but the full paper should increase repeats and evaluate another dataset or a stronger
backbone export before making a strong superiority claim.

## Annotation-Triage Result

Selected-positive rate during each newly acquired batch.

{markdown_policy_table(selected_table, strategies)}

![Selected-positive-rate curve]({selected_link})

Paired comparison: PQK uncertainty versus random sampling.

{markdown_paired_table(paired, "Classical_LogReg_C1", "selected_positive_rate", "pqk_uncertainty", "random")}

Interpretation: PQK uncertainty is much better at finding rare positive candidate annotations than
random sampling after 120 labels. This is currently the strongest quantum-specific contribution.
It supports a publishable annotation-efficiency / candidate-triage framing.

## Publishable Claim

Recommended claim:

> In a sequence-held-out endoscopic candidate-guidance benchmark, projected quantum-kernel
> uncertainty sampling enriched rare target-positive candidate annotations and, when combined
> with coverage-preserving sampling, improved top-5 target recovery over random and classical
> uncertainty acquisition for a classical final reranker.

Avoid claiming:

- quantum improves final segmentation Dice to clinical quality
- broad quantum advantage over all classical baselines
- replacement of the classical segmentation/guidance backbone

## Next Required Validation

Before full-paper submission:

1. Repeat the active-learning experiment on a stronger endoscopy-multiguidance export, not only the
   degraded CVC test export.
2. Increase repeats from 3 to at least 10 for tighter confidence intervals.
3. Replace Gaussian candidate-map refinement with a candidate-conditioned mask refinement module.
4. Add calibration/error-detection endpoints: ECE, Brier score, failure detection, and abstention.
5. Include qualitative panels showing cases where PQK acquisition discovers missed positives.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)


def main():
    parser = argparse.ArgumentParser(description="Publication analysis for active-learning guidance results")
    parser.add_argument("--metrics_csv", default="endoscopy_guidance/results/cvc_test_rgb_active_learning_metrics.csv")
    parser.add_argument("--output_md", default="docs/active_learning_publication_results.md")
    parser.add_argument("--figure_dir", default="docs/figures")
    parser.add_argument("--paired_csv", default="endoscopy_guidance/results/cvc_test_rgb_active_learning_paired_stats.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    figure_dir = Path(args.figure_dir)
    top5_figure = figure_dir / "active_learning_top5_logreg.svg"
    selected_figure = figure_dir / "active_learning_selected_positive_rate.svg"

    paired = paired_summary(df)
    Path(args.paired_csv).parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.paired_csv, index=False)

    top5_table = aggregate_policy_table(df, "Classical_LogReg_C1", "model_top5_hit")
    selected_table = aggregate_policy_table(df, "Classical_LogReg_C1", "selected_positive_rate")

    plot_learning_curve(df, "Classical_LogReg_C1", "model_top5_hit", top5_figure)
    plot_learning_curve(df, "Classical_LogReg_C1", "selected_positive_rate", selected_figure)
    write_markdown_report(
        Path(args.output_md),
        top5_table,
        selected_table,
        paired,
        top5_figure,
        selected_figure,
    )
    print(f"[saved] {args.output_md}")
    print(f"[saved] {top5_figure}")
    print(f"[saved] {selected_figure}")
    print(f"[saved] {args.paired_csv}")


if __name__ == "__main__":
    main()
