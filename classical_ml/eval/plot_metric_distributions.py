from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "f2",
    "mae",
    "pointing_game",
    "peak_center_dist",
)


def load_frames(
    inputs: list[Path],
    labels: list[str],
    group_column: str | None,
) -> pd.DataFrame:
    frames = []
    for path, label in zip(inputs, labels):
        frame = pd.read_csv(path).copy()
        frame["source"] = label
        if group_column is not None and group_column in frame.columns:
            frame["distribution_group"] = (
                frame["source"].astype(str) + ": " + frame[group_column].astype(str)
            )
        else:
            frame["distribution_group"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_distribution_summary(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for group_name, group in frame.groupby("distribution_group"):
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            values = values[values.notna() & np.isfinite(values)]
            if values.empty:
                continue
            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "count": int(len(values)),
                    "mean": round(float(values.mean()), 4),
                    "std": round(float(values.std(ddof=0)), 4),
                    "median": round(float(values.median()), 4),
                    "min": round(float(values.min()), 4),
                    "max": round(float(values.max()), 4),
                }
            )
    return pd.DataFrame(rows)


def plot_boxplots(
    frame: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    title: str,
) -> None:
    groups = sorted(frame["distribution_group"].unique())
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(4.2 * len(metrics), 4.8),
        squeeze=False,
    )
    axes = axes.reshape(-1)

    for axis, metric in zip(axes, metrics):
        if metric not in frame.columns:
            axis.set_visible(False)
            continue
        data = []
        labels = []
        for group in groups:
            values = pd.to_numeric(
                frame.loc[frame["distribution_group"] == group, metric],
                errors="coerce",
            )
            values = values[values.notna() & np.isfinite(values)]
            if values.empty:
                continue
            data.append(values.to_numpy())
            labels.append(group)
        if not data:
            axis.set_visible(False)
            continue
        axis.boxplot(data, showmeans=True)
        axis.set_xticklabels(labels)
        axis.set_title(metric, fontweight="bold")
        axis.tick_params(axis="x", rotation=35, labelsize=8)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_histograms(
    frame: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    title: str,
) -> None:
    groups = sorted(frame["distribution_group"].unique())
    rows = int(np.ceil(len(metrics) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4.2 * rows), squeeze=False)
    axes = axes.reshape(-1)

    for axis, metric in zip(axes, metrics):
        if metric not in frame.columns:
            axis.set_visible(False)
            continue
        plotted = False
        for group in groups:
            values = pd.to_numeric(
                frame.loc[frame["distribution_group"] == group, metric],
                errors="coerce",
            )
            values = values[values.notna() & np.isfinite(values)]
            if values.empty:
                continue
            axis.hist(values, bins=15, alpha=0.45, label=group)
            plotted = True
        if not plotted:
            axis.set_visible(False)
            continue
        axis.set_title(metric, fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)

    for axis in axes[len(metrics):]:
        axis.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-sample metric distributions for one or more eval CSVs."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    parser.add_argument("--group-column", type=str, default=None)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Metric Distributions")
    args = parser.parse_args()

    labels = args.labels or [path.stem for path in args.inputs]
    if len(labels) != len(args.inputs):
        raise ValueError("--labels must match the number of inputs")

    frame = load_frames(args.inputs, labels, args.group_column)
    summary = build_distribution_summary(frame, args.metrics)
    if summary.empty:
        raise ValueError("No valid numeric metric values were found.")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.csv")
    boxplot_path = args.output_prefix.with_name(f"{args.output_prefix.name}_boxplots.png")
    histogram_path = args.output_prefix.with_name(f"{args.output_prefix.name}_histograms.png")

    summary.to_csv(summary_path, index=False)
    plot_boxplots(frame, args.metrics, boxplot_path, args.title)
    plot_histograms(frame, args.metrics, histogram_path, args.title)

    print(f"Saved distribution summary -> {summary_path}")
    print(f"Saved boxplots             -> {boxplot_path}")
    print(f"Saved histograms           -> {histogram_path}")


if __name__ == "__main__":
    main()
