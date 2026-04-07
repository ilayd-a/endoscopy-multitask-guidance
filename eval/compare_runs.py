from __future__ import annotations

import argparse
import json
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
    "boundary_f1",
    "pointing_game",
    "peak_center_dist",
    "coherence_mer",
)


def load_summary(path: Path) -> dict[str, dict[str, float]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["metrics"]

    frame = pd.read_csv(path)
    if "metric" in frame.columns and {"mean", "std"}.issubset(frame.columns):
        return {
            row["metric"]: {"mean": float(row["mean"]), "std": float(row["std"])}
            for _, row in frame.iterrows()
        }

    return {
        metric: {
            "mean": float(pd.to_numeric(frame[metric], errors="coerce").mean()),
            "std": float(pd.to_numeric(frame[metric], errors="coerce").std(ddof=0)),
        }
        for metric in DEFAULT_METRICS
        if metric in frame.columns
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple evaluated runs and generate a summary chart."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Run Comparison")
    args = parser.parse_args()

    labels = args.labels or [path.stem for path in args.inputs]
    if len(labels) != len(args.inputs):
        raise ValueError("--labels must match the number of inputs")

    rows = []
    for label, path in zip(labels, args.inputs):
        summary = load_summary(path)
        for metric in args.metrics:
            if metric not in summary:
                continue
            mean = summary[metric]["mean"]
            std = summary[metric]["std"]
            if mean is None or not np.isfinite(mean):
                continue
            if std is None or not np.isfinite(std):
                std = 0.0
            rows.append(
                {
                    "run": label,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                }
            )

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError("No valid summary metrics were found for comparison.")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    comparison.to_csv(csv_path, index=False)

    wide_rows = []
    for run in labels:
        run_df = comparison[comparison["run"] == run]
        row = {"run": run}
        for metric in args.metrics:
            metric_df = run_df[run_df["metric"] == metric]
            if metric_df.empty:
                continue
            row[metric] = (
                f"{metric_df.iloc[0]['mean']:.4f} +/- {metric_df.iloc[0]['std']:.4f}"
            )
            row[f"{metric}_mean"] = metric_df.iloc[0]["mean"]
            row[f"{metric}_std"] = metric_df.iloc[0]["std"]
        wide_rows.append(row)

    wide_comparison = pd.DataFrame(wide_rows)
    wide_csv_path = args.output_prefix.with_name(f"{args.output_prefix.name}_wide.csv")
    wide_comparison.to_csv(wide_csv_path, index=False)

    fig, axes = plt.subplots(
        1, len(args.metrics), figsize=(4.5 * len(args.metrics), 4.5)
    )
    if len(args.metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, args.metrics):
        metric_df = comparison[comparison["metric"] == metric]
        axis.bar(
            metric_df["run"],
            metric_df["mean"],
            yerr=metric_df["std"],
            capsize=4,
            color="#4472C4",
            alpha=0.85,
        )
        axis.set_title(metric, fontweight="bold")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    plot_path = args.output_prefix.with_suffix(".png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved comparison CSV  -> {csv_path}")
    print(f"Saved wide CSV        -> {wide_csv_path}")
    print(f"Saved comparison plot -> {plot_path}")


if __name__ == "__main__":
    main()
