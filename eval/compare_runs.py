from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_METRICS = (
    "dice",
    "iou",
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
        metric: {"mean": float(frame[metric].mean()), "std": float(frame[metric].std())}
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
            rows.append(
                {
                    "run": label,
                    "metric": metric,
                    "mean": summary[metric]["mean"],
                    "std": summary[metric]["std"],
                }
            )

    comparison = pd.DataFrame(rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    comparison.to_csv(csv_path, index=False)

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
    print(f"Saved comparison plot -> {plot_path}")


if __name__ == "__main__":
    main()
