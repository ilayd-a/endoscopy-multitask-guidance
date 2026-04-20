from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import PROJECT_ROOT, load_binary_mask, load_rgb_image, load_samples
from evaluate_outputs import evaluate_matches
from generate_panels import normalize_mode, overlay_heatmap, select_samples
from predictions import (
    discover_outputs_root,
    discover_prediction_dirs,
    load_prediction_heatmap,
    load_prediction_mask,
    match_predictions_to_samples,
)


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_DISPLAY_METRICS = ("dice", "iou", "boundary_f1", "pointing_game")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate side-by-side qualitative comparisons across multiple runs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cvc",
        choices=["cvc", "kvasir", "split_folder"],
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--split-column", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--outputs-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument(
        "--eval-csvs",
        nargs="*",
        type=Path,
        default=None,
        help="Optional per-run evaluation CSVs matching --outputs-dirs order.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", type=str, default="run_comparison")
    parser.add_argument(
        "--mode",
        type=str,
        default="weak",
        choices=["best", "strong", "median", "worst", "weak", "random"],
    )
    parser.add_argument("--metric", type=str, default="dice")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="Which run to use for selecting strong/weak cases.",
    )
    parser.add_argument("--allow-index-fallback", action="store_true")
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help="Optional output path for the selected cases with per-run metrics.",
    )
    return parser.parse_args()


def format_metric_summary(metrics_row: pd.Series) -> str:
    parts = []
    for metric in DEFAULT_DISPLAY_METRICS:
        if metric not in metrics_row or pd.isna(metrics_row[metric]):
            continue
        value = metrics_row[metric]
        if metric == "pointing_game":
            parts.append(f"PG {int(value)}")
        else:
            short = {
                "dice": "D",
                "iou": "I",
                "boundary_f1": "BF1",
            }.get(metric, metric)
            parts.append(f"{short} {float(value):.3f}")
    return " | ".join(parts)


def load_run_bundle(
    label: str,
    outputs_dir: Path,
    eval_csv: Path | None,
    args: argparse.Namespace,
    samples: list,
) -> dict:
    outputs_root = discover_outputs_root(PROJECT_ROOT, outputs_dir)
    predictions = discover_prediction_dirs(outputs_root)
    matches = match_predictions_to_samples(
        samples=samples,
        predictions=predictions,
        allow_index_fallback=args.allow_index_fallback,
    )
    prediction_map = {sample.sample_id: prediction for sample, prediction in matches}

    if eval_csv is not None:
        eval_df = pd.read_csv(eval_csv)
    else:
        eval_df = evaluate_matches(matches, args.dataset, args.split, label)
    eval_index = eval_df.set_index("sample_id")
    return {
        "label": label,
        "prediction_map": prediction_map,
        "eval_df": eval_df,
        "eval_index": eval_index,
    }


def build_selection_frame(
    selected_ids: list[str],
    runs: list[dict],
) -> pd.DataFrame:
    rows = []
    for sample_id in selected_ids:
        row = {"sample_id": sample_id}
        for run in runs:
            if sample_id not in run["eval_index"].index:
                continue
            metrics_row = run["eval_index"].loc[sample_id]
            for metric in DEFAULT_DISPLAY_METRICS + ("peak_center_dist", "coherence_mer"):
                if metric in metrics_row.index:
                    row[f"{run['label']}_{metric}"] = metrics_row[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    labels = args.labels or [path.name for path in args.outputs_dirs]
    if len(labels) != len(args.outputs_dirs):
        raise ValueError("--labels must match the number of --outputs-dirs")

    eval_csvs = args.eval_csvs or []
    if eval_csvs and len(eval_csvs) != len(args.outputs_dirs):
        raise ValueError("--eval-csvs must match the number of --outputs-dirs")
    if not eval_csvs:
        eval_csvs = [None] * len(args.outputs_dirs)

    samples = load_samples(
        dataset=args.dataset,
        split=args.split,
        dataset_root=args.dataset_root,
        metadata_path=args.metadata_path,
        split_manifest=args.split_manifest,
        split_column=args.split_column,
    )

    runs = [
        load_run_bundle(label, outputs_dir, eval_csv, args, samples)
        for label, outputs_dir, eval_csv in zip(labels, args.outputs_dirs, eval_csvs)
    ]
    if not 0 <= args.reference_index < len(runs):
        raise ValueError("--reference-index is out of range")

    normalized_mode = normalize_mode(args.mode)
    reference_df = runs[args.reference_index]["eval_df"]
    if normalized_mode == "random":
        selected_df = reference_df.sample(
            n=min(args.samples, len(reference_df)),
            random_state=42,
        )
    else:
        selected_df = select_samples(
            reference_df,
            args.metric,
            normalized_mode,
            args.samples,
        ).copy()
    selected_ids = selected_df["sample_id"].tolist()
    if not selected_ids:
        raise ValueError("No samples selected for qualitative comparison.")

    sample_map = {sample.sample_id: sample for sample in samples}
    selection_frame = build_selection_frame(selected_ids, runs)

    column_count = 2 + 2 * len(runs)
    fig_width = 4.0 * column_count
    fig_height = 3.0 * max(1, len(selected_ids))
    fig, axes = plt.subplots(len(selected_ids), column_count, figsize=(fig_width, fig_height))
    axes = np.atleast_2d(axes)

    titles = ["Input", "Ground Truth"]
    for run in runs:
        titles.extend([f"{run['label']} Pred", f"{run['label']} Heatmap"])
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for row_index, sample_id in enumerate(selected_ids):
        sample = sample_map[sample_id]
        first_run_prediction = runs[0]["prediction_map"][sample_id]
        target_shape = load_prediction_heatmap(first_run_prediction.pred_heatmap_path).shape
        image_rgb = load_rgb_image(sample.image_path, target_shape=target_shape)
        gt_mask = load_binary_mask(sample.mask_path, target_shape=target_shape)

        axes[row_index, 0].imshow(image_rgb)
        axes[row_index, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 0].set_ylabel(sample_id, fontsize=8, rotation=90, labelpad=10)

        for run_index, run in enumerate(runs):
            prediction = run["prediction_map"][sample_id]
            heatmap = load_prediction_heatmap(
                prediction.pred_heatmap_path,
                target_shape=target_shape,
            )
            pred_mask = load_prediction_mask(
                prediction.pred_mask_path,
                target_shape=target_shape,
            )
            pred_axis = axes[row_index, 2 + run_index * 2]
            heat_axis = axes[row_index, 3 + run_index * 2]
            pred_axis.imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
            heat_axis.imshow(overlay_heatmap(image_rgb, heatmap))

            metrics_row = run["eval_index"].loc[sample_id]
            summary_text = format_metric_summary(metrics_row)
            pred_axis.set_xlabel(summary_text, fontsize=7, labelpad=4)

        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("black")

    fig.tight_layout()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.results_dir / (
        f"{args.dataset}_{args.split}_{args.run_name}_{args.mode}_{args.metric}_comparison.png"
    )
    fig.savefig(output_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    selection_csv = args.selection_csv or args.results_dir / (
        f"{args.dataset}_{args.split}_{args.run_name}_{args.mode}_{args.metric}_comparison_selection.csv"
    )
    selection_frame.to_csv(selection_csv, index=False)

    print(f"Saved qualitative comparison -> {output_path}")
    print(f"Saved selected cases CSV     -> {selection_csv}")


if __name__ == "__main__":
    main()
