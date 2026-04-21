from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import PROJECT_ROOT, load_binary_mask, load_rgb_image, load_samples
from evaluate_outputs import evaluate_matches
from generate_panels import overlay_heatmap
from predictions import (
    discover_outputs_root,
    discover_prediction_dirs,
    load_prediction_heatmap,
    load_prediction_mask,
    match_predictions_to_samples,
)


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
HIGHER_IS_BETTER = {"dice", "iou", "precision", "recall", "boundary_f1", "pointing_game", "coherence_mer"}
LOWER_IS_BETTER = {"peak_center_dist", "abs_area_error"}
PATTERN_COLUMNS = (
    "small_region",
    "low_contrast",
    "boundary_error",
    "over_segmentation",
    "under_segmentation",
    "fragmented_prediction",
    "missed_target",
    "border_touching_target",
    "heatmap_miss",
)
def load_or_compute_eval_frame(
    args: argparse.Namespace,
    matches: list[tuple],
) -> pd.DataFrame:
    if args.eval_csv is not None and args.eval_csv.exists():
        return pd.read_csv(args.eval_csv)
    return evaluate_matches(matches, args.dataset, args.split, args.run_name)


def select_extreme_cases(
    frame: pd.DataFrame,
    metric: str,
    count: int,
    mode: str,
) -> pd.DataFrame:
    if metric in HIGHER_IS_BETTER:
        ascending = mode == "worst"
    elif metric in LOWER_IS_BETTER:
        ascending = mode != "worst"
    else:
        raise ValueError(f"Unsupported metric for failure analysis: {metric}")

    ranked = frame.sort_values(metric, ascending=ascending).reset_index(drop=True)
    return ranked.head(min(count, len(ranked)))


def touches_border(mask: np.ndarray) -> bool:
    mask_bool = np.asarray(mask) > 0
    if not mask_bool.any():
        return False
    return bool(
        mask_bool[0, :].any()
        or mask_bool[-1, :].any()
        or mask_bool[:, 0].any()
        or mask_bool[:, -1].any()
    )


def build_failure_frame(
    eval_df: pd.DataFrame,
    matches: list[tuple],
) -> pd.DataFrame:
    metric_map = eval_df.set_index("sample_id").to_dict(orient="index")
    records: list[dict] = []

    for sample, prediction in matches:
        heatmap = load_prediction_heatmap(prediction.pred_heatmap_path)
        pred_mask = load_prediction_mask(prediction.pred_mask_path, target_shape=heatmap.shape)
        gt_mask = load_binary_mask(sample.mask_path, target_shape=heatmap.shape)
        image_rgb = load_rgb_image(sample.image_path, target_shape=heatmap.shape)

        metrics_row = metric_map.get(sample.sample_id, {}).copy()
        gray = np.dot(image_rgb.astype(np.float32) / 255.0, [0.299, 0.587, 0.114])
        gt_bool = gt_mask > 0
        outside = ~gt_bool
        contrast_delta = (
            float(abs(gray[gt_bool].mean() - gray[outside].mean()))
            if gt_bool.any() and outside.any()
            else 0.0
        )

        gt_area_ratio = float(metrics_row.get("gt_area_ratio", 0.0))
        pred_area_ratio = float(metrics_row.get("pred_area_ratio", 0.0))
        dice = float(metrics_row.get("dice", 0.0))
        boundary_f1 = float(metrics_row.get("boundary_f1", 0.0))
        fp_pixels = int(metrics_row.get("fp_pixels", 0))
        fn_pixels = int(metrics_row.get("fn_pixels", 0))
        gt_pixels = int(metrics_row.get("gt_pixels", int(gt_bool.sum())))
        pred_pixels = int(metrics_row.get("pred_pixels", int((pred_mask > 0).sum())))

        record = {
            **metrics_row,
            "sample_id": sample.sample_id,
            "contrast_delta": round(contrast_delta, 4),
            "gt_touches_border": touches_border(gt_mask),
            "pred_touches_border": touches_border(pred_mask),
            "small_region": gt_area_ratio <= 0.02,
            "low_contrast": contrast_delta <= 0.08,
            "boundary_error": gt_pixels > 0 and pred_pixels > 0 and boundary_f1 + 0.1 < dice,
            "over_segmentation": pred_area_ratio > max(gt_area_ratio * 1.5, gt_area_ratio + 0.01) and fp_pixels > fn_pixels,
            "under_segmentation": gt_area_ratio > 0.0 and pred_area_ratio < gt_area_ratio * 0.67 and fn_pixels >= fp_pixels,
            "fragmented_prediction": int(metrics_row.get("pred_components", 0)) > 1,
            "missed_target": pred_pixels == 0 and gt_pixels > 0,
            "border_touching_target": touches_border(gt_mask),
            "heatmap_miss": int(metrics_row.get("pointing_game", 0)) == 0,
        }
        records.append(record)

    return pd.DataFrame(records)


def save_case_figure(
    output_path: Path,
    sample_id: str,
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    heatmap: np.ndarray,
    metrics_row: pd.Series,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    titles = ["Input", "Prediction", "Ground Truth", "Heatmap"]
    panels = [
        image_rgb,
        pred_mask,
        gt_mask,
        overlay_heatmap(image_rgb, heatmap),
    ]
    cmaps = [None, "gray", "gray", None]

    for axis, title, panel, cmap in zip(axes, titles, panels, cmaps):
        axis.set_title(title, fontweight="bold")
        axis.imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(
        f"{sample_id} | Dice {metrics_row['dice']:.4f} | IoU {metrics_row['iou']:.4f} | Boundary F1 {metrics_row['boundary_f1']:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_failure_summary(
    analysis_df: pd.DataFrame,
    worst_df: pd.DataFrame,
    metric: str,
) -> dict:
    summary = {
        "num_samples": int(len(analysis_df)),
        "selection_metric": metric,
        "worst_case_count": int(len(worst_df)),
        "pattern_counts_all": {},
        "pattern_counts_worst": {},
        "pattern_metric_means": {},
    }

    for column in PATTERN_COLUMNS:
        summary["pattern_counts_all"][column] = int(analysis_df[column].sum())
        summary["pattern_counts_worst"][column] = int(worst_df[column].sum())
        flagged = analysis_df[analysis_df[column]]
        summary["pattern_metric_means"][column] = {
            "count": int(len(flagged)),
            "dice_mean": round(float(flagged["dice"].mean()), 4) if len(flagged) else None,
            "iou_mean": round(float(flagged["iou"].mean()), 4) if len(flagged) else None,
            "boundary_f1_mean": round(float(flagged["boundary_f1"].mean()), 4) if len(flagged) else None,
        }
    return summary


def write_failure_markdown(
    output_path: Path,
    worst_df: pd.DataFrame,
    summary: dict,
) -> None:
    lines = [
        "# Failure Analysis",
        "",
        f"- Samples analyzed: {summary['num_samples']}",
        f"- Worst-case metric: `{summary['selection_metric']}`",
        f"- Worst cases exported: {summary['worst_case_count']}",
        "",
        "## Worst Cases",
        "",
    ]

    for _, row in worst_df.iterrows():
        active_patterns = [name for name in PATTERN_COLUMNS if bool(row[name])]
        pattern_text = ", ".join(active_patterns) if active_patterns else "none flagged"
        lines.append(
            f"- `{row['sample_id']}`: dice={row['dice']:.4f}, iou={row['iou']:.4f}, "
            f"boundary_f1={row['boundary_f1']:.4f}, patterns={pattern_text}"
        )

    lines.extend(["", "## Pattern Counts Among Worst Cases", ""])
    for name, count in summary["pattern_counts_worst"].items():
        lines.append(f"- `{name}`: {count}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze failure patterns and export worst-case qualitative examples."
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
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="active_run")
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--metric", type=str, default="dice")
    parser.add_argument("--worst-k", type=int, default=12)
    parser.add_argument("--allow-index-fallback", action="store_true")
    args = parser.parse_args()

    samples = load_samples(
        dataset=args.dataset,
        split=args.split,
        dataset_root=args.dataset_root,
        metadata_path=args.metadata_path,
        split_manifest=args.split_manifest,
        split_column=args.split_column,
    )
    outputs_root = discover_outputs_root(PROJECT_ROOT, args.outputs_dir)
    predictions = discover_prediction_dirs(outputs_root)
    matches = match_predictions_to_samples(
        samples=samples,
        predictions=predictions,
        allow_index_fallback=args.allow_index_fallback,
    )

    eval_df = load_or_compute_eval_frame(args, matches)
    analysis_df = build_failure_frame(eval_df, matches)
    worst_df = select_extreme_cases(analysis_df, args.metric, args.worst_k, mode="worst")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{args.dataset}_{args.split}_{args.run_name}"
    analysis_csv_path = args.results_dir / f"{output_prefix}_failure_analysis.csv"
    worst_csv_path = args.results_dir / f"{output_prefix}_worst_{args.metric}.csv"
    summary_json_path = args.results_dir / f"{output_prefix}_failure_summary.json"
    summary_md_path = args.results_dir / f"{output_prefix}_failure_summary.md"
    cases_dir = args.results_dir / f"{output_prefix}_worst_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    analysis_df.to_csv(analysis_csv_path, index=False)
    worst_df.to_csv(worst_csv_path, index=False)

    match_map = {sample.sample_id: prediction for sample, prediction in matches}
    sample_map = {sample.sample_id: sample for sample in samples}
    for rank, (_, row) in enumerate(worst_df.iterrows(), start=1):
        sample = sample_map[row["sample_id"]]
        prediction = match_map[row["sample_id"]]
        heatmap = load_prediction_heatmap(prediction.pred_heatmap_path)
        pred_mask = load_prediction_mask(prediction.pred_mask_path, target_shape=heatmap.shape)
        gt_mask = load_binary_mask(sample.mask_path, target_shape=heatmap.shape)
        image_rgb = load_rgb_image(sample.image_path, target_shape=heatmap.shape)
        safe_id = Path(str(row["sample_id"])).stem.replace("/", "_")
        figure_path = cases_dir / f"{rank:02d}_{safe_id}.png"
        save_case_figure(figure_path, row["sample_id"], image_rgb, gt_mask, pred_mask, heatmap, row)

    summary = build_failure_summary(analysis_df, worst_df, args.metric)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_failure_markdown(summary_md_path, worst_df, summary)

    print(f"Saved failure analysis CSV -> {analysis_csv_path}")
    print(f"Saved worst-case CSV       -> {worst_csv_path}")
    print(f"Saved failure summary JSON -> {summary_json_path}")
    print(f"Saved failure summary MD   -> {summary_md_path}")
    print(f"Saved worst-case figures   -> {cases_dir}")


if __name__ == "__main__":
    main()
