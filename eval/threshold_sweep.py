from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import PROJECT_ROOT, load_binary_mask, load_samples
from metrics import (
    boundary_f1_score,
    component_count,
    confusion_counts,
    dice_score,
    f_beta_score,
    heatmap_peak_value,
    iou_score,
    mae_score,
    mask_area_ratio,
    mask_energy_ratio,
    peak_to_center_distance,
    pointing_game,
    precision_score,
    recall_score,
)
from predictions import (
    discover_outputs_root,
    discover_prediction_dirs,
    load_prediction_heatmap,
    match_predictions_to_samples,
)


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "f2",
    "mae",
    "boundary_f1",
    "pointing_game",
    "peak_center_dist",
    "coherence_mer",
)


def image_confidence_score(
    heatmap: np.ndarray,
    method: str,
    topk_percent: float,
) -> float:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.size == 0:
        return 0.0
    if method == "mean":
        return float(heatmap.mean())
    if method == "max":
        return float(heatmap.max())
    if method == "topk_mean":
        fraction = max(topk_percent, 0.0) / 100.0
        count = max(1, int(np.ceil(heatmap.size * fraction)))
        count = min(count, heatmap.size)
        top_values = np.partition(heatmap.reshape(-1), -count)[-count:]
        return float(top_values.mean())
    raise ValueError(f"Unsupported image score method: {method}")


def build_threshold_mask(
    heatmap: np.ndarray,
    pixel_threshold: float,
    image_threshold: float | None,
    image_score_method: str,
    topk_percent: float,
) -> tuple[np.ndarray, float, bool]:
    score = image_confidence_score(heatmap, image_score_method, topk_percent)
    gate_positive = True if image_threshold is None else score >= image_threshold
    if not gate_positive:
        return np.zeros_like(heatmap, dtype=np.uint8), score, False
    return (heatmap >= pixel_threshold).astype(np.uint8), score, True


def evaluate_threshold_setting(
    matches: list[tuple],
    dataset: str,
    split: str,
    run_name: str,
    pixel_threshold: float,
    image_threshold: float | None,
    image_score_method: str,
    topk_percent: float,
) -> pd.DataFrame:
    records: list[dict] = []
    approach = "baseline" if image_threshold is None else "two_stage"
    if image_threshold is None:
        setting = f"baseline_p{pixel_threshold:.2f}"
    else:
        setting = (
            f"two_stage_p{pixel_threshold:.2f}_"
            f"{image_score_method}_{image_threshold:.3f}"
        )

    for sample, prediction in matches:
        heatmap = load_prediction_heatmap(prediction.pred_heatmap_path)
        pred_mask, image_score, gate_positive = build_threshold_mask(
            heatmap=heatmap,
            pixel_threshold=pixel_threshold,
            image_threshold=image_threshold,
            image_score_method=image_score_method,
            topk_percent=topk_percent,
        )
        gt_mask = load_binary_mask(sample.mask_path, target_shape=heatmap.shape)
        tp, fp, fn, tn = confusion_counts(pred_mask, gt_mask)
        gt_pixels = int(gt_mask.sum())
        pred_pixels = int(pred_mask.sum())
        gt_area = mask_area_ratio(gt_mask)
        pred_area = mask_area_ratio(pred_mask)
        sample_has_target = gt_pixels > 0
        sample_pred_positive = pred_pixels > 0
        gt_bool = gt_mask > 0

        records.append(
            {
                "dataset": dataset,
                "split": split,
                "run_name": run_name,
                "approach": approach,
                "setting": setting,
                "pixel_threshold": pixel_threshold,
                "image_score_method": image_score_method,
                "image_threshold": image_threshold,
                "image_score": round(image_score, 6),
                "gate_positive": gate_positive,
                "sample_id": sample.sample_id,
                "prediction_dir": prediction.sample_dir.name,
                "sample_height": int(heatmap.shape[0]),
                "sample_width": int(heatmap.shape[1]),
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
                "tn_pixels": tn,
                "gt_area_ratio": round(gt_area, 6),
                "pred_area_ratio": round(pred_area, 6),
                "abs_area_error": round(abs(pred_area - gt_area), 6),
                "gt_components": component_count(gt_mask),
                "pred_components": component_count(pred_mask),
                "sample_has_target": sample_has_target,
                "sample_pred_positive": sample_pred_positive,
                "sample_missed_target": sample_has_target and not sample_pred_positive,
                "sample_false_positive": (not sample_has_target) and sample_pred_positive,
                "dice": round(dice_score(pred_mask, gt_mask), 4),
                "iou": round(iou_score(pred_mask, gt_mask), 4),
                "precision": round(precision_score(pred_mask, gt_mask), 4),
                "recall": round(recall_score(pred_mask, gt_mask), 4),
                "f2": round(f_beta_score(pred_mask, gt_mask, beta=2.0), 4),
                "mae": round(mae_score(pred_mask, gt_mask), 4),
                "boundary_f1": round(boundary_f1_score(pred_mask, gt_mask), 4),
                "pointing_game": pointing_game(heatmap, gt_mask),
                "peak_center_dist": round(peak_to_center_distance(heatmap, gt_mask), 2),
                "coherence_mer": round(mask_energy_ratio(heatmap, gt_mask), 4),
                "heatmap_peak_value": round(heatmap_peak_value(heatmap), 4),
                "heatmap_mean": round(float(heatmap.mean()), 4),
                "heatmap_inside_mean": round(
                    float(heatmap[gt_bool].mean()) if gt_bool.any() else 0.0,
                    4,
                ),
                "heatmap_outside_mean": round(
                    float(heatmap[~gt_bool].mean()) if (~gt_bool).any() else 0.0,
                    4,
                ),
            }
        )

    return pd.DataFrame(records)


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def summarize_thresholds(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_columns = [
        "approach",
        "pixel_threshold",
        "image_score_method",
        "image_threshold",
    ]

    for keys, group in per_sample.groupby(group_columns, dropna=False):
        approach, pixel_threshold, image_score_method, image_threshold = keys
        has_target = group["sample_has_target"].astype(bool)
        pred_positive = group["sample_pred_positive"].astype(bool)

        image_tp = int((has_target & pred_positive).sum())
        image_fp = int((~has_target & pred_positive).sum())
        image_fn = int((has_target & ~pred_positive).sum())
        image_tn = int((~has_target & ~pred_positive).sum())

        row = {
            "approach": approach,
            "pixel_threshold": pixel_threshold,
            "image_score_method": image_score_method,
            "image_threshold": None if pd.isna(image_threshold) else image_threshold,
            "num_samples": int(len(group)),
            "pred_positive_count": int(pred_positive.sum()),
            "missed_target_count": image_fn,
            "false_positive_sample_count": image_fp,
            "image_sensitivity": safe_ratio(image_tp, image_tp + image_fn),
            "image_precision": safe_ratio(image_tp, image_tp + image_fp),
            "image_specificity": safe_ratio(image_tn, image_tn + image_fp),
            "fp_pixels_mean": round(float(group["fp_pixels"].mean()), 2),
            "fn_pixels_mean": round(float(group["fn_pixels"].mean()), 2),
        }
        for metric in SUMMARY_METRICS:
            series = pd.to_numeric(group[metric], errors="coerce")
            series = series[series.notna() & np.isfinite(series)]
            row[f"{metric}_mean"] = round(float(series.mean()), 4) if len(series) else None
            row[f"{metric}_std"] = round(float(series.std(ddof=0)), 4) if len(series) else None
            row[f"{metric}_min"] = round(float(series.min()), 4) if len(series) else None
            row[f"{metric}_max"] = round(float(series.max()), 4) if len(series) else None
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["approach", "pixel_threshold", "image_score_method", "image_threshold"],
        na_position="first",
    )


def json_ready_records(frame: pd.DataFrame) -> list[dict]:
    records = frame.replace({np.nan: None}).to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, np.generic):
                record[key] = value.item()
    return records


def setting_label(row: pd.Series) -> str:
    if row["approach"] == "baseline":
        return f"baseline p={row['pixel_threshold']:.2f}"
    return (
        f"two-stage p={row['pixel_threshold']:.2f}, "
        f"{row['image_score_method']}>={row['image_threshold']:.3f}"
    )


def plot_threshold_curves(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = ("dice", "iou", "precision", "recall", "f2", "mae")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.reshape(-1)

    for axis, metric in zip(axes, metrics):
        mean_column = f"{metric}_mean"
        for label_keys, group in summary.groupby(
            ["approach", "image_score_method", "image_threshold"],
            dropna=False,
        ):
            approach, image_score_method, image_threshold = label_keys
            if approach == "baseline":
                label = "baseline"
            else:
                label = f"{image_score_method}>={image_threshold:.3f}"
            group = group.sort_values("pixel_threshold")
            axis.plot(
                group["pixel_threshold"],
                group[mean_column],
                marker="o",
                linewidth=1.8,
                label=label,
            )
        axis.set_title(metric, fontweight="bold")
        axis.set_xlabel("Pixel threshold")
        axis.grid(alpha=0.25)
        if metric == "mae":
            axis.set_ylabel("Lower is better")
        else:
            axis.set_ylabel("Higher is better")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle("Threshold Sweep Summary", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_precision_recall(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 5.5))
    for _, row in summary.iterrows():
        axis.scatter(row["recall_mean"], row["precision_mean"], s=55)
        axis.annotate(
            setting_label(row),
            (row["recall_mean"], row["precision_mean"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.set_xlabel("Recall / sensitivity")
    axis.set_ylabel("Precision")
    axis.set_title("Precision vs Recall Trade-off", fontweight="bold")
    axis.grid(alpha=0.25)
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pixel-threshold and two-stage image+pixel decision strategies."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cvc",
        choices=["cvc", "kvasir", "split_csv", "split_folder"],
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--split-column", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="threshold_sweep")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-index-fallback", action="store_true")
    parser.add_argument(
        "--pixel-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5],
        help="Pixel-level probability thresholds to evaluate.",
    )
    parser.add_argument(
        "--image-thresholds",
        type=float,
        nargs="*",
        default=[],
        help="Image-level confidence thresholds for the two-stage pipeline.",
    )
    parser.add_argument(
        "--image-score-methods",
        type=str,
        nargs="+",
        choices=["mean", "max", "topk_mean"],
        default=["mean"],
    )
    parser.add_argument(
        "--topk-percent",
        type=float,
        default=5.0,
        help="Percent of pixels used by --image-score-methods topk_mean.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Only evaluate two-stage settings.",
    )
    args = parser.parse_args()

    samples = load_samples(
        dataset=args.dataset,
        split=args.split,
        dataset_root=args.dataset_root,
        metadata_path=args.metadata_path,
        split_manifest=args.split_manifest,
        split_column=args.split_column,
    )
    if args.limit is not None:
        samples = samples[: args.limit]

    outputs_root = discover_outputs_root(PROJECT_ROOT, args.outputs_dir)
    predictions = discover_prediction_dirs(outputs_root)
    matches = match_predictions_to_samples(
        samples=samples,
        predictions=predictions,
        allow_index_fallback=args.allow_index_fallback,
    )

    frames: list[pd.DataFrame] = []
    if not args.skip_baseline:
        for pixel_threshold in args.pixel_thresholds:
            frames.append(
                evaluate_threshold_setting(
                    matches=matches,
                    dataset=args.dataset,
                    split=args.split,
                    run_name=args.run_name,
                    pixel_threshold=pixel_threshold,
                    image_threshold=None,
                    image_score_method=args.image_score_methods[0],
                    topk_percent=args.topk_percent,
                )
            )

    for pixel_threshold in args.pixel_thresholds:
        for image_score_method in args.image_score_methods:
            for image_threshold in args.image_thresholds:
                frames.append(
                    evaluate_threshold_setting(
                        matches=matches,
                        dataset=args.dataset,
                        split=args.split,
                        run_name=args.run_name,
                        pixel_threshold=pixel_threshold,
                        image_threshold=image_threshold,
                        image_score_method=image_score_method,
                        topk_percent=args.topk_percent,
                    )
                )

    if not frames:
        raise ValueError("No threshold settings were requested.")

    per_sample = pd.concat(frames, ignore_index=True)
    summary = summarize_thresholds(per_sample)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.results_dir / f"{args.dataset}_{args.split}_{args.run_name}"
    per_sample_path = output_prefix.with_name(f"{output_prefix.name}_threshold_per_sample.csv")
    summary_csv_path = output_prefix.with_name(f"{output_prefix.name}_threshold_summary.csv")
    summary_json_path = output_prefix.with_name(f"{output_prefix.name}_threshold_summary.json")
    curves_path = output_prefix.with_name(f"{output_prefix.name}_threshold_curves.png")
    pr_path = output_prefix.with_name(f"{output_prefix.name}_precision_recall_tradeoff.png")

    per_sample.to_csv(per_sample_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(
        json.dumps(json_ready_records(summary), indent=2),
        encoding="utf-8",
    )
    plot_threshold_curves(summary, curves_path)
    plot_precision_recall(summary, pr_path)

    print(f"Saved threshold per-sample CSV -> {per_sample_path}")
    print(f"Saved threshold summary CSV    -> {summary_csv_path}")
    print(f"Saved threshold summary JSON   -> {summary_json_path}")
    print(f"Saved threshold curves         -> {curves_path}")
    print(f"Saved precision/recall plot    -> {pr_path}")


if __name__ == "__main__":
    main()
