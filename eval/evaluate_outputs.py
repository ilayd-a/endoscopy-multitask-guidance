from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import PROJECT_ROOT, load_binary_mask, load_samples
from metrics import (
    boundary_f1_score,
    component_count,
    confusion_counts,
    dice_score,
    heatmap_peak_value,
    iou_score,
    mask_energy_ratio,
    mask_area_ratio,
    peak_to_center_distance,
    pointing_game,
    precision_score,
    recall_score,
)
from predictions import (
    discover_outputs_root,
    discover_prediction_dirs,
    load_prediction_heatmap,
    load_prediction_mask,
    match_predictions_to_samples,
)


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "boundary_f1",
    "pointing_game",
    "peak_center_dist",
    "coherence_mer",
)


def evaluate_matches(
    matches: list[tuple],
    dataset: str,
    split: str,
    run_name: str,
) -> pd.DataFrame:
    records: list[dict] = []

    for sample, prediction in matches:
        heatmap = load_prediction_heatmap(prediction.pred_heatmap_path)
        pred_mask = load_prediction_mask(prediction.pred_mask_path, target_shape=heatmap.shape)
        gt_mask = load_binary_mask(sample.mask_path, target_shape=heatmap.shape)
        tp, fp, fn, tn = confusion_counts(pred_mask, gt_mask)
        gt_pixels = int(gt_mask.sum())
        pred_pixels = int(pred_mask.sum())
        total_pixels = int(gt_mask.size)
        intersection_pixels = tp
        union_pixels = int(((pred_mask > 0) | (gt_mask > 0)).sum())
        gt_area = mask_area_ratio(gt_mask)
        pred_area = mask_area_ratio(pred_mask)
        gt_bool = gt_mask > 0

        records.append(
            {
                "dataset": dataset,
                "split": split,
                "run_name": run_name,
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
                "intersection_pixels": intersection_pixels,
                "union_pixels": union_pixels,
                "gt_area_ratio": round(gt_area, 6),
                "pred_area_ratio": round(pred_area, 6),
                "abs_area_error": round(abs(pred_area - gt_area), 6),
                "gt_components": component_count(gt_mask),
                "pred_components": component_count(pred_mask),
                "dice": round(dice_score(pred_mask, gt_mask), 4),
                "iou": round(iou_score(pred_mask, gt_mask), 4),
                "precision": round(precision_score(pred_mask, gt_mask), 4),
                "recall": round(recall_score(pred_mask, gt_mask), 4),
                "boundary_f1": round(boundary_f1_score(pred_mask, gt_mask), 4),
                "pointing_game": pointing_game(heatmap, gt_mask),
                "peak_center_dist": round(peak_to_center_distance(heatmap, gt_mask), 2),
                "coherence_mer": round(mask_energy_ratio(heatmap, gt_mask), 4),
                "heatmap_peak_value": round(heatmap_peak_value(heatmap), 4),
                "heatmap_mean": round(float(heatmap.mean()), 4),
                "heatmap_inside_mean": round(float(heatmap[gt_bool].mean()) if gt_bool.any() else 0.0, 4),
                "heatmap_outside_mean": round(float(heatmap[~gt_bool].mean()) if (~gt_bool).any() else 0.0, 4),
            }
        )

    return pd.DataFrame(records)


def summarize_results(df: pd.DataFrame, dataset: str, split: str, run_name: str) -> dict:
    summary = {
        "dataset": dataset,
        "split": split,
        "run_name": run_name,
        "num_samples": int(len(df)),
        "metrics": {},
    }
    for metric in SUMMARY_METRICS:
        series = pd.to_numeric(df[metric], errors="coerce")
        series = series[series.notna() & np.isfinite(series)]
        if series.empty:
            summary["metrics"][metric] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
            continue
        summary["metrics"][metric] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std(ddof=0)), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
        }
    return summary


def build_summary_table(summary: dict) -> pd.DataFrame:
    rows = []
    for metric, stats in summary["metrics"].items():
        rows.append({"metric": metric, **stats})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved segmentation/heatmap outputs against GT masks."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cvc",
        choices=["cvc", "kvasir", "split_folder"],
        help="Dataset loader to use. Active default is CVC-ClinicDB.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Official split file from the data team (for example splits/val.txt or splits/val.csv).",
    )
    parser.add_argument(
        "--split-column",
        type=str,
        default=None,
        help="Metadata column containing the official split labels.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Directory containing sample folders with pred_mask/pred_heatmap files.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="active_run",
        help="Short tag used for saved evaluation files.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow-index-fallback",
        action="store_true",
        help="Use sample ordering to match legacy numbered output folders when exact ids are unavailable.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
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

    df = evaluate_matches(matches, args.dataset, args.split, args.run_name)
    summary = summarize_results(df, args.dataset, args.split, args.run_name)
    summary_table = build_summary_table(summary)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{args.dataset}_{args.split}_{args.run_name}"
    per_sample_path = args.results_dir / f"{output_prefix}_per_sample.csv"
    summary_csv_path = args.results_dir / f"{output_prefix}_summary.csv"
    summary_json_path = args.results_dir / f"{output_prefix}_summary.json"

    df.to_csv(per_sample_path, index=False)
    summary_table.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved per-sample metrics -> {per_sample_path}")
    print(f"Saved summary CSV        -> {summary_csv_path}")
    print(f"Saved summary JSON       -> {summary_json_path}")
    print()
    for metric, stats in summary["metrics"].items():
        if stats["mean"] is None:
            print(f"{metric:20s} unavailable")
            continue
        print(
            f"{metric:20s} "
            f"{stats['mean']:.4f} +/- {stats['std']:.4f} "
            f"(min {stats['min']:.4f}, max {stats['max']:.4f})"
        )


if __name__ == "__main__":
    main()
