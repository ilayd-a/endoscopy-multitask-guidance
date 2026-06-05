from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import PROJECT_ROOT, load_binary_mask, load_rgb_image, load_samples
from predictions import (
    discover_outputs_root,
    discover_prediction_dirs,
    load_prediction_heatmap,
    load_prediction_mask,
    match_predictions_to_samples,
)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
HIGHER_IS_BETTER = {
    "dice",
    "iou",
    "precision",
    "recall",
    "f2",
    "boundary_f1",
    "pointing_game",
    "coherence_mer",
}
LOWER_IS_BETTER = {"mae", "peak_center_dist", "abs_area_error"}
MODE_ALIASES = {
    "strong": "best",
    "weak": "worst",
}


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    gray_rgb = np.stack([gray, gray, gray], axis=-1)

    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    return np.clip(0.45 * gray_rgb + 0.55 * heat_rgb, 0.0, 1.0)


def select_samples(
    eval_df: pd.DataFrame,
    metric: str,
    mode: str,
    samples: int,
) -> pd.DataFrame:
    mode = normalize_mode(mode)
    if mode == "random":
        return eval_df.sample(n=min(samples, len(eval_df)), random_state=42)

    if metric in HIGHER_IS_BETTER:
        best_ranked = eval_df.sort_values(metric, ascending=False).reset_index(drop=True)
        worst_ranked = eval_df.sort_values(metric, ascending=True).reset_index(drop=True)
    elif metric in LOWER_IS_BETTER:
        best_ranked = eval_df.sort_values(metric, ascending=True).reset_index(drop=True)
        worst_ranked = eval_df.sort_values(metric, ascending=False).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported metric for selection: {metric}")

    if mode == "best":
        return best_ranked.head(samples)
    if mode == "worst":
        return worst_ranked.head(samples)
    if mode == "median":
        center = len(best_ranked) // 2
        half = samples // 2
        start = max(0, center - half)
        end = min(len(best_ranked), start + samples)
        return best_ranked.iloc[start:end]
    raise ValueError(f"Unsupported selection mode: {mode}")


def normalize_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def build_row_label(sample_id: str, metrics_row: pd.Series | None) -> str:
    if metrics_row is None:
        return sample_id

    parts = [sample_id]
    metric_specs = (
        ("dice", "D"),
        ("iou", "I"),
        ("boundary_f1", "BF1"),
        ("pointing_game", "PG"),
    )
    for column, short_name in metric_specs:
        if column not in metrics_row or pd.isna(metrics_row[column]):
            continue
        value = metrics_row[column]
        if column == "pointing_game":
            parts.append(f"{short_name} {int(value)}")
        else:
            parts.append(f"{short_name} {float(value):.3f}")
    return "\n".join([parts[0], " | ".join(parts[1:])]) if len(parts) > 1 else parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate qualitative panels from saved model outputs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cvc",
        choices=["cvc", "kvasir", "split_csv", "split_folder"],
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
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", type=str, default="active_run")
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help="Per-sample CSV from evaluate_outputs.py. Required for best/worst/median selection.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="worst",
        choices=["best", "strong", "median", "worst", "weak", "random"],
    )
    parser.add_argument("--metric", type=str, default="dice")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--allow-index-fallback", action="store_true")
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the selected strong/weak cases.",
    )
    args = parser.parse_args()
    normalized_mode = normalize_mode(args.mode)

    samples = load_samples(
        dataset=args.dataset,
        split=args.split,
        dataset_root=args.dataset_root,
        metadata_path=args.metadata_path,
        split_manifest=args.split_manifest,
        split_column=args.split_column,
    )
    sample_map = {sample.sample_id: sample for sample in samples}

    outputs_root = discover_outputs_root(PROJECT_ROOT, args.outputs_dir)
    predictions = discover_prediction_dirs(outputs_root)
    matches = match_predictions_to_samples(
        samples=samples,
        predictions=predictions,
        allow_index_fallback=args.allow_index_fallback,
    )
    prediction_map = {sample.sample_id: prediction for sample, prediction in matches}

    if args.eval_csv is None and normalized_mode != "random":
        default_csv = (
            args.results_dir
            / f"{args.dataset}_{args.split}_{args.run_name}_per_sample.csv"
        )
        args.eval_csv = default_csv

    if normalized_mode == "random":
        random_df = pd.DataFrame(
            {"sample_id": [sample.sample_id for sample in samples]}
        )
        selected_df = random_df.sample(
            n=min(args.samples, len(random_df)),
            random_state=42,
        )
    else:
        if args.eval_csv is None or not args.eval_csv.exists():
            raise FileNotFoundError(
                "Selection mode requires an evaluation CSV. "
                "Run evaluate_outputs.py first or pass --eval-csv."
            )
        eval_df = pd.read_csv(args.eval_csv)
        selected_df = select_samples(
            eval_df, args.metric, normalized_mode, args.samples
        ).copy()
    selected_ids = selected_df["sample_id"].tolist()
    if not selected_ids:
        raise ValueError("No samples selected for panel generation.")
    selected_metric_map = (
        selected_df.set_index("sample_id").to_dict(orient="index")
        if len(selected_df.columns) > 1
        else {}
    )

    panel_rows = []
    for sample_id in selected_ids:
        sample = sample_map[sample_id]
        prediction = prediction_map[sample_id]
        heatmap = load_prediction_heatmap(prediction.pred_heatmap_path)
        pred_mask = load_prediction_mask(
            prediction.pred_mask_path, target_shape=heatmap.shape
        )
        image_rgb = load_rgb_image(sample.image_path, target_shape=heatmap.shape)
        gt_mask = load_binary_mask(sample.mask_path, target_shape=heatmap.shape)
        panel_rows.append(
            (
                sample_id,
                image_rgb,
                gt_mask,
                pred_mask,
                heatmap,
                selected_metric_map.get(sample_id),
            )
        )

    fig, axes = plt.subplots(len(panel_rows), 4, figsize=(12, 3 * len(panel_rows)))
    axes = np.atleast_2d(axes)
    titles = ["Input", "GT Mask", "Pred Mask", "Guidance Heatmap"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=10)

    for row_index, (sample_id, image_rgb, gt_mask, pred_mask, heatmap, metrics_row) in enumerate(
        panel_rows
    ):
        axes[row_index, 0].imshow(image_rgb)
        axes[row_index, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 3].imshow(overlay_heatmap(image_rgb, heatmap))
        axes[row_index, 0].set_ylabel(
            build_row_label(sample_id, pd.Series(metrics_row) if metrics_row else None),
            fontsize=8,
            rotation=90,
            labelpad=10,
        )

        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("black")

    fig.tight_layout()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{args.dataset}_{args.split}_{args.run_name}_{args.mode}_{args.metric}_panel.png"
    output_path = args.results_dir / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    selection_csv = args.selection_csv or args.results_dir / (
        f"{args.dataset}_{args.split}_{args.run_name}_{args.mode}_{args.metric}_selected.csv"
    )
    selected_df.to_csv(selection_csv, index=False)
    print(f"Saved qualitative panel -> {output_path}")
    print(f"Saved selected cases    -> {selection_csv}")


if __name__ == "__main__":
    main()
