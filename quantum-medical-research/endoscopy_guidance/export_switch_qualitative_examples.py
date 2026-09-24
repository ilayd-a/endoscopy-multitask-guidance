"""
export_switch_qualitative_examples.py
=====================================
Export visual examples where a learned SAM switch improves over UNet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sam_prompt_quality_ranker import sam_prompt_masks_all
from sam_quantum_prompt_benchmark import choose_device, load_sam_predictor, set_cached_or_compute_image


def dim_image(image: np.ndarray) -> np.ndarray:
    """Slightly dim the image so colored masks/contours remain unambiguous."""
    return np.clip(image.astype(np.float32) * 0.72, 0, 255).astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = dim_image(image).astype(np.float32)
    tint = np.asarray(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * tint
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_contours(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 3) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def prediction_panel(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    pred_color: tuple[int, int, int],
) -> np.ndarray:
    """Show prediction and ground truth as distinct overlays.

    Prediction is filled and contoured with pred_color; ground truth is always
    only a green contour. This avoids making the ground truth look like the
    model prediction in failure cases.
    """
    out = overlay(image, pred, pred_color, alpha=0.62)
    out = draw_contours(out, pred, pred_color, thickness=3)
    out = draw_contours(out, gt, (0, 255, 0), thickness=3)
    return out


def baseline_sample_map(baseline_dir: Path) -> dict[str, str]:
    metrics = pd.read_csv(baseline_dir / "baseline_metrics.csv")
    return {str(row.source_file): str(row.sample_id) for row in metrics.itertuples(index=False)}


def main():
    parser = argparse.ArgumentParser(description="Export qualitative UNet-vs-SAM switch rescue examples")
    parser.add_argument("--per_sample_csv", default="endoscopy_guidance/results/residual_gain_kvasir_hard_enriched_300_seed0_per_sample.csv")
    parser.add_argument("--policy", default="classical_histgb:absolute_sam_dice:meta_rf_gain_switch")
    parser.add_argument("--data_dir", default="endoscopy_guidance/exports/kvasir_hard_enriched_300")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test")
    parser.add_argument("--checkpoint", default="models/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--embedding_cache_dir", default="endoscopy_guidance/results/sam_embedding_cache_kvasir_hard_enriched_300")
    parser.add_argument("--output_dir", default="docs/figures/kvasir_hard_enriched_switch_examples")
    parser.add_argument("--max_examples", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    per = pd.read_csv(args.per_sample_csv)
    if "policy" not in per.columns:
        raise ValueError("Per-sample CSV must include policy rows. Rerun residual_gain_prompt_fusion.py first.")
    rows = per[
        per["policy"].eq(args.policy)
        & per["use_sam"].astype(bool)
        & per["hard_unet"].astype(bool)
        & per["delta_vs_unet"].gt(0)
    ].sort_values("delta_vs_unet", ascending=False).head(args.max_examples)
    if rows.empty:
        raise ValueError(f"No positive hard-frame switch examples found for policy={args.policy}")

    data_dir = Path(args.data_dir)
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_to_baseline = baseline_sample_map(baseline_dir)
    predictor = load_sam_predictor(Path(args.checkpoint), args.model_type, choose_device(args.device))
    embedding_cache = Path(args.embedding_cache_dir) if args.embedding_cache_dir else None

    manifest = []
    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        image = np.load(data_dir / f"image_{row.sample_id}.npy")
        gt = np.load(data_dir / f"gt_mask_{row.sample_id}.npy")
        baseline_id = source_to_baseline[str(row.source_file)]
        unet = np.load(baseline_dir / "pred_masks" / f"{baseline_id}.npy")
        set_cached_or_compute_image(predictor, image, str(row.sample_id), embedding_cache)
        masks, _ = sam_prompt_masks_all(predictor, [(int(row.y), int(row.x))], image.shape[:2], int(row.radius))
        sam = masks[0].astype(np.uint8)

        gt_overlay = draw_contours(overlay(image, gt, (40, 210, 90), alpha=0.56), gt, (0, 255, 0))
        unet_overlay = prediction_panel(image, gt, unet, (255, 45, 45))
        sam_overlay = prediction_panel(image, gt, sam, (30, 160, 255))

        fig, axes = plt.subplots(1, 4, figsize=(12, 3.35), dpi=180)
        panels = [
            (image, "Image"),
            (gt_overlay, "Ground truth (green)"),
            (unet_overlay, f"UNet pred. red | GT green\nDice {row.unet_dice:.3f}"),
            (sam_overlay, f"Switched SAM pred. blue | GT green\nDice {row.sam_dice:.3f}"),
        ]
        for ax, (panel, title) in zip(axes, panels):
            ax.imshow(panel)
            ax.set_title(title, fontsize=8.5)
            ax.axis("off")
        fig.suptitle(
            f"{row.source_file} | Dice gain +{row.delta_vs_unet:.3f} | green = ground truth, red/blue = prediction",
            fontsize=9.5,
        )
        fig.tight_layout(rect=[0, 0.02, 1, 0.92])
        out_path = output_dir / f"switch_rescue_{idx:02d}_{Path(str(row.source_file)).stem}.png"
        fig.savefig(out_path)
        plt.close(fig)
        manifest.append({
            "figure": str(out_path),
            "sample_id": row.sample_id,
            "source_file": row.source_file,
            "unet_dice": float(row.unet_dice),
            "sam_dice": float(row.sam_dice),
            "delta_vs_unet": float(row.delta_vs_unet),
            "radius": int(row.radius),
            "policy": args.policy,
        })
        print(f"[saved] {out_path}")

    manifest_path = output_dir / "manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"[saved] {manifest_path}")


if __name__ == "__main__":
    main()
