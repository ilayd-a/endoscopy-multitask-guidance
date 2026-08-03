"""
export_oracle_portfolio.py
==========================
Save the per-frame oracle portfolio for a fixed UNet plus cached SAM candidates.

The oracle table is a feasibility upper bound: it uses ground-truth Dice to
identify the best cached SAM candidate and the better of UNet/SAM for each
frame. It must not be reported as a deployable policy, but it is useful for
quantifying whether a second-opinion pathway has enough headroom to justify
learning a switch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Export UNet/SAM oracle portfolio")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/kvasir_external_120_prompt_quality.csv")
    parser.add_argument("--unet_metrics_csv", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test/baseline_metrics.csv")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/kvasir120_unet_sam_oracle_portfolio.csv")
    parser.add_argument("--summary_csv", default="endoscopy_guidance/results/kvasir120_unet_sam_oracle_summary.csv")
    parser.add_argument("--hard_threshold", type=float, default=0.80)
    args = parser.parse_args()

    qdf = pd.read_csv(args.prompt_quality_csv)
    unet = pd.read_csv(args.unet_metrics_csv)
    unet = unet[["sample_id", "source_file", "split", "dice", "iou"]].rename(
        columns={"sample_id": "unet_sample_id", "dice": "unet_dice", "iou": "unet_iou"}
    )
    rows = qdf.merge(unet, on="source_file", how="inner", suffixes=("", "_unet"))
    if rows.empty:
        raise ValueError("No overlap between prompt-quality CSV and UNet metrics")

    best_sam = rows.sort_values("sam_dice", ascending=False).groupby("source_file", as_index=False).head(1).copy()
    best_sam["sam_gain_vs_unet"] = best_sam["sam_dice"] - best_sam["unet_dice"]
    best_sam["sam_beats_unet"] = best_sam["sam_gain_vs_unet"] > 0
    best_sam["oracle_dice"] = np.maximum(best_sam["unet_dice"], best_sam["sam_dice"])
    best_sam["oracle_iou"] = np.maximum(best_sam["unet_iou"], best_sam["sam_iou"])
    best_sam["oracle_uses_sam"] = best_sam["sam_beats_unet"]
    best_sam["hard_unet_frame"] = best_sam["unet_dice"] < args.hard_threshold

    output_cols = [
        "sample_id",
        "source_file",
        "unet_sample_id",
        "split_unet",
        "unet_dice",
        "unet_iou",
        "sam_dice",
        "sam_iou",
        "sam_gain_vs_unet",
        "sam_beats_unet",
        "oracle_dice",
        "oracle_iou",
        "oracle_uses_sam",
        "hard_unet_frame",
        "candidate_index",
        "radius",
        "heatmap_score",
        "sam_score",
        "center_dist",
        "point_hit",
    ]
    output_cols = [column for column in output_cols if column in best_sam.columns]

    summary_rows = []
    for label, subset in [("all", best_sam), ("hard_unet", best_sam[best_sam["hard_unet_frame"]])]:
        if subset.empty:
            continue
        summary_rows.append({
            "subset": label,
            "frames": int(len(subset)),
            "unet_dice": float(subset["unet_dice"].mean()),
            "best_sam_dice": float(subset["sam_dice"].mean()),
            "oracle_dice": float(subset["oracle_dice"].mean()),
            "oracle_gain_vs_unet": float((subset["oracle_dice"] - subset["unet_dice"]).mean()),
            "sam_beats_unet_rate": float(subset["sam_beats_unet"].mean()),
            "sam_beats_unet_frames": int(subset["sam_beats_unet"].sum()),
        })

    output = Path(args.output_csv)
    summary_output = Path(args.summary_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_sam[output_cols].sort_values("sam_gain_vs_unet", ascending=False).to_csv(output, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_output, index=False)
    print(f"[saved] {output}")
    print(f"[saved] {summary_output}")
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
