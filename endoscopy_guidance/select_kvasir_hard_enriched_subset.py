"""
select_kvasir_hard_enriched_subset.py
=====================================
Select a reproducible Kvasir subset enriched for hard UNet frames.

The output source_files.txt can be passed to export_kvasir_external.py via
--source_list so prompt-quality caches line up with a fixed UNet baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Select hard-enriched Kvasir source filenames from UNet baseline metrics")
    parser.add_argument("--baseline_csv", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test/baseline_metrics.csv")
    parser.add_argument("--output_dir", default="endoscopy_guidance/exports/kvasir_hard_enriched_300")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--n_medium", type=int, default=120, help="Number of sub-0.90 Dice frames to add after hard frames.")
    parser.add_argument("--n_uncertain", type=int, default=120, help="Number of high-uncertainty frames to add.")
    parser.add_argument("--n_random", type=int, default=120, help="Number of random frames to add for coverage.")
    parser.add_argument("--hard_threshold", type=float, default=0.80)
    parser.add_argument("--medium_threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_csv)
    baseline = baseline.sort_values(["dice", "mean_uncertainty", "boundary_uncertainty"], ascending=[True, False, False])
    hard = baseline[baseline["dice"] < args.hard_threshold]
    medium = baseline[(baseline["dice"] >= args.hard_threshold) & (baseline["dice"] < args.medium_threshold)].head(args.n_medium)
    uncertain = baseline.sort_values(["mean_uncertainty", "boundary_uncertainty"], ascending=False).head(args.n_uncertain)
    random_pool = baseline.sample(n=min(args.n_random, len(baseline)), random_state=args.seed)
    selected = pd.concat([hard, medium, uncertain, random_pool], ignore_index=True)
    selected = selected.drop_duplicates("source_file")
    selected = selected.sort_values(["dice", "mean_uncertainty"], ascending=[True, False]).head(args.n_samples)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_list = out / "source_files.txt"
    source_list.write_text("\n".join(selected["source_file"].astype(str)) + "\n")
    selected[[
        "sample_id",
        "source_file",
        "split",
        "dice",
        "iou",
        "mean_uncertainty",
        "boundary_uncertainty",
        "mask_area_frac",
        "gt_area_frac",
    ]].to_csv(out / "selected_from_unet_baseline.csv", index=False)

    print(f"[saved] {source_list} sources={len(selected)}")
    print(f"hard<{args.hard_threshold}: {int((selected['dice'] < args.hard_threshold).sum())}")
    print(f"medium<{args.medium_threshold}: {int((selected['dice'] < args.medium_threshold).sum())}")


if __name__ == "__main__":
    main()
