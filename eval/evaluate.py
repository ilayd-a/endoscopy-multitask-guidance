import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
from metrics import dice_score, iou_score, pointing_game, peak_to_center_distance

def discover_samples(data_dir: str) -> list[str]:
    pattern = os.path.join(data_dir, "gt_mask_*.npy")
    ids: list[str] = []
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"gt_mask_(\d+)\.npy$", path)
        if match:
            ids.append(match.group(1))
    return ids

def evaluate(data_dir: str, output_csv: str) -> pd.DataFrame:
    sample_ids = discover_samples(data_dir)
    if not sample_ids:
        raise FileNotFoundError(
            f"No gt_mask_*.npy files found in {data_dir}. "
        )

    records: list[dict] = []
    for sid in sample_ids:
        gt_mask = np.load(os.path.join(data_dir, f"gt_mask_{sid}.npy"))
        pred_mask = np.load(os.path.join(data_dir, f"pred_mask_{sid}.npy"))
        pred_heatmap = np.load(os.path.join(data_dir, f"pred_heatmap_{sid}.npy"))

        records.append({
            "sample_id": sid,
            "dice": dice_score(pred_mask, gt_mask),
            "iou": iou_score(pred_mask, gt_mask),
            "pointing_game": pointing_game(pred_heatmap, gt_mask),
            "peak_center_dist": peak_to_center_distance(pred_heatmap, gt_mask),
        })

    df = pd.DataFrame(records)

    summary = {
        "sample_id": "MEAN±STD",
        "dice": f"{df['dice'].mean():.4f} ± {df['dice'].std():.4f}",
        "iou": f"{df['iou'].mean():.4f} ± {df['iou'].std():.4f}",
        "pointing_game": f"{df['pointing_game'].mean():.4f} ± {df['pointing_game'].std():.4f}",
        "peak_center_dist": f"{df['peak_center_dist'].mean():.2f} ± {df['peak_center_dist'].std():.2f}",
    }
    df_with_summary = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    df_with_summary.to_csv(output_csv, index=False)
    print(f"Results written to {output_csv}  ({len(sample_ids)} samples)")
    print()
    print("Summary: ")
    for key in ["dice", "iou", "pointing_game", "peak_center_dist"]:
        mean, std = df[key].mean(), df[key].std()
        print(f"  {key:20s}  {mean:.4f} ± {std:.4f}")
    print()
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation and heatmap predictions")
    parser.add_argument("--datadir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data"),
                        help="Directory with gt/pred .npy files")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv"),
                        help="Path for the output CSV")
    args = parser.parse_args()

    evaluate(args.data_dir, args.output)

if __name__ == "__main__":
    main()
