"""
Kvasir-SEG model evaluation on the held-out test set.

Loads the trained UNet (pretrained ResNet34 encoder) and evaluates
segmentation quality on test images using four metrics:
  - Dice coefficient
  - IoU (Jaccard index)
  - Pointing Game (does heatmap peak fall inside GT mask?)
  - Peak-to-center distance (pixels between heatmap peak and GT centroid)

Usage:
    cd eval
    python eval_kvasir_model.py
    python eval_kvasir_model.py --split val          # evaluate on val set
    python eval_kvasir_model.py --threshold 0.4      # custom threshold
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS_DIR, get_device

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    ScaleIntensityRange, Resize, ToTensor,
)

DEFAULT_DATA   = PROJECT_ROOT / "dataset" / "data"
DEFAULT_MODEL  = MODELS_DIR / "unet_pretrained.pth"
RESULTS_DIR    = Path(__file__).resolve().parent / "results"
TARGET_SIZE    = (256, 256)


# Metrics

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    intersection = (pred_b & gt_b).sum()
    return float(2.0 * intersection / (pred_b.sum() + gt_b.sum()))


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    intersection = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def pointing_game(heatmap: np.ndarray, gt: np.ndarray) -> int:
    peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(gt[peak[0], peak[1]] > 0)


def peak_to_center_distance(heatmap: np.ndarray, gt: np.ndarray) -> float:
    peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    ys, xs = np.where(gt > 0)
    if len(ys) == 0:
        return float("inf")
    cy, cx = ys.mean(), xs.mean()
    return float(np.sqrt((peak[0] - cy) ** 2 + (peak[1] - cx) ** 2))


# Model loading

def load_model(weight_path: Path, device: torch.device):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"  Loaded model from {weight_path}")
    return model


def get_img_transform():
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensityRange(
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Resize(TARGET_SIZE),
        ToTensor(),
    ])


def get_mask_transform():
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensityRange(
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Resize(TARGET_SIZE),
        ToTensor(),
    ])


# Evaluation

@torch.no_grad()
def evaluate(model, data_dir: Path, split: str, threshold: float, device):
    img_dir = data_dir / split / "images"
    mask_dir = data_dir / split / "masks"

    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found")
        sys.exit(1)

    all_names = sorted([
        f.name for f in img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".png", ".jpeg"}
    ])
    print(f"  Found {len(all_names)} images in {split} set")

    img_transform = get_img_transform()
    mask_transform = get_mask_transform()
    records = []

    for i, name in enumerate(all_names):
        # Load and preprocess image (MONAI pipeline, matches training)
        img_tensor = img_transform(str(img_dir / name)).unsqueeze(0).to(device)

        # Model prediction
        output = model(img_tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

        # Load GT mask (MONAI pipeline, same spatial orientation as image)
        gt_tensor = mask_transform(str(mask_dir / name))
        gt_np = gt_tensor.squeeze().cpu().numpy()
        if gt_np.ndim == 3:
            gt_np = gt_np[0]  # take first channel if RGB mask
        gt_mask = (gt_np > 0.5).astype(np.uint8)

        # Binary prediction mask
        pred_mask = (prob_map >= threshold).astype(np.uint8)

        # Compute metrics
        d = dice_score(pred_mask, gt_mask)
        iou = iou_score(pred_mask, gt_mask)
        pg = pointing_game(prob_map, gt_mask)
        pcd = peak_to_center_distance(prob_map, gt_mask)

        records.append({
            "sample_id": name,
            "dice": round(d, 4),
            "iou": round(iou, 4),
            "pointing_game": pg,
            "peak_center_dist": round(pcd, 2),
        })

        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(all_names)}")

    return pd.DataFrame(records)


# Summary

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["dice", "iou", "pointing_game", "peak_center_dist"]
    rows = []
    for m in metrics:
        rows.append({
            "metric": m,
            "mean": round(df[m].mean(), 4),
            "std": round(df[m].std(), 4),
            "min": round(df[m].min(), 4),
            "max": round(df[m].max(), 4),
        })
    return pd.DataFrame(rows)


# Plots

def plot_results_table(summary: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")
    ax.set_title("UNet (ResNet34) — Test Set Metrics", fontsize=13, pad=12)

    metric_labels = {
        "dice": "Dice",
        "iou": "IoU",
        "pointing_game": "Pointing Game",
        "peak_center_dist": "Peak-to-Center (px)",
    }

    table_data = []
    for _, row in summary.iterrows():
        label = metric_labels.get(row["metric"], row["metric"])
        table_data.append([
            label,
            f"{row['mean']:.4f}",
            f"{row['std']:.4f}",
            f"{row['min']:.4f}",
            f"{row['max']:.4f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Mean", "Std", "Min", "Max"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    for j in range(5):
        table[0, j].set_facecolor("#4e79a7")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(table_data) + 1):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(5):
            table[i, j].set_facecolor(color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Results table    → {save_path}")


def plot_localization(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Dice & IoU distribution
    ax = axes[0]
    ax.hist(df["dice"], bins=25, alpha=0.7, label="Dice", color="#4e79a7")
    ax.hist(df["iou"], bins=25, alpha=0.7, label="IoU", color="#f28e2b")
    ax.axvline(df["dice"].mean(), color="#4e79a7", ls="--", lw=1.5,
               label=f'Dice μ={df["dice"].mean():.3f}')
    ax.axvline(df["iou"].mean(), color="#f28e2b", ls="--", lw=1.5,
               label=f'IoU μ={df["iou"].mean():.3f}')
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Dice & IoU Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Peak-to-center distance
    ax = axes[1]
    ax.hist(df["peak_center_dist"], bins=25, alpha=0.7, color="#e15759")
    ax.axvline(df["peak_center_dist"].mean(), color="black", ls="--", lw=1.5,
               label=f'mean={df["peak_center_dist"].mean():.1f}px')
    ax.set_xlabel("Distance (pixels)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Peak-to-Center Distance", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Dice vs peak-to-center scatter
    ax = axes[2]
    sc = ax.scatter(df["peak_center_dist"], df["dice"],
                    c=df["pointing_game"], cmap="RdYlGn",
                    alpha=0.6, edgecolors="gray", linewidths=0.3, s=30)
    ax.set_xlabel("Peak-to-Center Distance (px)", fontsize=11)
    ax.set_ylabel("Dice Score", fontsize=11)
    ax.set_title("Localization Quality", fontsize=12)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Pointing Game (0/1)")
    ax.grid(alpha=0.3)

    fig.suptitle("UNet (ResNet34) — Test Set Localization Analysis",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Localization plot → {save_path}")


def plot_comparison(baseline_csv: Path, model_df: pd.DataFrame, save_path: Path):
    """Compare model results vs simulated baseline (if baseline CSV exists)."""
    if not baseline_csv.exists():
        return

    baseline = pd.read_csv(baseline_csv)
    metrics = ["dice", "iou"]
    labels = ["Dice", "IoU"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, label in zip(axes, metrics, labels):
        b_vals = baseline[metric].values
        m_vals = model_df[metric].values

        bp = ax.boxplot([b_vals, m_vals], patch_artist=True,
                        labels=["Simulated\nBaseline", "UNet\n(ResNet34)"])
        bp["boxes"][0].set_facecolor("#e15759")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#4e79a7")
        bp["boxes"][1].set_alpha(0.6)

        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label} Comparison", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        b_mean = b_vals.mean()
        m_mean = m_vals.mean()
        ax.text(1, b_mean + 0.02, f"μ={b_mean:.3f}", ha="center", fontsize=9, color="#e15759")
        ax.text(2, m_mean + 0.02, f"μ={m_mean:.3f}", ha="center", fontsize=9, color="#4e79a7")

    fig.suptitle("Model vs Simulated Baseline", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot  → {save_path}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate UNet segmentation model on Kvasir-SEG test set")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binary mask")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        print("Run  PYTHONPATH=. python models/train_pretrained.py  first")
        sys.exit(1)

    device = get_device()
    print(f"\n{'='*60}")
    print("KVASIR-SEG MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"  Device:    {device}")
    print(f"  Model:     {model_path}")
    print(f"  Data:      {data_dir}")
    print(f"  Split:     {args.split}")
    print(f"  Threshold: {args.threshold}")

    print("\nLoading model...")
    model = load_model(model_path, device)

    print(f"\nRunning evaluation on {args.split} set...")
    df = evaluate(model, data_dir, args.split, args.threshold, device)

    # Per-sample CSV
    csv_path = RESULTS_DIR / f"kvasir_model_{args.split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Per-sample CSV   → {csv_path}  ({len(df)} rows)")

    # Summary
    summary = make_summary(df)
    summary_path = RESULTS_DIR / f"kvasir_model_{args.split}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Summary CSV      → {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS ({args.split} set, {len(df)} images)")
    print(f"{'='*60}")
    for _, row in summary.iterrows():
        print(f"  {row['metric']:20s}  {row['mean']:.4f} ± {row['std']:.4f}")
    print(f"\n  Pointing Game accuracy: {df['pointing_game'].mean()*100:.1f}%")

    # Plots
    print(f"\nGenerating plots...")
    plot_results_table(summary, RESULTS_DIR / f"kvasir_model_{args.split}_table.png")
    plot_localization(df, RESULTS_DIR / f"kvasir_model_{args.split}_localization.png")

    # Compare with baseline if available
    baseline_csv = RESULTS_DIR / "kvasir_per_sample.csv"
    plot_comparison(baseline_csv, df, RESULTS_DIR / "kvasir_model_vs_baseline.png")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
