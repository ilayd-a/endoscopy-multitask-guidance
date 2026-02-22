"""
Kvasir-SEG baseline evaluation with simulated GradCAM predictions
against real ground truth masks. Computes Dice, IoU, Pointing Game,
and peak-to-center distance.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, shift

DEFAULT_DATA = PROJECT_ROOT / "dataset" / "Kvasir-SEG"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"
TARGET_SIZE  = (256, 256)


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
    return float(intersection / union)


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


# Load Kvasir-SEG data
def load_kvasir(kvasir_dir: Path, n_samples: int, seed: int):
    img_dir = kvasir_dir / "images"
    mask_dir = kvasir_dir / "masks"

    all_names = sorted([
        f.name for f in img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".png", ".jpeg", ".bmp"}
    ])

    rng = np.random.default_rng(seed)
    if n_samples > 0 and n_samples < len(all_names):
        chosen = rng.choice(all_names, n_samples, replace=False)
    else:
        chosen = all_names

    images, masks, names = [], [], []
    for name in sorted(chosen):
        img = cv2.imread(str(img_dir / name))
        mask = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img = cv2.resize(img, TARGET_SIZE)
        mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)
        images.append(img)
        masks.append(mask)
        names.append(name)

    print(f"  Loaded {len(names)} image-mask pairs from {kvasir_dir}")
    return images, masks, names


# Simulate GradCAM-like heatmap from GT mask
def simulate_gradcam(gt_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = gt_mask.shape
    heatmap = gt_mask.astype(np.float64)

    # Heavy blur (GradCAM is typically 7x7 or 14x14 resolution)
    sigma = rng.uniform(15, 30)
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Random shift (model doesn't perfectly locate)
    dy = rng.integers(-int(h * 0.1), int(h * 0.1) + 1)
    dx = rng.integers(-int(w * 0.1), int(w * 0.1) + 1)
    heatmap = shift(heatmap, [dy, dx], mode="constant", cval=0)

    # Add noise
    noise = rng.normal(0, 0.08, heatmap.shape)
    heatmap = heatmap + noise

    # Normalize to [0, 1]
    heatmap = np.clip(heatmap, 0, None)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


# Evaluation
def evaluate_kvasir(kvasir_dir: Path, n_samples: int, seed: int, threshold: float):
    images, masks, names = load_kvasir(kvasir_dir, n_samples, seed)
    rng = np.random.default_rng(seed)

    records = []
    for i, (img, gt_mask, name) in enumerate(zip(images, masks, names)):
        heatmap = simulate_gradcam(gt_mask, rng)
        pred_mask = (heatmap >= threshold).astype(np.uint8)

        d = dice_score(pred_mask, gt_mask)
        iou = iou_score(pred_mask, gt_mask)
        pg = pointing_game(heatmap, gt_mask)
        pcd = peak_to_center_distance(heatmap, gt_mask)

        records.append({
            "sample_id": name,
            "dice": round(d, 4),
            "iou": round(iou, 4),
            "pointing_game": pg,
            "peak_center_dist": round(pcd, 2),
        })

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
def plot_baseline_table(summary: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")
    ax.set_title("Kvasir-SEG Baseline Metrics", fontsize=13, pad=12)

    metric_labels = {
        "dice": "Dice",
        "iou": "IoU",
        "pointing_game": "Pointing Game",
        "peak_center_dist": "Peak-to-Center",
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
    print(f"  Baseline table   → {save_path}")


def plot_localization(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Dice & IoU distribution
    ax = axes[0]
    ax.hist(df["dice"], bins=25, alpha=0.7, label="Dice", color="#4e79a7")
    ax.hist(df["iou"], bins=25, alpha=0.7, label="IoU", color="#f28e2b")
    ax.axvline(df["dice"].mean(), color="#4e79a7", ls="--", lw=1.5,
               label=f'Dice mean={df["dice"].mean():.3f}')
    ax.axvline(df["iou"].mean(), color="#f28e2b", ls="--", lw=1.5,
               label=f'IoU mean={df["iou"].mean():.3f}')
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
                    alpha=0.6, edgecolors="gray", linewidths=0.3, s=20)
    ax.set_xlabel("Peak-to-Center Distance (px)", fontsize=11)
    ax.set_ylabel("Dice Score", fontsize=11)
    ax.set_title("Localization Quality", fontsize=12)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Pointing Game (0/1)")
    ax.grid(alpha=0.3)

    fig.suptitle("Kvasir-SEG Baseline — Localization Quality Analysis",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Localization plot → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Kvasir-SEG baseline evaluation")
    parser.add_argument("--kvasir_dir", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--n_samples", type=int, default=0,
                        help="Number of samples to evaluate (0 = all 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Heatmap threshold for binary mask prediction")
    args = parser.parse_args()

    kvasir_dir = Path(args.kvasir_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not kvasir_dir.exists():
        print(f"ERROR: Kvasir-SEG not found at {kvasir_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("KVASIR-SEG BASELINE EVALUATION")
    print(f"{'='*60}")
    print(f"  Data:      {kvasir_dir}")
    print(f"  Samples:   {'all' if args.n_samples <= 0 else args.n_samples}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Seed:      {args.seed}")

    print("\nLoading data...")
    df = evaluate_kvasir(kvasir_dir, args.n_samples, args.seed, args.threshold)

    # Per-sample CSV
    csv_path = RESULTS_DIR / "kvasir_per_sample.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Per-sample CSV   → {csv_path}  ({len(df)} rows)")

    # Summary
    summary = make_summary(df)
    summary_path = RESULTS_DIR / "kvasir_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Summary CSV      → {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for _, row in summary.iterrows():
        print(f"  {row['metric']:20s}  {row['mean']:.4f} ± {row['std']:.4f}")
    print(f"\n  Pointing Game accuracy: {df['pointing_game'].mean()*100:.1f}%")

    # Plots
    print(f"\nGenerating plots...")
    plot_baseline_table(summary, RESULTS_DIR / "kvasir_baseline_table.png")
    plot_localization(df, RESULTS_DIR / "kvasir_localization.png")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
