"""
Full-dataset evaluation of the Models group's ResNet18 bladder-tissue
classifier trained via models/train.py.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import MODELS_DIR, get_device

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

DEFAULT_MODEL  = MODELS_DIR / "bladder_model.pth"
DEFAULT_DATA   = PROJECT_ROOT / "dataset" / "EndoscopicBladderTissue"
RESULTS_DIR    = Path(__file__).resolve().parent / "results"
CLASS_NAMES = ["HGC", "LGC", "NST", "NTL"]


# Dataset
class BladderImageDataset(Dataset):
    # Reads images from the four class folders
    def __init__(self, data_dir: Path, transform):
        self.transform = transform
        self.samples: list[tuple[str, int, str]] = []  # (path, label, class_name)
        for idx, cls in enumerate(CLASS_NAMES):
            cls_dir = data_dir / cls
            if not cls_dir.is_dir():
                print(f"WARNING: {cls_dir} not found, skipping")
                continue
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((str(img), idx, cls))
        print(f"  Loaded {len(self.samples)} images across {len(CLASS_NAMES)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


# Build model identical to train.py
def build_model(n_classes: int, weight_path: Path, device: torch.device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Loaded weights from {weight_path}")
    return model


# Evaluation loop
@torch.no_grad()
def evaluate_all(model, dataloader, device) -> pd.DataFrame:
    records = []
    for imgs, labels, paths in dataloader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        labels_np = labels.numpy()

        for i in range(len(labels_np)):
            fname = Path(paths[i]).name
            true_cls = CLASS_NAMES[labels_np[i]]
            pred_cls = CLASS_NAMES[preds[i]]
            conf = probs[i, preds[i]]
            records.append({
                "filename": fname,
                "true_class": true_cls,
                "pred_class": pred_cls,
                "correct": int(true_cls == pred_cls),
                "confidence": float(conf),
                **{f"prob_{c}": float(probs[i, j]) for j, c in enumerate(CLASS_NAMES)},
            })
    return pd.DataFrame(records)


# Per-class summary
def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls in CLASS_NAMES:
        sub = df[df["true_class"] == cls]
        n = len(sub)
        acc = sub["correct"].mean() if n else 0
        conf = sub["confidence"].mean() if n else 0
        rows.append({
            "class": cls,
            "n_samples": n,
            "accuracy": round(acc, 4),
            "mean_confidence": round(conf, 4),
        })
    # overall row
    rows.append({
        "class": "OVERALL",
        "n_samples": len(df),
        "accuracy": round(df["correct"].mean(), 4),
        "mean_confidence": round(df["confidence"].mean(), 4),
    })
    return pd.DataFrame(rows)


# Plots
def plot_confusion(df: pd.DataFrame, save_path: Path):
    y_true = df["true_class"]
    y_pred = df["pred_class"]
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Bladder Tissue Classification — Confusion Matrix", fontsize=13, pad=10)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=10, color=color)

    fig.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {save_path}")


def plot_confidence(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box-plot by class
    ax = axes[0]
    data = [df[df["true_class"] == c]["confidence"].values for c in CLASS_NAMES]
    bp = ax.boxplot(data, patch_artist=True, labels=CLASS_NAMES)
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Confidence", fontsize=11)
    ax.set_title("Confidence by True Class", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Histogram correct vs incorrect
    ax = axes[1]
    correct = df[df["correct"] == 1]["confidence"]
    wrong   = df[df["correct"] == 0]["confidence"]
    ax.hist(correct, bins=30, alpha=0.6, label=f"Correct  (n={len(correct)})", color="#4e79a7")
    if len(wrong) > 0:
        ax.hist(wrong, bins=30, alpha=0.6, label=f"Wrong  (n={len(wrong)})", color="#e15759")
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Confidence Distribution", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Bladder Model — Confidence Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confidence plot  → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate bladder classifier on full dataset")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--data_dir",   type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    data_dir   = Path(args.data_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        print("Run  train.py  first to generate bladder_model.pth")
        sys.exit(1)
    if not data_dir.exists():
        print(f"ERROR: data not found at {data_dir}")
        sys.exit(1)

    device = get_device()
    print(f"\n{'='*60}")
    print(f"BLADDER MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Model:  {model_path}")
    print(f"  Data:   {data_dir}")

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("\nLoading dataset...")
    ds = BladderImageDataset(data_dir, eval_transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Loading model...")
    model = build_model(len(CLASS_NAMES), model_path, device)

    print("\nRunning evaluation on all images...\n")
    df = evaluate_all(model, loader, device)

    # Save per-sample CSV
    csv_path = RESULTS_DIR / "bladder_per_sample.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Per-sample CSV   → {csv_path}  ({len(df)} rows)")

    # Summary
    summary = make_summary(df)
    summary_path = RESULTS_DIR / "bladder_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Summary CSV      → {summary_path}")

    # Classification report
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(df["true_class"], df["pred_class"],
                                target_names=CLASS_NAMES, digits=4))

    # Summary table
    print(f"\n{'='*60}")
    print("PER-CLASS SUMMARY")
    print(f"{'='*60}")
    print(summary.to_string(index=False))

    # Plots
    print(f"\nGenerating plots...")
    plot_confusion(df, RESULTS_DIR / "bladder_confusion.png")
    plot_confidence(df, RESULTS_DIR / "bladder_confidence.png")

    # Highlight errors
    errors = df[df["correct"] == 0]
    if len(errors) > 0:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)} misclassifications)")
        print(f"{'='*60}")
        for _, row in errors.iterrows():
            print(f"  {row['filename']:50s}  true={row['true_class']}  pred={row['pred_class']}  conf={row['confidence']:.4f}")
    else:
        print(f"\n PERFECT — zero misclassifications on {len(df)} images")

    print(f"\n{'='*60}")
    overall_acc = df["correct"].mean()
    print(f"OVERALL ACCURACY: {overall_acc:.4f}  ({df['correct'].sum()}/{len(df)})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
