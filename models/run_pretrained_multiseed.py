"""
Run repeated pretrained U-Net experiments for BHI reporting.

This trains the ResNet34 encoder-pretrained U-Net on the fixed Kvasir-SEG
train/val split for multiple random seeds, then evaluates the best-val checkpoint
for each seed on:

- Kvasir test
- external CVC-ClinicDB without retraining

Outputs include per-seed metrics and model-level mean +/- SD tables.

Usage:
    PYTHONPATH=. python3 models/run_pretrained_multiseed.py --seeds 0 1 2 --epochs 20
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    Resized,
    ScaleIntensityRanged,
    ToTensord,
)
from monai.utils import set_determinism
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import segmentation_models_pytorch as smp

from config import DATA_DIR, PROJECT_DIR


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def confusion_counts(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[int, int, int, int]:
    pred = _as_bool_mask(pred_mask)
    gt = _as_bool_mask(gt_mask)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return tp, fp, fn, tn


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = _as_bool_mask(pred_mask)
    gt = _as_bool_mask(gt_mask)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    intersection = (pred & gt).sum()
    return float(2.0 * intersection / (pred.sum() + gt.sum()))


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = _as_bool_mask(pred_mask)
    gt = _as_bool_mask(gt_mask)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    union = (pred | gt).sum()
    return float((pred & gt).sum() / union) if union > 0 else 0.0


def precision_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    tp, fp, _, _ = confusion_counts(pred_mask, gt_mask)
    if tp == 0 and fp == 0:
        return 1.0 if not _as_bool_mask(gt_mask).any() else 0.0
    return float(tp / (tp + fp))


def recall_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    tp, _, fn, _ = confusion_counts(pred_mask, gt_mask)
    if tp == 0 and fn == 0:
        return 1.0
    return float(tp / (tp + fn))


def f_beta_score(pred_mask: np.ndarray, gt_mask: np.ndarray, beta: float = 2.0) -> float:
    precision = precision_score(pred_mask, gt_mask)
    recall = recall_score(pred_mask, gt_mask)
    beta_sq = beta**2
    denominator = beta_sq * precision + recall
    if denominator == 0.0:
        return 0.0
    return float((1.0 + beta_sq) * precision * recall / denominator)


def pointing_game(heatmap: np.ndarray, gt_mask: np.ndarray) -> int:
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    gt = _as_bool_mask(gt_mask)
    return int(gt[int(peak_y), int(peak_x)])


def largest_component_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    mask_u8 = _as_bool_mask(mask).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return None
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centroids[largest_label]
    return float(cy), float(cx)


def peak_to_center_distance(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    centroid = largest_component_centroid(gt_mask)
    if centroid is None:
        return float("inf")
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    cy, cx = centroid
    return float(np.sqrt((float(peak_y) - cy) ** 2 + (float(peak_x) - cx) ** 2))


class SegmentationDataset(Dataset):
    def __init__(self, image_paths: list[Path], mask_paths: list[Path], transform=None):
        if len(image_paths) != len(mask_paths):
            raise ValueError("image_paths and mask_paths must have the same length")
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        sample_id = self.image_paths[idx].stem
        data = {"image": str(self.image_paths[idx]), "mask": str(self.mask_paths[idx])}
        if self.transform:
            data = self.transform(data)

        mask = data["mask"]
        if mask.shape[0] == 3:
            mask = mask.mean(dim=0, keepdim=True)
        mask = (mask > 0.5).float()

        return data["image"], mask, sample_id


def build_transforms(train: bool) -> Compose:
    transforms = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["mask"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image", "mask"], spatial_size=(256, 256)),
    ]
    if train:
        transforms.extend(
            [
                RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
                RandRotated(keys=["image", "mask"], range_x=15, prob=0.5),
                RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
                RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
            ]
        )
    transforms.append(ToTensord(keys=["image", "mask"]))
    return Compose(transforms)


def load_kvasir_split(split: str, transform: Compose) -> SegmentationDataset:
    image_dir = DATA_DIR / split / "images"
    mask_dir = DATA_DIR / split / "masks"
    image_paths = sorted(path for path in image_dir.iterdir() if path.name != ".DS_Store")
    mask_paths = [mask_dir / path.name for path in image_paths]
    missing = [str(path) for path in mask_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Kvasir masks: {missing[:5]}")
    return SegmentationDataset(image_paths, mask_paths, transform=transform)


def load_cvc_external(transform: Compose) -> SegmentationDataset:
    image_dir = PROJECT_DIR / "dataset" / "dataset_b" / "PNG" / "Original"
    mask_dir = PROJECT_DIR / "dataset" / "dataset_b" / "PNG" / "Ground Truth"
    image_paths = sorted(image_dir.glob("*.png"), key=lambda path: int(path.stem))
    mask_paths = [mask_dir / path.name for path in image_paths]
    missing = [str(path) for path in mask_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CVC masks: {missing[:5]}")
    return SegmentationDataset(image_paths, mask_paths, transform=transform)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(device: torch.device) -> torch.nn.Module:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model.to(device)


def dice_from_logits(outputs: torch.Tensor, masks: torch.Tensor) -> float:
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return float((2 * intersection / (union + 1e-8)).mean().item())


def train_one_seed(
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
    progress: bool,
) -> tuple[dict, list[dict], torch.nn.Module]:
    set_seed(seed)

    train_ds = load_kvasir_split("train", build_transforms(train=True))
    val_ds = load_kvasir_split("val", build_transforms(train=False))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(device)
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = nn.BCEWithLogitsLoss()

    def criterion(outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return dice_loss(outputs, masks) + bce_loss(outputs, masks)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_val_dice = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for images, masks, _ in tqdm(
            train_loader,
            desc=f"seed {seed} epoch {epoch}/{epochs}",
            leave=False,
            disable=not progress,
        ):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * images.size(0)
            train_count += images.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_dice_values = []
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss_sum += float(loss.item()) * images.size(0)
                val_count += images.size(0)
                val_dice_values.append(dice_from_logits(outputs, masks))

        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count
        val_dice = float(np.mean(val_dice_values))
        scheduler.step(val_loss)
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_dice": round(val_dice, 6),
            }
        )
        print(
            f"seed {seed:02d} epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}",
            flush=True,
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best_state is None:
        raise RuntimeError(f"No best state captured for seed {seed}")

    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    metadata = {
        "seed": seed,
        "best_val_dice": round(best_val_dice, 6),
        "best_epoch": max(history, key=lambda row: row["val_dice"])["epoch"],
    }
    pd.DataFrame(history).to_csv(output_dir / f"seed_{seed:02d}_training_history.csv", index=False)
    return metadata, history, model


def evaluate_model(
    model: torch.nn.Module,
    dataset: SegmentationDataset,
    dataset_name: str,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    model.eval()
    with torch.no_grad():
        for images, masks, sample_ids in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()

            for idx, sample_id in enumerate(sample_ids):
                heatmap = probs[idx, 0].astype(np.float32).T
                pred_mask = (heatmap >= 0.5).astype(np.uint8)
                gt_mask = masks_np[idx, 0].astype(np.float32).T
                gt_mask = (gt_mask > 0.5).astype(np.uint8)
                rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset_name,
                        "sample_id": sample_id,
                        "dice": dice_score(pred_mask, gt_mask),
                        "iou": iou_score(pred_mask, gt_mask),
                        "precision": precision_score(pred_mask, gt_mask),
                        "recall": recall_score(pred_mask, gt_mask),
                        "f2": f_beta_score(pred_mask, gt_mask, beta=2.0),
                        "pointing_game": pointing_game(heatmap, gt_mask),
                        "peak_center_dist": peak_to_center_distance(heatmap, gt_mask),
                    }
                )

    df = pd.DataFrame(rows)
    metrics = ["dice", "iou", "precision", "recall", "f2", "pointing_game", "peak_center_dist"]
    summary = {"seed": seed, "dataset": dataset_name, "n": int(len(df))}
    for metric in metrics:
        summary[metric] = round(float(df[metric].mean()), 6)
    return summary


def summarize_across_seeds(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "dice",
        "iou",
        "precision",
        "recall",
        "f2",
        "pointing_game",
        "peak_center_dist",
    ]
    rows = []
    for dataset_name, group in per_seed_df.groupby("dataset", sort=False):
        row = {"dataset": dataset_name, "seeds": int(group["seed"].nunique()), "n_per_seed": int(group["n"].iloc[0])}
        for metric in metric_cols:
            row[f"{metric}_mean"] = round(float(group[metric].mean()), 4)
            row[f"{metric}_sd"] = round(float(group[metric].std(ddof=1)), 4) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed pretrained U-Net evaluation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results" / "bhi_multiseed")
    parser.add_argument("--progress", action="store_true", help="Show per-batch tqdm progress bars.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed seed rows in output-dir and only run missing seeds.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    print(f"Device: {device}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)

    kvasir_test = load_kvasir_split("test", build_transforms(train=False))
    cvc_external = load_cvc_external(build_transforms(train=False))

    per_seed_path = args.output_dir / "pretrained_multiseed_per_seed.csv"
    summary_path = args.output_dir / "pretrained_multiseed_summary.csv"
    metadata_path = args.output_dir / "pretrained_multiseed_training_metadata.csv"
    json_path = args.output_dir / "pretrained_multiseed_summary.json"

    per_seed_rows = []
    metadata_rows = []
    completed_seeds: set[int] = set()
    if args.resume and per_seed_path.exists() and metadata_path.exists():
        existing_per_seed = pd.read_csv(per_seed_path)
        existing_metadata = pd.read_csv(metadata_path)
        completed_seeds = set(existing_metadata["seed"].astype(int).tolist())
        per_seed_rows = existing_per_seed.to_dict(orient="records")
        metadata_rows = existing_metadata.to_dict(orient="records")
        print(f"Resuming from completed seeds: {sorted(completed_seeds)}", flush=True)

    for seed in args.seeds:
        if seed in completed_seeds:
            print(f"Skipping completed seed {seed}", flush=True)
            continue

        metadata, _, model = train_one_seed(
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            output_dir=args.output_dir,
            progress=args.progress,
        )
        metadata_rows.append(metadata)
        per_seed_rows.append(evaluate_model(model, kvasir_test, "Kvasir test", seed, args.batch_size, device))
        per_seed_rows.append(evaluate_model(model, cvc_external, "CVC external", seed, args.batch_size, device))

        per_seed_df = pd.DataFrame(per_seed_rows)
        summary_df = summarize_across_seeds(per_seed_df)
        metadata_df = pd.DataFrame(metadata_rows)
        per_seed_df.to_csv(per_seed_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        metadata_df.to_csv(metadata_path, index=False)
        json_path.write_text(
            json.dumps(
                {
                    "seeds": args.seeds,
                    "completed_seeds": sorted(metadata_df["seed"].astype(int).tolist()),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "summary": summary_df.to_dict(orient="records"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved intermediate results after seed {seed} -> {summary_path}", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_seed_df = pd.DataFrame(per_seed_rows)
    summary_df = summarize_across_seeds(per_seed_df)
    metadata_df = pd.DataFrame(metadata_rows)

    per_seed_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "summary": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nPer-seed metrics:", flush=True)
    print(per_seed_df.to_string(index=False), flush=True)
    print("\nMean +/- SD across seeds:", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\nSaved: {per_seed_path}", flush=True)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
