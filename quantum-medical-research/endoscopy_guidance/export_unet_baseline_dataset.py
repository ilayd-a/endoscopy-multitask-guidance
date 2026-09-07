"""
export_unet_baseline_dataset.py
================================
Export UNet/ResNet34 probability maps for split-folder or split-CSV datasets.

This is the headline exporter for the quantum residual study: it can reproduce
the strong Kvasir validation/test baseline from the endoscopy repo checkpoint
and produce the probability maps needed for residual quantum refinement.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp

from export_classical_cvc_baseline import boundary_band, choose_device, dice_iou, sample_features


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    mask_path: Path
    split: str


def load_model(checkpoint: Path, device: torch.device):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_image(path: Path, image_size: int) -> tuple[np.ndarray, torch.Tensor]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return rgb, tensor


def load_mask(path: Path, image_size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def load_split_csv(root: Path, splits: list[str]) -> list[Sample]:
    frame = pd.read_csv(root / "splits.csv")
    image_col = next(col for col in ["image_path", "image", "image_file", "png_image_path"] if col in frame.columns)
    mask_col = next(col for col in ["mask_path", "mask", "mask_file"] if col in frame.columns)
    wanted = {split.lower() for split in splits}
    samples: list[Sample] = []
    for row in frame.itertuples(index=False):
        split = str(getattr(row, "split")).strip().lower()
        if split not in wanted:
            continue
        image_rel = Path(str(getattr(row, image_col)))
        mask_rel = Path(str(getattr(row, mask_col)))
        stem = str(getattr(row, "stem")) if hasattr(row, "stem") else image_rel.stem
        samples.append(Sample(stem, root / image_rel, root / mask_rel, split))
    return sorted(samples, key=lambda item: (item.split, item.sample_id))


def load_split_folders(root: Path, splits: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for split in splits:
        image_dir = root / split / "images"
        mask_dir = root / split / "masks"
        for image_path in sorted(image_dir.iterdir()):
            if image_path.name.startswith(".") or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            samples.append(Sample(image_path.stem, image_path, mask_path, split))
    return samples


def load_samples(root: Path, splits: list[str]) -> list[Sample]:
    if (root / "splits.csv").exists():
        return load_split_csv(root, splits)
    return load_split_folders(root, splits)


def main():
    parser = argparse.ArgumentParser(description="Export UNet baseline predictions for split datasets")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--checkpoint", default="/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance/models/unet_pretrained.pth")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--output_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_val_test")
    parser.add_argument("--dataset_name", default="kvasir")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_sweep", type=float, nargs="*", default=[])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    checkpoint = Path(args.checkpoint)
    samples = load_samples(root, args.splits)
    if not samples:
        raise ValueError(f"No samples found for splits={args.splits} under {root}")

    out = Path(args.output_dir)
    for subdir in ["images", "gt_masks", "prob_maps", "pred_masks"]:
        (out / subdir).mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = load_model(checkpoint, device)
    rows = []
    for idx, sample in enumerate(samples):
        sid = f"{args.dataset_name}_{sample.split}_{idx:04d}_{sample.sample_id}"
        image, tensor = load_image(sample.image_path, args.image_size)
        gt = load_mask(sample.mask_path, args.image_size)
        with torch.no_grad():
            logits = model(tensor.to(device))
            prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
        pred = (prob >= args.threshold).astype(np.uint8)
        dice, iou = dice_iou(pred, gt)

        np.save(out / "images" / f"{sid}.npy", image)
        np.save(out / "gt_masks" / f"{sid}.npy", gt)
        np.save(out / "prob_maps" / f"{sid}.npy", prob)
        np.save(out / "pred_masks" / f"{sid}.npy", pred)

        record = {
            "sample_id": sid,
            "source_file": sample.image_path.name,
            "sequence_id": -1,
            "frame_id": idx,
            "split": sample.split,
            "dice": float(dice),
            "iou": float(iou),
        }
        record.update(sample_features(prob, pred, gt))
        for threshold in args.threshold_sweep:
            sweep_pred = (prob >= threshold).astype(np.uint8)
            sweep_dice, sweep_iou = dice_iou(sweep_pred, gt)
            record[f"dice_t{threshold:.2f}"] = float(sweep_dice)
            record[f"iou_t{threshold:.2f}"] = float(sweep_iou)
        rows.append(record)
        if (idx + 1) % 25 == 0 or idx + 1 == len(samples):
            print(f"[baseline:{args.dataset_name}] {idx + 1}/{len(samples)} frames", flush=True)

    summary_path = out / "baseline_metrics.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    print(f"[saved] {summary_path}")
    print(df.groupby("split")[["dice", "iou", "boundary_error_frac", "mean_uncertainty"]].mean().to_string())
    sweep_cols = [col for col in df.columns if col.startswith("dice_t")]
    if sweep_cols:
        print("threshold sweep Dice:")
        print(df.groupby("split")[sweep_cols].mean().to_string())
    print("hardest frames:")
    print(df.sort_values("dice").head(10)[["sample_id", "split", "source_file", "dice", "iou", "boundary_error_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()
