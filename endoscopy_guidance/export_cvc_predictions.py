"""
export_cvc_predictions.py
=========================
Export CVC-ClinicDB guidance predictions for quantum candidate ranking.

This script reads the local endoscopy-multitask-guidance repo, loads its
UNet/ResNet34 checkpoint, and writes NumPy triplets compatible with
candidate_ranking_benchmark.py:

  gt_mask_*.npy
  pred_mask_*.npy
  pred_heatmap_*.npy

By default it exports the sequence-held-out test split (sequences 27-29).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp


INPUT_SIZE = (256, 256)


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def split_frame_names(metadata: pd.DataFrame, split: str) -> list[str]:
    if split == "train":
        df = metadata[metadata.sequence_id <= 23]
    elif split == "val":
        df = metadata[(metadata.sequence_id >= 24) & (metadata.sequence_id <= 26)]
    elif split == "test":
        df = metadata[metadata.sequence_id >= 27]
    elif split == "val_test":
        df = metadata[metadata.sequence_id >= 24]
    elif split == "all":
        df = metadata
    else:
        raise ValueError(f"Unknown split: {split}")
    return df["png_image_path"].apply(lambda p: Path(p).name).tolist()


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


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    denom = pred_bool.sum() + gt_bool.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred_bool, gt_bool).sum() / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_bool, gt_bool).sum() / union)


def pointing_game(heatmap: np.ndarray, gt: np.ndarray) -> int:
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(gt[y, x] > 0)


def export_predictions(args):
    endoscopy_repo = Path(args.endoscopy_repo)
    cvc_dir = endoscopy_repo / "dataset" / "CVC-ClinicDB"
    img_dir = cvc_dir / "PNG" / "Original"
    mask_dir = cvc_dir / "PNG" / "Ground Truth"
    metadata = pd.read_csv(cvc_dir / "metadata.csv")
    frame_names = split_frame_names(metadata, args.split)
    if args.max_samples:
        frame_names = frame_names[:args.max_samples]

    device = device_from_name(args.device)
    model = load_model(Path(args.checkpoint), device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[model] {args.checkpoint}")
    print(f"[data] split={args.split} frames={len(frame_names)} device={device}")
    print(f"[out] {out_dir}")

    rows = []
    for idx, name in enumerate(frame_names):
        image_tensor = load_image_tensor(img_dir / name, device)
        gt = load_mask(mask_dir / name)
        with torch.no_grad():
            logits = model(image_tensor)
            heatmap = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
        pred = (heatmap >= args.threshold).astype(np.uint8)

        sid = f"{idx:04d}_{Path(name).stem}"
        np.save(out_dir / f"gt_mask_{sid}.npy", gt)
        np.save(out_dir / f"pred_mask_{sid}.npy", pred)
        np.save(out_dir / f"pred_heatmap_{sid}.npy", heatmap)
        row = {
            "sample_id": sid,
            "source_file": name,
            "split": args.split,
            "dice": dice_score(pred, gt),
            "iou": iou_score(pred, gt),
            "pointing_game": pointing_game(heatmap, gt),
            "gt_pixels": int(gt.sum()),
            "pred_pixels": int(pred.sum()),
            "heatmap_max": float(heatmap.max()),
            "heatmap_mean": float(heatmap.mean()),
        }
        rows.append(row)
        if (idx + 1) % 10 == 0 or idx + 1 == len(frame_names):
            print(f"[export] {idx + 1}/{len(frame_names)}")

    summary_path = out_dir / "export_summary.csv"
    with summary_path.open("w", newline="") as f:
        fieldnames = [
            "sample_id",
            "source_file",
            "split",
            "dice",
            "iou",
            "pointing_game",
            "gt_pixels",
            "pred_pixels",
            "heatmap_max",
            "heatmap_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[summary] dice={np.mean([r['dice'] for r in rows]):.4f}")
    print(f"[summary] iou={np.mean([r['iou'] for r in rows]):.4f}")
    print(f"[summary] pointing={np.mean([r['pointing_game'] for r in rows]):.4f}")
    print(f"[saved] {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Export CVC predictions for quantum guidance ranking")
    parser.add_argument(
        "--endoscopy_repo",
        default="/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance",
    )
    parser.add_argument(
        "--checkpoint",
        default="/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance/models/unet_cvc.pth",
    )
    parser.add_argument(
        "--output_dir",
        default="endoscopy_guidance/exports/cvc_test",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "val_test", "all"], default="test")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    export_predictions(args)


if __name__ == "__main__":
    main()
