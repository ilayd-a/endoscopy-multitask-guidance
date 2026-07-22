"""
export_classical_cvc_baseline.py
================================
Export predictions from the strong classical UNet/ResNet34 baseline in the
endoscopy-multitask-guidance repo.

The output is intentionally stored in the quantum repo so quantum residual and
hard-case experiments can build on the classical baseline without modifying the
source segmentation repo.
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


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def split_mask(metadata: pd.DataFrame, split: str) -> pd.Series:
    if split == "train":
        return metadata["sequence_id"].le(23)
    if split == "val":
        return metadata["sequence_id"].between(24, 26)
    if split == "test":
        return metadata["sequence_id"].ge(27)
    if split == "val_test":
        return metadata["sequence_id"].ge(24)
    if split == "all":
        return pd.Series(True, index=metadata.index)
    raise ValueError(f"Unknown split={split}")


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


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    dice = (2.0 * inter) / (pred_sum + gt_sum + 1e-8)
    union = float(np.logical_or(pred, gt).sum())
    iou = inter / (union + 1e-8)
    return dice, iou


def boundary_band(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    return (dilated != eroded)


def sample_features(prob: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    uncertainty = 1.0 - np.abs(prob - 0.5) * 2.0
    band = boundary_band(pred, radius=3) | boundary_band(gt, radius=3)
    err = pred.astype(np.uint8) != gt.astype(np.uint8)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    return {
        "mean_prob": float(prob.mean()),
        "mean_uncertainty": float(uncertainty.mean()),
        "boundary_uncertainty": float(uncertainty[band].mean()) if band.any() else 0.0,
        "mask_area_frac": float(pred.mean()),
        "gt_area_frac": float(gt.mean()),
        "error_frac": float(err.mean()),
        "boundary_error_frac": float(err[band].mean()) if band.any() else 0.0,
        "false_positive_frac": float(fp.mean()),
        "false_negative_frac": float(fn.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Export strong classical CVC baseline predictions")
    parser.add_argument("--source_repo", default="/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", choices=["train", "val", "test", "val_test", "all"], default="val_test")
    parser.add_argument("--output_dir", default="endoscopy_guidance/results/strong_unet_pretrained_cvc_val_test")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_sweep", type=float, nargs="*", default=[])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    source = Path(args.source_repo)
    cvc_dir = source / "dataset" / "CVC-ClinicDB"
    checkpoint = Path(args.checkpoint) if args.checkpoint else source / "models" / "unet_pretrained.pth"
    metadata = pd.read_csv(cvc_dir / "metadata.csv")
    metadata = metadata.loc[split_mask(metadata, args.split)].sort_values(["sequence_id", "frame_id"]).reset_index(drop=True)

    out = Path(args.output_dir)
    for subdir in ["images", "gt_masks", "prob_maps", "pred_masks"]:
        (out / subdir).mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = load_model(checkpoint, device)

    rows = []
    for idx, row in metadata.iterrows():
        image_name = Path(row.png_image_path).name
        sid = f"cvc_{args.split}_{idx:04d}"
        image, tensor = load_image(cvc_dir / row.png_image_path, args.image_size)
        gt = load_mask(cvc_dir / row.png_mask_path, args.image_size)
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
            "source_file": image_name,
            "sequence_id": int(row.sequence_id),
            "frame_id": int(row.frame_id),
            "split": "val" if int(row.sequence_id) <= 26 else "test",
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
        if (idx + 1) % 25 == 0 or idx + 1 == len(metadata):
            print(f"[baseline] {idx + 1}/{len(metadata)} frames", flush=True)

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
    print(df.sort_values("dice").head(10)[["sample_id", "split", "source_file", "sequence_id", "dice", "iou", "boundary_error_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()
