"""
export_unet_tta_uncertainty.py
==============================
Export inference-time UNet uncertainty features from test-time augmentation.

These features are meant for downstream switch policies that decide whether a
fixed UNet mask should be trusted or whether a second expert path should be
consulted. They intentionally avoid ground-truth masks, so they can be used in a
deployment-style evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp

from export_classical_cvc_baseline import choose_device


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


def entropy_binary(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    up = np.zeros_like(mask)
    down = np.zeros_like(mask)
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    up[1:, :] = mask[:-1, :]
    down[:-1, :] = mask[1:, :]
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]
    return mask ^ (up & down & left & right)


def tta_batch(tensor: torch.Tensor) -> list[tuple[torch.Tensor, str]]:
    return [
        (tensor, "none"),
        (torch.flip(tensor, dims=[3]), "h"),
        (torch.flip(tensor, dims=[2]), "v"),
        (torch.flip(tensor, dims=[2, 3]), "hv"),
    ]


def undo_tta(prob: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "h":
        return torch.flip(prob, dims=[3])
    if kind == "v":
        return torch.flip(prob, dims=[2])
    if kind == "hv":
        return torch.flip(prob, dims=[2, 3])
    return prob


def summarize_tta(probs: np.ndarray, saved_prob: np.ndarray | None = None) -> dict[str, float]:
    mean_prob = probs.mean(axis=0)
    std_prob = probs.std(axis=0)
    masks = probs >= 0.5
    vote = masks.mean(axis=0)
    mask_areas = masks.mean(axis=(1, 2))
    mean_mask = mean_prob >= 0.5
    boundaries = np.stack([boundary(mask) for mask in masks], axis=0)
    boundary_vote = boundaries.mean(axis=0)
    record = {
        "tta_prob_std_mean": float(std_prob.mean()),
        "tta_prob_std_p95": float(np.percentile(std_prob, 95)),
        "tta_prob_std_max": float(std_prob.max()),
        "tta_disagreement_mean": float(np.abs(probs - mean_prob[None, :, :]).mean()),
        "tta_vote_entropy": float(entropy_binary(vote).mean()),
        "tta_area_mean": float(mask_areas.mean()),
        "tta_area_std": float(mask_areas.std()),
        "tta_area_range": float(mask_areas.max() - mask_areas.min()),
        "tta_boundary_vote_entropy": float(entropy_binary(boundary_vote).mean()),
        "tta_boundary_area_frac": float(boundary(mean_mask).mean()),
    }
    if saved_prob is not None:
        record["tta_mean_prob_shift"] = float(np.abs(mean_prob - saved_prob).mean())
    return record


def main():
    parser = argparse.ArgumentParser(description="Export UNet TTA uncertainty features")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test")
    parser.add_argument("--checkpoint", default="/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance/models/unet_pretrained.pth")
    parser.add_argument("--prompt_quality_csv", default="")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/kvasir120_unet_tta_uncertainty.csv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    base = Path(args.baseline_dir)
    metrics = pd.read_csv(base / "baseline_metrics.csv")
    if args.prompt_quality_csv:
        prompt_sources = set(pd.read_csv(args.prompt_quality_csv)["source_file"].astype(str))
        metrics = metrics.loc[metrics["source_file"].astype(str).isin(prompt_sources)].copy()
    metrics = metrics.sort_values("sample_id").reset_index(drop=True)

    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    rows = []
    for start in range(0, len(metrics), args.batch_size):
        batch = metrics.iloc[start : start + args.batch_size]
        images = [
            np.load(base / "images" / f"{row.sample_id}.npy").astype(np.float32) / 255.0
            for row in batch.itertuples(index=False)
        ]
        tensor = torch.from_numpy(np.stack([img.transpose(2, 0, 1) for img in images], axis=0)).float().to(device)
        aug_probs = []
        with torch.no_grad():
            for aug, kind in tta_batch(tensor):
                prob = torch.sigmoid(model(aug))
                aug_probs.append(undo_tta(prob, kind).squeeze(1).detach().cpu().numpy().astype(np.float32))
        probs_by_sample = np.stack(aug_probs, axis=1)
        for idx, row in enumerate(batch.itertuples(index=False)):
            saved_path = base / "prob_maps" / f"{row.sample_id}.npy"
            saved_prob = np.load(saved_path).astype(np.float32) if saved_path.exists() else None
            record = {
                "sample_id": row.sample_id,
                "source_file": row.source_file,
                "split": row.split,
            }
            record.update(summarize_tta(probs_by_sample[idx], saved_prob))
            rows.append(record)
        done = start + len(batch)
        if done == len(metrics) or (done // args.batch_size) % 10 == 0:
            print(f"[tta] {done}/{len(metrics)} frames", flush=True)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
