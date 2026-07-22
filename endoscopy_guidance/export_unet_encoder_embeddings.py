"""
export_unet_encoder_embeddings.py
=================================
Export pooled encoder embeddings from the strong UNet/ResNet34 baseline.

These embeddings give downstream quantum-kernel selectors richer learned
context than probability-map morphology alone, while keeping the segmentation
backbone fixed.
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


def main():
    parser = argparse.ArgumentParser(description="Export pooled UNet encoder embeddings")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test")
    parser.add_argument("--checkpoint", default="/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance/models/unet_pretrained.pth")
    parser.add_argument("--output_npz", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_encoder_embeddings.npz")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    base = Path(args.baseline_dir)
    metrics = pd.read_csv(base / "baseline_metrics.csv")
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)

    sample_ids = metrics["sample_id"].tolist()
    embeddings = []
    for start in range(0, len(sample_ids), args.batch_size):
        batch_ids = sample_ids[start : start + args.batch_size]
        # Images are already stored as RGB npy arrays in the baseline export.
        arrays = [np.load(base / "images" / f"{sample_id}.npy").astype(np.float32) / 255.0 for sample_id in batch_ids]
        tensor = torch.from_numpy(np.stack([arr.transpose(2, 0, 1) for arr in arrays], axis=0)).float().to(device)
        with torch.no_grad():
            features = model.encoder(tensor)[-1]
            avg = features.mean(dim=(2, 3))
            max_pool = features.amax(dim=(2, 3))
            emb = torch.cat([avg, max_pool], dim=1).detach().cpu().numpy().astype(np.float32)
        embeddings.append(emb)
        if start + len(batch_ids) >= len(sample_ids) or (start // args.batch_size + 1) % 10 == 0:
            print(f"[embeddings] {start + len(batch_ids)}/{len(sample_ids)} frames", flush=True)

    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_ids=np.asarray(sample_ids, dtype=object),
        embeddings=np.concatenate(embeddings, axis=0),
    )
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
