#!/usr/bin/env python3
"""
export_cvc_failure_features.py
==============================
Export one row of label-free features per CVC-ClinicDB frame for the
failure-detection benchmark, using the Kvasir-trained U-Net (domain shift).

Place this file in quantum-medical-research/endoscopy_guidance/ (it reuses
heatmap_features / pooled_encoder_features / load_model from that folder) and
run it from the repository root:

  python quantum-medical-research/endoscopy_guidance/export_cvc_failure_features.py \
      --cvc-root dataset/dataset_b \
      --checkpoint models/unet_pretrained.pth \
      --official-csv eval/results/cvc_test_kvasir_to_cvc_per_sample.csv \
      --out failure_detection/features_cvc.csv

Output columns
  sample_id, sequence_id, dice (label source, official eval CSV by default),
  dice_recomputed, dice_official, u_* (training-free uncertainty scores used as
  baselines, higher = more suspicious), f_hm_* (27 heatmap/image statistics),
  f_enc_* (pooled encoder activations).

The ground-truth mask is used only for the Dice label, never for features.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from encoder_embedding_failure_benchmark import (  # noqa: E402
    heatmap_features, load_rgb, pooled_encoder_features)
from export_cvc_predictions import INPUT_SIZE, device_from_name, load_model  # noqa: E402

HM_NAMES = (["hm_mean", "hm_std", "hm_min", "hm_max"]
            + [f"hm_q{q}" for q in (10, 25, 50, 75, 90, 95, 99)]
            + ["grad_mean", "grad_std", "pred_area", "n_components", "largest_comp_area", "mean_comp_area",
               "pos_prob_mean", "pos_prob_std", "gray_mean", "gray_std",
               "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"])


def uncertainty_scores(hm: np.ndarray, pred: np.ndarray) -> dict:
    """Training-free failure scores from the U-Net output; higher = more likely failure.
    Directions are fixed in advance (not fit to labels)."""
    from scipy import ndimage
    p = np.clip(hm.astype(np.float64), 1e-7, 1 - 1e-7)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)  # binary entropy in bits, 0..1
    m = pred.astype(bool)
    n_comp = ndimage.label(m)[1]
    return {
        "u_low_max_confidence": float(1 - p.max()),
        "u_low_mask_confidence": float(1 - p[m].mean()) if m.any() else 1.0,
        "u_mean_entropy": float(ent.mean()),
        "u_mask_entropy": float(ent[m].mean()) if m.any() else 1.0,
        "u_fragmentation": float(n_comp) if n_comp > 0 else 99.0,  # empty prediction = maximal suspicion
    }


def official_transform():
    """Exactly the preprocessing of models/heatmap_pretrained.py, which produced the official CVC CSV.
    MONAI's LoadImage returns PNGs transposed relative to cv2/PIL; training used the same loader,
    so the model expects the transposed input and its output is transposed back afterwards."""
    from monai.transforms import Compose, EnsureChannelFirst, LoadImage, Resize, ScaleIntensity, ToTensor
    return Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity(), Resize((256, 256)),
                    ToTensor()])


def load_binary_mask_official(path: Path, shape=(256, 256)) -> np.ndarray:
    """Same as eval/datasets.py::load_binary_mask (PIL, NEAREST, >127)."""
    from PIL import Image
    m = Image.open(path).convert("L").resize((shape[1], shape[0]), Image.NEAREST)
    a = np.asarray(m, dtype=np.float32)
    return a > (127.0 if a.max() > 1.0 else 0.5)


def find_cvc_root(arg: Path | None) -> Path:
    cands = [arg] if arg else [Path("dataset/dataset_b"), Path("dataset/CVC-ClinicDB")]
    for c in cands:
        if c and (c / "metadata.csv").exists() and (c / "PNG" / "Original").exists():
            return c
    sys.exit(f"CVC root not found (need metadata.csv and PNG/Original) in {cands}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cvc-root", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=Path("models/unet_pretrained.pth"))
    ap.add_argument("--official-csv", type=Path,
                    default=Path("eval/results/cvc_test_kvasir_to_cvc_per_sample.csv"))
    ap.add_argument("--label-source", choices=["official", "recomputed"], default="official")
    ap.add_argument("--include-scales", choices=["bottleneck", "deep", "all"], default="deep")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--preprocess", choices=["official", "cv2"], default="official",
                    help="official = MONAI pipeline of models/heatmap_pretrained.py (matches training and the "
                         "official eval CSV); cv2 = legacy loader of the quantum scripts (transposed relative to training)")
    ap.add_argument("--out", type=Path, default=Path("failure_detection/features_cvc.csv"))
    args = ap.parse_args()

    if "cvc" in args.checkpoint.name.lower():
        sys.exit("This checkpoint looks CVC-trained. Use the Kvasir-trained models/unet_pretrained.pth.")
    root = find_cvc_root(args.cvc_root)
    meta = pd.read_csv(root / "metadata.csv")
    if not {"png_image_path", "sequence_id"} <= set(meta.columns):
        sys.exit("metadata.csv needs png_image_path and sequence_id columns")
    img_dir, gt_dir = root / "PNG" / "Original", root / "PNG" / "Ground Truth"

    device = device_from_name(args.device)
    model = load_model(args.checkpoint, device)
    print(f"Model {args.checkpoint} on {device}; {len(meta)} frames from {root}")

    tf = official_transform() if args.preprocess == "official" else None
    print(f"Preprocessing: {args.preprocess}")
    rows, enc_dim = [], None
    for i, r in enumerate(meta.itertuples(index=False)):
        name = Path(r.png_image_path).name
        image = load_rgb(img_dir / name)  # natural orientation, 256x256 RGB; used for colour statistics only
        if tf is not None:
            x = tf(str(img_dir / name))
            x = (x.as_tensor() if hasattr(x, "as_tensor") else x).float().unsqueeze(0).to(device)
        else:
            x = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model.encoder(x)
            hm = torch.sigmoid(model.segmentation_head(model.decoder(feats))).squeeze().cpu().numpy()
        if tf is not None:
            hm = np.ascontiguousarray(hm.T)  # back to natural image coordinates, as heatmap_pretrained.py does
        pred = (hm >= args.threshold).astype(np.uint8)
        enc = pooled_encoder_features(feats, args.include_scales)
        hmf = heatmap_features(hm, pred, image)
        enc_dim = len(enc)

        if not (gt_dir / name).exists():
            sys.exit(f"Missing ground-truth mask for {name}")
        gt = load_binary_mask_official(gt_dir / name, (INPUT_SIZE[1], INPUT_SIZE[0]))
        p = pred.astype(bool)
        denom = p.sum() + gt.sum()
        dice = 1.0 if denom == 0 else 2.0 * (p & gt).sum() / denom

        row = dict(sample_id=name, sequence_id=int(r.sequence_id), dice_recomputed=float(dice))
        row.update(uncertainty_scores(hm, pred))
        row.update({f"f_hm_{n}": v for n, v in zip(HM_NAMES, hmf)})
        row.update({f"f_enc_{k:04d}": float(v) for k, v in enumerate(enc)})
        rows.append(row)
        if (i + 1) % 100 == 0 or i + 1 == len(meta):
            print(f"  {i + 1}/{len(meta)}", flush=True)

    df = pd.DataFrame(rows)
    if args.official_csv.exists():
        off = pd.read_csv(args.official_csv)[["sample_id", "dice"]].rename(columns={"dice": "dice_official"})
        off["sample_id"] = off.sample_id.astype(str).map(lambda s: Path(s).name)
        df = df.merge(off, on="sample_id", how="left")
        both = df.dropna(subset=["dice_official"])
        corr = np.corrcoef(both.dice_recomputed, both.dice_official)[0, 1]
        print(f"\nDice check vs {args.official_csv.name}: matched {len(both)}/{len(df)} frames, "
              f"mean recomputed {both.dice_recomputed.mean():.4f} vs official {both.dice_official.mean():.4f}, "
              f"correlation {corr:.4f}, max |diff| {np.abs(both.dice_recomputed - both.dice_official).max():.4f}")
        if len(both) < 0.95 * len(df) or corr < 0.95 or abs(both.dice_recomputed.mean() - both.dice_official.mean()) > 0.02:
            print("WARNING: recomputed Dice does not match the official evaluation closely. "
                  "Check the checkpoint and data before running the benchmark.")
    else:
        df["dice_official"] = np.nan
        print(f"(official CSV {args.official_csv} not found; using recomputed Dice)")
    use_official = args.label_source == "official" and df.dice_official.notna().all()
    df.insert(2, "dice", df.dice_official if use_official else df.dice_recomputed)
    print(f"Label source: {'official eval CSV' if use_official else 'recomputed'}; "
          f"failures Dice<0.5: {(df.dice < 0.5).sum()}, Dice<0.2: {(df.dice < 0.2).sum()} of {len(df)}")
    print(f"Features: {len(HM_NAMES)} f_hm_* + {enc_dim} f_enc_*; sequences: {df.sequence_id.nunique()}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
