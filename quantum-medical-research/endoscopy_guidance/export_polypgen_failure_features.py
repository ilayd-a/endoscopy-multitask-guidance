#!/usr/bin/env python3
"""
export_polypgen_failure_features.py
===================================
Second external test set for the failure-detection benchmark: PolypGen 2021
(positive frames), with the Kvasir-trained U-Net and exactly the same
preprocessing, features and uncertainty scores as export_cvc_failure_features.py.

Pre-specified rules (fixed before any PolypGen result is seen)
  * Eligible frames: positive/images with a mask that has at least one pixel > 127
    (masks are JPEG, so they are thresholded at 127 like eval/datasets.py). Frames
    whose file name contains "negative" are excluded (3 mislabelled files), and so
    are frames whose mask becomes empty after resizing to 256x256.
  * Dice label: recomputed (no official PolypGen eval exists), official
    preprocessing (MONAI, as in models/heatmap_pretrained.py), threshold 0.5.
  * Groups for CV: video sequence for sequence frames (seqN_...), medical centre
    for single frames (C1..C6). No frame shares a group with a frame from the same
    video/centre in another fold.
  * All eligible frames are used (no subsampling). --max-frames > 0 would draw a
    fixed random subsample (seed 0); it is off by default.

Place in quantum-medical-research/endoscopy_guidance/ next to
export_cvc_failure_features.py and run from the repo root:

  python quantum-medical-research/endoscopy_guidance/export_polypgen_failure_features.py \
      --root dataset/polypgen/PolypGen2021_MultiCenterData_v3 \
      --checkpoint models/unet_pretrained.pth \
      --out failure_detection/features_polypgen.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def polypgen_group(name: str) -> str:
    stem = Path(name).stem
    m = re.match(r"seq(\d+)_", stem)
    if m:
        return f"seq{int(m.group(1)):02d}"
    m = re.match(r"(C\d)_", stem)
    if m:
        return m.group(1)
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="Folder containing positive/images and positive/masks")
    ap.add_argument("--checkpoint", type=Path, default=Path("models/unet_pretrained.pth"))
    ap.add_argument("--max-frames", type=int, default=0, help="0 = use all eligible frames")
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--include-scales", choices=["bottleneck", "deep", "all"], default="deep")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("failure_detection/features_polypgen.csv"))
    args = ap.parse_args()

    import torch
    from encoder_embedding_failure_benchmark import heatmap_features, load_rgb, pooled_encoder_features
    from export_cvc_failure_features import (HM_NAMES, load_binary_mask_official, official_transform,
                                             uncertainty_scores)
    from export_cvc_predictions import INPUT_SIZE, device_from_name, load_model
    from PIL import Image

    if "cvc" in args.checkpoint.name.lower():
        sys.exit("Use the Kvasir-trained models/unet_pretrained.pth.")
    img_dir, mask_dir = args.root / "positive" / "images", args.root / "positive" / "masks"
    names = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    excluded = {"negative_in_name": 0, "missing_mask": 0, "empty_mask": 0, "empty_after_resize": 0}
    eligible = []
    for n in names:
        if "negative" in n.lower():
            excluded["negative_in_name"] += 1
            continue
        mp = mask_dir / n
        if not mp.exists():
            excluded["missing_mask"] += 1
            continue
        if np.asarray(Image.open(mp).convert("L")).max() <= 127:
            excluded["empty_mask"] += 1
            continue
        if not load_binary_mask_official(mp, (INPUT_SIZE[1], INPUT_SIZE[0])).any():
            excluded["empty_after_resize"] += 1
            continue
        eligible.append(n)
    print(f"{len(names)} positive images; eligible {len(eligible)}; excluded {excluded}")

    device = device_from_name(args.device)
    model = load_model(args.checkpoint, device)
    tf = official_transform()
    rows = []
    for i, n in enumerate(eligible):
        image = load_rgb(img_dir / n)
        x = tf(str(img_dir / n))
        x = (x.as_tensor() if hasattr(x, "as_tensor") else x).float().unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model.encoder(x)
            hm = torch.sigmoid(model.segmentation_head(model.decoder(feats))).squeeze().cpu().numpy()
        hm = np.ascontiguousarray(hm.T)
        pred = (hm >= args.threshold).astype(np.uint8)
        gt = load_binary_mask_official(mask_dir / n, (INPUT_SIZE[1], INPUT_SIZE[0]))
        p = pred.astype(bool)
        dice = 2.0 * (p & gt).sum() / (p.sum() + gt.sum())
        g = polypgen_group(n)
        row = dict(sample_id=n, group=g, dice=float(dice))
        row.update(uncertainty_scores(hm, pred))
        row.update({f"f_hm_{k}": v for k, v in zip(HM_NAMES, heatmap_features(hm, pred, image))})
        row.update({f"f_enc_{k:04d}": float(v) for k, v in enumerate(pooled_encoder_features(feats, args.include_scales))})
        rows.append(row)
        if (i + 1) % 250 == 0 or i + 1 == len(eligible):
            print(f"  {i + 1}/{len(eligible)}", flush=True)

    df = pd.DataFrame(rows)
    codes = {g: k for k, g in enumerate(sorted(df.group.unique()))}
    df.insert(1, "sequence_id", df.group.map(codes))  # integer group id used by the benchmark
    full_path = args.out.with_name(args.out.stem + "_all.csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)
    sub = df
    if args.max_frames > 0 and len(df) > args.max_frames:
        rng = np.random.default_rng(args.subsample_seed)
        sub = df.iloc[np.sort(rng.choice(len(df), args.max_frames, replace=False))].reset_index(drop=True)
    sub.to_csv(args.out, index=False)

    def summary(d, label):
        print(f"\n[{label}] frames {len(d)}, groups {d.group.nunique()}, mean Dice {d.dice.mean():.4f}, "
              f"failures Dice<0.5: {(d.dice < 0.5).sum()}, Dice<0.2: {(d.dice < 0.2).sum()}")
        print("  frames per group: " + ", ".join(f"{g}={c}" for g, c in d.group.value_counts().sort_index().items()))
    summary(df, f"all eligible -> {full_path}")
    summary(sub, f"benchmark set -> {args.out}")


if __name__ == "__main__":
    main()
