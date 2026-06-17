"""
heatmap_pretrained.py
Generates pred_mask.png and pred_heatmap.npy for validation or test images
using UNet with pretrained ResNet34 encoder (val Dice: 0.8893).

Output structure:
    output/
        000_pretrained/
            pred_mask.png
            pred_heatmap.npy
        001_pretrained/
        ...
        019_pretrained/

Usage:
    PYTHONPATH=. python3 models/heatmap_pretrained.py
    PYTHONPATH=. python3 models/heatmap_pretrained.py --split test --num-samples 100
"""

import argparse
import os
import torch
import numpy as np
import cv2
from pathlib import Path
import segmentation_models_pytorch as smp
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor

from config import MODELS_DIR, DATA_DIR, OUTPUT_DIR, get_device

device = get_device()

HEATMAP_THRESHOLD = 0.5

parser = argparse.ArgumentParser(
    description="Export pretrained UNet masks and heatmaps for evaluation."
)
parser.add_argument("--split", choices=["train", "val", "test"], default="val")
parser.add_argument("--num-samples", type=int, default=20)
parser.add_argument("--output-dir", default=None)
parser.add_argument(
    "--image-dir",
    type=Path,
    default=None,
    help="Optional image folder override for external/domain-shift datasets.",
)
args = parser.parse_args()

output_dir = OUTPUT_DIR if args.output_dir is None else Path(args.output_dir)

# ---------------------------------------------------------------------------
# Load pretrained UNet
# ---------------------------------------------------------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)
model.load_state_dict(torch.load(MODELS_DIR / "unet_pretrained.pth", map_location=device))
model = model.to(device)
model.eval()

transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((256, 256)),
    ToTensor()
])

# ---------------------------------------------------------------------------
# Collect split images
# ---------------------------------------------------------------------------
val_img_dir = args.image_dir if args.image_dir is not None else DATA_DIR / args.split / "images"

if not val_img_dir.exists():
    raise FileNotFoundError(f"Images not found at {val_img_dir}")

all_images = sorted([
    f for f in os.listdir(val_img_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])[:args.num_samples]

print(f"Generating pretrained outputs for {len(all_images)} images from {val_img_dir}...\n")

# ---------------------------------------------------------------------------
# Output goes to output/000_pretrained/, 001_pretrained/, etc.
# ---------------------------------------------------------------------------
for idx, img_name in enumerate(all_images):
    img_path = str(val_img_dir / img_name)

    try:
        img_tensor = transform(img_path).unsqueeze(0).to(device)

        with torch.no_grad():
            output   = model(img_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy().astype(np.float32)

        # MONAI's image-only loading path used here matches training, but its
        # spatial axes are transposed relative to PIL/cv2 image coordinates.
        # Save outputs in natural image coordinates for eval and visualization.
        pred_heatmap = prob_map.T
        pred_mask    = (prob_map >= HEATMAP_THRESHOLD).astype(np.uint8) * 255
        pred_mask    = pred_mask.T

        # Folder named NNN_pretrained
        sample_dir = output_dir / f"{idx:03d}_pretrained"
        sample_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(sample_dir / "pred_mask.png"), pred_mask)
        np.save(str(sample_dir / "pred_heatmap.npy"), pred_heatmap)
        (sample_dir / "source_image.txt").write_text(img_name, encoding="utf-8")

        print(f"[{idx:03d}] {img_name}  ->  {sample_dir.name}/")

    except Exception as e:
        print(f"  ERROR on {img_path}: {e}")

print(f"\nDone! Outputs saved to '{output_dir}'")
print(f"\nModel notes:")
print(f"  Architecture : UNet + ResNet34 encoder pretrained on ImageNet")
print(f"  Loss         : Dice Loss + Binary Cross Entropy")
print(f"  Val Dice     : 0.8893")
print(f"  Heatmap      : UNet sigmoid output (per-pixel polyp probability, float32 0-1)")
print(f"  Mask         : Thresholded at {HEATMAP_THRESHOLD} -> binary PNG (0=background, 255=polyp)")
