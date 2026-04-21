"""
heatmap_deeplabv3.py
Generates pred_mask.png and pred_heatmap.npy for 20 validation images
using DeepLabV3 with pretrained ResNet50 encoder.

Output structure:
    output/
        000_deeplabv3/
            pred_mask.png
            pred_heatmap.npy
        001_deeplabv3/
        ...
        019_deeplabv3/

Usage:
    PYTHONPATH=. python3 models/heatmap_deeplabv3.py
"""

import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    ScaleIntensityRange, Resize, ToTensor
)

from config import MODELS_DIR, DATA_DIR, OUTPUT_DIR, get_device

device = get_device()

HEATMAP_THRESHOLD = 0.5
NUM_SAMPLES = 20

# ---------------------------------------------------------------------------
# Load trained DeepLabV3
# ---------------------------------------------------------------------------
model = smp.DeepLabV3(
    encoder_name="resnet50",
    encoder_weights=None,
    encoder_output_stride=16,
    in_channels=3,
    classes=1,
    activation=None,
)

model.load_state_dict(
    torch.load(MODELS_DIR / "deeplabv3_pretrained.pth", map_location=device)
)
model = model.to(device)
model.eval()

transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resize((256, 256)),
    ToTensor()
])

# ---------------------------------------------------------------------------
# Collect val images
# ---------------------------------------------------------------------------
val_img_dir = DATA_DIR / "val" / "images"

if not val_img_dir.exists():
    raise FileNotFoundError(f"Val images not found at {val_img_dir}")

all_images = sorted([
    f for f in os.listdir(val_img_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])[:NUM_SAMPLES]

print(f"Generating DeepLabV3 outputs for {len(all_images)} validation images...\n")

# ---------------------------------------------------------------------------
# Output goes to output/000_deeplabv3/, 001_deeplabv3/, etc.
# ---------------------------------------------------------------------------
for idx, img_name in enumerate(all_images):
    img_path = str(val_img_dir / img_name)

    try:
        img_tensor = transform(img_path).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy().astype(np.float32)

        pred_heatmap = prob_map
        pred_mask = (prob_map >= HEATMAP_THRESHOLD).astype(np.uint8) * 255

        sample_dir = OUTPUT_DIR / f"{idx:03d}_deeplabv3"
        sample_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(sample_dir / "pred_mask.png"), pred_mask)
        np.save(str(sample_dir / "pred_heatmap.npy"), pred_heatmap)

        print(f"[{idx:03d}] {img_name}  ->  {sample_dir.name}/")

    except Exception as e:
        print(f"  ERROR on {img_path}: {e}")

print(f"\nDone! Outputs saved to '{OUTPUT_DIR}'")
print("\nModel notes:")
print("  Architecture : DeepLabV3 + ResNet50 encoder pretrained on ImageNet")
print("  Loss         : Dice Loss + Binary Cross Entropy")
print(f"  Heatmap      : DeepLabV3 sigmoid output (per-pixel polyp probability, float32 0-1)")
print(f"  Mask         : Thresholded at {HEATMAP_THRESHOLD} -> binary PNG (0=background, 255=polyp)")