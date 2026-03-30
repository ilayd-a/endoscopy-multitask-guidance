"""
heatmap_CVCClinicDB.py
Generates pred_mask.png and pred_heatmap.npy for 20 validation images.
Uses sequence-based val split (sequences 24-26).

Output structure:
    output/
        000_CVCClinicDB/
            original_XXX.png
            overlay_XXX.png
            pred_mask.png
            pred_heatmap.npy

Usage:
    PYTHONPATH=. python3 models/heatmap_CVCClinicDB.py
"""

import os
import torch
import numpy as np
import cv2
import pandas as pd
import segmentation_models_pytorch as smp
from pathlib import Path
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor

from config import MODELS_DIR, DATA_DIR, OUTPUT_DIR, get_device

device = get_device()

HEATMAP_THRESHOLD = 0.3   # tuned from threshold sweep
INPUT_SIZE        = (256, 256)
OUTPUT_SIZE       = (256, 256)  # eval team expects 256x256
NUM_SAMPLES       = 20

CVC_DIR  = DATA_DIR.parent / "CVC-ClinicDB"
IMG_DIR  = CVC_DIR / "PNG" / "Original"
META     = CVC_DIR / "metadata.csv"

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)
model.load_state_dict(torch.load(MODELS_DIR / "unet_cvc.pth", map_location=device))
model = model.to(device)
model.eval()

transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize(INPUT_SIZE),
    ToTensor()
])

# ---------------------------------------------------------------------------
# Val images from sequences 24-26
# ---------------------------------------------------------------------------
df = pd.read_csv(META)
val_imgs = df[(df.sequence_id >= 24) & (df.sequence_id <= 26)]["png_image_path"].apply(lambda x: Path(x).name).tolist()
val_imgs = val_imgs[:NUM_SAMPLES]

print(f"Generating CVC outputs for {len(val_imgs)} validation images (seq 24-26)...\n")

# ---------------------------------------------------------------------------
# Generate outputs
# ---------------------------------------------------------------------------
for idx, img_name in enumerate(val_imgs):
    img_path = str(IMG_DIR / img_name)

    try:
        img_tensor = transform(img_path).unsqueeze(0).to(device)

        with torch.no_grad():
            output   = model(img_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy().astype(np.float32)

        # Resize prob_map to eval team format (256x256)
        prob_map_256 = cv2.resize(prob_map, OUTPUT_SIZE).astype(np.float32)
        pred_heatmap = prob_map_256
        pred_mask    = (prob_map_256 >= HEATMAP_THRESHOLD).astype(np.uint8) * 255

        sample_dir = OUTPUT_DIR / f"{idx:03d}_CVCClinicDB"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Required outputs (256x256)
        cv2.imwrite(str(sample_dir / "pred_mask.png"), pred_mask)
        np.save(str(sample_dir / "pred_heatmap.npy"), pred_heatmap)

        # Original image resized to 256x256
        original = cv2.imread(img_path)
        original_256 = cv2.resize(original, OUTPUT_SIZE)
        cv2.imwrite(str(sample_dir / f"original_{img_name}"), original_256)

        # Overlay at 256x256
        heatmap_color = cv2.applyColorMap(np.uint8(pred_heatmap * 255), cv2.COLORMAP_JET)
        overlay = (0.4 * heatmap_color + 0.6 * original_256).astype(np.uint8)
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(str(sample_dir / f"overlay_{img_name}"), overlay)

        print(f"[{idx:03d}] {img_name}  ->  polyp pixels: {(pred_mask == 255).sum()}")

    except Exception as e:
        print(f"  ERROR on {img_path}: {e}")

print(f"\nDone! Outputs saved to '{OUTPUT_DIR}'")
print(f"\nModel notes:")
print(f"  Architecture : UNet + ResNet34 encoder pretrained on ImageNet")
print(f"  Dataset      : CVC-ClinicDB, sequence-based split (val: seq 24-26)")
print(f"  Loss         : Dice Loss + BCE")
print(f"  Heatmap      : UNet sigmoid output (per-pixel polyp probability, float32 0-1)")
print(f"  Mask         : Thresholded at {HEATMAP_THRESHOLD} -> binary PNG (0=background, 255=polyp)")
