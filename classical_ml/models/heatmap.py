import os
import torch
import numpy as np
import cv2
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor

from config import MODELS_DIR, DATA_DIR, OUTPUT_DIR, get_device

device = get_device()

HEATMAP_THRESHOLD = 0.5   # probability threshold for binary mask
OUTPUT_SIZE       = (256, 256)
NUM_SAMPLES       = 20    # minimum 20 test images as required

# ---------------------------------------------------------------------------
# Load trained UNet
# ---------------------------------------------------------------------------
model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load(MODELS_DIR / "unet_model.pth", map_location=device))
model.eval()

transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((256, 256)),
    ToTensor()
])

# ---------------------------------------------------------------------------
# Use VAL images (as required — "20 test images")
# ---------------------------------------------------------------------------
img_dir = DATA_DIR / "test" / "images"

if not img_dir.exists():
    raise FileNotFoundError(f"Val images not found at {img_dir}")

all_images = sorted([
    f for f in os.listdir(img_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])[:NUM_SAMPLES]   # take first 20

print(f"Generating outputs for {len(all_images)} validation images...\n")

# ---------------------------------------------------------------------------
# Generate outputs
# Output structure per sample:
#   output/
#       000/
#           pred_mask.png
#           pred_heatmap.npy
# ---------------------------------------------------------------------------
for idx, img_name in enumerate(all_images):
    img_path = str(img_dir / img_name)

    try:
        # --- Load & transform ---
        img_tensor = transform(img_path).unsqueeze(0).to(device)

        # --- UNet prediction ---
        with torch.no_grad():
            output   = model(img_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy().astype(np.float32)

        # --- pred_heatmap: raw probability map (float32, 256x256) ---
        pred_heatmap = prob_map

        # --- pred_mask: binary (uint8, 256x256, values 0 or 255 for PNG) ---
        pred_mask = (prob_map >= HEATMAP_THRESHOLD).astype(np.uint8) * 255

        # --- pred_mask: heatmap ---
        heatmap_uint8 = (pred_heatmap * 255).clip(0, 255).astype(np.uint8)
        pred_heatmap_png = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # --- Create per-sample output folder: output/000/ ---
        sample_dir = OUTPUT_DIR / f"{idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # --- Save pred_mask.png ---
        cv2.imwrite(str(sample_dir / "pred_mask.png"), pred_mask)

        # --- Save pred_heatmap.npy ---
        np.save(str(sample_dir / "pred_heatmap.npy"), pred_heatmap)

        cv2.imwrite(str(sample_dir / "pred_heatmap.png"), pred_heatmap_png)


        print(f"[{idx:03d}] {img_name} -> {sample_dir}/")

    except Exception as e:
        print(f"  ERROR on {img_path}: {e}")

print(f"\nDone! Outputs saved to '{OUTPUT_DIR}/'")
print(f"\nNotes for eval team:")
print(f"  Loss:    DiceLoss (MONAI) — directly optimises pixel overlap with ground truth mask")
print(f"  Heatmap: UNet sigmoid output (raw per-pixel polyp probability, float32 0-1)")
print(f"  Mask:    Heatmap thresholded at {HEATMAP_THRESHOLD} → binary PNG (0=background, 255=polyp)")
