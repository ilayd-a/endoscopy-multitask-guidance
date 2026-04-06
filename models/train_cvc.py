import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import cv2
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, Resized, ToTensord,
    RandFlipd, RandRotated, RandGaussianNoised,
    RandAdjustContrastd, RandGaussianSmoothd
)

from config import DATA_DIR, MODELS_DIR, get_device

# Paths

CVC_DIR  = DATA_DIR.parent / "CVC-ClinicDB"
IMG_DIR  = CVC_DIR / "PNG" / "Original"
MASK_DIR = CVC_DIR / "PNG" / "Ground Truth"
META     = CVC_DIR / "metadata.csv"

INPUT_SIZE = (256, 256)



# ---------------------------------------------------------------------------
# Dataset
# Load BOTH image and mask through MONAI so paired spatial transforms
# are applied consistently.
# ---------------------------------------------------------------------------
class CVCDataset(Dataset):
    def __init__(self, image_names, transform=None):
        self.image_names = image_names
        self.transform   = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name      = self.image_names[idx]
        img_path  = str(IMG_DIR  / name)
        mask_path = str(MASK_DIR / name)

        data = {"image": img_path, "mask": mask_path}
        if self.transform:
            data = self.transform(data)

        image = data["image"].float()
        mask  = data["mask"].float()

        if mask.shape[0] > 1:
            mask = mask[:1]

        mask = (mask > 0.5).float()

        return image, mask




# ---------------------------------------------------------------------------
# MONAI Transforms
# Spatial transforms apply to BOTH image and mask.
# Intensity transforms apply to IMAGE only.
# ---------------------------------------------------------------------------
train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=0, a_max=255,
        b_min=0.0, b_max=1.0,
        clip=True
    ),

    Resized(
        keys=["image", "mask"],
        spatial_size=(256, 256),
        mode=("bilinear", "nearest")
    ),

    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),

    RandRotated(
        keys=["image", "mask"],
        range_x=np.pi / 12,   # 15 degrees
        prob=0.5,
        mode=("bilinear", "nearest"),
        padding_mode="zeros"
    ),

    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
    RandGaussianSmoothd(
        keys=["image"],
        prob=0.2,
        sigma_x=(0.25, 1.5),
        sigma_y=(0.25, 1.5)
    ),

    ToTensord(keys=["image", "mask"])
])

val_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=0, a_max=255,
        b_min=0.0, b_max=1.0,
        clip=True
    ),

    Resized(
        keys=["image", "mask"],
        spatial_size=(256, 256),
        mode=("bilinear", "nearest")
    ),

    ToTensord(keys=["image", "mask"])
])


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
df = pd.read_csv(META)
train_imgs = df[df.sequence_id <= 23]["png_image_path"].apply(lambda x: Path(x).name).tolist()
val_imgs   = df[(df.sequence_id >= 24) & (df.sequence_id <= 26)]["png_image_path"].apply(lambda x: Path(x).name).tolist()
test_imgs  = df[df.sequence_id >= 27]["png_image_path"].apply(lambda x: Path(x).name).tolist()

print(f"Sequence-based split: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")
print(f"Input size: {INPUT_SIZE}")

print(f"Loading dataset from: {DATA_DIR}")

train_ds = CVCDataset(train_imgs, transform=train_transforms)
val_ds   = CVCDataset(val_imgs,   transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

print(f"  Train: {len(train_ds)} images")
print(f"  Val:   {len(val_ds)} images")

# Optional sanity check
sample_images, sample_masks = next(iter(train_loader))
print("Sample batch image shape:", sample_images.shape)  # [B, 3, 256, 256]
print("Sample batch mask shape: ", sample_masks.shape)   # [B, 1, 256, 256]


# ---------------------------------------------------------------------------
# Model — MONAI UNet for binary segmentation
# ---------------------------------------------------------------------------
device = get_device()
print(f"Using device: {device}")

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

criterion = DiceLoss(sigmoid=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
num_epochs = 20


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
best_val_loss = float("inf")

for epoch in range(num_epochs):
    # --- Train ---
    model.train()
    running_loss = 0.0
    total = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    train_loss = running_loss / total if total > 0 else 0.0

    # --- Validate ---
    model.eval()
    val_loss = 0.0
    val_total = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * images.size(0)
            val_total += images.size(0)

    val_loss = val_loss / val_total if val_total > 0 else 0.0
    scheduler.step()

    print(f"Epoch {epoch+1:02d}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = MODELS_DIR / "unet_model_cvc.pth"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  ✅ Best model saved → {save_path}  (val loss: {val_loss:.4f})")

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

