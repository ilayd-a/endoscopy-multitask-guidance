import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, Resized, ToTensord,
    RandFlipd, RandRotated, RandGaussianNoised,
    RandAdjustContrastd, RandGaussianSmoothd
)

from config import DATA_DIR, MODELS_DIR, get_device

# ---------------------------------------------------------------------------
# Dataset
# Mask is loaded as grayscale and given a channel dim manually so MONAI
# doesn't try to treat it as RGB (which caused the shape mismatch error)
# ---------------------------------------------------------------------------
class KvasirSegDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir  = Path(root_dir)
        self.img_dir   = self.root_dir / split / "images"
        self.mask_dir  = self.root_dir / split / "masks"
        self.transform = transform

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Split '{split}' not found in {root_dir}")

        self.images = sorted([
            f.name for f in self.img_dir.iterdir()
            if f.name != ".DS_Store"
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        img_path  = str(self.img_dir  / img_name)
        mask_path = str(self.mask_dir / img_name)

        # Load mask as grayscale with cv2 and add channel dim → (1, H, W)
        # This avoids MONAI loading it as 3-channel RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32)     # normalise to 0-1
        mask = np.expand_dims(mask, axis=0)           # (1, H, W)

        # MONAI handles the image via transform pipeline
        data = {"image": img_path}
        if self.transform:
            data = self.transform(data)

        img_tensor  = data["image"]
        mask_tensor = torch.from_numpy(mask)

        # Resize mask to match image size (256x256)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0), size=(256, 256), mode="nearest"
        ).squeeze(0)

        return img_tensor, mask_tensor


# ---------------------------------------------------------------------------
# MONAI Transforms — image only (mask handled in __getitem__)
# ---------------------------------------------------------------------------
train_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=0, a_max=255,
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    Resized(keys=["image"], spatial_size=(256, 256)),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
    RandRotated(keys=["image"], range_x=15, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
    RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
    ToTensord(keys=["image"])
])

val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=0, a_max=255,
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    Resized(keys=["image"], spatial_size=(256, 256)),
    ToTensord(keys=["image"])
])

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
print(f"Loading dataset from: {DATA_DIR}")
train_ds = KvasirSegDataset(root_dir=DATA_DIR, split="train", transform=train_transforms)
val_ds   = KvasirSegDataset(root_dir=DATA_DIR, split="val",   transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

print(f"  Train: {len(train_ds)} images")
print(f"  Val:   {len(val_ds)} images")

# ---------------------------------------------------------------------------
# Model — MONAI UNet for binary segmentation
# ---------------------------------------------------------------------------
device = get_device()
print(f"Using device: {device}")

model = UNet(
    spatial_dims=2,
    in_channels=3,       # RGB input
    out_channels=1,      # single channel: polyp probability per pixel
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
    running_loss = 0
    total = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    train_loss = running_loss / total

    # --- Validate ---
    model.eval()
    val_loss  = 0
    val_total = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss  += loss.item() * images.size(0)
            val_total += images.size(0)

    val_loss = val_loss / val_total if val_total > 0 else 0
    scheduler.step()

    print(f"Epoch {epoch+1:02d}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = MODELS_DIR / "unet_model.pth"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  ✅ Best model saved → {save_path}  (val loss: {val_loss:.4f})")

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")