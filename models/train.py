import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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
# Load BOTH image and mask through MONAI so paired spatial transforms
# are applied consistently.
# ---------------------------------------------------------------------------
class KvasirSegDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / "masks"
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
        img_name = self.images[idx]

        data = {
            "image": str(self.img_dir / img_name),
            "mask": str(self.mask_dir / img_name),
        }

        if self.transform:
            data = self.transform(data)

        image = data["image"].float()
        mask = data["mask"].float()

        # If mask somehow comes in as 3 channels, keep one channel only
        if mask.shape[0] > 1:
            mask = mask[:1]

        # Force binary mask
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
print(f"Loading dataset from: {DATA_DIR}")
train_ds = KvasirSegDataset(root_dir=DATA_DIR, split="train", transform=train_transforms)
val_ds = KvasirSegDataset(root_dir=DATA_DIR, split="val", transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

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
best_val_dice = 0.0

train_loss_history = []
val_loss_history = []
dice_score_history = []
epochs = [x for x in range(num_epochs)]
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

    dice_scores = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * images.size(0)
            val_total += images.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2 * intersection / (union + 1e-8)).mean().item()
            dice_scores.append(dice)


    val_loss = val_loss / val_total if val_total > 0 else 0.0
    val_dice = np.mean(dice_scores)
    scheduler.step()

    print(f"Epoch {epoch+1:02d}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Dice {val_dice:.4f}")

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    dice_score_history.append(val_dice)

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        best_val_loss = val_loss
        save_path = MODELS_DIR / "unet_model.pth"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  ✅ Best model saved → {save_path}  (val loss: {val_loss:.4f})")

print(f"\nTraining complete. Best val dice: {best_val_dice:.4f}")


plt.plot(epochs, train_loss_history, label="Train Loss")
plt.plot(epochs, val_loss_history, label="Validation Loss")
plt.plot(epochs, dice_score_history, label="Dice Score")
plt.xlabel("Epochs")
plt.ylabel("Loss / Dice Score")
plt.legend()
os.makedirs("results", exist_ok=True)
plt.savefig("results/kvasir_results.png")
plt.show()