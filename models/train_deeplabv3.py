"""
train_deeplabv3.py
DeepLabV3 with pretrained ResNet50 encoder (ImageNet weights)
for polyp segmentation on Kvasir-SEG.

Usage:
    PYTHONPATH=. python3 models/train_deeplabv3.py

Requirements:
    pip install segmentation-models-pytorch
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import segmentation_models_pytorch as smp
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, Resized, ToTensord,
    RandFlipd, RandRotated, RandGaussianNoised,
    RandAdjustContrastd, RandGaussianSmoothd
)

from config import DATA_DIR, MODELS_DIR, get_device

# ---------------------------------------------------------------------------
# Dataset
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
        img_path = str(self.img_dir / img_name)
        mask_path = str(self.mask_dir / img_name)

        data = {"image": img_path, "mask": mask_path}
        if self.transform:
            data = self.transform(data)

        mask = data["mask"]
        if mask.shape[0] == 3:
            mask = mask.mean(dim=0, keepdim=True)
        mask = (mask > 0.5).float()

        return data["image"], mask


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=["mask"],  a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image", "mask"], spatial_size=(256, 256)),
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
    RandRotated(keys=["image", "mask"], range_x=15, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
    RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
    ToTensord(keys=["image", "mask"])
])

val_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=["mask"],  a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image", "mask"], spatial_size=(256, 256)),
    ToTensord(keys=["image", "mask"])
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
# Model — DeepLabV3 with pretrained ResNet50 encoder
# ---------------------------------------------------------------------------
device = get_device()
print(f"Using device: {device}")

model = smp.DeepLabV3(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    encoder_output_stride=16,
    in_channels=3,
    classes=1,
    activation=None,
)
model = model.to(device)

# ---------------------------------------------------------------------------
# Loss / optimizer
# ---------------------------------------------------------------------------
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss  = nn.BCEWithLogitsLoss()

def criterion(outputs, masks):
    return dice_loss(outputs, masks) + bce_loss(outputs, masks)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)
num_epochs = 40

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
best_val_dice = 0.0

for epoch in range(num_epochs):
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

    train_loss = running_loss / total

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

    val_loss = val_loss / val_total
    val_dice = np.mean(dice_scores)
    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1:02d}: "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Dice {val_dice:.4f}"
    )

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        save_path = MODELS_DIR / "deeplabv3_pretrained.pth"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  Best model saved -> {save_path} (val dice: {val_dice:.4f})")

print(f"\nTraining complete. Best val Dice: {best_val_dice:.4f}")
print("\nModel notes for paper:")
print("  Architecture : DeepLabV3 with ResNet50 encoder pretrained on ImageNet")
print("  Loss         : Dice Loss + Binary Cross Entropy")
print("  Optimiser    : Adam (lr=1e-4) with ReduceLROnPlateau scheduler")
print("  Dataset      : Kvasir-SEG, 800 train / 100 val / 100 test")
print("  Input size   : 256x256")