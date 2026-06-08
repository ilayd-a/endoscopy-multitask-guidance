"""
train_deeplabv3.py
DeepLabV3 with pretrained ResNet50 encoder trained on CVC-ClinicDB.

Usage:
    PYTHONPATH=. python models/train_deeplabv3.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    RandGaussianNoised, RandAdjustContrastd, RandGaussianSmoothd
)

from config import DATA_DIR, MODELS_DIR, get_device

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CVC_DIR = BASE_DIR / "archive"
IMG_DIR = CVC_DIR / "PNG" / "Original"
MASK_DIR = CVC_DIR / "PNG" / "Ground Truth"
META     = CVC_DIR / "metadata.csv"

INPUT_SIZE = (256, 256)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CVCDataset(Dataset):
    def __init__(self, image_names, transform=None):
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = str(IMG_DIR / name)
        mask_path = str(MASK_DIR / name)

        # Load mask with cv2 to avoid channel issues
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        data = {"image": img_path}
        if self.transform:
            data = self.transform(data)

        return data["image"], mask

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
train_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image"], spatial_size=INPUT_SIZE),
    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
    RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
    ToTensord(keys=["image"])
])

val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image"], spatial_size=INPUT_SIZE),
    ToTensord(keys=["image"])
])

# ---------------------------------------------------------------------------
# Sequence-based split
# ---------------------------------------------------------------------------
df = pd.read_csv(META)

train_imgs = df[df.sequence_id <= 23]["png_image_path"].apply(lambda x: Path(x).name).tolist()
val_imgs   = df[(df.sequence_id >= 24) & (df.sequence_id <= 26)]["png_image_path"].apply(lambda x: Path(x).name).tolist()
test_imgs  = df[df.sequence_id >= 27]["png_image_path"].apply(lambda x: Path(x).name).tolist()

print(f"Sequence-based split: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")
print(f"Input size: {INPUT_SIZE}")

train_ds = CVCDataset(train_imgs, transform=train_transforms)
val_ds   = CVCDataset(val_imgs, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

# ---------------------------------------------------------------------------
# Model
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
bce_loss = nn.BCEWithLogitsLoss()

def criterion(outputs, masks):
    return dice_loss(outputs, masks) + bce_loss(outputs, masks)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

num_epochs = 20

train_loss_history = []
val_loss_history = []
val_dice_history = []

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

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    val_dice_history.append(val_dice)

    print(
        f"Epoch {epoch+1:02d}: "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Dice {val_dice:.4f}"
    )

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        save_path = MODELS_DIR / "deeplabv3_cvc.pth"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  Best model saved -> {save_path} (val dice: {val_dice:.4f})")


avg_train_loss = np.mean(train_loss_history)
avg_val_loss = np.mean(val_loss_history)
avg_val_dice = np.mean(val_dice_history)

print(f"\nAverage Train Loss across epochs: {avg_train_loss:.4f}")
print(f"Average Val Loss across epochs:   {avg_val_loss:.4f}")
print(f"Average Val Dice across epochs:   {avg_val_dice:.4f}")

print(f"\nTraining complete. Best val Dice: {best_val_dice:.4f}")
print("\nModel notes for paper:")
print("  Architecture : DeepLabV3 with ResNet50 encoder pretrained on ImageNet")
print("  Loss         : Dice Loss + Binary Cross Entropy")
print("  Optimiser    : Adam (lr=1e-4) with ReduceLROnPlateau scheduler")
print("  Dataset      : CVC-ClinicDB, sequence-based split")
print(f"                 Train: seq 1-23 ({len(train_imgs)} frames)")
print(f"                 Val:   seq 24-26 ({len(val_imgs)} frames)")
print(f"                 Test:  seq 27-29 ({len(test_imgs)} frames)")
print(f"  Input size   : {INPUT_SIZE}")


epochs = np.arange(1, num_epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss_history, marker="o", label="Train Loss")
plt.plot(epochs, val_loss_history, marker="o", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss and Val Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = MODELS_DIR / "deeplabv3_cvc_loss_curve.png"
plt.savefig(loss_plot_path, dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, val_dice_history, marker="o", label="Val Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Val Dice vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
dice_plot_path = MODELS_DIR / "deeplabv3_cvc_dice_curve.png"
plt.savefig(dice_plot_path, dpi=300)
plt.show()

print(f"Saved loss plot to: {loss_plot_path}")
print(f"Saved dice plot to: {dice_plot_path}")