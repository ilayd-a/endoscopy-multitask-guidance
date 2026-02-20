import ssl

from config import DATA_DIR, get_device

ssl._create_default_https_context = ssl._create_unverified_context
import os
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityRanged, RandGaussianNoised, RandAdjustContrastd,
    RandGaussianSmoothd, Resized, RandFlipd, RandRotated, ToTensord
)
from monai.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm


dataset_path = DATA_DIR/"EndoscopicBladderTissue"

class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
class_folders.sort()

print(f"Found {len(class_folders)} classes: {class_folders}")

data_list = []
for idx, cls in enumerate(class_folders):
    folder_path = os.path.join(dataset_path, cls)
    
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    for img_name in img_files:
        data_list.append({"image": os.path.join(folder_path, img_name), "label": idx})

print(f"Total images: {len(data_list)}")

train_files, val_files = train_test_split(data_list, test_size=0.2, stratify=[d["label"] for d in data_list])


train_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    
    ScaleIntensityRanged(
        keys=["image"],
        a_min=0,      
        a_max=255,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),
    
    Resized(keys=["image"], spatial_size=(224, 224)),
    
    
    RandGaussianNoised(
        keys=["image"],
        prob=0.1,
        mean=0.0,
        std=0.01
    ),
    
    RandAdjustContrastd(
        keys=["image"],
        prob=0.3,
        gamma=(0.7, 1.3)
    ),
    
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandRotated(keys=["image"], range_x=15, prob=0.5),
    
    RandGaussianSmoothd(
        keys=["image"],
        prob=0.2,
        sigma_x=(0.25, 1.5),
        sigma_y=(0.25, 1.5)
    ),
    
    ToTensord(keys=["image", "label"])
])

val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=0,
        a_max=255,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),
    Resized(keys=["image"], spatial_size=(224, 224)),
    ToTensord(keys=["image", "label"])
])


train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, num_workers=0)


device = get_device()
print(f"Using device: {device}")

model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(class_folders))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    train_acc = correct / total
    train_loss = running_loss / total
    
    
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total if val_total > 0 else 0
    
    print(f"Epoch {epoch+1}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "bladder_model.pth")
print("✅ Model saved as bladder_model.pth")