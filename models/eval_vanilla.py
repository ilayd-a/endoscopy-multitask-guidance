
from pathlib import Path
import torch
from monai.networks.nets import UNet
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, Resized, ToTensord,
)

from config import DATA_DIR, MODELS_DIR, get_device
from model_metrics import MetricTracker


class Dataset(Dataset):
    def __init__(self, root_dir, split, transform):
        self.img_dir = Path(root_dir) / split / "images"
        self.mask_dir = Path(root_dir) / split / "masks"
        self.transform = transform
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Split '{split}' not found in {root_dir}")
        self.images = sorted(f.name for f in self.img_dir.iterdir() if f.name != ".DS_Store")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        data = self.transform({
            "image": str(self.img_dir / name),
            "mask": str(self.mask_dir / name),
        })
        mask = data["mask"]
        if mask.shape[0] == 3:
            mask = mask.mean(dim=0, keepdim=True)
        mask = (mask > 0.5).float()
        return data["image"], mask


eval_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=["mask"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image", "mask"], spatial_size=(256, 256)),
    ToTensord(keys=["image", "mask"]),
])

device = get_device()
print(f"Using device: {device}")

test_ds = Dataset(root_dir=DATA_DIR, split="test", transform=eval_transforms)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
print(f"Test set: {len(test_ds)} images")


model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

ckpt_path = MODELS_DIR / "unet_model.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

tracker = MetricTracker()
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        probs = torch.sigmoid(model(images))
        tracker.update(probs, masks)

print("\n=== Test set results: unet_model.pth ===")
tracker.report()
