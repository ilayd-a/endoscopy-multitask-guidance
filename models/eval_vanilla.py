
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, Resized, ToTensord,
)

from config import DATA_DIR, MODELS_DIR, get_device
from model_metrics import MetricTracker

CVC_DIR  = DATA_DIR.parent / "CVC-ClinicDB"
IMG_DIR  = CVC_DIR / "PNG" / "Original"
MASK_DIR = CVC_DIR / "PNG" / "Ground Truth"
META     = CVC_DIR / "metadata.csv"

class Dataset(Dataset):
    def __init__(self, image_names, transform):
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        data = self.transform({
            "image": str(IMG_DIR / name),
            "mask": str(MASK_DIR / name),
        })
        image = data["image"].float()
        mask = data["mask"].float()
        if mask.shape[0] > 1:
            mask = mask[:1]
        mask = (mask > 0.5).float()
        return image, mask


eval_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=["mask"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image", "mask"], spatial_size=(256, 256)),
    ToTensord(keys=["image", "mask"]),
])

df = pd.read_csv(META)
test_imgs = df[df.sequence_id >= 27]["png_image_path"].apply(lambda x: Path(x).name).tolist()
print(f"Test set: {len(test_imgs)} images")

device = get_device()
print(f"Using device: {device}")

test_ds = Dataset(test_imgs, transform=eval_transforms)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

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
