"""
data_loader.py
==============
Kvasir-SEG dataset loader and feature extractor for QML experiments.

The dataset can be downloaded from:
  https://datasets.simula.no/kvasir-seg/

Expected directory structure:
  data/
    Kvasir-SEG/
      images/   *.jpg
      masks/    *.jpg

Since QML is limited to a small number of qubits (n_qubits << classical feature dim),
we use a classical CNN (ResNet-18) as a frozen feature extractor and then apply PCA
to reduce to n_features = n_qubits.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# Binary classification label assignment
# Strategy: use mask coverage to split into
#   class 1 (polyp-rich)  : mask ratio >= threshold
#   class 0 (polyp-sparse): mask ratio <  threshold
# ─────────────────────────────────────────────
MASK_RATIO_THRESHOLD = 0.02   # ~2 % of pixels are polyp


class KvasirSEGDataset(Dataset):
    """Raw image + mask dataset."""

    def __init__(self, data_dir: str, img_size: int = 224):
        self.data_dir = Path(data_dir)
        self.img_dir  = self.data_dir / "images"
        self.msk_dir  = self.data_dir / "masks"

        self.filenames = sorted([
            f.stem for f in self.img_dir.glob("*.jpg")
        ])
        assert len(self.filenames) > 0, (
            f"No .jpg images found in {self.img_dir}. "
            "Please download Kvasir-SEG and place it at data/Kvasir-SEG/."
        )

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img = Image.open(self.img_dir / f"{name}.jpg").convert("RGB")
        msk = Image.open(self.msk_dir / f"{name}.jpg").convert("L")

        # Binary label from mask coverage
        msk_arr = np.array(msk, dtype=np.float32) / 255.0
        ratio   = msk_arr.mean()
        label   = 1 if ratio >= MASK_RATIO_THRESHOLD else 0

        return self.transform(img), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────
# CNN feature extractor  (ResNet-18, no head)
# ─────────────────────────────────────────────

def build_feature_extractor(device: torch.device) -> nn.Module:
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Remove final FC layer → outputs 512-dim vector
    extractor = nn.Sequential(*list(resnet.children())[:-1])
    extractor.eval()
    extractor.to(device)
    return extractor


def extract_features(
    data_dir: str,
    n_features: int = 4,
    max_samples: int = 200,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : np.ndarray, shape (N, n_features)  – PCA-reduced, normalised features
    y : np.ndarray, shape (N,)             – binary labels {0, 1}
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KvasirSEGDataset(data_dir)

    # Subsample for QML speed
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:max_samples].tolist()
    subset  = torch.utils.data.Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=batch_size, shuffle=False)

    extractor = build_feature_extractor(device)
    feats, labels = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out  = extractor(imgs).squeeze(-1).squeeze(-1)   # (B, 512)
            feats.append(out.cpu().numpy())
            labels.append(lbls.numpy())

    X_raw = np.concatenate(feats,  axis=0)   # (N, 512)
    y     = np.concatenate(labels, axis=0)   # (N,)

    # Standardise → PCA → re-normalise to [-π, π] for angle encoding
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X_raw)

    pca = PCA(n_components=n_features, random_state=seed)
    X_pca = pca.fit_transform(X_std)

    explained = pca.explained_variance_ratio_.sum()
    print(f"[data_loader] PCA retained {explained*100:.1f}% variance "
          f"({n_features} components from 512)")

    # Scale each feature to [-π, π]
    from sklearn.preprocessing import MinMaxScaler
    mms   = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_out = mms.fit_transform(X_pca)

    return X_out.astype(np.float64), y.astype(int)
