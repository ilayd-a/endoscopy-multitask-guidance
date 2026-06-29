"""
data_loader_ebtc.py
===================
Data loader for Endoscopic Bladder Tissue Classification (EBTC) dataset.

Directory structure expected:
    data/EBTC/
        HGC/  *.png   (High-Grade Cancer)
        LGC/  *.png   (Low-Grade Cancer)
        NST/  *.png   (Non-Suspicious Tissue)
        NTL/  *.png   (No Tumor Lesion)
        annotations.csv

Option A — Binary classification:
    class 1 (cancer)     : HGC + LGC   (1,116 images)
    class 0 (non-cancer) : NST + NTL   (  638 images)

Pipeline:
    PNG images (300-350px)
        -> ResNet-18 frozen feature extractor  (512-dim)
        -> StandardScaler + PCA                (-> n_features dim)
        -> MinMaxScaler [-pi, pi]              (angle encoding)
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample


# ─────────────────────────────────────────────────────────
# Label mapping  (Option A: binary)
# ─────────────────────────────────────────────────────────

BINARY_MAP = {
    "HGC": 1,   # cancer
    "LGC": 1,   # cancer
    "NST": 0,   # non-cancer
    "NTL": 0,   # non-cancer
}

CLASS_NAMES = {0: "non-cancer (NST+NTL)", 1: "cancer (HGC+LGC)"}


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────

class EBTCDataset(Dataset):
    """
    Loads all PNG images from HGC/ LGC/ NST/ NTL/ subdirectories.
    Returns (tensor_image, binary_label).
    """

    def __init__(self, data_dir: str, img_size: int = 224):
        self.samples: list[tuple[Path, int]] = []
        data_dir = Path(data_dir)

        for class_name, label in BINARY_MAP.items():
            class_dir = data_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected directory not found: {class_dir}\n"
                    "Please place the EBTC dataset at data/EBTC/ with subdirs "
                    "HGC/, LGC/, NST/, NTL/."
                )
            pngs = sorted(class_dir.glob("*.png"))
            for p in pngs:
                self.samples.append((p, label))

        assert len(self.samples) > 0, f"No PNG images found under {data_dir}"
        print(f"[EBTCDataset] Loaded {len(self.samples)} images from {data_dir}")

        # Class breakdown
        from collections import Counter
        cnt = Counter(lbl for _, lbl in self.samples)
        print(f"  class 0 (non-cancer): {cnt[0]}  |  class 1 (cancer): {cnt[1]}")

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────
# CNN feature extractor
# ─────────────────────────────────────────────────────────

def _build_extractor(device: torch.device) -> nn.Module:
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    extractor = nn.Sequential(*list(resnet.children())[:-1])   # drop FC → 512-dim
    extractor.eval().to(device)
    return extractor


# ─────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────

def extract_features(
    data_dir: str,
    n_features: int = 6,
    max_samples: int | None = 300,
    balance: bool = True,
    seed: int = 42,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : np.ndarray  shape (N, n_features)   angle-encoded features in [-π, π]
    y : np.ndarray  shape (N,)              binary labels {0, 1}
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EBTCDataset(data_dir)

    # Optional subsampling (keep class ratio before balancing)
    indices = list(range(len(dataset)))
    if max_samples and max_samples < len(dataset):
        rng     = np.random.default_rng(seed)
        indices = rng.permutation(indices)[:max_samples].tolist()

    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    extractor = _build_extractor(device)
    feats, labels = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            out = extractor(imgs.to(device)).squeeze(-1).squeeze(-1)  # (B, 512)
            feats.append(out.cpu().numpy())
            labels.append(lbls.numpy())

    X_raw = np.concatenate(feats,  axis=0)
    y     = np.concatenate(labels, axis=0)

    # Class balancing via oversampling
    if balance:
        idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
        n = max(len(idx0), len(idx1))
        if len(idx0) < n:
            idx0 = resample(idx0, replace=True, n_samples=n, random_state=seed)
        else:
            idx1 = resample(idx1, replace=True, n_samples=n, random_state=seed)
        all_idx = np.concatenate([idx0, idx1])
        np.random.default_rng(seed).shuffle(all_idx)
        X_raw, y = X_raw[all_idx], y[all_idx]
        print(f"[extract_features] After balancing: N={len(y)}, "
              f"class ratio={y.mean():.3f}")

    # Standardise → PCA → scale to [-π, π]
    X_std = StandardScaler().fit_transform(X_raw)

    pca = PCA(n_components=n_features, random_state=seed)
    X_pca = pca.fit_transform(X_std)
    print(f"[extract_features] PCA variance retained: "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% "
          f"({n_features} components from 512)")

    X_out = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit_transform(X_pca)
    return X_out.astype(np.float64), y.astype(int)


# ─────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/EBTC"
    X, y = extract_features(data_dir, n_features=6, max_samples=200)
    print(f"X shape: {X.shape},  y shape: {y.shape}")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
