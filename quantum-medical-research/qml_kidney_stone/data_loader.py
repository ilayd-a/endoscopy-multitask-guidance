"""
data_loader.py — Kidney Stone (merged COCO export) data loading + feature extraction

Pipeline (mirrors the Kvasir-SEG / EBTC QML benchmark pipeline):
    image -> frozen ResNet-18 (512-dim) -> StandardScaler -> PCA(n_qubits)
          -> MinMaxScaler([-pi, pi]) -> angle-encoding-ready vectors

Label design:
    "Stone Present"  : image_id has >= 1 COCO annotation (polygon mask)
    "Stone Absent"   : image_id has 0 annotations

Important note on this dataset:
    The dataset now combines THREE source videos:
        - video_A       (294 frames, original)
        - video_B_seg1  (300 frames, new1)
        - video_B_seg2  (300 frames, new2)
    Frame numbers reset to 1 within each source, so a single global
    "sort by frame number" split is no longer valid across the merged set.
    Instead, the train/test split is precomputed PER SOURCE VIDEO at merge
    time (see merge_and_split.py) using contiguous temporal blocks, and
    stored directly in merged_instances.json as a "split" field on each
    image. This loader reads that field rather than recomputing a split.
"""

import json
import os
import random
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


# --------------------------------------------------------------------------- #
# 1. COCO annotation parsing -> labels + split + source metadata
# --------------------------------------------------------------------------- #

def load_records_from_merged_coco(coco_json_path):
    """
    Returns a list of dicts, one per image:
        {
            "file_name": "kidney_stone/ezgif-frame-001.jpg",   # includes subfolder
            "label": 0 or 1,
            "source_video": "video_A" / "video_B_seg1" / "video_B_seg2",
            "split": "train" / "test",
        }

    Requires merged_instances.json produced by merge_and_split.py, which already
    embeds "source_video" and "split" per image.
    """
    with open(coco_json_path) as f:
        data = json.load(f)

    img_to_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    records = []
    for img in data["images"]:
        records.append({
            "file_name": img["file_name"],
            "label": 1 if img["id"] in img_to_anns else 0,
            "source_video": img.get("source_video", "unknown"),
            "split": img.get("split", "unknown"),
        })

    return records


# --------------------------------------------------------------------------- #
# 2. Balanced sampling (applied WITHIN a split, so train/test boundaries
#    set by merge_and_split.py are never crossed)
# --------------------------------------------------------------------------- #

def balanced_sample(records: list, max_samples=100, seed=42, present_absent_ratio=1.0):
    """
    Build a (roughly) balanced subset from a list of records (all assumed to
    belong to the same split, e.g. all "train" or all "test").

    present_absent_ratio: how many present samples per absent sample.
        1.0 -> as close to 50/50 as the absent-class size allows.
    """
    random.seed(seed)

    present = [r for r in records if r["label"] == 1]
    absent = [r for r in records if r["label"] == 0]

    random.shuffle(present)
    random.shuffle(absent)

    n_absent = len(absent)
    n_present_target = min(len(present), int(n_absent * present_absent_ratio))

    total_target = n_absent + n_present_target
    if total_target > max_samples:
        n_absent_use = max_samples // 2
        n_present_use = max_samples - n_absent_use
        n_absent_use = min(n_absent_use, n_absent)
        n_present_use = min(n_present_use, n_present_target)
    else:
        n_absent_use = n_absent
        n_present_use = n_present_target

    chosen = absent[:n_absent_use] + present[:n_present_use]
    random.shuffle(chosen)

    return chosen


# --------------------------------------------------------------------------- #
# 3. Frozen ResNet-18 feature extractor
# --------------------------------------------------------------------------- #

class ResNet18FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()  # strip classification head -> 512-dim output
        backbone.eval()
        self.model = backbone.to(self.device)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image_paths, batch_size=16, verbose=True):
        feats = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            imgs = [self.transform(Image.open(p).convert("RGB")) for p in batch_paths]
            batch = torch.stack(imgs).to(self.device)
            out = self.model(batch)  # (B, 512)
            feats.append(out.cpu().numpy())
            if verbose:
                print(f"  [feature extraction] {min(i + batch_size, len(image_paths))}/{len(image_paths)}")
        return np.concatenate(feats, axis=0)


# --------------------------------------------------------------------------- #
# 4. Full loading pipeline (train + test, split-aware)
# --------------------------------------------------------------------------- #

def load_kidney_stone_dataset(
    merged_json_path,
    image_dir,
    n_qubits=4,
    max_train_samples=100,
    max_test_samples=40,
    seed=42,
    present_absent_ratio=1.0,
    verbose=True,
):
    """
    Loads the merged dataset, applies balanced sampling SEPARATELY within
    the train split and the test split (so the split boundary from
    merge_and_split.py is preserved), then fits ResNet-18 -> PCA -> MinMax
    scaling on train and applies the same fitted transforms to test.

    Returns a dict with:
        X_train, y_train, files_train,
        X_test,  y_test,  files_test,
        raw_512_train, raw_512_test,
        pca (fitted), scaler (fitted), mm_scaler (fitted)
    """
    all_records = load_records_from_merged_coco(merged_json_path)

    train_records = [r for r in all_records if r["split"] == "train"]
    test_records = [r for r in all_records if r["split"] == "test"]

    train_sample = balanced_sample(
        train_records, max_samples=max_train_samples, seed=seed,
        present_absent_ratio=present_absent_ratio,
    )
    test_sample = balanced_sample(
        test_records, max_samples=max_test_samples, seed=seed,
        present_absent_ratio=present_absent_ratio,
    )

    if verbose:
        for name, sample in [("train", train_sample), ("test", test_sample)]:
            n_pos = sum(r["label"] for r in sample)
            print(f"[data] {name}: {len(sample)} samples -> present={n_pos}, absent={len(sample) - n_pos}")

    def resolve_paths(sample):
        paths = [os.path.join(image_dir, r["file_name"]) for r in sample]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[:3]}")
        return paths

    train_paths = resolve_paths(train_sample)
    test_paths = resolve_paths(test_sample)

    extractor = ResNet18FeatureExtractor()
    raw_512_train = extractor.extract(train_paths, verbose=verbose)
    raw_512_test = extractor.extract(test_paths, verbose=verbose)

    # Fit StandardScaler + PCA + MinMaxScaler on TRAIN ONLY, then transform test
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(raw_512_train)
    X_test_std = scaler.transform(raw_512_test)

    pca = PCA(n_components=n_qubits, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    if verbose:
        print(f"[data] PCA explained variance ratio (n_qubits={n_qubits}): "
              f"{pca.explained_variance_ratio_.sum():.3f}")

    mm_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_quantum = mm_scaler.fit_transform(X_train_pca)
    X_test_quantum = mm_scaler.transform(X_test_pca)

    y_train = np.array([r["label"] for r in train_sample])
    y_test = np.array([r["label"] for r in test_sample])
    files_train = [r["file_name"] for r in train_sample]
    files_test = [r["file_name"] for r in test_sample]

    return {
        "X_train": X_train_quantum, "y_train": y_train, "files_train": files_train,
        "X_test": X_test_quantum, "y_test": y_test, "files_test": files_test,
        "raw_512_train": raw_512_train, "raw_512_test": raw_512_test,
        "pca": pca, "scaler": scaler, "mm_scaler": mm_scaler,
    }


if __name__ == "__main__":
    # Smoke test
    result = load_kidney_stone_dataset(
        merged_json_path="../training_data/merged_instances.json",
        image_dir="../images",
        n_qubits=4,
        max_train_samples=100,
        max_test_samples=40,
    )
    print("X_train shape:", result["X_train"].shape)
    print("X_test shape: ", result["X_test"].shape)
    print("train class balance:", np.bincount(result["y_train"]))
    print("test class balance: ", np.bincount(result["y_test"]))