"""
patch_benchmark.py
==================
Kidney-stone target-vs-background patch benchmark.

This reframes the task from whole-frame "stone present/absent" classification
to an image-guidance-style patch task:

  positive samples: crops around annotated stone bounding boxes
  negative samples: random crops from unannotated frames in the same split

The resulting task uses the available segmentation labels more directly and
creates more training samples than whole-frame classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "EndoscopicBladderTissue" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from publication_benchmark_ebtc import ProjectedQuantumKernelSVC, model_kernel_diagnostics


def load_coco_records(coco_json: Path):
    data = json.loads(coco_json.read_text())
    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    images = {img["id"]: img for img in data["images"]}
    return images, anns_by_image


def padded_box(bbox, width, height, pad_frac):
    x, y, w, h = bbox
    pad = pad_frac * max(w, h)
    x0 = max(0, int(round(x - pad)))
    y0 = max(0, int(round(y - pad)))
    x1 = min(width, int(round(x + w + pad)))
    y1 = min(height, int(round(y + h + pad)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def random_crop_box(width, height, crop_w, crop_h, rng):
    crop_w = min(int(round(crop_w)), width)
    crop_h = min(int(round(crop_h)), height)
    x0 = rng.randint(0, max(0, width - crop_w))
    y0 = rng.randint(0, max(0, height - crop_h))
    return x0, y0, x0 + crop_w, y0 + crop_h


def build_patch_specs(coco_json: Path, image_dir: Path, neg_per_pos: int, pad_frac: float, seed: int):
    images, anns_by_image = load_coco_records(coco_json)
    rng = random.Random(seed)
    specs = []
    pos_sizes_by_split = defaultdict(list)
    absent_by_split = defaultdict(list)

    for image_id, img in images.items():
        split = img.get("split", "unknown")
        if image_id in anns_by_image:
            for ann in anns_by_image[image_id]:
                box = padded_box(ann["bbox"], img["width"], img["height"], pad_frac)
                if box is None:
                    continue
                specs.append({
                    "image_file": img["file_name"],
                    "split": split,
                    "label": 1,
                    "box": box,
                    "source": "annotation",
                })
                pos_sizes_by_split[split].append((box[2] - box[0], box[3] - box[1]))
        else:
            absent_by_split[split].append(img)

    for split, sizes in pos_sizes_by_split.items():
        if not sizes or not absent_by_split[split]:
            continue
        n_neg = neg_per_pos * len(sizes)
        for i in range(n_neg):
            img = rng.choice(absent_by_split[split])
            crop_w, crop_h = sizes[i % len(sizes)]
            jitter = rng.uniform(0.75, 1.25)
            box = random_crop_box(img["width"], img["height"], crop_w * jitter, crop_h * jitter, rng)
            specs.append({
                "image_file": img["file_name"],
                "split": split,
                "label": 0,
                "box": box,
                "source": "background",
            })

    missing = [s["image_file"] for s in specs if not (image_dir / s["image_file"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} patch source images missing, e.g. {missing[:3]}")
    return specs


class ResNet18CropExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        backbone.eval().to(self.device)
        self.model = backbone
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, specs, image_dir: Path, batch_size: int):
        features = []
        labels = []
        splits = []
        ids = []
        for start in range(0, len(specs), batch_size):
            batch_specs = specs[start:start + batch_size]
            tensors = []
            for spec in batch_specs:
                img = Image.open(image_dir / spec["image_file"]).convert("RGB")
                crop = img.crop(spec["box"])
                tensors.append(self.transform(crop))
                labels.append(spec["label"])
                splits.append(spec["split"])
                ids.append(f"{spec['image_file']}:{spec['box']}:{spec['source']}")
            out = self.model(torch.stack(tensors).to(self.device))
            features.append(out.cpu().numpy())
            print(f"[features] {min(start + batch_size, len(specs))}/{len(specs)}")
        return np.concatenate(features, axis=0), np.asarray(labels), np.asarray(splits), ids


def preprocess_features(X_train_raw, X_test_raw, n_components, seed):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=seed)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train = angle_scaler.fit_transform(pca.fit_transform(scaler.fit_transform(X_train_raw)))
    X_test = angle_scaler.transform(pca.transform(scaler.transform(X_test_raw)))
    return X_train.astype(np.float64), X_test.astype(np.float64), float(pca.explained_variance_ratio_.sum())


def evaluate(name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    pred = model.predict(X_test)
    try:
        train_score = model.predict_proba(X_train)[:, 1]
        score = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, score)
    except Exception:
        train_score = None
        score = None
        auc = float("nan")
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": auc,
        "train_time_sec": elapsed,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    if train_score is not None and score is not None:
        thresholds = np.unique(train_score)
        best_threshold = 0.5
        best_train_balanced = -1.0
        for threshold in thresholds:
            train_pred = (train_score >= threshold).astype(int)
            train_balanced = balanced_accuracy_score(y_train, train_pred)
            if train_balanced > best_train_balanced:
                best_train_balanced = train_balanced
                best_threshold = float(threshold)
        tuned_pred = (score >= best_threshold).astype(int)
        row.update({
            "tuned_threshold": best_threshold,
            "tuned_train_balanced_accuracy": best_train_balanced,
            "tuned_accuracy": accuracy_score(y_test, tuned_pred),
            "tuned_balanced_accuracy": balanced_accuracy_score(y_test, tuned_pred),
            "tuned_f1": f1_score(y_test, tuned_pred, zero_division=0),
            "tuned_confusion_matrix": confusion_matrix(y_test, tuned_pred).tolist(),
        })
    row.update(model_kernel_diagnostics(model, y_train))
    return row


def main():
    parser = argparse.ArgumentParser(description="Kidney stone patch classification benchmark")
    parser.add_argument("--coco_json", default="qml_kidney_stone/training_data/dataset4_instances_split.json")
    parser.add_argument("--image_dir", default="/Users/ilaydadilek/Downloads/kidney video.jpg")
    parser.add_argument("--results_csv", default="qml_kidney_stone/results/dataset4_patch_benchmark.csv")
    parser.add_argument("--n_components", type=int, default=6)
    parser.add_argument("--neg_per_pos", type=int, default=2)
    parser.add_argument("--pad_frac", type=float, default=0.12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    specs = build_patch_specs(
        coco_json=Path(args.coco_json),
        image_dir=Path(args.image_dir),
        neg_per_pos=args.neg_per_pos,
        pad_frac=args.pad_frac,
        seed=args.seed,
    )
    print(f"[data] patches={len(specs)} positives={sum(s['label'] for s in specs)}")

    extractor = ResNet18CropExtractor()
    X_raw, y, splits, ids = extractor.extract(specs, Path(args.image_dir), args.batch_size)
    train_idx = np.where(splits == "train")[0]
    test_idx = np.where(splits == "test")[0]
    X_train, X_test, pca_var = preprocess_features(
        X_raw[train_idx], X_raw[test_idx], args.n_components, args.seed
    )
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"[split] train={len(y_train)} pos={int(y_train.sum())} test={len(y_test)} pos={int(y_test.sum())}")
    print(f"[features] PCA variance retained={pca_var:.3f}")

    models_to_run = {
        "Classical_LogReg_C1": LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=args.seed),
        "Classical_LinearSVM_C1": SVC(C=1, kernel="linear", probability=True, class_weight="balanced", random_state=args.seed),
        "Classical_RBFSVM_C1_gammaScale": SVC(C=1, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=args.seed),
        "Classical_RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=args.seed),
        "QML_PQK_6q_reps1_gammaScale": ProjectedQuantumKernelSVC(gamma="scale", reps=1),
        "QML_PQK_6q_reps2_gammaScale": ProjectedQuantumKernelSVC(gamma="scale", reps=2),
        "QML_PQK_6q_reps3_gammaScale": ProjectedQuantumKernelSVC(gamma="scale", reps=3),
    }

    rows = []
    for name, model in models_to_run.items():
        print(f"[run] {name}")
        row = evaluate(name, model, X_train, y_train, X_test, y_test)
        row.update({
            "train_count": len(y_train),
            "test_count": len(y_test),
            "train_present": int(y_train.sum()),
            "test_present": int(y_test.sum()),
            "pca_variance_retained": pca_var,
        })
        rows.append(row)
        print(row)

    out = Path(args.results_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "train_time_sec",
        "confusion_matrix",
        "train_count",
        "test_count",
        "train_present",
        "test_present",
        "pca_variance_retained",
        "kernel_target_alignment",
        "kernel_diag_mean",
        "kernel_offdiag_mean",
        "kernel_offdiag_std",
        "tuned_threshold",
        "tuned_train_balanced_accuracy",
        "tuned_accuracy",
        "tuned_balanced_accuracy",
        "tuned_f1",
        "tuned_confusion_matrix",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
