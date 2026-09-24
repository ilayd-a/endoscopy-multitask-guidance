"""
encoder_embedding_failure_benchmark.py
======================================
Test whether projected quantum kernels add value for endoscopic model
reliability monitoring when using frozen U-Net encoder representations.

The task is intentionally leakage-controlled: features come from the RGB frame,
the model heatmap, and internal encoder activations only. Ground-truth masks are
used only to define retrospective failure labels such as Dice < 0.2.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
EBTC_EXPERIMENTS = ROOT / "EndoscopicBladderTissue" / "experiments"
if str(EBTC_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EBTC_EXPERIMENTS))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from publication_benchmark_ebtc import ProjectedQuantumKernelSVC, model_kernel_diagnostics
from export_cvc_predictions import INPUT_SIZE, device_from_name, load_model


def source_sequence_id(metadata: pd.DataFrame) -> dict[str, int]:
    return {
        Path(row.png_image_path).name: int(row.sequence_id)
        for row in metadata.itertuples(index=False)
    }


def split_from_sequence(sequence_id: int) -> str:
    if sequence_id <= 23:
        return "train"
    if sequence_id <= 26:
        return "val"
    return "test"


def heatmap_features(heatmap: np.ndarray, pred: np.ndarray, image: np.ndarray) -> list[float]:
    hm = heatmap.astype(np.float32)
    pred_bool = pred.astype(bool)
    labeled, n_components = ndimage.label(pred_bool)
    component_sizes = np.bincount(labeled.ravel())[1:] if n_components else np.asarray([], dtype=int)
    grad_y, grad_x = np.gradient(hm)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    q = np.quantile(hm, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    positive = hm[hm > 0.5]
    rgb = image.astype(np.float32) / 255.0
    gray = rgb.mean(axis=2)
    return [
        float(hm.mean()),
        float(hm.std()),
        float(hm.min()),
        float(hm.max()),
        *[float(v) for v in q],
        float(grad_mag.mean()),
        float(grad_mag.std()),
        float(pred_bool.mean()),
        float(n_components),
        float(component_sizes.max()) / hm.size if component_sizes.size else 0.0,
        float(component_sizes.mean()) / hm.size if component_sizes.size else 0.0,
        float(positive.mean()) if positive.size else 0.0,
        float(positive.std()) if positive.size else 0.0,
        float(gray.mean()),
        float(gray.std()),
        *[float(v) for v in rgb.mean(axis=(0, 1))],
        *[float(v) for v in rgb.std(axis=(0, 1))],
    ]


def pooled_encoder_features(features: list[torch.Tensor], include_scales: str) -> np.ndarray:
    if include_scales == "bottleneck":
        selected = features[-1:]
    elif include_scales == "deep":
        selected = features[-3:]
    elif include_scales == "all":
        selected = features[1:]
    else:
        raise ValueError(f"Unknown include_scales={include_scales}")
    pieces = []
    for feat in selected:
        arr = feat.detach().cpu().numpy()[0].astype(np.float32)
        pieces.append(arr.mean(axis=(1, 2)))
        pieces.append(arr.std(axis=(1, 2)))
        pieces.append(arr.max(axis=(1, 2)))
    return np.concatenate(pieces).astype(np.float32)


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)


def build_features(args) -> tuple[pd.DataFrame, np.ndarray]:
    endoscopy_repo = Path(args.endoscopy_repo)
    cvc_dir = endoscopy_repo / "dataset" / "CVC-ClinicDB"
    image_dir = cvc_dir / "PNG" / "Original"
    metadata = pd.read_csv(cvc_dir / "metadata.csv")
    sequence_map = source_sequence_id(metadata)
    summary = pd.read_csv(args.export_summary)
    device = device_from_name(args.device)
    model = load_model(Path(args.checkpoint), device)

    rows = []
    matrix = []
    for idx, row in enumerate(summary.itertuples(index=False)):
        source_file = row.source_file
        sequence_id = sequence_map[source_file]
        image = load_rgb(image_dir / source_file)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            encoder_features = model.encoder(tensor)
            logits = model.decoder(encoder_features)
            logits = model.segmentation_head(logits)
            heatmap = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
        pred = (heatmap >= args.threshold).astype(np.uint8)
        frame_features = np.concatenate([
            pooled_encoder_features(encoder_features, args.include_scales),
            np.asarray(heatmap_features(heatmap, pred, image), dtype=np.float32),
        ])
        matrix.append(frame_features)
        rows.append({
            "sample_id": row.sample_id,
            "source_file": source_file,
            "sequence_id": sequence_id,
            "split": split_from_sequence(sequence_id),
            "dice": float(row.dice),
            "iou": float(row.iou),
        })
        if (idx + 1) % 50 == 0 or idx + 1 == len(summary):
            print(f"[features] {idx + 1}/{len(summary)}")
    return pd.DataFrame(rows), np.vstack(matrix).astype(np.float32)


def low_dim_fit_transform(X_train, X_eval, n_components: int, seed: int):
    n_components = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=seed)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_pca = pca.fit_transform(scaler.fit_transform(X_train))
    X_eval_pca = pca.transform(scaler.transform(X_eval))
    return (
        angle_scaler.fit_transform(X_train_pca).astype(np.float64),
        angle_scaler.transform(X_eval_pca).astype(np.float64),
        float(pca.explained_variance_ratio_.sum()),
    )


def models(seed: int) -> dict:
    return {
        "Classical_LogReg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed),
        ),
        "Classical_RBFSVM": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=seed),
        ),
        "Classical_RF": RandomForestClassifier(n_estimators=160, class_weight="balanced", random_state=seed),
        "Classical_ExtraTrees": ExtraTreesClassifier(n_estimators=220, class_weight="balanced", random_state=seed),
        "Classical_HistGB": HistGradientBoostingClassifier(max_iter=160, learning_rate=0.05, random_state=seed),
    }


def scores_from_model(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "predict_proba"):
        return inner.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def evaluate_once(name: str, model, X_train, y_train, X_eval, y_eval, split_name: str) -> dict:
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)
    score = scores_from_model(model, X_eval)
    try:
        auc = roc_auc_score(y_eval, score)
    except ValueError:
        auc = np.nan
    out = {
        "model": name,
        "eval_split": split_name,
        "n_train": int(len(y_train)),
        "n_eval": int(len(y_eval)),
        "eval_positive_rate": float(np.mean(y_eval)),
        "accuracy": accuracy_score(y_eval, pred),
        "balanced_accuracy": balanced_accuracy_score(y_eval, pred),
        "f1": f1_score(y_eval, pred, zero_division=0),
        "roc_auc": auc,
    }
    out.update(model_kernel_diagnostics(model, y_train))
    return out


def run_benchmark(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / f"{args.prefix}_features.npy"
    metadata_path = out_dir / f"{args.prefix}_metadata.csv"
    if args.use_cache and feature_path.exists() and metadata_path.exists():
        print(f"[cache] loading {feature_path}")
        X = np.load(feature_path)
        rows = pd.read_csv(metadata_path)
    else:
        rows, X = build_features(args)
        np.save(feature_path, X)
        rows.to_csv(metadata_path, index=False)

    result_rows = []
    for threshold in args.failure_thresholds:
        y = (rows["dice"].to_numpy(float) < threshold).astype(int)
        train_mask = rows["split"].eq("train").to_numpy()
        val_mask = rows["split"].eq("val").to_numpy()
        test_mask = rows["split"].eq("test").to_numpy()
        if len(np.unique(y[train_mask])) < 2:
            print(f"[skip] threshold={threshold} has one train class")
            continue

        for seed in args.seeds:
            classical_models = models(seed)
            for model_name, model in classical_models.items():
                for split_name, eval_mask in [("val", val_mask), ("test", test_mask), ("val_test", val_mask | test_mask)]:
                    if len(np.unique(y[eval_mask])) < 2:
                        continue
                    row = evaluate_once(model_name, model, X[train_mask], y[train_mask], X[eval_mask], y[eval_mask], split_name)
                    row.update({"failure_threshold": threshold, "seed": seed, "pca_variance_retained": np.nan})
                    result_rows.append(row)

            for n_components in args.pqk_components:
                X_train_q, X_all_q, variance = low_dim_fit_transform(X[train_mask], X, n_components, seed)
                for reps in args.pqk_reps:
                    for C in args.pqk_c:
                        model_name = f"QML_PQK_{n_components}pc_reps{reps}_C{C:g}"
                        for split_name, eval_mask in [("val", val_mask), ("test", test_mask), ("val_test", val_mask | test_mask)]:
                            if len(np.unique(y[eval_mask])) < 2:
                                continue
                            model = ProjectedQuantumKernelSVC(gamma="scale", reps=reps, C=C, class_weight="balanced")
                            row = evaluate_once(
                                model_name,
                                model,
                                X_train_q,
                                y[train_mask],
                                X_all_q[eval_mask],
                                y[eval_mask],
                                split_name,
                            )
                            row.update({"failure_threshold": threshold, "seed": seed, "pca_variance_retained": variance})
                            result_rows.append(row)
        print(f"[threshold] {threshold} complete")
        interim = pd.DataFrame(result_rows)
        if not interim.empty:
            interim.to_csv(out_dir / f"{args.prefix}_results.partial.csv", index=False)

    result_path = out_dir / f"{args.prefix}_results.csv"
    with result_path.open("w", newline="") as f:
        fieldnames = sorted({key for row in result_rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)
    results = pd.DataFrame(result_rows)
    summary = (
        results.groupby(["failure_threshold", "eval_split", "model"], dropna=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            f1_mean=("f1", "mean"),
            eval_positive_rate=("eval_positive_rate", "mean"),
            pca_variance_retained=("pca_variance_retained", "mean"),
        )
        .reset_index()
        .sort_values(["failure_threshold", "eval_split", "roc_auc_mean"], ascending=[True, True, False])
    )
    summary_path = out_dir / f"{args.prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[saved] {result_path}")
    print(f"[saved] {summary_path}")
    print(summary.groupby(["failure_threshold", "eval_split"]).head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Frozen U-Net encoder reliability benchmark")
    parser.add_argument("--export_summary", default="endoscopy_guidance/exports/cvc_all_rgb/export_summary.csv")
    parser.add_argument("--endoscopy_repo", default="/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance")
    parser.add_argument("--checkpoint", default="/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance/models/unet_cvc.pth")
    parser.add_argument("--output_dir", default="endoscopy_guidance/results/encoder_failure")
    parser.add_argument("--prefix", default="cvc_all_encoder")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--include_scales", choices=["bottleneck", "deep", "all"], default="deep")
    parser.add_argument("--failure_thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--pqk_components", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--pqk_reps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--pqk_c", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--use_cache", action="store_true")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
