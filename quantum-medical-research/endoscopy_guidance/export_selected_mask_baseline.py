"""
export_selected_mask_baseline.py
================================
Materialize an adaptive mask-hypothesis selector as a new baseline directory.

The input baseline directory contains images, ground truth masks, probability
maps, and fixed-threshold masks. This script trains a selector on one split,
chooses a threshold per frame, and writes a new baseline directory whose
`pred_masks/` are the selected-threshold masks. Downstream residual-refinement
scripts can then train and evaluate on top of the selected masks.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from export_classical_cvc_baseline import dice_iou, sample_features
from quantum_mask_hypothesis_selector import (
    QuantumClassifier,
    load_frame_table,
    risk_features,
    split_arrays,
)


def make_selector(name: str, components: int, reps: int, seed: int):
    if name == "classical_logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    if name == "classical_histgb":
        return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, random_state=seed)
    if name == "classical_random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    if name == "projected_quantum_logistic":
        return QuantumClassifier(components, reps, seed, hybrid=False, head="logistic")
    if name == "hybrid_quantum_logistic":
        return QuantumClassifier(components, reps, seed, hybrid=True, head="logistic")
    if name == "projected_quantum_histgb":
        return QuantumClassifier(components, reps, seed, hybrid=False)
    if name == "hybrid_quantum_histgb":
        return QuantumClassifier(components, reps, seed, hybrid=True)
    raise ValueError(f"Unknown selector_model={name}")


def copy_static_arrays(source: Path, output: Path, sample_id: str):
    for subdir in ["images", "gt_masks", "prob_maps"]:
        src = source / subdir / f"{sample_id}.npy"
        dst = output / subdir / f"{sample_id}.npy"
        if not dst.exists():
            shutil.copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Export selected-threshold masks as a baseline directory")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test")
    parser.add_argument("--output_dir", default="endoscopy_guidance/results/selected_threshold_classical_logistic_kvasir_train_val_test")
    parser.add_argument("--selector_model", default="classical_logistic")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--pqk_components", type=int, default=10)
    parser.add_argument("--pqk_reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    source = Path(args.baseline_dir)
    output = Path(args.output_dir)
    for subdir in ["images", "gt_masks", "prob_maps", "pred_masks"]:
        (output / subdir).mkdir(parents=True, exist_ok=True)

    thresholds = sorted(args.thresholds)
    frame = load_frame_table(source, thresholds)
    X_train, y_train, _ = split_arrays(frame, args.train_split)
    selector = make_selector(args.selector_model, args.pqk_components, args.pqk_reps, args.seed)
    selector.fit(X_train, y_train)

    source_metrics = pd.read_csv(source / "baseline_metrics.csv")
    rows = []
    for split in args.splits:
        split_df = source_metrics.loc[source_metrics["split"].eq(split)].copy().reset_index(drop=True)
        if split_df.empty:
            continue
        features = []
        for record in split_df.itertuples(index=False):
            prob = np.load(source / "prob_maps" / f"{record.sample_id}.npy").astype(np.float32)
            features.append(risk_features(prob, threshold=0.5))
        selected_idx = selector.predict(np.asarray(features, dtype=np.float32))
        for record, threshold_idx in zip(split_df.itertuples(index=False), selected_idx):
            threshold = thresholds[int(threshold_idx)]
            sid = record.sample_id
            prob = np.load(source / "prob_maps" / f"{sid}.npy").astype(np.float32)
            gt = np.load(source / "gt_masks" / f"{sid}.npy").astype(np.uint8)
            pred = (prob >= threshold).astype(np.uint8)
            dice, iou = dice_iou(pred, gt)
            copy_static_arrays(source, output, sid)
            np.save(output / "pred_masks" / f"{sid}.npy", pred)
            row = {
                "sample_id": sid,
                "source_file": record.source_file,
                "sequence_id": getattr(record, "sequence_id", -1),
                "frame_id": getattr(record, "frame_id", -1),
                "split": split,
                "dice": float(dice),
                "iou": float(iou),
                "selected_threshold": float(threshold),
                "source_baseline_dice": float(record.dice),
                "source_delta_dice": float(dice - record.dice),
            }
            row.update(sample_features(prob, pred, gt))
            rows.append(row)
        print(f"[selected:{split}] {len(split_df)} frames", flush=True)

    summary_path = output / "baseline_metrics.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    df = pd.DataFrame(rows)
    print(f"[saved] {summary_path}")
    print(df.groupby("split")[["dice", "iou", "source_baseline_dice", "source_delta_dice"]].mean().to_string())
    print("selected thresholds:")
    print(df.groupby(["split", "selected_threshold"]).size().to_string())


if __name__ == "__main__":
    main()
