"""
quantum_mask_hypothesis_selector.py
===================================
Select a per-frame segmentation threshold/hypothesis for a fixed UNet output.

This tests a higher-leverage quantum role than residual pixel cleanup: use
frame-level probability-map and morphology features to choose which mask
hypothesis should be trusted for each frame. Training uses only the train split,
model/threshold policy selection uses validation, and the final table reports
held-out test Dice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_residual_patch_dataset import boundary
from export_classical_cvc_baseline import dice_iou
from sam_prompt_quality_ranker import projected_quantum_features


def risk_features(prob: np.ndarray, threshold: float = 0.5) -> list[float]:
    pred = (prob >= threshold).astype(np.uint8)
    uncertainty = 1.0 - np.abs(prob - 0.5) * 2.0
    pred_boundary = boundary(pred, radius=3)
    n_components, _, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if n_components > 1 else np.asarray([], dtype=np.float32)
    return [
        float(prob.mean()),
        float(prob.std()),
        float(prob.max()),
        float(np.quantile(prob, 0.50)),
        float(np.quantile(prob, 0.75)),
        float(np.quantile(prob, 0.90)),
        float(np.quantile(prob, 0.95)),
        float(np.quantile(prob, 0.99)),
        float(pred.mean()),
        float(pred_boundary.mean()),
        float(uncertainty.mean()),
        float(uncertainty.std()),
        float(np.quantile(uncertainty, 0.90)),
        float(np.quantile(uncertainty, 0.95)),
        float(n_components - 1),
        float(component_areas.max() / pred.size) if len(component_areas) else 0.0,
        float(component_areas.mean() / pred.size) if len(component_areas) else 0.0,
    ]


def load_frame_table(base: Path, thresholds: list[float]) -> pd.DataFrame:
    metrics = pd.read_csv(base / "baseline_metrics.csv")
    rows = []
    for record in metrics.itertuples(index=False):
        prob = np.load(base / "prob_maps" / f"{record.sample_id}.npy").astype(np.float32)
        gt = np.load(base / "gt_masks" / f"{record.sample_id}.npy").astype(np.uint8)
        threshold_dice = []
        threshold_iou = []
        for threshold in thresholds:
            pred = (prob >= threshold).astype(np.uint8)
            dice, iou = dice_iou(pred, gt)
            threshold_dice.append(dice)
            threshold_iou.append(iou)
        best_idx = int(np.argmax(threshold_dice))
        row = {
            "sample_id": record.sample_id,
            "split": record.split,
            "source_file": record.source_file,
            "baseline_dice": threshold_dice[thresholds.index(0.5)] if 0.5 in thresholds else float(record.dice),
            "oracle_dice": float(threshold_dice[best_idx]),
            "oracle_threshold": float(thresholds[best_idx]),
            "best_threshold_index": best_idx,
        }
        for idx, threshold in enumerate(thresholds):
            row[f"dice_t{threshold:.2f}"] = float(threshold_dice[idx])
            row[f"iou_t{threshold:.2f}"] = float(threshold_iou[idx])
            row[f"features_t{threshold:.2f}"] = risk_features(prob, threshold=threshold)
        row["features"] = risk_features(prob, threshold=0.5)
        rows.append(row)
    return pd.DataFrame(rows)


def split_arrays(frame: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    subset = frame.loc[frame["split"].eq(split)].copy().reset_index(drop=True)
    X = np.asarray(subset["features"].tolist(), dtype=np.float32)
    y = subset["best_threshold_index"].to_numpy(dtype=np.int64)
    return X, y, subset


class QuantumClassifier:
    def __init__(self, components: int, reps: int, seed: int, hybrid: bool = False, head: str = "histgb"):
        self.components = components
        self.reps = reps
        self.seed = seed
        self.hybrid = hybrid
        self.head = head
        self.scaler = StandardScaler()
        self.pca = None
        self.angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        if head == "logistic":
            self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        else:
            self.model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, random_state=seed)

    def _quantum_features(self, X: np.ndarray) -> np.ndarray:
        low = self.angle_scaler.transform(self.pca.transform(self.scaler.transform(X)))
        Z = projected_quantum_features(low, self.reps)
        if self.hybrid:
            return np.concatenate([X, Z], axis=1)
        return Z

    def fit(self, X: np.ndarray, y: np.ndarray):
        max_components = max(1, min(self.components, X.shape[1], X.shape[0] - 1))
        self.pca = PCA(n_components=max_components, random_state=self.seed)
        X_std = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_std)
        self.angle_scaler.fit(X_pca)
        self.model.fit(self._quantum_features(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._quantum_features(X))


def selected_dice(eval_df: pd.DataFrame, selected_idx: np.ndarray, thresholds: list[float]) -> np.ndarray:
    dice = []
    for row_idx, idx in enumerate(selected_idx):
        threshold = thresholds[int(idx)]
        dice.append(eval_df.iloc[row_idx][f"dice_t{threshold:.2f}"])
    return np.asarray(dice, dtype=np.float32)


def evaluate_selector(name: str, eval_df: pd.DataFrame, selected_idx: np.ndarray, thresholds: list[float], hard_threshold: float) -> dict:
    dice = selected_dice(eval_df, selected_idx, thresholds)
    baseline = eval_df["baseline_dice"].to_numpy(dtype=np.float32)
    hard = baseline < hard_threshold
    return {
        "model": name,
        "frames": int(len(eval_df)),
        "selected_dice": float(dice.mean()),
        "baseline_dice": float(baseline.mean()),
        "delta_dice": float((dice - baseline).mean()),
        "hard_frames": int(hard.sum()),
        "hard_selected_dice": float(dice[hard].mean()) if hard.any() else np.nan,
        "hard_baseline_dice": float(baseline[hard].mean()) if hard.any() else np.nan,
        "hard_delta_dice": float((dice[hard] - baseline[hard]).mean()) if hard.any() else np.nan,
        "changed_frames": int((selected_idx != thresholds.index(0.5)).sum()) if 0.5 in thresholds else int(len(eval_df)),
    }


def selector_frame_rows(name: str, eval_df: pd.DataFrame, selected_idx: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    dice = selected_dice(eval_df, selected_idx, thresholds)
    baseline = eval_df["baseline_dice"].to_numpy(dtype=np.float32)
    rows = eval_df[["sample_id", "split", "source_file", "baseline_dice", "oracle_dice", "oracle_threshold"]].copy()
    rows["model"] = name
    rows["selected_threshold"] = [thresholds[int(idx)] for idx in selected_idx]
    rows["selected_dice"] = dice
    rows["delta_dice"] = dice - baseline
    return rows


def main():
    parser = argparse.ArgumentParser(description="Quantum/classical mask-threshold hypothesis selector")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/quantum_mask_hypothesis_selector_kvasir.csv")
    parser.add_argument("--per_frame_csv", default="")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--hard_dice_threshold", type=float, default=0.80)
    parser.add_argument("--pqk_components", type=int, default=10)
    parser.add_argument("--pqk_reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds)
    frame = load_frame_table(Path(args.baseline_dir), thresholds)
    X_train, y_train, train_df = split_arrays(frame, args.train_split)
    X_eval, _, eval_df = split_arrays(frame, args.eval_split)

    results = []
    frame_outputs = []
    fixed_val = np.full(len(eval_df), thresholds.index(0.5), dtype=np.int64)
    results.append(evaluate_selector("fixed_threshold_0.50", eval_df, fixed_val, thresholds, args.hard_dice_threshold))
    frame_outputs.append(selector_frame_rows("fixed_threshold_0.50", eval_df, fixed_val, thresholds))
    fixed_best = int(np.argmax([train_df[f"dice_t{threshold:.2f}"].mean() for threshold in thresholds]))
    results.append(evaluate_selector(f"fixed_train_best_t{thresholds[fixed_best]:.2f}", eval_df, np.full(len(eval_df), fixed_best), thresholds, args.hard_dice_threshold))
    frame_outputs.append(selector_frame_rows(f"fixed_train_best_t{thresholds[fixed_best]:.2f}", eval_df, np.full(len(eval_df), fixed_best), thresholds))
    oracle_idx = eval_df["best_threshold_index"].to_numpy(dtype=np.int64)
    results.append(evaluate_selector("oracle_threshold", eval_df, oracle_idx, thresholds, args.hard_dice_threshold))
    frame_outputs.append(selector_frame_rows("oracle_threshold", eval_df, oracle_idx, thresholds))

    models = {
        "classical_logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "classical_histgb": HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, random_state=args.seed),
        "classical_random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1),
        "projected_quantum_histgb": QuantumClassifier(args.pqk_components, args.pqk_reps, args.seed, hybrid=False),
        "hybrid_quantum_histgb": QuantumClassifier(args.pqk_components, args.pqk_reps, args.seed, hybrid=True),
        "projected_quantum_logistic": QuantumClassifier(args.pqk_components, args.pqk_reps, args.seed, hybrid=False, head="logistic"),
        "hybrid_quantum_logistic": QuantumClassifier(args.pqk_components, args.pqk_reps, args.seed, hybrid=True, head="logistic"),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        selected = model.predict(X_eval)
        results.append(evaluate_selector(name, eval_df, selected, thresholds, args.hard_dice_threshold))
        frame_outputs.append(selector_frame_rows(name, eval_df, selected, thresholds))

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(results).sort_values(["selected_dice", "hard_selected_dice"], ascending=False)
    summary.to_csv(output, index=False)
    per_frame = Path(args.per_frame_csv) if args.per_frame_csv else output.with_name(output.stem + "_per_frame.csv")
    pd.concat(frame_outputs, ignore_index=True).to_csv(per_frame, index=False)
    print(f"[saved] {output}")
    print(f"[saved] {per_frame}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
