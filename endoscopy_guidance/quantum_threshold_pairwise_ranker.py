"""
quantum_threshold_pairwise_ranker.py
====================================
Pairwise quantum-kernel threshold ranker for mask-hypothesis selection.

For each frame and candidate threshold, train a classifier to predict whether
that threshold improves Dice over the fixed 0.50 mask. At inference, score all
thresholds for a frame and choose the highest-scoring beneficial hypothesis,
falling back to 0.50 when no candidate looks beneficial.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

EBTC_EXPERIMENTS = THIS_DIR.parents[0] / "EndoscopicBladderTissue" / "experiments"
if str(EBTC_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EBTC_EXPERIMENTS))

from publication_benchmark_ebtc import ProjectedQuantumKernelSVC
from quantum_mask_hypothesis_selector import load_frame_table, selected_dice


def attach_embeddings(frame: pd.DataFrame, embedding_npz: str, embedding_weight: float) -> pd.DataFrame:
    if not embedding_npz:
        return frame
    payload = np.load(embedding_npz, allow_pickle=True)
    ids = payload["sample_ids"].astype(str).tolist()
    embeddings = payload["embeddings"].astype(np.float32)
    lookup = {sample_id: embeddings[idx] for idx, sample_id in enumerate(ids)}
    enriched = frame.copy()
    enriched["features"] = [
        np.concatenate([np.asarray(features, dtype=np.float32), lookup[str(sample_id)] * embedding_weight])
        for sample_id, features in zip(enriched["sample_id"], enriched["features"])
    ]
    return enriched


def candidate_features(frame: pd.DataFrame, thresholds: list[float]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = []
    X = []
    y = []
    for frame_idx, record in frame.iterrows():
        base_dice = float(record["baseline_dice"])
        oracle_dice = float(record["oracle_dice"])
        oracle_gain = oracle_dice - base_dice
        base_features = np.asarray(record["features"], dtype=np.float32)
        for threshold_idx, threshold in enumerate(thresholds):
            dice = float(record[f"dice_t{threshold:.2f}"])
            gain = dice - base_dice
            candidate_morphology = np.asarray(record[f"features_t{threshold:.2f}"], dtype=np.float32)
            threshold_features = np.asarray([
                threshold,
                abs(threshold - 0.5),
                threshold < 0.5,
                threshold > 0.5,
            ], dtype=np.float32)
            X.append(np.concatenate([
                base_features,
                candidate_morphology,
                candidate_morphology - base_features[: len(candidate_morphology)],
                threshold_features,
            ]))
            y.append(int(gain > 1e-6))
            rows.append({
                "frame_idx": frame_idx,
                "sample_id": record["sample_id"],
                "threshold_idx": threshold_idx,
                "threshold": threshold,
                "dice": dice,
                "gain": gain,
                "oracle_gain": oracle_gain,
            })
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), rows


class LowDimProjectedQuantumSVC:
    def __init__(self, components: int, reps: int, seed: int, C: float = 1.0):
        self.components = components
        self.reps = reps
        self.seed = seed
        self.C = C
        self.scaler = StandardScaler()
        self.pca = None
        self.angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        self.model = ProjectedQuantumKernelSVC(gamma="scale", C=C, reps=reps, class_weight="balanced")

    def fit(self, X: np.ndarray, y: np.ndarray):
        max_components = max(1, min(self.components, X.shape[1], X.shape[0] - 1))
        self.pca = PCA(n_components=max_components, random_state=self.seed)
        X_std = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_std)
        X_angle = self.angle_scaler.fit_transform(X_pca)
        self.model.fit(X_angle, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.angle_scaler.transform(self.pca.transform(self.scaler.transform(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.transform(X))


def make_model(name: str, components: int, reps: int, seed: int):
    if name == "classical_logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    if name == "classical_histgb":
        return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, random_state=seed)
    if name == "classical_random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    if name == "projected_quantum_kernel_svc":
        return LowDimProjectedQuantumSVC(components, reps, seed, C=1.0)
    raise ValueError(f"Unknown model={name}")


def choose_thresholds(model, X_eval: np.ndarray, rows: list[dict], n_frames: int, thresholds: list[float], min_score: float) -> np.ndarray:
    probs = model.predict_proba(X_eval)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.model.model.classes_)
    positive_col = classes.index(1) if 1 in classes else int(np.argmax(classes))
    scores = probs[:, positive_col]
    fixed_idx = thresholds.index(0.5)
    selected = np.full(n_frames, fixed_idx, dtype=np.int64)
    best_scores = np.full(n_frames, min_score, dtype=np.float32)
    for row, score in zip(rows, scores):
        frame_idx = int(row["frame_idx"])
        threshold_idx = int(row["threshold_idx"])
        if threshold_idx == fixed_idx:
            continue
        if score > best_scores[frame_idx]:
            best_scores[frame_idx] = score
            selected[frame_idx] = threshold_idx
    return selected


def evaluate(name: str, eval_df: pd.DataFrame, selected: np.ndarray, thresholds: list[float], hard_threshold: float) -> dict:
    dice = selected_dice(eval_df, selected, thresholds)
    baseline = eval_df["baseline_dice"].to_numpy(dtype=np.float32)
    hard = baseline < hard_threshold
    return {
        "model": name,
        "selected_dice": float(dice.mean()),
        "baseline_dice": float(baseline.mean()),
        "delta_dice": float((dice - baseline).mean()),
        "hard_selected_dice": float(dice[hard].mean()) if hard.any() else np.nan,
        "hard_baseline_dice": float(baseline[hard].mean()) if hard.any() else np.nan,
        "hard_delta_dice": float((dice[hard] - baseline[hard]).mean()) if hard.any() else np.nan,
        "changed_frames": int((selected != thresholds.index(0.5)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Pairwise quantum-kernel threshold ranker")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/quantum_threshold_pairwise_ranker_kvasir.csv")
    parser.add_argument("--embedding_npz", default="")
    parser.add_argument("--embedding_weight", type=float, default=1.0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--models", nargs="+", default=["classical_logistic", "projected_quantum_kernel_svc", "classical_histgb"])
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--tune_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--min_score_grid", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    parser.add_argument("--pqk_components", type=int, default=8)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--hard_dice_threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds)
    frame = attach_embeddings(load_frame_table(Path(args.baseline_dir), thresholds), args.embedding_npz, args.embedding_weight)
    train_df = frame.loc[frame["split"].eq(args.train_split)].copy().reset_index(drop=True)
    tune_df = frame.loc[frame["split"].eq(args.tune_split)].copy().reset_index(drop=True)
    test_df = frame.loc[frame["split"].eq(args.test_split)].copy().reset_index(drop=True)
    X_train, y_train, _ = candidate_features(train_df, thresholds)
    X_tune, _, tune_rows = candidate_features(tune_df, thresholds)
    X_test, _, test_rows = candidate_features(test_df, thresholds)

    results = []
    fixed = np.full(len(test_df), thresholds.index(0.5), dtype=np.int64)
    results.append(evaluate("fixed_threshold_0.50", test_df, fixed, thresholds, args.hard_dice_threshold))
    oracle = test_df["best_threshold_index"].to_numpy(dtype=np.int64)
    results.append(evaluate("oracle_threshold", test_df, oracle, thresholds, args.hard_dice_threshold))

    for model_name in args.models:
        model = make_model(model_name, args.pqk_components, args.pqk_reps, args.seed)
        model.fit(X_train, y_train)
        best = None
        for min_score in args.min_score_grid:
            tune_selected = choose_thresholds(model, X_tune, tune_rows, len(tune_df), thresholds, min_score)
            row = evaluate(model_name, tune_df, tune_selected, thresholds, args.hard_dice_threshold)
            score = (row["hard_selected_dice"], row["selected_dice"])
            if best is None or score > best[0]:
                best = (score, min_score)
        min_score = best[1]
        test_selected = choose_thresholds(model, X_test, test_rows, len(test_df), thresholds, min_score)
        row = evaluate(model_name, test_df, test_selected, thresholds, args.hard_dice_threshold)
        row["min_score"] = float(min_score)
        results.append(row)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(results).sort_values(["selected_dice", "hard_selected_dice"], ascending=False)
    summary.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
