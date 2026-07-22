"""
residual_patch_quantum_benchmark.py
===================================
Compare classical and projected-quantum residual patch classifiers.

This is an intermediate benchmark: it tests whether a quantum feature map helps
identify classical segmentation errors at the patch/pixel level. A later script
should apply the residual predictions back to probability maps and measure Dice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from candidate_ranking_benchmark import fit_low_dim
from sam_prompt_quality_ranker import projected_quantum_features


def limit_training(X: np.ndarray, y: np.ndarray, max_train: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_train <= 0 or len(y) <= max_train:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    labels = np.unique(y)
    per_class = max(1, max_train // len(labels))
    for label in labels:
        idx = np.where(y == label)[0]
        keep.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    if len(keep) < max_train:
        remaining = np.setdiff1d(np.arange(len(y)), np.asarray(keep), assume_unique=False)
        extra = rng.choice(remaining, size=min(max_train - len(keep), len(remaining)), replace=False)
        keep.extend(extra.tolist())
    keep = np.asarray(sorted(set(keep)), dtype=int)
    return X[keep], y[keep]


def metrics(name: str, y_true: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict:
    row = {
        "model": name,
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "error_f1": float(f1_score((y_true > 0).astype(int), (pred > 0).astype(int))),
    }
    if score is not None:
        try:
            row["error_auc"] = float(roc_auc_score((y_true > 0).astype(int), score))
        except ValueError:
            row["error_auc"] = np.nan
    return row


def error_score(model, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = list(model.classes_)
        error_cols = [i for i, label in enumerate(classes) if label > 0]
        return probs[:, error_cols].sum(axis=1)
    return None


def main():
    parser = argparse.ArgumentParser(description="Residual patch quantum benchmark")
    parser.add_argument("--dataset_npz", default="endoscopy_guidance/results/residual_patch_dataset_cvc_val_test.npz")
    parser.add_argument("--rows_csv", default="endoscopy_guidance/results/residual_patch_dataset_cvc_val_test.csv")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/residual_patch_quantum_benchmark.csv")
    parser.add_argument("--max_train", type=int, default=6000)
    parser.add_argument("--pqk_components", type=int, default=8)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--train_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    payload = np.load(args.dataset_npz, allow_pickle=True)
    X = payload["X"].astype(np.float32)
    y = payload["y"].astype(np.int64)
    rows = pd.read_csv(args.rows_csv)
    train_mask = rows["split"].eq(args.train_split).to_numpy()
    test_mask = rows["split"].eq(args.test_split).to_numpy()
    if not train_mask.any() or not test_mask.any():
        raise ValueError(f"Missing train/test split rows: train={args.train_split}, test={args.test_split}")
    X_train, y_train = limit_training(X[train_mask], y[train_mask], args.max_train, args.seed)
    X_test, y_test = X[test_mask], y[test_mask]

    results = []
    models = {
        "classical_logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "classical_histgb": HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05, random_state=args.seed),
        "classical_random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append(metrics(name, y_test, pred, error_score(model, X_test)))

    X_train_q, X_test_q, pca_info = fit_low_dim(X_train, X_test, args.pqk_components, args.seed)
    Z_train = projected_quantum_features(X_train_q, args.pqk_reps)
    Z_test = projected_quantum_features(X_test_q, args.pqk_reps)
    q_model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05, random_state=args.seed)
    q_model.fit(Z_train, y_train)
    q_pred = q_model.predict(Z_test)
    q_row = metrics("projected_quantum_histgb", y_test, q_pred, error_score(q_model, Z_test))
    q_row["pca_variance"] = pca_info["pca_variance_retained"]
    results.append(q_row)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(results).sort_values("error_f1", ascending=False)
    summary.to_csv(output, index=False)
    print(f"[saved] {output}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
