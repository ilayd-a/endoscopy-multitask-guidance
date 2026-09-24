"""
validation_tuned_triage_ebtc.py
===============================
Tune PQK second-reader triage on the official validation split, then evaluate
the selected policy once on the official test split.

This is stricter than selecting the best hybrid directly on the test set.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader_ebtc import extract_resnet18_features_from_paths, list_ebtc_samples_with_metadata
from publication_benchmark_ebtc import classical_models, limit_split, pqk_models, predict_scores

RESULTS_DIR = ROOT / "results" / "publication_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fit_feature_views(X_train_raw, X_val_raw, X_test_raw, n_components, seed):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=seed)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))

    X_train_std = scaler.fit_transform(X_train_raw)
    X_train_pca = pca.fit_transform(X_train_std)
    X_train = angle_scaler.fit_transform(X_train_pca)

    X_val = angle_scaler.transform(pca.transform(scaler.transform(X_val_raw)))
    X_test = angle_scaler.transform(pca.transform(scaler.transform(X_test_raw)))
    return X_train.astype(float), X_val.astype(float), X_test.astype(float)


def binary_metrics(y_true, y_pred, scores=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else np.nan,
        "false_positive_rate": fp / (tn + fp) if (tn + fp) else np.nan,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if scores is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, scores)
        except Exception:
            out["roc_auc"] = np.nan
    return out


def model_predictions(model, X_train, y_train, X_eval):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    scores = predict_scores(model, X_eval, y_pred)
    return y_pred.astype(int), np.asarray(scores, dtype=float)


def triage_predictions(base_pred, base_score, second_pred, second_score, coverage, second_min_margin):
    uncertainty = np.abs(base_score - 0.5)
    second_margin = np.abs(second_score - 0.5)
    n = len(base_pred)
    k = min(max(int(round(n * coverage)), 0), n)
    selected = np.zeros(n, dtype=bool)
    if k:
        selected[np.argsort(uncertainty)[:k]] = True
    switched = selected & (second_margin >= second_min_margin)

    hybrid_pred = base_pred.copy()
    hybrid_score = base_score.copy()
    hybrid_pred[switched] = second_pred[switched]
    hybrid_score[switched] = second_score[switched]
    return hybrid_pred, hybrid_score, int(switched.sum())


def pref_key(metrics):
    return (
        metrics["balanced_accuracy"],
        metrics["sensitivity"],
        metrics["specificity"],
        metrics.get("roc_auc", -np.inf),
    )


def pref_key_no_auc(metrics):
    return (
        metrics["balanced_accuracy"],
        metrics["sensitivity"],
        metrics["specificity"],
    )


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("No rows to write")
    fields = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_prefixed(prefix, metrics):
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Validation-tuned EBTC PQK triage")
    parser.add_argument("--data_dir", default=str(ROOT / "dataset" / "baldder_tissue_classification"))
    parser.add_argument("--label_mode", default="hgc_vs_lgc", choices=["hgc_vs_lgc", "cancer_vs_noncancer"])
    parser.add_argument("--train_sizes", nargs="*", type=int, default=[40, 80, 160, 240])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--coverage", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--second_min_margin", nargs="*", type=float, default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--output_csv", default=str(RESULTS_DIR / "ebtc_validation_tuned_triage.csv"))
    args = parser.parse_args()

    paths, y, meta = list_ebtc_samples_with_metadata(args.data_dir, label_mode=args.label_mode)
    split_labels = np.asarray([row["sub_dataset"] for row in meta])
    paths = np.asarray(paths, dtype=object)
    y = np.asarray(y, dtype=int)

    print(f"[data] Extracting ResNet-18 features for {len(paths)} EBTC images...")
    X_raw = extract_resnet18_features_from_paths(list(paths), batch_size=args.batch_size)

    train_pool_idx = np.where(split_labels == "train")[0]
    val_idx = np.where(split_labels == "val")[0]
    test_idx = np.where(split_labels == "test")[0]
    rows = []

    for train_limit in args.train_sizes:
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            X_train_pool = X_raw[train_pool_idx]
            y_train_pool = y[train_pool_idx]
            train_ids = [str(paths[i].relative_to(args.data_dir)) for i in train_pool_idx]
            X_train_raw, y_train, _ = limit_split(
                X_train_pool, y_train_pool, train_ids, train_limit, seed, balance=True
            )
            X_val_raw, y_val = X_raw[val_idx], y[val_idx]
            X_test_raw, y_test = X_raw[test_idx], y[test_idx]

            X_train, X_val, X_test = fit_feature_views(X_train_raw, X_val_raw, X_test_raw, 6, seed)

            model_specs = {}
            model_specs.update(classical_models(seed, fast=False, grid=True))
            model_specs.update(pqk_models(["pqk"], qsvm_grid=True))

            val_preds = {}
            test_preds = {}
            print(f"[split] train_limit={train_limit} repeat={repeat} train={len(y_train)} val={len(y_val)} test={len(y_test)}")
            for name, model in model_specs.items():
                y_val_pred, val_score = model_predictions(model, X_train, y_train, X_val)
                y_test_pred = model.predict(X_test).astype(int)
                test_score = predict_scores(model, X_test, y_test_pred)
                val_preds[name] = (y_val_pred, val_score)
                test_preds[name] = (y_test_pred, np.asarray(test_score, dtype=float))

            classical_names = [name for name in model_specs if not name.startswith("QML")]
            pqk_names = [name for name in model_specs if name.startswith("QML_PQK")]

            best_classical_name = max(
                classical_names,
                key=lambda name: pref_key(binary_metrics(y_val, val_preds[name][0], val_preds[name][1])),
            )
            best_pqk_name = max(
                pqk_names,
                key=lambda name: pref_key(binary_metrics(y_val, val_preds[name][0], val_preds[name][1])),
            )

            best_hybrid = None
            for base_name in classical_names:
                base_val_pred, base_val_score = val_preds[base_name]
                for second_name in pqk_names:
                    second_val_pred, second_val_score = val_preds[second_name]
                    for coverage in args.coverage:
                        for margin in args.second_min_margin:
                            hybrid_val_pred, hybrid_val_score, n_switched_val = triage_predictions(
                                base_val_pred,
                                base_val_score,
                                second_val_pred,
                                second_val_score,
                                coverage,
                                margin,
                            )
                            metrics = binary_metrics(y_val, hybrid_val_pred, hybrid_val_score)
                            candidate = {
                                "base_model": base_name,
                                "second_model": second_name,
                                "coverage": coverage,
                                "second_min_margin": margin,
                                "n_switched_val": n_switched_val,
                                "val_metrics": metrics,
                            }
                            if best_hybrid is None or pref_key_no_auc(metrics) > pref_key_no_auc(best_hybrid["val_metrics"]):
                                best_hybrid = candidate

            # Evaluate selected policies on test.
            best_classical_test = binary_metrics(y_test, test_preds[best_classical_name][0], test_preds[best_classical_name][1])
            best_pqk_test = binary_metrics(y_test, test_preds[best_pqk_name][0], test_preds[best_pqk_name][1])

            base_test_pred, base_test_score = test_preds[best_hybrid["base_model"]]
            second_test_pred, second_test_score = test_preds[best_hybrid["second_model"]]
            hybrid_test_pred, hybrid_test_score, n_switched_test = triage_predictions(
                base_test_pred,
                base_test_score,
                second_test_pred,
                second_test_score,
                best_hybrid["coverage"],
                best_hybrid["second_min_margin"],
            )
            hybrid_test = binary_metrics(y_test, hybrid_test_pred, hybrid_test_score)

            row = {
                "train_limit": train_limit,
                "repeat": repeat,
                "seed": seed,
                "train_count": len(y_train),
                "val_count": len(y_val),
                "test_count": len(y_test),
                "selected_classical_model": best_classical_name,
                "selected_pqk_model": best_pqk_name,
                "selected_hybrid_base": best_hybrid["base_model"],
                "selected_hybrid_second": best_hybrid["second_model"],
                "selected_hybrid_coverage": best_hybrid["coverage"],
                "selected_hybrid_second_min_margin": best_hybrid["second_min_margin"],
                "hybrid_n_switched_val": best_hybrid["n_switched_val"],
                "hybrid_n_switched_test": n_switched_test,
            }
            row.update(metric_prefixed("test_classical", best_classical_test))
            row.update(metric_prefixed("test_pqk", best_pqk_test))
            row.update(metric_prefixed("test_hybrid", hybrid_test))
            row["delta_hybrid_vs_classical_balanced_accuracy"] = (
                hybrid_test["balanced_accuracy"] - best_classical_test["balanced_accuracy"]
            )
            row["delta_hybrid_vs_classical_sensitivity"] = (
                hybrid_test["sensitivity"] - best_classical_test["sensitivity"]
            )
            row["delta_hybrid_vs_classical_specificity"] = (
                hybrid_test["specificity"] - best_classical_test["specificity"]
            )
            rows.append(row)

    write_csv(Path(args.output_csv), rows)
    print(f"[results] {args.output_csv}")
    summary = pd.DataFrame(rows).groupby("train_limit").agg({
        "test_classical_balanced_accuracy": "mean",
        "test_hybrid_balanced_accuracy": "mean",
        "delta_hybrid_vs_classical_balanced_accuracy": "mean",
        "test_classical_sensitivity": "mean",
        "test_hybrid_sensitivity": "mean",
        "test_classical_specificity": "mean",
        "test_hybrid_specificity": "mean",
    })
    print(summary)


if __name__ == "__main__":
    main()
