"""
publication_benchmark_ebtc.py
=============================
Leakage-controlled EBTC benchmark for SPIE-style reporting.

Key differences from the exploratory benchmarks:
  - split before feature extraction/PCA/scaling decisions are used for modeling
  - fit StandardScaler, PCA, and MinMaxScaler on the training split only
  - optionally balance/downsample the training split only
  - include fair classical baselines on the same low-dimensional features
  - save machine-readable metrics plus a compact comparison figure

Examples
--------
Synthetic smoke test:
    python publication_benchmark_ebtc.py --synthetic --models classical --fast

Real EBTC, classical baselines plus QSVM:
    python publication_benchmark_ebtc.py --data_dir data/EBTC --models classical qsvm_v1 qsvm_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS_DIR = ROOT / "results" / "publication_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_synthetic(n_samples: int, n_features: int, seed: int):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=64,
        n_informative=16,
        n_redundant=8,
        class_sep=1.2,
        weights=[0.45, 0.55],
        random_state=seed,
    )
    return X.astype(np.float64), y.astype(int), [f"synthetic_{i:04d}" for i in range(len(y))]


def limit_split(X, y, ids, max_samples: int | None, seed: int, balance: bool = False):
    if max_samples is None or max_samples <= 0 or max_samples >= len(y):
        return X, y, ids

    rng = np.random.default_rng(seed)
    if balance:
        idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
        per_class = max_samples // 2
        use0 = rng.choice(idx0, size=min(per_class, len(idx0)), replace=False)
        use1 = rng.choice(idx1, size=min(max_samples - len(use0), len(idx1)), replace=False)
        idx = np.concatenate([use0, use1])
    else:
        idx, _ = train_test_split(
            np.arange(len(y)),
            train_size=max_samples,
            stratify=y,
            random_state=seed,
        )
    rng.shuffle(idx)
    return X[idx], y[idx], [ids[i] for i in idx]


def balance_train_only(X, y, seed: int):
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = max(len(idx0), len(idx1))
    if len(idx0) < n:
        idx0 = resample(idx0, replace=True, n_samples=n, random_state=seed)
    elif len(idx1) < n:
        idx1 = resample(idx1, replace=True, n_samples=n, random_state=seed)
    idx = np.concatenate([idx0, idx1])
    np.random.default_rng(seed).shuffle(idx)
    return X[idx], y[idx]


def fit_low_dim_features(X_train_raw, X_test_raw, n_components: int, seed: int):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=seed)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))

    X_train_std = scaler.fit_transform(X_train_raw)
    X_train_pca = pca.fit_transform(X_train_std)
    X_train = angle_scaler.fit_transform(X_train_pca)

    X_test_std = scaler.transform(X_test_raw)
    X_test_pca = pca.transform(X_test_std)
    X_test = angle_scaler.transform(X_test_pca)

    return X_train.astype(np.float64), X_test.astype(np.float64), {
        "pca_components": n_components,
        "pca_variance_retained": float(pca.explained_variance_ratio_.sum()),
    }


def classical_models(seed: int, fast: bool):
    max_iter = 300 if fast else 1000
    return {
        "Classical_LogReg": LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed),
        "Classical_LinearSVM": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=seed),
        "Classical_RBFSVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "Classical_RandomForest": RandomForestClassifier(
            n_estimators=80 if fast else 300,
            class_weight="balanced",
            random_state=seed,
        ),
        "Classical_MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=100 if fast else 400,
                random_state=seed,
            ),
        ),
    }


def qml_models(model_names: list[str], fast: bool):
    out = {}
    if "qsvm_v1" in model_names or "all_qml" in model_names:
        out["QML_QSVM_v1_4q"] = build_qsvm_classifier(n_qubits=4, version="v1")
    if "qsvm_v2" in model_names or "all_qml" in model_names:
        out["QML_QSVM_v2_6q"] = build_qsvm_classifier(n_qubits=6, version="v2")
    if "all_qml" in model_names:
        from models import VQCClassifier as VQC1, QKernelSVMClassifier as QK1, QNNClassifier as QNN1
        from models_v2 import VQCClassifier as VQC2, QKernelSVMClassifier as QK2, QNNClassifier as QNN2
        out.update({
            "QML_VQC_v1_4q": VQC1(n_qubits=4, max_iter=30 if fast else 100),
            "QML_QKernelSVM_v1_4q": QK1(n_qubits=4),
            "QML_QNN_v1_4q": QNN1(n_qubits=4, epochs=10 if fast else 30),
            "QML_VQC_v2_6q": VQC2(n_qubits=6, max_iter=30 if fast else 150),
            "QML_QKernelSVM_v2_6q": QK2(n_qubits=6),
            "QML_QNN_v2_6q": QNN2(n_qubits=6, epochs=10 if fast else 40),
        })
    return out


def build_qsvm_classifier(n_qubits: int, version: str):
    """
    Build QSVM without importing the repo's Torch-dependent model modules.

    This keeps QSVM runnable in lightweight Qiskit environments.
    """
    from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
    try:
        from qiskit.primitives import StatevectorSampler
        sampler = StatevectorSampler()
    except ImportError:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    try:
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
    except ImportError:
        try:
            from qiskit_algorithms.state_fidelities import ComputeUncompute
        except ImportError:
            from qiskit.algorithms.state_fidelities import ComputeUncompute

    if version == "v1":
        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
    else:
        feature_map = PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=3,
            paulis=["Z", "ZZ", "ZZZ"],
        )

    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    return QSVC(quantum_kernel=kernel)


def missing_qml_records(model_names: list[str], exc: Exception):
    requested = []
    if "qsvm_v1" in model_names:
        requested.append("QML_QSVM_v1_4q")
    if "qsvm_v2" in model_names:
        requested.append("QML_QSVM_v2_6q")
    if "all_qml" in model_names:
        requested.extend([
            "QML_QSVM_v1_4q",
            "QML_VQC_v1_4q",
            "QML_QKernelSVM_v1_4q",
            "QML_QNN_v1_4q",
            "QML_QSVM_v2_6q",
            "QML_VQC_v2_6q",
            "QML_QKernelSVM_v2_6q",
            "QML_QNN_v2_6q",
        ])
    return [{
        "model": name,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "train_time_sec": np.nan,
        "confusion_matrix": None,
        "error": str(exc),
    } for name in requested]


def predict_scores(model, X_test, y_pred):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "predict_proba"):
        return inner.predict_proba(X_test)[:, 1]
    if inner is not None and hasattr(inner, "decision_function"):
        return inner.decision_function(X_test)
    return y_pred.astype(float)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    y_pred = model.predict(X_test)
    try:
        scores = predict_scores(model, X_test, y_pred)
        auc = roc_auc_score(y_test, scores)
    except Exception:
        auc = float("nan")

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": auc,
        "train_time_sec": elapsed,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def plot_results(records: list[dict], save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["model"] for r in records]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    fig.patch.set_facecolor("white")
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Acc."),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
    ]
    colors = ["#2F6B8F" if n.startswith("Classical") else "#B35C44" for n in names]
    for ax, (key, title) in zip(axes, metrics):
        vals = np.asarray([float(r.get(key, np.nan)) for r in records])
        ax.bar(x, vals, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, min(v + 0.025, 1.02), f"{v:.2f}", ha="center", fontsize=8)
    fig.suptitle("EBTC Leakage-Controlled Benchmark")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_raw_features(args):
    if args.synthetic:
        return make_synthetic(args.max_samples, 64, args.seed)

    from data_loader_ebtc import extract_resnet18_features_from_paths, list_ebtc_samples

    paths, y = list_ebtc_samples(args.data_dir)
    ids = [str(p.relative_to(args.data_dir)) for p in paths]
    if args.max_samples and args.max_samples < len(y):
        idx, _ = train_test_split(
            np.arange(len(y)),
            train_size=args.max_samples,
            stratify=y,
            random_state=args.seed,
        )
        paths = [paths[i] for i in idx]
        y = y[idx]
        ids = [ids[i] for i in idx]

    print(f"[data] Extracting ResNet-18 features for {len(paths)} EBTC images...")
    X_raw = extract_resnet18_features_from_paths(paths, batch_size=args.batch_size)
    return X_raw, y, ids


def main():
    parser = argparse.ArgumentParser(description="Leakage-controlled EBTC publication benchmark")
    parser.add_argument("--data_dir", default=str(ROOT / "data" / "EBTC"))
    parser.add_argument("--models", nargs="+", default=["classical", "qsvm_v1", "qsvm_v2"],
                        choices=["classical", "qsvm_v1", "qsvm_v2", "all_qml"])
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--max_train_samples", type=int, default=120,
                        help="Training subset limit after split; set 0 to use all.")
    parser.add_argument("--max_test_samples", type=int, default=80,
                        help="Test subset limit after split; set 0 to use all.")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--balance_train", action="store_true",
                        help="Oversample minority class in training only.")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip PNG figure generation; CSV/JSON are still saved.")
    args = parser.parse_args()

    X_raw, y, ids = load_raw_features(args)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )

    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    ids_train = [ids[i] for i in train_idx]
    ids_test = [ids[i] for i in test_idx]

    X_train_raw, y_train, ids_train = limit_split(
        X_train_raw, y_train, ids_train, args.max_train_samples, args.seed, balance=True
    )
    X_test_raw, y_test, ids_test = limit_split(
        X_test_raw, y_test, ids_test, args.max_test_samples, args.seed, balance=False
    )

    print(f"[split] train={len(y_train)} test={len(y_test)}")
    print(f"[split] train class counts={np.bincount(y_train)} test class counts={np.bincount(y_test)}")

    records = []
    metadata = {
        "dataset": "synthetic" if args.synthetic else "EBTC",
        "seed": args.seed,
        "test_size": args.test_size,
        "train_count": int(len(y_train)),
        "test_count": int(len(y_test)),
        "train_class_counts": np.bincount(y_train).astype(int).tolist(),
        "test_class_counts": np.bincount(y_test).astype(int).tolist(),
        "models_requested": args.models,
        "preprocessing": "StandardScaler + PCA + MinMaxScaler fit on train only",
    }

    feature_sets = {}
    for n_components in [4, 6]:
        Xtr, Xte, info = fit_low_dim_features(X_train_raw, X_test_raw, n_components, args.seed)
        if args.balance_train:
            Xtr, ytr = balance_train_only(Xtr, y_train, args.seed)
        else:
            ytr = y_train
        feature_sets[n_components] = (Xtr, ytr, Xte, info)
        metadata[f"pca_{n_components}q"] = info
        print(f"[features] {n_components} components retain {info['pca_variance_retained']:.3f} variance")

    if "classical" in args.models:
        Xtr, ytr, Xte, _ = feature_sets[6]
        for name, model in classical_models(args.seed, args.fast).items():
            print(f"[classical] {name}")
            records.append(evaluate_model(name, model, Xtr, ytr, Xte, y_test))

    try:
        quantum_models = qml_models(args.models, args.fast)
    except ModuleNotFoundError as exc:
        print(f"[qml] QML dependencies unavailable: {exc}")
        records.extend(missing_qml_records(args.models, exc))
        quantum_models = {}

    for name, model in quantum_models.items():
        n_components = 4 if name.endswith("_4q") else 6
        Xtr, ytr, Xte, _ = feature_sets[n_components]
        print(f"[qml] {name}")
        try:
            records.append(evaluate_model(name, model, Xtr, ytr, Xte, y_test))
        except Exception as exc:
            print(f"[qml] {name} failed: {exc}")
            records.append({
                "model": name,
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "f1": np.nan,
                "roc_auc": np.nan,
                "train_time_sec": np.nan,
                "confusion_matrix": None,
                "error": str(exc),
            })

    metrics_path = RESULTS_DIR / "ebtc_publication_metrics.csv"
    metadata_path = RESULTS_DIR / "ebtc_publication_metadata.json"
    figure_path = RESULTS_DIR / "ebtc_publication_comparison.png"

    fieldnames = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "train_time_sec",
        "confusion_matrix",
        "error",
    ]
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    metadata["sample_ids"] = {"train": ids_train, "test": ids_test}
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    if not args.no_plot:
        plot_results(records, figure_path)

    print(f"[results] metrics: {metrics_path}")
    print(f"[results] metadata: {metadata_path}")
    if not args.no_plot:
        print(f"[results] figure: {figure_path}")
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
