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
    brier_score_loss,
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


def classical_models(seed: int, fast: bool, grid: bool = False):
    max_iter = 300 if fast else 1000
    models = {
        "Classical_LogReg_C1": LogisticRegression(
            C=1.0, max_iter=max_iter, class_weight="balanced", random_state=seed
        ),
        "Classical_LinearSVM_C1": SVC(
            C=1.0, kernel="linear", probability=True, class_weight="balanced", random_state=seed
        ),
        "Classical_RBFSVM_C1_gammaScale": SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=seed
        ),
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

    if grid:
        for c in [0.1, 10.0]:
            models[f"Classical_LogReg_C{c:g}"] = LogisticRegression(
                C=c, max_iter=max_iter, class_weight="balanced", random_state=seed
            )
            models[f"Classical_LinearSVM_C{c:g}"] = SVC(
                C=c, kernel="linear", probability=True, class_weight="balanced", random_state=seed
            )
        for c in [0.1, 10.0]:
            for gamma in ["scale", "auto"]:
                models[f"Classical_RBFSVM_C{c:g}_gamma{gamma.title()}"] = SVC(
                    C=c,
                    kernel="rbf",
                    gamma=gamma,
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                )

    return models


def qml_models(model_names: list[str], fast: bool, qsvm_grid: bool = False):
    out = {}
    v1_reps = [1, 2, 3] if qsvm_grid else [2]
    v2_reps = [1, 2, 3] if qsvm_grid else [3]
    if "qsvm_v1" in model_names or "all_qml" in model_names:
        for reps in v1_reps:
            out[f"QML_QSVM_v1_4q_reps{reps}"] = build_qsvm_classifier(
                n_qubits=4, version="v1", reps=reps
            )
    if "qsvm_v2" in model_names or "all_qml" in model_names:
        for reps in v2_reps:
            out[f"QML_QSVM_v2_6q_reps{reps}"] = build_qsvm_classifier(
                n_qubits=6, version="v2", reps=reps
            )
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


class QuantumKernelSVC:
    def __init__(self, kernel, C: float = 1.0):
        self.kernel = kernel
        self.model = SVC(kernel="precomputed", C=C, probability=True)
        self.X_train = None

    def fit(self, X, y):
        self.X_train = X
        self.K_train = self.kernel.evaluate(x_vec=X)
        self.model.fit(self.K_train, y)
        return self

    def predict(self, X):
        return self.model.predict(self.kernel.evaluate(x_vec=X, y_vec=self.X_train))

    def predict_proba(self, X):
        return self.model.predict_proba(self.kernel.evaluate(x_vec=X, y_vec=self.X_train))


class ProjectedQuantumKernelSVC:
    """
    Local-observable projected quantum kernel approximation.

    Instead of using pairwise state fidelities directly, this maps each sample
    into low-dimensional trigonometric features inspired by single-qubit
    expectation values after angle encoding, then applies an RBF kernel.
    This is a simulator-light proxy for projected quantum kernels and gives a
    concrete baseline for the "avoid kernel concentration" research question.
    """

    def __init__(self, gamma="scale", C: float = 1.0, reps: int = 1, class_weight=None):
        self.gamma = gamma
        self.reps = reps
        self.class_weight = class_weight
        self.model = SVC(kernel="precomputed", C=C, probability=True, class_weight=class_weight)

    def _project(self, X):
        features = [np.cos(X), np.sin(X)]
        if self.reps >= 2:
            features.extend([np.cos(2 * X), np.sin(2 * X)])
        if self.reps >= 3:
            features.extend([np.cos(3 * X), np.sin(3 * X)])
        return np.concatenate(features, axis=1)

    def _gamma_value(self, Z):
        if self.gamma == "scale":
            var = float(np.var(Z))
            return 1.0 / (Z.shape[1] * var) if var > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / Z.shape[1]
        return float(self.gamma)

    def _kernel(self, A, B):
        AA = np.sum(A * A, axis=1)[:, None]
        BB = np.sum(B * B, axis=1)[None, :]
        dist2 = np.maximum(AA + BB - 2 * A @ B.T, 0)
        return np.exp(-self.gamma_ * dist2)

    def fit(self, X, y):
        self.Z_train = self._project(X)
        self.gamma_ = self._gamma_value(self.Z_train)
        self.K_train = self._kernel(self.Z_train, self.Z_train)
        self.model.fit(self.K_train, y)
        return self

    def predict(self, X):
        return self.model.predict(self._kernel(self._project(X), self.Z_train))

    def predict_proba(self, X):
        return self.model.predict_proba(self._kernel(self._project(X), self.Z_train))


def pqk_models(model_names: list[str], qsvm_grid: bool = False):
    out = {}
    if "pqk" not in model_names:
        return out
    reps_list = [1, 2, 3] if qsvm_grid else [2]
    for reps in reps_list:
        out[f"QML_PQK_6q_reps{reps}_gammaScale"] = ProjectedQuantumKernelSVC(
            gamma="scale", reps=reps
        )
    return out


def build_qsvm_classifier(n_qubits: int, version: str, reps: int):
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
        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
    else:
        feature_map = PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            paulis=["Z", "ZZ", "ZZZ"],
        )

    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    return QuantumKernelSVC(kernel)


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
        "sensitivity": np.nan,
        "specificity": np.nan,
        "ppv": np.nan,
        "npv": np.nan,
        "false_negative_rate": np.nan,
        "false_positive_rate": np.nan,
        "brier_score": np.nan,
        "ece_10bin": np.nan,
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


def expected_calibration_error(y_true, scores, n_bins: int = 10):
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if len(scores) == 0 or np.any(np.isnan(scores)):
        return np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (scores >= lo) & (scores <= hi if i == n_bins - 1 else scores < hi)
        if not np.any(mask):
            continue
        confidence = float(np.mean(scores[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(confidence - accuracy)
    return ece


def medical_binary_metrics(y_true, y_pred, scores):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else np.nan,
        "false_positive_rate": fp / (tn + fp) if (tn + fp) else np.nan,
    }
    try:
        clipped = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        metrics["brier_score"] = brier_score_loss(y_true, clipped)
        metrics["ece_10bin"] = expected_calibration_error(y_true, clipped, n_bins=10)
    except Exception:
        metrics["brier_score"] = np.nan
        metrics["ece_10bin"] = np.nan
    return metrics


def kernel_target_alignment(K, y):
    y_pm = np.where(np.asarray(y) > 0, 1.0, -1.0)
    yy = np.outer(y_pm, y_pm)
    denom = np.linalg.norm(K, "fro") * np.linalg.norm(yy, "fro")
    return float(np.sum(K * yy) / denom) if denom > 0 else np.nan


def kernel_concentration(K):
    K = np.asarray(K, dtype=float)
    mask = ~np.eye(K.shape[0], dtype=bool)
    off_diag = K[mask]
    return {
        "kernel_diag_mean": float(np.mean(np.diag(K))),
        "kernel_offdiag_mean": float(np.mean(off_diag)) if off_diag.size else np.nan,
        "kernel_offdiag_std": float(np.std(off_diag)) if off_diag.size else np.nan,
    }


def model_kernel_diagnostics(model, y_train):
    K = getattr(model, "K_train", None)
    if K is None:
        inner = getattr(model, "model", None)
        K = getattr(inner, "K_train", None)
    if K is None:
        return {}
    diagnostics = kernel_concentration(K)
    diagnostics["kernel_target_alignment"] = kernel_target_alignment(K, y_train)
    return diagnostics


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    y_pred = model.predict(X_test)
    try:
        scores = predict_scores(model, X_test, y_pred)
        auc = roc_auc_score(y_test, scores)
    except Exception:
        scores = y_pred.astype(float)
        auc = float("nan")

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": auc,
        "train_time_sec": elapsed,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    result.update(medical_binary_metrics(y_test, y_pred, scores))
    result.update(model_kernel_diagnostics(model, y_train))
    return result


def write_csv(path: Path, records: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def aggregate_records(records: list[dict]):
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record.get("train_limit"), record.get("model"))
        grouped.setdefault(key, []).append(record)

    aggregate = []
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "false_negative_rate",
        "false_positive_rate",
        "brier_score",
        "ece_10bin",
        "train_time_sec",
        "kernel_target_alignment",
        "kernel_offdiag_mean",
        "kernel_offdiag_std",
    ]
    for (train_limit, model), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary = {
            "train_limit": train_limit,
            "model": model,
            "n_runs": len(rows),
        }
        for metric in metrics:
            vals = np.asarray([
                float(row.get(metric)) for row in rows
                if row.get(metric) is not None and not np.isnan(float(row.get(metric)))
            ])
            summary[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            summary[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) else np.nan
        aggregate.append(summary)
    return aggregate


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
        X, y, ids = make_synthetic(args.max_samples, 64, args.seed)
        return X, y, ids, None

    from data_loader_ebtc import extract_resnet18_features_from_paths, list_ebtc_samples_with_metadata

    paths, y, sample_metadata = list_ebtc_samples_with_metadata(args.data_dir, label_mode=args.label_mode)
    ids = [str(p.relative_to(args.data_dir)) for p in paths]
    split_labels = [row["sub_dataset"] for row in sample_metadata]
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
        split_labels = [split_labels[i] for i in idx]

    print(f"[data] Extracting ResNet-18 features for {len(paths)} EBTC images...")
    X_raw = extract_resnet18_features_from_paths(paths, batch_size=args.batch_size)
    return X_raw, y, ids, split_labels


def run_one_split(args, X_raw, y, ids, split_labels, repeat_index: int, train_limit: int):
    split_seed = args.seed + repeat_index
    if args.split_mode == "official":
        if split_labels is None or "unknown" in set(split_labels):
            raise ValueError("Official split mode requires annotations.csv with known sub_dataset values.")
        split_labels_arr = np.asarray(split_labels)
        train_mask = np.isin(split_labels_arr, args.official_train_parts)
        test_mask = split_labels_arr == "test"
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"Official split produced train={len(train_idx)} test={len(test_idx)}. "
                f"Check --official_train_parts={args.official_train_parts}."
            )
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=args.test_size,
            stratify=y,
            random_state=split_seed,
        )

    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    ids_train = [ids[i] for i in train_idx]
    ids_test = [ids[i] for i in test_idx]

    X_train_raw, y_train, ids_train = limit_split(
        X_train_raw, y_train, ids_train, train_limit, split_seed, balance=True
    )
    X_test_raw, y_test, ids_test = limit_split(
        X_test_raw, y_test, ids_test, args.max_test_samples, split_seed, balance=False
    )

    print(
        f"[split] repeat={repeat_index} seed={split_seed} "
        f"train_limit={train_limit} train={len(y_train)} test={len(y_test)}"
    )
    print(f"[split] train class counts={np.bincount(y_train)} test class counts={np.bincount(y_test)}")

    metadata = {
        "repeat": repeat_index,
        "seed": split_seed,
        "split_mode": args.split_mode,
        "official_train_parts": args.official_train_parts if args.split_mode == "official" else None,
        "train_limit": int(train_limit),
        "train_count": int(len(y_train)),
        "test_count": int(len(y_test)),
        "train_class_counts": np.bincount(y_train).astype(int).tolist(),
        "test_class_counts": np.bincount(y_test).astype(int).tolist(),
        "sample_ids": {"train": ids_train, "test": ids_test},
    }

    records = []
    feature_sets = {}
    for n_components in [4, 6]:
        Xtr, Xte, info = fit_low_dim_features(X_train_raw, X_test_raw, n_components, split_seed)
        if args.balance_train:
            Xtr, ytr = balance_train_only(Xtr, y_train, split_seed)
        else:
            ytr = y_train
        feature_sets[n_components] = (Xtr, ytr, Xte, info)
        metadata[f"pca_{n_components}q"] = info
        print(f"[features] {n_components} components retain {info['pca_variance_retained']:.3f} variance")

    if "classical" in args.models:
        Xtr, ytr, Xte, _ = feature_sets[6]
        for name, model in classical_models(split_seed, args.fast, args.classical_grid).items():
            print(f"[classical] {name}")
            result = evaluate_model(name, model, Xtr, ytr, Xte, y_test)
            records.append(result)

    try:
        quantum_models = qml_models(args.models, args.fast, args.qsvm_grid)
        quantum_models.update(pqk_models(args.models, args.qsvm_grid))
    except ModuleNotFoundError as exc:
        print(f"[qml] QML dependencies unavailable: {exc}")
        records.extend(missing_qml_records(args.models, exc))
        quantum_models = {}

    for name, model in quantum_models.items():
        n_components = 4 if "_4q" in name else 6
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
                "sensitivity": np.nan,
                "specificity": np.nan,
                "ppv": np.nan,
                "npv": np.nan,
                "false_negative_rate": np.nan,
                "false_positive_rate": np.nan,
                "brier_score": np.nan,
                "ece_10bin": np.nan,
                "train_time_sec": np.nan,
                "confusion_matrix": None,
                "error": str(exc),
            })

    for record in records:
        record["repeat"] = repeat_index
        record["seed"] = split_seed
        record["train_limit"] = train_limit
        record["train_count"] = int(len(y_train))
        record["test_count"] = int(len(y_test))

    return records, metadata


def main():
    parser = argparse.ArgumentParser(description="Leakage-controlled EBTC publication benchmark")
    parser.add_argument("--data_dir", default=str(ROOT / "data" / "EBTC"))
    parser.add_argument(
        "--label_mode",
        choices=["cancer_vs_noncancer", "hgc_vs_lgc"],
        default="cancer_vs_noncancer",
        help="Binary task: old cancer/non-cancer mode or README-focused HGC vs LGC.",
    )
    parser.add_argument("--models", nargs="+", default=["classical", "qsvm_v1", "qsvm_v2"],
                        choices=["classical", "qsvm_v1", "qsvm_v2", "pqk", "all_qml"])
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--max_train_samples", type=int, default=120,
                        help="Training subset limit after split; set 0 to use all.")
    parser.add_argument("--max_test_samples", type=int, default=80,
                        help="Test subset limit after split; set 0 to use all.")
    parser.add_argument("--train_sizes", nargs="*", type=int,
                        help="Optional low-data curve train limits, e.g. 20 40 80 120.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeated stratified splits.")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--split_mode", choices=["random", "official"], default="random",
                        help="Use repeated random stratified splits or annotations.csv train/test split.")
    parser.add_argument("--official_train_parts", nargs="+", default=["train"],
                        choices=["train", "val"],
                        help="Official sub_dataset parts used for training when --split_mode official.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--balance_train", action="store_true",
                        help="Oversample minority class in training only.")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--classical_grid", action="store_true",
                        help="Evaluate extra C/gamma variants for classical baselines.")
    parser.add_argument("--qsvm_grid", action="store_true",
                        help="Evaluate QSVM feature-map reps 1, 2, and 3.")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip PNG figure generation; CSV/JSON are still saved.")
    args = parser.parse_args()

    X_raw, y, ids, split_labels = load_raw_features(args)
    train_sizes = args.train_sizes or [args.max_train_samples]

    all_records = []
    split_metadata = []
    metadata = {
        "dataset": "synthetic" if args.synthetic else "EBTC",
        "label_mode": args.label_mode,
        "seed": args.seed,
        "repeats": args.repeats,
        "train_sizes": train_sizes,
        "test_size": args.test_size,
        "split_mode": args.split_mode,
        "official_train_parts": args.official_train_parts if args.split_mode == "official" else None,
        "models_requested": args.models,
        "classical_grid": args.classical_grid,
        "qsvm_grid": args.qsvm_grid,
        "preprocessing": "StandardScaler + PCA + MinMaxScaler fit on train only",
    }

    for train_limit in train_sizes:
        for repeat_index in range(args.repeats):
            records, split_info = run_one_split(args, X_raw, y, ids, split_labels, repeat_index, train_limit)
            all_records.extend(records)
            split_metadata.append(split_info)

    metrics_path = RESULTS_DIR / "ebtc_publication_metrics.csv"
    aggregate_path = RESULTS_DIR / "ebtc_publication_aggregate.csv"
    diagnostics_path = RESULTS_DIR / "ebtc_kernel_diagnostics.csv"
    metadata_path = RESULTS_DIR / "ebtc_publication_metadata.json"
    figure_path = RESULTS_DIR / "ebtc_publication_comparison.png"

    fieldnames = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "false_negative_rate",
        "false_positive_rate",
        "brier_score",
        "ece_10bin",
        "train_time_sec",
        "confusion_matrix",
        "error",
        "repeat",
        "seed",
        "train_limit",
        "train_count",
        "test_count",
        "kernel_target_alignment",
        "kernel_diag_mean",
        "kernel_offdiag_mean",
        "kernel_offdiag_std",
    ]
    write_csv(metrics_path, all_records, fieldnames)
    diagnostic_records = [
        record for record in all_records
        if record.get("kernel_target_alignment") is not None
        and not np.isnan(float(record.get("kernel_target_alignment")))
    ]
    diagnostic_fields = [
        "model",
        "repeat",
        "seed",
        "train_limit",
        "kernel_target_alignment",
        "kernel_diag_mean",
        "kernel_offdiag_mean",
        "kernel_offdiag_std",
        "accuracy",
        "roc_auc",
    ]
    write_csv(diagnostics_path, diagnostic_records, diagnostic_fields)
    aggregate = aggregate_records(all_records)
    aggregate_fields = [
        "train_limit",
        "model",
        "n_runs",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "sensitivity_mean",
        "sensitivity_std",
        "specificity_mean",
        "specificity_std",
        "ppv_mean",
        "ppv_std",
        "npv_mean",
        "npv_std",
        "false_negative_rate_mean",
        "false_negative_rate_std",
        "false_positive_rate_mean",
        "false_positive_rate_std",
        "brier_score_mean",
        "brier_score_std",
        "ece_10bin_mean",
        "ece_10bin_std",
        "train_time_sec_mean",
        "train_time_sec_std",
        "kernel_target_alignment_mean",
        "kernel_target_alignment_std",
        "kernel_offdiag_mean_mean",
        "kernel_offdiag_mean_std",
        "kernel_offdiag_std_mean",
        "kernel_offdiag_std_std",
    ]
    write_csv(aggregate_path, aggregate, aggregate_fields)
    metadata["splits"] = split_metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    if not args.no_plot:
        plot_results(all_records, figure_path)

    print(f"[results] metrics: {metrics_path}")
    print(f"[results] aggregate: {aggregate_path}")
    print(f"[results] diagnostics: {diagnostics_path}")
    print(f"[results] metadata: {metadata_path}")
    if not args.no_plot:
        print(f"[results] figure: {figure_path}")
    for record in aggregate:
        print(record)


if __name__ == "__main__":
    main()
