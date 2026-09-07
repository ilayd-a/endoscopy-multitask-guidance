"""
models.py — QSVM implementation for the Kidney Stone (present/absent) QML benchmark.

Mirrors the QSVM design from the previous Kvasir-SEG / EBTC benchmark:
    ZZFeatureMap -> quantum kernel K(x_i, x_j) = |<psi(x_i)|psi(x_j)>|^2 -> classical SVC

Compatible with qiskit 2.x / qiskit-machine-learning 0.9.x
(uses the non-deprecated `zz_feature_map` function form and AerSimulator-backed Sampler).
"""

import time
import numpy as np

from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def build_qsvm_kernel(n_qubits, reps=2, backend="aer", shots=1024, seed=42):
    """
    Build a FidelityQuantumKernel using a ZZFeatureMap.

    backend:
        "statevector" -> exact, noiseless StatevectorSampler (slow but exact)
        "aer"         -> AerSimulator-backed sampler with finite shots (more realistic)
    """
    feature_map = zz_feature_map(feature_dimension=n_qubits, reps=reps)

    if backend == "statevector":
        sampler = StatevectorSampler(seed=seed)
    elif backend == "aer":
        sampler = AerSamplerV2(default_shots=shots, seed=seed)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    return kernel


def run_qsvm(X_train, y_train, X_test, y_test, n_qubits, reps=2,
             backend="aer", shots=1024, seed=42, C=1.0, verbose=True):
    """
    Trains QSVM (quantum kernel + classical SVC) and evaluates on the test set.

    Returns a dict of metrics + the fitted kernel/model for further inspection.
    """
    if verbose:
        print(f"[QSVM] Building quantum kernel (n_qubits={n_qubits}, reps={reps}, backend={backend})")
    kernel = build_qsvm_kernel(n_qubits=n_qubits, reps=reps, backend=backend, shots=shots, seed=seed)

    t0 = time.time()
    if verbose:
        print(f"[QSVM] Computing training kernel matrix ({len(X_train)}x{len(X_train)})... "
              f"this is O(N^2) quantum circuit evaluations, may take a while.")
    K_train = kernel.evaluate(x_vec=X_train)
    train_kernel_time = time.time() - t0

    t0 = time.time()
    if verbose:
        print(f"[QSVM] Computing test kernel matrix ({len(X_test)}x{len(X_train)})...")
    K_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
    test_kernel_time = time.time() - t0

    t0 = time.time()
    clf = SVC(kernel="precomputed", C=C, probability=True, random_state=seed)
    clf.fit(K_train, y_train)
    fit_time = time.time() - t0

    y_pred = clf.predict(K_test)
    y_proba = clf.predict_proba(K_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")  # only one class present in y_test
    cm = confusion_matrix(y_test, y_pred)

    total_time = train_kernel_time + test_kernel_time + fit_time

    if verbose:
        print(f"[QSVM] train_kernel_time={train_kernel_time:.1f}s  "
              f"test_kernel_time={test_kernel_time:.1f}s  svc_fit_time={fit_time:.2f}s")
        print(f"[QSVM] Accuracy={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
        print(f"[QSVM] Confusion matrix:\n{cm}")

    return {
        "name": "QSVM",
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "train_time": total_time,
        "train_kernel_time": train_kernel_time,
        "test_kernel_time": test_kernel_time,
        "model": clf,
        "kernel": kernel,
        "K_train": K_train,
        "K_test": K_test,
    }
