"""
models_v2_aer.py
================
Improved QML classifiers (v2) using AerSimulator-based primitives.

Key difference from models_v2.py:
  - Uses AerSampler / AerEstimator instead of StatevectorSampler/Estimator
  - Shot-based sampling (default: 1024 shots) -> includes sampling noise
  - Circuits are transpiled via PassManager before execution on Aer
  - Optional noise model via FakeBackend for realistic hardware simulation

All v2 improvements retained:
  PauliFeatureMap(Z+ZZ+ZZZ), EfficientSU2 ansatz,
  Z-sum observable, n_qubits=6
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Qiskit
from qiskit.circuit.library import PauliFeatureMap, EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Qiskit Aer
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

# Qiskit ML
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel,
    TrainableFidelityQuantumKernel,
)
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


# ════════════════════════════════════════════════════════
# Backend and PassManager factory
# ════════════════════════════════════════════════════════

def _make_backend(noise_model=None) -> AerSimulator:
    """Build AerSimulator, optionally with a noise model."""
    if noise_model is not None:
        return AerSimulator(noise_model=noise_model)
    return AerSimulator()


def _make_pass_manager(backend: AerSimulator, optimization_level: int = 1):
    """Build a transpilation PassManager targeting the given backend."""
    return generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
    )


def _make_sampler(shots: int = 1024, noise_model=None):
    """AerSampler with optional noise model."""
    if noise_model is not None:
        return AerSampler(default_shots=shots,
                          options={"noise_model": noise_model})
    return AerSampler(default_shots=shots)


def _make_estimator(shots: int = 1024, noise_model=None):
    if noise_model is not None:
        return AerEstimator(options={
            "run_options": {"shots": shots},
            "backend_options": {"noise_model": noise_model},
        })
    return AerEstimator(options={"run_options": {"shots": shots}})

def _load_noise_model(backend_name: str):
    """
    Load a NoiseModel from a FakeBackend name, e.g. 'fake_nairobi'.
    Returns None on failure (falls back to ideal).
    """
    try:
        from qiskit_aer.noise import NoiseModel
        import importlib
        provider = importlib.import_module("qiskit_ibm_runtime.fake_provider")
        cls_name = "".join(w.capitalize() for w in backend_name.split("_"))
        backend  = getattr(provider, cls_name)()
        nm = NoiseModel.from_backend(backend)
        print(f"[noise] Loaded noise model: {cls_name}")
        return nm
    except Exception as e:
        print(f"[noise] Could not load '{backend_name}': {e}  -> ideal fallback")
        return None


# ════════════════════════════════════════════════════════
# Shared circuit helpers
# ════════════════════════════════════════════════════════

def _feature_map(n_qubits: int, reps: int = 3):
    return PauliFeatureMap(feature_dimension=n_qubits,
                           reps=reps, paulis=["Z", "ZZ", "ZZZ"])


def _zsum_observable(n_qubits: int) -> SparsePauliOp:
    terms = [("I" * (n_qubits - 1 - i) + "Z" + "I" * i, 1.0)
             for i in range(n_qubits)]
    return SparsePauliOp.from_list(terms)


# ════════════════════════════════════════════════════════
# 1. QSVM
# ════════════════════════════════════════════════════════

class QSVMClassifier:
    """
    Quantum SVM using FidelityQuantumKernel with AerSampler.
    Circuits are transpiled via PassManager before Aer execution.
    """

    def __init__(self, n_qubits: int = 6, reps: int = 3,
                 shots: int = 1024, noise_model=None):
        nm      = _load_noise_model(noise_model) if isinstance(noise_model, str) else noise_model
        backend = _make_backend(nm)
        pm      = _make_pass_manager(backend)
        sampler = _make_sampler(shots, nm)

        fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)
        kernel   = FidelityQuantumKernel(
            fidelity=fidelity,
            feature_map=_feature_map(n_qubits, reps),
        )
        self.model = QSVC(quantum_kernel=kernel)

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ════════════════════════════════════════════════════════
# 2. VQC
# ════════════════════════════════════════════════════════

class VQCClassifier:
    """
    Variational Quantum Classifier using AerSampler with PassManager.
    """

    def __init__(self, n_qubits: int = 6, reps: int = 3,
                 max_iter: int = 150, shots: int = 1024, noise_model=None):
        nm      = _load_noise_model(noise_model) if isinstance(noise_model, str) else noise_model
        backend = _make_backend(nm)
        pm      = _make_pass_manager(backend)
        ansatz  = EfficientSU2(n_qubits, reps=reps, entanglement="full")

        self.model = VQC(
            feature_map=_feature_map(n_qubits, reps),
            ansatz=ansatz,
            sampler=_make_sampler(shots, nm),
            optimizer=COBYLA(maxiter=max_iter),
            pass_manager=pm,
        )

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ════════════════════════════════════════════════════════
# 3. QKernel + SVM
# ════════════════════════════════════════════════════════

class QKernelSVMClassifier:
    """
    Trainable quantum kernel + sklearn SVC using AerSampler.
    """

    def __init__(self, n_qubits: int = 6, reps: int = 3,
                 C: float = 1.0, shots: int = 1024, noise_model=None):
        nm      = _load_noise_model(noise_model) if isinstance(noise_model, str) else noise_model
        backend = _make_backend(nm)
        pm      = _make_pass_manager(backend)

        fidelity = ComputeUncompute(sampler=_make_sampler(shots, nm),
                                    pass_manager=pm)
        self.q_kernel = TrainableFidelityQuantumKernel(
            fidelity=fidelity,
            feature_map=_feature_map(n_qubits, reps),
        )
        self.svc = SVC(kernel="precomputed", C=C, probability=True)

    def _gram(self, X1, X2):
        return self.q_kernel.evaluate(x_vec=X1, y_vec=X2)

    def fit(self, X, y):
        self.X_train = X
        self.svc.fit(self._gram(X, X), y); return self

    def predict(self, X):
        return self.svc.predict(self._gram(X, self.X_train))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ════════════════════════════════════════════════════════
# 4. QNN
# ════════════════════════════════════════════════════════

def _build_qnn(n_qubits: int, reps: int, estimator) -> EstimatorQNN:
    from qiskit.circuit import QuantumCircuit, ParameterVector

    feature_params = ParameterVector("x", n_qubits)
    enc = QuantumCircuit(n_qubits)
    for i, p in enumerate(feature_params):
        enc.ry(p, i)

    ansatz = EfficientSU2(n_qubits, reps=reps, entanglement="full")
    qc     = enc.compose(ansatz)

    return EstimatorQNN(
        circuit=qc,
        observables=_zsum_observable(n_qubits),
        input_params=list(feature_params),
        weight_params=list(ansatz.parameters),
        estimator=estimator,
    )


class _TorchQNNNet(nn.Module):
    def __init__(self, qnn):
        super().__init__()
        self.qnn_layer = TorchConnector(qnn)
        self.head      = nn.Linear(1, 2)

    def forward(self, x):
        return self.head(self.qnn_layer(x))


class QNNClassifier:
    """
    QNN via EstimatorQNN + TorchConnector using AerEstimator.
    """

    def __init__(self, n_qubits: int = 6, reps: int = 3,
                 epochs: int = 40, lr: float = 0.01, batch_size: int = 16,
                 shots: int = 1024, noise_model=None):
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.device     = torch.device("cpu")

        nm        = _load_noise_model(noise_model) if isinstance(noise_model, str) else noise_model
        estimator = _make_estimator(shots, nm)
        qnn       = _build_qnn(n_qubits, reps, estimator)
        self.model = _TorchQNNNet(qnn).to(self.device)

    def fit(self, X, y):
        X_t     = torch.tensor(X, dtype=torch.float32)
        y_t     = torch.tensor(y, dtype=torch.long)
        opt     = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        N       = len(X_t)

        self.model.train()
        for epoch in range(self.epochs):
            perm  = torch.randperm(N)
            total = 0.0
            for start in range(0, N, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            if (epoch + 1) % 10 == 0:
                print(f"    [QNN] epoch {epoch+1:3d}/{self.epochs} | loss={total/N:.4f}")
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
        return logits.argmax(dim=1).cpu().numpy()

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
