"""
models.py
=========
Four QML classifiers for endoscopic image classification.

1. QSVM   – Qiskit QSVC with FidelityQuantumKernel (ZZFeatureMap)
2. VQC    – Qiskit VQC  with ZZFeatureMap + RealAmplitudes ansatz
3. QKernel SVM – TrainableFidelityQuantumKernel + sklearn SVC
4. QNN    – EstimatorQNN connected to PyTorch via TorchConnector

All models expose a unified interface:
    model.fit(X_train, y_train)
    model.predict(X_test)          -> np.ndarray
    model.score(X_test, y_test)    -> float
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Qiskit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler, StatevectorEstimator

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
# Helper: build standard ZZFeatureMap
# ════════════════════════════════════════════════════════

def _feature_map(n_qubits: int, reps: int = 2):
    return ZZFeatureMap(feature_dimension=n_qubits, reps=reps)


# ════════════════════════════════════════════════════════
# 1. QSVM  (FidelityQuantumKernel + QSVC)
# ════════════════════════════════════════════════════════

class QSVMClassifier:
    """
    Quantum Support Vector Machine.
    Uses the ZZ feature map kernel matrix (no trainable parameters).
    """

    def __init__(self, n_qubits: int = 4, reps: int = 2):
        self.n_qubits = n_qubits
        sampler       = StatevectorSampler()
        fidelity      = ComputeUncompute(sampler=sampler)
        feature_map   = _feature_map(n_qubits, reps)
        kernel        = FidelityQuantumKernel(
            fidelity=fidelity,
            feature_map=feature_map,
        )
        self.model = QSVC(quantum_kernel=kernel)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ════════════════════════════════════════════════════════
# 2. VQC  (Variational Quantum Classifier)
# ════════════════════════════════════════════════════════

class VQCClassifier:
    """
    Variational Quantum Classifier.
    ZZFeatureMap encoding + RealAmplitudes ansatz.
    Optimised with COBYLA (gradient-free).
    """

    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 2,
        max_iter: int = 100,
    ):
        self.n_qubits = n_qubits
        feature_map   = _feature_map(n_qubits, reps)
        ansatz        = RealAmplitudes(n_qubits, reps=reps)
        sampler       = StatevectorSampler()

        self.model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            sampler=sampler,
            optimizer=COBYLA(maxiter=max_iter),
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ════════════════════════════════════════════════════════
# 3. Trainable Quantum Kernel + classical SVM
# ════════════════════════════════════════════════════════

class QKernelSVMClassifier:
    """
    Trainable quantum kernel aligned with training labels
    (kernel target alignment), then passed to sklearn SVC.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 2,
        C: float = 1.0,
    ):
        self.n_qubits = n_qubits
        sampler       = StatevectorSampler()
        fidelity      = ComputeUncompute(sampler=sampler)
        feature_map   = _feature_map(n_qubits, reps)

        # TrainableFidelityQuantumKernel adds trainable parameters to the
        # feature map, which can be optimised via kernel target alignment.
        self.q_kernel = TrainableFidelityQuantumKernel(
            fidelity=fidelity,
            feature_map=feature_map,
        )
        self.svc = SVC(kernel="precomputed", C=C, probability=True)

    def _gram(self, X1, X2):
        return self.q_kernel.evaluate(x_vec=X1, y_vec=X2)

    def fit(self, X, y):
        # Note: full kernel alignment training requires QKA optimizer (not
        # included here for simplicity). We use the kernel as-is (same as
        # FidelityQuantumKernel but with a parameterised map that could be
        # trained separately).
        K_train = self._gram(X, X)
        self.X_train = X
        self.svc.fit(K_train, y)
        return self

    def predict(self, X):
        K_test = self._gram(X, self.X_train)
        return self.svc.predict(K_test)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ════════════════════════════════════════════════════════
# 4. QNN  (EstimatorQNN → TorchConnector → PyTorch)
# ════════════════════════════════════════════════════════

def _build_qnn(n_qubits: int, reps: int) -> EstimatorQNN:
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    # Build circuit: feature encoding + variational ansatz
    feature_params = ParameterVector("x", n_qubits)
    weight_params  = ParameterVector("w", n_qubits * reps)

    qc = QuantumCircuit(n_qubits)
    # Angle encoding
    for i, p in enumerate(feature_params):
        qc.ry(p, i)
    # Variational layers
    idx = 0
    for _ in range(reps):
        for i in range(n_qubits):
            qc.rz(weight_params[idx], i)
            idx += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    # Observable: Z on first qubit
    observable = SparsePauliOp("Z" + "I" * (n_qubits - 1))

    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=list(feature_params),
        weight_params=list(weight_params),
        estimator=estimator,
    )
    return qnn


class _TorchQNNNet(nn.Module):
    def __init__(self, qnn: EstimatorQNN):
        super().__init__()
        self.qnn_layer = TorchConnector(qnn)
        # Map single QNN output → 2-class logits
        self.head = nn.Linear(1, 2)

    def forward(self, x):
        q_out = self.qnn_layer(x)          # (B, 1)
        return self.head(q_out)            # (B, 2)


class QNNClassifier:
    """
    Quantum Neural Network via EstimatorQNN + TorchConnector.
    Trained end-to-end with Adam + CrossEntropyLoss.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 2,
        epochs: int = 30,
        lr: float = 0.01,
        batch_size: int = 16,
    ):
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.device     = torch.device("cpu")   # QNN runs on CPU

        qnn        = _build_qnn(n_qubits, reps)
        self.model = _TorchQNNNet(qnn).to(self.device)

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        opt      = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn  = nn.CrossEntropyLoss()
        N        = len(X_t)

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(N)
            total_loss = 0.0
            for start in range(0, N, self.batch_size):
                idx    = perm[start:start + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                opt.zero_grad()
                logits = self.model(xb)
                loss   = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(xb)
            avg = total_loss / N
            if (epoch + 1) % 5 == 0:
                print(f"    [QNN] epoch {epoch+1:3d}/{self.epochs} | loss = {avg:.4f}")
        return self

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
