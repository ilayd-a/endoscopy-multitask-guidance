"""
models_v2.py
============
Improved QML classifiers — all four enhancements applied:

  1. PauliFeatureMap(paulis=["Z","ZZ","ZZZ"])  instead of ZZFeatureMap
  2. EfficientSU2 ansatz                        instead of RealAmplitudes
  3. Full-qubit Z-sum observable                instead of single-qubit Z  (QNN)
  4. n_qubits = 6, reps = 3 throughout
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
# Helper: PauliFeatureMap with higher-order interactions
# ════════════════════════════════════════════════════════

def _feature_map(n_qubits: int, reps: int = 3):
    """PauliFeatureMap with Z, ZZ, ZZZ interactions."""
    return PauliFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        paulis=["Z", "ZZ", "ZZZ"],
    )


# ════════════════════════════════════════════════════════
# Helper: full-qubit Z-sum observable
# ════════════════════════════════════════════════════════

def _zsum_observable(n_qubits: int) -> SparsePauliOp:
    """Sum of Z_i over all qubits: Z⊗I...I + I⊗Z⊗I...I + ..."""
    terms = []
    for i in range(n_qubits):
        pauli_str = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
        terms.append((pauli_str, 1.0))
    return SparsePauliOp.from_list(terms)


# ════════════════════════════════════════════════════════
# 1. QSVM
# ════════════════════════════════════════════════════════

class QSVMClassifier:
    def __init__(self, n_qubits: int = 6, reps: int = 3):
        sampler     = StatevectorSampler()
        fidelity    = ComputeUncompute(sampler=sampler)
        kernel      = FidelityQuantumKernel(
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
# 2. VQC  — EfficientSU2 ansatz
# ════════════════════════════════════════════════════════

class VQCClassifier:
    def __init__(self, n_qubits: int = 6, reps: int = 3, max_iter: int = 150):
        ansatz = EfficientSU2(n_qubits, reps=reps, entanglement="full")
        self.model = VQC(
            feature_map=_feature_map(n_qubits, reps),
            ansatz=ansatz,
            sampler=StatevectorSampler(),
            optimizer=COBYLA(maxiter=max_iter),
        )

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ════════════════════════════════════════════════════════
# 3. QKernel + SVM  — TrainableFidelityQuantumKernel
# ════════════════════════════════════════════════════════

class QKernelSVMClassifier:
    def __init__(self, n_qubits: int = 6, reps: int = 3, C: float = 1.0):
        fidelity      = ComputeUncompute(sampler=StatevectorSampler())
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
# 4. QNN  — full-qubit Z-sum observable + EfficientSU2
# ════════════════════════════════════════════════════════

def _build_qnn(n_qubits: int, reps: int) -> EstimatorQNN:
    from qiskit.circuit import QuantumCircuit, ParameterVector

    feature_params = ParameterVector("x", n_qubits)

    # Angle-encoding layer
    enc = QuantumCircuit(n_qubits)
    for i, p in enumerate(feature_params):
        enc.ry(p, i)

    # EfficientSU2 variational ansatz
    ansatz = EfficientSU2(n_qubits, reps=reps, entanglement="full")

    qc = enc.compose(ansatz)

    observable = _zsum_observable(n_qubits)

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=list(feature_params),
        weight_params=list(ansatz.parameters),
        estimator=StatevectorEstimator(),
    )
    return qnn


class _TorchQNNNet(nn.Module):
    def __init__(self, qnn: EstimatorQNN):
        super().__init__()
        self.qnn_layer = TorchConnector(qnn)
        self.head      = nn.Linear(1, 2)

    def forward(self, x):
        return self.head(self.qnn_layer(x))


class QNNClassifier:
    def __init__(
        self,
        n_qubits: int = 6,
        reps: int = 3,
        epochs: int = 40,
        lr: float = 0.01,
        batch_size: int = 16,
    ):
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.device     = torch.device("cpu")
        self.model      = _TorchQNNNet(_build_qnn(n_qubits, reps)).to(self.device)

    def fit(self, X, y):
        X_t    = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t    = torch.tensor(y, dtype=torch.long).to(self.device)
        opt    = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        N      = len(X_t)

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(N)
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
