#!/usr/bin/env python3
"""
failure_quantum_benchmark.py
============================
Does a quantum circuit add anything to segmentation-failure detection?

Task
    For each frame, predict whether the fixed U-Net failed (Dice below a
    threshold) from label-free features of that frame (heatmap statistics and
    pooled encoder activations). Ground-truth masks are used only to define the
    failure labels.

Compared models (all see exactly the same 8-dimensional input, the same folds
and the same hyperparameter budget of 8 configurations each)
    logreg, rbf_svm, gbt, mlp      classical baselines
    q_entangled                    fidelity-kernel SVM, ZZ feature map
                                   (Havlicek et al. 2019), simulated exactly
    q_product                      identical circuit with every entangling
                                   ZZ gate removed (product-state ablation)
Reference rows (not part of the matched comparison)
    logreg_full, gbt_full          classical models on all raw features, to
                                   show what is lost by compressing to 8 dims
    u_* columns                    training-free U-Net uncertainty scores
                                   (low confidence, entropy, fragmentation),
                                   used as-is with a fixed direction

Protocol
    * Outer: StratifiedGroupKFold (default 5 folds) grouped by video sequence,
      repeated over seeds. Every frame gets one out-of-fold score per seed.
    * StandardScaler + PCA(n_qubits) are fit on the outer training fold only.
    * Inner: grouped 3-fold CV on the training fold picks the configuration
      with the best mean AUROC. The test fold is never used for selection.
    * Metrics: AUROC, AUPRC per seed (mean +/- SD), cluster bootstrap over
      sequences for paired differences, and selective segmentation (mean Dice
      of frames kept after rejecting the most suspicious 10/20/30%).

Quantum circuit
    |psi(x)> = (U_Phi(x) H^n)^reps |0^n>,
    U_Phi(x) = exp(-i [ sum_i x_i Z_i + sum_(i,j) (pi - x_i)(pi - x_j) Z_i Z_j ]).
    U_Phi is diagonal, so the state is simulated exactly with diagonal phases
    and fast Walsh-Hadamard transforms. --self-test checks this against an
    explicit gate-by-gate simulation (H, Rz, CNOT-Rz-CNOT). Kernel:
    k(x, x') = |<psi(x)|psi(x')>|^2. Angles are x = alpha * pi * u, with u the
    min-max scaled PCA features (fit on the training fold) and alpha a tuned
    bandwidth, which counters kernel concentration.

Usage
    python failure_quantum_benchmark.py --self-test
    python failure_quantum_benchmark.py --features features_cvc.csv --out results/
    python failure_quantum_benchmark.py --synthetic --out synthetic_check/ --seeds 0 1

Input CSV: one row per frame with columns
    sample_id, sequence_id, dice, and feature columns prefixed with "f_".
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:  # pragma: no cover
    pass

MATCHED = ["logreg", "rbf_svm", "gbt", "mlp", "q_entangled", "q_product"]
CLASSICAL_8D = ["logreg", "rbf_svm", "gbt", "mlp"]
REFERENCE = ["logreg_full", "gbt_full"]


# =========================================================================== #
# Quantum circuit simulation
# =========================================================================== #
def entangling_pairs(n: int, topology: str) -> list[tuple[int, int]]:
    if topology == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if topology == "ring":
        return [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]
    if topology == "full":
        return list(itertools.combinations(range(n), 2))
    raise ValueError(topology)


def z_table(n: int) -> np.ndarray:
    """Z[b, i] = +1 if qubit i of basis state b is 0 else -1 (qubit 0 = most significant bit)."""
    b = np.arange(2 ** n)[:, None]
    bits = (b >> (n - 1 - np.arange(n))[None, :]) & 1
    return (1 - 2 * bits).astype(np.float64)


def walsh_hadamard(psi: np.ndarray, n: int) -> np.ndarray:
    """Apply H on every qubit to a batch of states, shape (m, 2**n)."""
    m = psi.shape[0]
    out = psi.reshape((m,) + (2,) * n)
    for ax in range(1, n + 1):
        a = np.take(out, 0, axis=ax)
        b = np.take(out, 1, axis=ax)
        out = np.stack([a + b, a - b], axis=ax) / np.sqrt(2.0)
    return out.reshape(m, 2 ** n)


def feature_map_states(X: np.ndarray, reps: int, pairs, entangle: bool) -> np.ndarray:
    """Exact statevectors (m, 2**n) of the ZZ feature map for angle inputs X (m, n)."""
    m, n = X.shape
    Z = z_table(n)
    phase = X @ Z.T                                     # sum_i x_i z_i
    if entangle and pairs:
        P = np.pi - X
        for i, j in pairs:
            phase += np.outer(P[:, i] * P[:, j], Z[:, i] * Z[:, j])
    D = np.exp(-1j * phase)
    psi = np.full((m, 2 ** n), 1.0 / np.sqrt(2 ** n), dtype=np.complex128)  # H^n |0>
    psi = D * psi
    for _ in range(reps - 1):
        psi = D * walsh_hadamard(psi, n)
    return psi


def fidelity_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.abs(A.conj() @ B.T) ** 2


def mean_single_qubit_purity(psi: np.ndarray, n: int) -> float:
    """Average purity Tr(rho_i^2) of the one-qubit reduced states (1 = no entanglement)."""
    m = psi.shape[0]
    t = psi.reshape((m,) + (2,) * n)
    vals = []
    for q in range(n):
        v = np.moveaxis(t, q + 1, 1).reshape(m, 2, -1)
        rho = np.einsum("mai,mbi->mab", v, v.conj())
        vals.append(np.real(np.einsum("mab,mba->m", rho, rho)))
    return float(np.mean(vals))


# ---- explicit gate-level reference simulator, used only by --self-test ---- #
def _gate_level_state(x: np.ndarray, reps: int, pairs, entangle: bool) -> np.ndarray:
    n = len(x)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I = np.eye(2)

    def on(op, q):
        mats = [op if k == q else I for k in range(n)]
        out = mats[0]
        for mm in mats[1:]:
            out = np.kron(out, mm)
        return out

    def rz(theta):
        return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])

    def cnot(c, t):
        dim = 2 ** n
        U = np.zeros((dim, dim))
        for b in range(dim):
            bits = [(b >> (n - 1 - k)) & 1 for k in range(n)]
            if bits[c]:
                bits[t] ^= 1
            U[sum(bit << (n - 1 - k) for k, bit in enumerate(bits)), b] = 1
        return U

    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1
    for _ in range(reps):
        for q in range(n):
            psi = on(H, q) @ psi
        for q in range(n):
            psi = on(rz(2 * x[q]), q) @ psi
        if entangle:
            for i, j in pairs:
                phi = (np.pi - x[i]) * (np.pi - x[j])
                psi = cnot(i, j) @ psi
                psi = on(rz(2 * phi), j) @ psi
                psi = cnot(i, j) @ psi
    return psi


def self_test() -> bool:
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    print("Self-test: fast simulator vs explicit gate-by-gate circuit")
    for n, reps, topo in [(2, 1, "linear"), (3, 2, "linear"), (4, 2, "ring"), (4, 3, "full")]:
        pairs = entangling_pairs(n, topo)
        X = rng.uniform(0, np.pi, size=(5, n))
        for ent in (True, False):
            fast = feature_map_states(X, reps, pairs, ent)
            ref = np.stack([_gate_level_state(x, reps, pairs, ent) for x in X])
            fid = np.abs(np.sum(fast.conj() * ref, axis=1)) ** 2   # equal up to global phase
            check(f"n={n} reps={reps} {topo} entangle={ent}", np.allclose(fid, 1, atol=1e-10),
                  f"(min fidelity {fid.min():.12f})")

    print("Self-test: kernel properties")
    n, pairs = 6, entangling_pairs(6, "linear")
    X = rng.uniform(0, np.pi, size=(40, n))
    for ent in (True, False):
        S = feature_map_states(X, 2, pairs, ent)
        K = fidelity_kernel(S, S)
        check(f"entangle={ent}: states normalized", np.allclose(np.linalg.norm(S, axis=1), 1))
        check(f"entangle={ent}: diag(K)=1, symmetric, 0<=K<=1",
              np.allclose(np.diag(K), 1) and np.allclose(K, K.T) and K.min() > -1e-12 and K.max() < 1 + 1e-12)
        check(f"entangle={ent}: K positive semidefinite", np.linalg.eigvalsh(K).min() > -1e-9)

    print("Self-test: ablation really is a product state")
    S = feature_map_states(X, 2, pairs, entangle=False)
    K = fidelity_kernel(S, S)
    Kprod = np.ones_like(K)
    for q in range(n):
        s1 = feature_map_states(X[:, [q]], 2, [], entangle=False)
        Kprod *= fidelity_kernel(s1, s1)
    check("no-entanglement kernel == product of one-qubit kernels", np.allclose(K, Kprod))
    check("no-entanglement purity == 1", abs(mean_single_qubit_purity(S, n) - 1) < 1e-10)
    pe = mean_single_qubit_purity(feature_map_states(X, 2, pairs, True), n)
    check("entangled circuit purity < 1 (entanglement present)", pe < 0.99, f"(mean purity {pe:.3f})")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# =========================================================================== #
# Models
# =========================================================================== #
class QuantumKernelSVC:
    """SVM on the exact fidelity kernel of the ZZ feature map (or its product-state ablation)."""

    def __init__(self, alpha=0.5, C=1.0, reps=2, topology="linear", entangle=True):
        self.alpha, self.C, self.reps, self.topology, self.entangle = alpha, C, reps, topology, entangle

    def _angles(self, X):
        u = (X - self.lo_) / self.span_
        return self.alpha * np.pi * np.clip(u, -0.1, 1.1)

    def _states(self, X):
        return feature_map_states(self._angles(X), self.reps, self.pairs_, self.entangle)

    def fit(self, X, y):
        self.lo_ = X.min(axis=0)
        self.span_ = np.maximum(X.max(axis=0) - self.lo_, 1e-12)
        self.pairs_ = entangling_pairs(X.shape[1], self.topology)
        self.S_ = self._states(X)
        K = fidelity_kernel(self.S_, self.S_)
        self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight="balanced").fit(K, y)
        off = K[~np.eye(len(K), dtype=bool)]
        self.diag_ = dict(kernel_offdiag_mean=float(off.mean()), kernel_offdiag_std=float(off.std()),
                          mean_qubit_purity=mean_single_qubit_purity(self.S_[:200], X.shape[1]))
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(fidelity_kernel(self._states(X), self.S_))


def model_grid(name: str, n_features: int, seed: int, reps: int, topology: str, x_var: float = 1.0) -> list:
    """Every matched model gets exactly 8 configurations."""
    if name in ("logreg", "logreg_full"):
        return [LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
                for c in np.logspace(-3, 2, 8)]
    if name == "rbf_svm":
        g0 = 1.0 / (n_features * max(x_var, 1e-12))  # sklearn 'scale' rule: adapts to feature variance
        return [SVC(C=c, gamma=g, class_weight="balanced")
                for c in (0.1, 1, 10, 100) for g in (0.3 * g0, 3 * g0)]
    if name in ("gbt", "gbt_full"):
        return [HistGradientBoostingClassifier(learning_rate=lr, max_depth=d, max_iter=it,
                                               class_weight="balanced", random_state=seed)
                for lr in (0.03, 0.1) for d in (2, 3) for it in (100, 300)]
    if name == "mlp":
        return [MLPClassifier(hidden_layer_sizes=h, alpha=a, learning_rate_init=lr, max_iter=2000,
                              early_stopping=False, random_state=seed)
                for h in ((16,), (32,)) for a in (1e-4, 1e-2) for lr in (1e-3, 1e-2)]
    if name in ("q_entangled", "q_product"):
        ent = name == "q_entangled"
        return [QuantumKernelSVC(alpha=a, C=c, reps=reps, topology=topology, entangle=ent)
                for a in (0.1, 0.25, 0.5, 1.0) for c in (1.0, 10.0)]
    raise ValueError(name)


def score(model, X) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def safe_auc(y, s) -> float:
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def clone_cfg(m):
    from sklearn.base import clone
    if isinstance(m, QuantumKernelSVC):
        return QuantumKernelSVC(m.alpha, m.C, m.reps, m.topology, m.entangle)
    return clone(m)


def tune_and_fit(name, X, y, groups, seed, args):
    grid = model_grid(name, X.shape[1], seed, args.reps, args.topology, x_var=float(X.var()))
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + 1000)
    splits = [(a, b) for a, b in inner.split(X, y, groups) if 0 < y[a].sum() < len(a) and 0 < y[b].sum() < len(b)]
    best, best_auc = 0, -np.inf
    if len(grid) > 1 and splits:
        for k, cfg in enumerate(grid):
            aucs = []
            for a, b in splits:
                m = clone_cfg(cfg).fit(X[a], y[a])
                aucs.append(safe_auc(y[b], score(m, X[b])))
            mean_auc = np.nanmean(aucs)
            if mean_auc > best_auc:
                best, best_auc = k, mean_auc
    return clone_cfg(grid[best]).fit(X, y), best


# =========================================================================== #
# Experiment
# =========================================================================== #
def run_fold(Xraw, y, groups, tr, te, seed, args):
    scaler = StandardScaler().fit(Xraw[tr])
    pca = PCA(n_components=args.n_qubits, random_state=seed).fit(scaler.transform(Xraw[tr]))
    Z = pca.transform(scaler.transform(Xraw))
    Xs_full = scaler.transform(Xraw)
    out, diag = {}, {}
    for name in MATCHED + (REFERENCE if not args.no_reference else []):
        X = Xs_full if name.endswith("_full") else Z
        model, k = tune_and_fit(name, X[tr], y[tr], groups[tr], seed, args)
        out[name] = score(model, X[te])
        diag[name] = dict(chosen_config=k, **getattr(model, "diag_", {}))
    return out, diag, float(pca.explained_variance_ratio_.sum())


def cluster_bootstrap_delta(y, s1, s2, groups, n_boot, rng, metric):
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    base = metric(y, s1) - metric(y, s2)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        if 0 < y[idx].sum() < len(idx):
            deltas.append(metric(y[idx], s1[idx]) - metric(y[idx], s2[idx]))
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return base, lo, hi, min(p, 1.0)


def make_synthetic(n=612, n_groups=29, n_feat=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = rng.integers(1, n_groups + 1, size=n)
    group_shift = rng.normal(0, 0.5, size=n_groups + 1)[groups]
    latent = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, n_feat))
    F = latent @ W + rng.normal(scale=1.5, size=(n, n_feat))
    logit = 1.4 * latent[:, 0] - 1.0 * latent[:, 1] * latent[:, 2] + group_shift + 1.3
    dice = 1 / (1 + np.exp(-logit))
    dice = np.clip(dice + rng.normal(0, 0.05, n), 0, 1)
    df = pd.DataFrame(F, columns=[f"f_{i:03d}" for i in range(n_feat)])
    df["u_noisy_confidence"] = -logit + rng.normal(0, 1.5, n)
    df["u_useless"] = rng.normal(size=n)
    df.insert(0, "dice", dice)
    df.insert(0, "sequence_id", groups)
    df.insert(0, "sample_id", [f"syn_{i}.png" for i in range(n)])
    return df


def plot_all(out: Path, y_by_t, scores_by_t, dice, summary, selective, primary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plots skipped: {e})")
        return
    names = MATCHED
    colors = {"logreg": "#7f7f7f", "rbf_svm": "#1f77b4", "gbt": "#2ca02c", "mlp": "#9467bd",
              "q_entangled": "#d62728", "q_product": "#ff7f0e"}
    y, sc = y_by_t[primary], scores_by_t[primary]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for nme in names:
        fpr, tpr, _ = roc_curve(y, sc[nme])
        ax.plot(fpr, tpr, color=colors[nme], lw=2 if nme.startswith("q_") else 1.3,
                label=f"{nme} ({roc_auc_score(y, sc[nme]):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="k", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"Failure detection (Dice < {primary})")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "roc_primary.png", dpi=200); plt.close(fig)

    s = summary[(summary.threshold == primary) & summary.model.isin(names)].set_index("model").loc[names]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(range(len(names)), s.auroc_mean, yerr=s.auroc_sd, color=[colors[n] for n in names], capsize=3)
    ax.axhline(0.5, ls=":", color="k", lw=0.8)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="AUROC (mean ± SD over seeds)", ylim=(0.4, 1.0), title=f"Dice < {primary}")
    fig.tight_layout(); fig.savefig(out / "auroc_bar.png", dpi=200); plt.close(fig)

    sel = selective[selective.threshold == primary]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for nme in names + ["oracle"]:
        r = sel[sel.model == nme]
        ax.plot(r.reject_rate * 100, r.mean_dice_kept, marker="o", ms=3,
                color=colors.get(nme, "k"), ls="--" if nme == "oracle" else "-", label=nme)
    ax.axhline(dice.mean(), color="k", lw=0.8, ls=":", label="random rejection")
    ax.set(xlabel="Frames rejected (%)", ylabel="Mean Dice of kept frames", title="Selective segmentation")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "selective_dice.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("failure_results"))
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.2],
                    help="Failure = Dice < threshold. The first one is the primary endpoint.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--topology", choices=["linear", "ring", "full"], default="linear")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-reference", action="store_true", help="Skip logreg_full / gbt_full")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel (seed, fold) jobs")
    ap.add_argument("--feature-prefix", default="f_",
                    help="Use feature columns with this prefix, e.g. f_hm_ (heatmap/image stats) or f_ (all)")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.self_test and not self_test():
        sys.exit("Self-test failed; not running the benchmark.")

    if args.synthetic:
        df = make_synthetic()
    elif args.features:
        df = pd.read_csv(args.features)
    else:
        sys.exit("Give --features CSV, --synthetic or --self-test")
    feat_cols = [c for c in df.columns if c.startswith(args.feature_prefix)]
    for col in ("sample_id", "sequence_id", "dice"):
        if col not in df.columns:
            sys.exit(f"Missing column {col!r}")
    if len(feat_cols) < args.n_qubits:
        sys.exit(f"Need at least {args.n_qubits} feature columns starting with {args.feature_prefix!r}, "
                 f"found {len(feat_cols)}")
    df = df.dropna(subset=["dice"]).reset_index(drop=True)
    unc_cols = [c for c in df.columns if c.startswith("u_")]  # training-free scores, higher = more suspicious
    Xraw = np.nan_to_num(df[feat_cols].to_numpy(np.float64))
    groups = df.sequence_id.to_numpy()
    dice = df.dice.to_numpy(np.float64)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"{len(df)} frames, {len(np.unique(groups))} sequences, {len(feat_cols)} raw features, "
          f"{args.n_qubits} qubits, seeds={args.seeds}, folds={args.folds}", flush=True)

    rows, diag_rows, oof_rows = [], [], []
    y_by_t, scores_by_t = {}, {}
    t_start = time.time()
    for t in args.thresholds:
        y = (dice < t).astype(int)
        print(f"\n=== failure = Dice < {t}: {y.sum()} failures / {len(y)} ({y.mean():.1%})", flush=True)
        if y.sum() < 2 * args.folds:
            print("  too few failures for this threshold, skipped")
            continue
        jobs = []
        for seed in args.seeds:
            cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            for f, (tr, te) in enumerate(cv.split(Xraw, y, groups)):
                jobs.append((seed, f, tr, te))
        if args.jobs > 1:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=args.jobs)(delayed(run_fold)(Xraw, y, groups, tr, te, s, args)
                                             for s, f, tr, te in jobs)
        else:
            res = []
            for s, f, tr, te in jobs:
                res.append(run_fold(Xraw, y, groups, tr, te, s, args))
                print(f"  seed {s} fold {f} done ({time.time() - t_start:.0f}s)", flush=True)

        model_names = list(res[0][0].keys())
        oof = {s: {m: np.full(len(y), np.nan) for m in model_names} for s in args.seeds}
        for (s, f, tr, te), (sc, dg, var) in zip(jobs, res):
            for m in model_names:
                oof[s][m][te] = sc[m]
                diag_rows.append(dict(threshold=t, seed=s, fold=f, model=m, pca_var_retained=var, **dg[m]))
        for s in args.seeds:
            for m in model_names:
                rows.append(dict(threshold=t, seed=s, model=m, n_failures=int(y.sum()),
                                 auroc=safe_auc(y, oof[s][m]),
                                 auprc=average_precision_score(y, oof[s][m])))
        # seed-averaged scores: z-score within seed so scales are comparable, then average
        avg = {m: np.mean([(oof[s][m] - oof[s][m].mean()) / (oof[s][m].std() + 1e-12) for s in args.seeds], axis=0)
               for m in model_names}
        # training-free uncertainty baselines: fixed direction, nothing fit, identical across seeds
        for c in unc_cols:
            raw = np.nan_to_num(df[c].to_numpy(np.float64))
            for s in args.seeds:
                rows.append(dict(threshold=t, seed=s, model=c, n_failures=int(y.sum()),
                                 auroc=safe_auc(y, raw), auprc=average_precision_score(y, raw)))
            avg[c] = (raw - raw.mean()) / (raw.std() + 1e-12)
        y_by_t[t], scores_by_t[t] = y, avg
        for i in range(len(y)):
            oof_rows.append(dict(threshold=t, sample_id=df.sample_id[i], sequence_id=groups[i], dice=dice[i],
                                 failure=y[i], **{m: avg[m][i] for m in avg}))

    per_seed = pd.DataFrame(rows)
    summary = (per_seed.groupby(["threshold", "model"])
               .agg(n_failures=("n_failures", "first"),
                    auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                    auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"))
               .reset_index())
    summary["prevalence"] = summary.n_failures / len(df)

    # paired comparisons on seed-averaged scores, cluster bootstrap over sequences
    rng = np.random.default_rng(12345)
    comp_rows, sel_rows = [], []
    for t in y_by_t:
        y, sc = y_by_t[t], scores_by_t[t]
        s_t = summary[summary.threshold == t].set_index("model")
        best_cl = s_t.loc[CLASSICAL_8D].auroc_mean.idxmax()
        pairs = [("q_entangled", "q_product", "Does entanglement help? (main ablation)"),
                 ("q_entangled", best_cl, f"Quantum vs best classical on the same {args.n_qubits} features"),
                 ("q_entangled", "rbf_svm", "Quantum kernel vs classical kernel"),
                 ("q_product", "rbf_svm", "Product-state circuit vs classical kernel")]
        if unc_cols:
            best_unc = s_t.loc[unc_cols].auroc_mean.idxmax()
            pairs += [("q_entangled", best_unc, "Quantum vs best training-free uncertainty score"),
                      (best_cl, best_unc, "Best classical vs best training-free uncertainty score")]
        for a, b, question in pairs:
            for metric_name, metric in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
                d, lo, hi, p = cluster_bootstrap_delta(y, sc[a], sc[b], groups, args.n_boot, rng, metric)
                wins = int(sum(per_seed[(per_seed.threshold == t) & (per_seed.model == a)][metric_name].values >
                               per_seed[(per_seed.threshold == t) & (per_seed.model == b)][metric_name].values))
                comp_rows.append(dict(threshold=t, question=question, model_a=a, model_b=b, metric=metric_name,
                                      delta=d, ci95_low=lo, ci95_high=hi, p_value=p,
                                      seeds_a_better=f"{wins}/{len(args.seeds)}",
                                      ci_excludes_zero=bool(lo > 0 or hi < 0)))
        order = np.argsort(dice)  # oracle rejects the truly worst frames
        for m in list(sc.keys()) + ["oracle"]:
            for r in (0.0, 0.1, 0.2, 0.3):
                k = int(round(r * len(y)))
                if m == "oracle":
                    keep = np.ones(len(y), bool); keep[order[:k]] = False
                else:
                    keep = np.ones(len(y), bool); keep[np.argsort(-sc[m])[:k]] = False
                sel_rows.append(dict(threshold=t, model=m, reject_rate=r, mean_dice_kept=dice[keep].mean()))
    comparisons, selective = pd.DataFrame(comp_rows), pd.DataFrame(sel_rows)
    diagnostics = pd.DataFrame(diag_rows)

    per_seed.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    comparisons.to_csv(out / "comparisons.csv", index=False)
    selective.to_csv(out / "selective_dice.csv", index=False)
    diagnostics.to_csv(out / "diagnostics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out / "oof_scores.csv", index=False)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update(n_frames=len(df), n_sequences=int(len(np.unique(groups))), n_raw_features=len(feat_cols),
               python=platform.python_version(), numpy=np.__version__,
               sklearn=__import__("sklearn").__version__, runtime_s=round(time.time() - t_start, 1))
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2))
    primary = args.thresholds[0]
    if primary in y_by_t:
        plot_all(out, y_by_t, scores_by_t, dice, summary, selective, primary)
    write_report(out, summary, comparisons, selective, diagnostics, cfg)
    print(f"\nDone in {time.time() - t_start:.0f}s. See {out / 'REPORT.md'}")


def write_report(out, summary, comparisons, selective, diagnostics, cfg):
    L = ["# Failure-detection benchmark: what does the quantum circuit add?", "",
         f"{cfg['n_frames']} frames, {cfg['n_sequences']} sequences, {cfg['n_raw_features']} raw features "
         f"compressed to {cfg['n_qubits']} PCA dims (= qubits). {cfg['folds']}-fold CV grouped by sequence, "
         f"seeds {cfg['seeds']}. Circuit: ZZ feature map, reps={cfg['reps']}, {cfg['topology']} entanglement.", ""]
    for t in sorted(summary.threshold.unique(), reverse=True):
        s = summary[summary.threshold == t].copy()
        L += [f"## Failure = Dice < {t}  ({int(s.n_failures.iloc[0])} failures, prevalence {s.prevalence.iloc[0]:.1%})", "",
              "| Model | AUROC | AUPRC |", "|---|---|---|"]
        unc = sorted(m for m in s.model if m.startswith("u_"))
        for m in MATCHED + REFERENCE + unc:
            r = s[s.model == m]
            if len(r):
                r = r.iloc[0]
                tag = (" *(reference: all raw features)*" if m in REFERENCE else
                       " *(training-free U-Net uncertainty)*" if m in unc else "")
                L.append(f"| {m}{tag} | {r.auroc_mean:.3f} ± {r.auroc_sd:.3f} | {r.auprc_mean:.3f} ± {r.auprc_sd:.3f} |")
        L += ["", "AUPRC of a random guess equals the prevalence.", "",
              "**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):", "",
              "| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |",
              "|---|---|---|---|---|---|---|"]
        for r in comparisons[comparisons.threshold == t].itertuples():
            L.append(f"| {r.question}: {r.model_a} vs {r.model_b} | {r.metric} | {r.delta:+.3f} | "
                     f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] | {r.p_value:.3f} | {r.seeds_a_better} | "
                     f"{'yes' if r.ci_excludes_zero else 'no'} |")
        sel = selective[(selective.threshold == t) & selective.model.isin(MATCHED + unc + ["oracle"])]
        piv = sel.pivot(index="model", columns="reject_rate", values="mean_dice_kept")
        L += ["", "**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction", "",
              "| Model | " + " | ".join(f"reject {int(c * 100)}%" for c in piv.columns) + " |",
              "|---|" + "---|" * len(piv.columns)]
        for m in MATCHED + unc + ["oracle"]:
            if m in piv.index:
                L.append(f"| {m} | " + " | ".join(f"{v:.3f}" for v in piv.loc[m]) + " |")
        dq = diagnostics[(diagnostics.threshold == t) & diagnostics.model.isin(["q_entangled", "q_product"])]
        if "mean_qubit_purity" in dq:
            g = dq.groupby("model")[["kernel_offdiag_mean", "kernel_offdiag_std", "mean_qubit_purity"]].mean()
            L += ["", "**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; "
                  "off-diagonal kernel mean near 0 with tiny SD = kernel concentration.", "",
                  "| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |", "|---|---|---|---|"]
            for m, r in g.iterrows():
                L.append(f"| {m} | {r.kernel_offdiag_mean:.3f} | {r.kernel_offdiag_std:.3f} | {r.mean_qubit_purity:.3f} |")
        L.append("")
    L += ["## How to read this", "",
          "* The main question is the first comparison row: q_entangled vs q_product. Same circuit, same inputs, "
          "same tuning; the only difference is the entangling gates. If its CI includes 0, the data show no "
          "evidence that entanglement helps.",
          "* 'Best classical' is chosen after seeing the results, which favours the classical side; that makes a "
          "positive quantum result more convincing, not less.",
          "* The *_full rows use all raw features and are not matched; they show the cost of compressing to "
          f"{cfg['n_qubits']} dims.",
          "* All quantum results are exact statevector simulations, not quantum hardware.", ""]
    (out / "REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
