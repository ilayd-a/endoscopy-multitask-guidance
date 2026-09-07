"""
benchmark_v2.py
===============
Head-to-head comparison: v1 (baseline) vs v2 (improved).

Quick run (synthetic):
    python benchmark_v2.py --synthetic --max_samples 80 --fast
"""

import argparse, time, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.utils import resample

warnings.filterwarnings("ignore")
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def make_synthetic(n_samples, n_features, seed=42):
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import MinMaxScaler
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_features, n_redundant=0, random_state=seed,
    )
    return MinMaxScaler(feature_range=(-np.pi, np.pi)).fit_transform(X).astype(np.float64), y.astype(int)


def balance_classes(X, y, seed=42):
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = max(len(idx0), len(idx1))
    if len(idx0) < n:
        idx0 = resample(idx0, replace=True, n_samples=n, random_state=seed)
    else:
        idx1 = resample(idx1, replace=True, n_samples=n, random_state=seed)
    idx = np.concatenate([idx0, idx1])
    np.random.default_rng(seed).shuffle(idx)
    return X[idx], y[idx]


# ─────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────

def get_v1_models(fast):
    # Original implementation (n_qubits=4, ZZFeatureMap, RealAmplitudes)
    sys.path.insert(0, str(Path(__file__).parent))
    from models import (QSVMClassifier as QSVM1, VQCClassifier as VQC1,
                        QKernelSVMClassifier as QK1, QNNClassifier as QNN1)
    return {
        "QSVM":        QSVM1(n_qubits=4),
        "VQC":         VQC1 (n_qubits=4, max_iter=30 if fast else 100),
        "QKernel+SVM": QK1  (n_qubits=4),
        "QNN":         QNN1 (n_qubits=4, epochs=10 if fast else 30),
    }


def get_v2_models(fast):
    # Improved (n_qubits=6, PauliFeatureMap, EfficientSU2, Z-sum obs)
    from models_v2 import (QSVMClassifier as QSVM2, VQCClassifier as VQC2,
                           QKernelSVMClassifier as QK2, QNNClassifier as QNN2)
    return {
        "QSVM":        QSVM2(n_qubits=6),
        "VQC":         VQC2 (n_qubits=6, max_iter=30 if fast else 150),
        "QKernel+SVM": QK2  (n_qubits=6),
        "QNN":         QNN2 (n_qubits=6, epochs=10 if fast else 40),
    }


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t  = time.time() - t0
    yp = model.predict(X_te)
    acc = accuracy_score(y_te, yp)
    f1  = f1_score(y_te, yp, zero_division=0)
    try:
        m = getattr(model, "model", None)
        if m and hasattr(m, "predict_proba"):
            prob = m.predict_proba(X_te)[:, 1]
        elif m and hasattr(m, "decision_function"):
            prob = m.decision_function(X_te)
        else:
            prob = yp.astype(float)
        auc = roc_auc_score(y_te, prob)
    except Exception:
        auc = float("nan")
    return dict(acc=acc, f1=f1, auc=auc, time=t,
                yp=yp, cm=confusion_matrix(y_te, yp))


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

MODEL_NAMES = ["QSVM", "VQC", "QKernel+SVM", "QNN"]
V1_COLOR    = "#4C72B0"
V2_COLOR    = "#DD8452"
BG          = "#0f0f1a"
GRID_C      = "#2a2a40"
LABEL_C     = "#aaaacc"


def _ax_style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=LABEL_C, labelsize=8)
    ax.yaxis.grid(True, color=GRID_C, linestyle="--", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_C)


def plot_comparison(v1: dict, v2: dict, save: Path):
    metrics = ["acc", "f1", "auc", "time"]
    mlabels = ["Accuracy", "F1 Score", "ROC-AUC", "Train Time (s)"]

    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        "QML Endoscopy — Baseline (v1) vs Improved (v2)",
        color="white", fontsize=16, fontweight="bold", y=0.97,
    )

    outer = gridspec.GridSpec(3, 1, figure=fig,
                              hspace=0.55, top=0.92, bottom=0.06,
                              left=0.06, right=0.97)

    # ── Row 0: grouped bar charts (4 metrics) ─────────────────────────
    bar_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.35)
    x      = np.arange(len(MODEL_NAMES))
    w      = 0.35

    for col, (met, mlab) in enumerate(zip(metrics, mlabels)):
        ax = fig.add_subplot(bar_gs[col])
        v1_vals = [v1[n][met] for n in MODEL_NAMES]
        v2_vals = [v2[n][met] for n in MODEL_NAMES]
        b1 = ax.bar(x - w/2, v1_vals, w, color=V1_COLOR, label="v1 baseline", zorder=3)
        b2 = ax.bar(x + w/2, v2_vals, w, color=V2_COLOR, label="v2 improved", zorder=3)
        _ax_style(ax, mlab)
        ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES, rotation=20, ha="right", fontsize=8)
        if met != "time": ax.set_ylim(0, 1.18)
        for bar, v in zip(list(b1)+list(b2), v1_vals+v2_vals):
            if not np.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{v:.2f}", ha="center", va="bottom", color="white", fontsize=7)
        if col == 0:
            ax.legend(fontsize=8, labelcolor="white",
                      facecolor="#1a1a2e", edgecolor=GRID_C, loc="upper right")

    # ── Row 1: delta-accuracy bar (v2 - v1) ───────────────────────────
    delta_ax = fig.add_subplot(outer[1])
    deltas   = [v2[n]["acc"] - v1[n]["acc"] for n in MODEL_NAMES]
    bar_colors = [V2_COLOR if d >= 0 else "#C44E52" for d in deltas]
    bars = delta_ax.bar(MODEL_NAMES, deltas, color=bar_colors, width=0.45, zorder=3)
    _ax_style(delta_ax, "Accuracy Improvement  (v2 − v1)")
    delta_ax.axhline(0, color=LABEL_C, lw=0.8)
    for bar, d in zip(bars, deltas):
        ypos = bar.get_height() + 0.003 if d >= 0 else bar.get_height() - 0.015
        delta_ax.text(bar.get_x()+bar.get_width()/2, ypos,
                      f"{d:+.3f}", ha="center", va="bottom", color="white", fontsize=9)

    # ── Row 2: confusion matrices (v2 only) ───────────────────────────
    cm_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], wspace=0.4)
    for col, name in enumerate(MODEL_NAMES):
        ax = fig.add_subplot(cm_gs[col])
        ax.set_facecolor(BG)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=v2[name]["cm"],
            display_labels=["sparse", "rich"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(f"v2 · {name}\nConfusion Matrix",
                     color="white", fontsize=9, fontweight="bold", pad=6)
        ax.tick_params(colors=LABEL_C, labelsize=8)
        ax.xaxis.label.set_color(LABEL_C)
        ax.yaxis.label.set_color(LABEL_C)
        for txt in ax.texts: txt.set_color("white")

    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n[benchmark_v2] Plot saved → {save}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/Kvasir-SEG")
    parser.add_argument("--max_samples", type=int,   default=200)
    parser.add_argument("--test_size",   type=float, default=0.25)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--synthetic",   action="store_true")
    parser.add_argument("--fast",        action="store_true")
    args = parser.parse_args()

    N_QUBITS_V1 = 4
    N_QUBITS_V2 = 6

    # ── Load data ─────────────────────────────
    if args.synthetic:
        print("[data] Synthetic mode.")
        # v1 uses 4-dim features, v2 uses 6-dim
        X4, y = make_synthetic(args.max_samples, N_QUBITS_V1, args.seed)
        X6, _ = make_synthetic(args.max_samples, N_QUBITS_V2, args.seed)
    else:
        from data_loader import extract_features
        X4, y = extract_features(args.data_dir, N_QUBITS_V1, args.max_samples, args.seed)
        X6, _ = extract_features(args.data_dir, N_QUBITS_V2, args.max_samples, args.seed)

    # Balance classes (improvement #4)
    X4b, y4b = balance_classes(X4, y, args.seed)
    X6b, y6b = balance_classes(X6, y, args.seed)
    print(f"[data] After balancing: N={len(y4b)}, class ratio={y4b.mean():.2f}")

    def split(X, y):
        return train_test_split(X, y, test_size=args.test_size,
                                stratify=y, random_state=args.seed)

    X4_tr, X4_te, y4_tr, y4_te = split(X4b, y4b)
    X6_tr, X6_te, y6_tr, y6_te = split(X6b, y6b)

    # ── Run v1 ────────────────────────────────
    print("\n" + "="*55)
    print(" V1  BASELINE  (n_qubits=4, ZZFeatureMap, RealAmplitudes)")
    print("="*55)
    v1_results = {}
    for name, model in get_v1_models(args.fast).items():
        print(f"\n[v1] {name} ...")
        try:
            r = evaluate(model, X4_tr, y4_tr, X4_te, y4_te)
            v1_results[name] = r
            print(f"  acc={r['acc']:.3f}  f1={r['f1']:.3f}  auc={r['auc']:.3f}  t={r['time']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            v1_results[name] = dict(acc=0, f1=0, auc=0, time=0,
                                    yp=np.array([]), cm=np.zeros((2,2)))

    # ── Run v2 ────────────────────────────────
    print("\n" + "="*55)
    print(" V2  IMPROVED  (n_qubits=6, PauliFeatureMap, EfficientSU2)")
    print("="*55)
    v2_results = {}
    for name, model in get_v2_models(args.fast).items():
        print(f"\n[v2] {name} ...")
        try:
            r = evaluate(model, X6_tr, y6_tr, X6_te, y6_te)
            v2_results[name] = r
            print(f"  acc={r['acc']:.3f}  f1={r['f1']:.3f}  auc={r['auc']:.3f}  t={r['time']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            v2_results[name] = dict(acc=0, f1=0, auc=0, time=0,
                                    yp=np.array([]), cm=np.zeros((2,2)))

    # ── Summary table ─────────────────────────
    rows = []
    for n in MODEL_NAMES:
        r1, r2 = v1_results[n], v2_results[n]
        rows.append({
            "Model": n,
            "v1 Acc": round(r1["acc"], 4), "v2 Acc": round(r2["acc"], 4),
            "Δ Acc":  round(r2["acc"] - r1["acc"], 4),
            "v1 F1":  round(r1["f1"],  4), "v2 F1":  round(r2["f1"],  4),
            "v2 AUC": round(r2["auc"], 4),
        })
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "comparison_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[results] CSV saved → {csv_path}")
    print("\n" + df.to_string(index=False))

    # ── Plot ──────────────────────────────────
    plot_comparison(v1_results, v2_results, RESULTS_DIR / "comparison.png")


if __name__ == "__main__":
    main()
