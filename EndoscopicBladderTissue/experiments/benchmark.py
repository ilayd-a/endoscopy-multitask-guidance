"""
benchmark.py
============
Compare all four QML models on Kvasir-SEG.

Usage
-----
# With real Kvasir-SEG data:
python benchmark.py --data_dir data/Kvasir-SEG --n_qubits 4 --max_samples 200

# Quick smoke-test with synthetic data (no dataset download needed):
python benchmark.py --synthetic --n_qubits 4 --max_samples 80

Results are saved to results/benchmark_results.png and results/metrics.csv.
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Synthetic data fallback
# ─────────────────────────────────────────────

def make_synthetic(n_samples: int = 120, n_features: int = 4, seed: int = 42):
    """Two Gaussian blobs scaled to [-π, π] as a stand-in for real features."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        random_state=seed,
    )
    # Rescale to [-π, π]
    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit_transform(X)
    return X.astype(np.float64), y.astype(int)


# ─────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────

def get_models(n_qubits: int, fast: bool = False):
    from models import QSVMClassifier, VQCClassifier, QKernelSVMClassifier, QNNClassifier

    max_iter = 30 if fast else 100
    epochs   = 10 if fast else 30

    return {
        "QSVM":         QSVMClassifier(n_qubits=n_qubits),
        "VQC":          VQCClassifier (n_qubits=n_qubits, max_iter=max_iter),
        "QKernel+SVM":  QKernelSVMClassifier(n_qubits=n_qubits),
        "QNN":          QNNClassifier (n_qubits=n_qubits, epochs=epochs),
    }


# ─────────────────────────────────────────────
# Single model evaluation
# ─────────────────────────────────────────────

def evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, zero_division=0)

    # AUC – fall back gracefully if model has no probability output
    try:
        if hasattr(model.model, "predict_proba"):
            proba = model.model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "model") and hasattr(model.model, "decision_function"):
            proba = model.model.decision_function(X_test)
        else:
            proba = y_pred.astype(float)
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy":   acc,
        "f1":         f1,
        "auc":        auc,
        "train_time": train_time,
        "y_pred":     y_pred,
        "cm":         cm,
    }


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

COLORS = {
    "QSVM":        "#4C72B0",
    "VQC":         "#DD8452",
    "QKernel+SVM": "#55A868",
    "QNN":         "#C44E52",
}


def plot_results(records: list[dict], save_path: Path):
    names    = [r["name"]       for r in records]
    accs     = [r["accuracy"]   for r in records]
    f1s      = [r["f1"]         for r in records]
    aucs     = [r["auc"]        for r in records]
    times    = [r["train_time"] for r in records]
    cms      = [r["cm"]         for r in records]
    colors   = [COLORS[n]       for n in names]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f0f1a")

    title_kw  = dict(color="white",  fontsize=13, fontweight="bold", pad=10)
    label_kw  = dict(color="#aaaacc", fontsize=10)
    tick_kw   = dict(colors="#aaaacc", labelsize=9)
    grid_kw   = dict(color="#2a2a40", linestyle="--", linewidth=0.6)

    # Layout: 2 rows × 4 cols; last row has confusion matrices spanning 1 col each
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.88, bottom=0.08)

    # ── Row 0: bar charts ──────────────────────────────────────
    def bar_panel(ax, values, title, fmt="{:.3f}", ylim=None):
        bars = ax.bar(names, values, color=colors, width=0.5, zorder=3)
        ax.set_facecolor("#0f0f1a")
        ax.set_title(title, **title_kw)
        ax.tick_params(axis="x", **tick_kw)
        ax.tick_params(axis="y", **tick_kw)
        ax.yaxis.grid(True, **grid_kw)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a40")
        if ylim:
            ax.set_ylim(*ylim)
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        fmt.format(v), ha="center", va="bottom",
                        color="white", fontsize=9)

    bar_panel(fig.add_subplot(gs[0, 0]), accs,  "Accuracy",         ylim=(0, 1.15))
    bar_panel(fig.add_subplot(gs[0, 1]), f1s,   "F1 Score",         ylim=(0, 1.15))
    bar_panel(fig.add_subplot(gs[0, 2]), aucs,  "ROC-AUC",          ylim=(0, 1.15))
    bar_panel(fig.add_subplot(gs[0, 3]), times, "Train Time (sec)", fmt="{:.1f}")

    # ── Row 1: confusion matrices ──────────────────────────────
    for col, (name, cm) in enumerate(zip(names, cms)):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor("#0f0f1a")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["sparse", "rich"])
        disp.plot(ax=ax, colorbar=False,
                  cmap="Blues", values_format="d")
        ax.set_title(f"{name}\nConfusion Matrix", **title_kw)
        ax.tick_params(axis="both", **tick_kw)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        for txt in ax.texts:
            txt.set_color("white")

    # ── Super title ───────────────────────────────────────────
    fig.suptitle(
        "QML Benchmark — Kvasir-SEG Polyp Classification",
        color="white", fontsize=16, fontweight="bold", y=0.96,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n[benchmark] Plot saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QML Endoscopy Benchmark")
    parser.add_argument("--data_dir",    type=str,  default="data/Kvasir-SEG",
                        help="Path to Kvasir-SEG root (containing images/ and masks/)")
    parser.add_argument("--n_qubits",    type=int,  default=4)
    parser.add_argument("--max_samples", type=int,  default=200,
                        help="Max samples to use (QML is slow)")
    parser.add_argument("--test_size",   type=float, default=0.25)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--synthetic",   action="store_true",
                        help="Use synthetic data (no Kvasir download needed)")
    parser.add_argument("--fast",        action="store_true",
                        help="Reduce iterations for quick testing")
    args = parser.parse_args()

    # ── Data ──────────────────────────────────────────────────
    if args.synthetic:
        print("[benchmark] Using synthetic data (smoke-test mode).")
        X, y = make_synthetic(n_samples=args.max_samples,
                              n_features=args.n_qubits,
                              seed=args.seed)
    else:
        print("[benchmark] Loading Kvasir-SEG features...")
        from data_loader import extract_features
        X, y = extract_features(
            data_dir=args.data_dir,
            n_features=args.n_qubits,
            max_samples=args.max_samples,
            seed=args.seed,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size,
        stratify=y, random_state=args.seed,
    )
    print(f"[benchmark] Train={len(X_train)}, Test={len(X_test)}, "
          f"Class balance (train): {y_train.mean():.2f}")

    # ── Models ────────────────────────────────────────────────
    models  = get_models(n_qubits=args.n_qubits, fast=args.fast)
    records = []

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"[benchmark] Running {name} ...")
        try:
            result = evaluate(model, X_train, y_train, X_test, y_test)
            result["name"] = name
            records.append(result)
            print(f"  ✓ Accuracy={result['accuracy']:.3f} | "
                  f"F1={result['f1']:.3f} | "
                  f"AUC={result['auc']:.3f} | "
                  f"Time={result['train_time']:.1f}s")
        except Exception as exc:
            print(f"  ✗ {name} failed: {exc}")

    if not records:
        print("\n[benchmark] All models failed. Exiting.")
        return

    # ── Save metrics CSV ──────────────────────────────────────
    df = pd.DataFrame([{
        "Model":      r["name"],
        "Accuracy":   round(r["accuracy"],   4),
        "F1":         round(r["f1"],         4),
        "ROC-AUC":    round(r["auc"],        4),
        "Train(sec)": round(r["train_time"], 2),
    } for r in records])
    csv_path = RESULTS_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[benchmark] Metrics saved → {csv_path}")
    print(df.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────
    plot_results(records, RESULTS_DIR / "benchmark_results.png")


if __name__ == "__main__":
    main()
