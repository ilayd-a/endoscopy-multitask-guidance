"""
benchmark_ebtc.py
=================
QML benchmark on the EBTC dataset (Option A: binary cancer vs non-cancer).
Compares v1 (baseline, 4 qubits) against v2 (improved, 6 qubits).

Usage
-----
# With real EBTC data:
python benchmark_ebtc.py --data_dir data/EBTC --max_samples 300

# Smoke test (synthetic):
python benchmark_ebtc.py --synthetic --max_samples 80 --fast
"""

import argparse, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

warnings.filterwarnings("ignore")
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────

def make_synthetic(n, nf, seed=42):
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import MinMaxScaler
    X, y = make_classification(n_samples=n, n_features=nf,
                               n_informative=nf, n_redundant=0,
                               random_state=seed)
    return MinMaxScaler((-np.pi, np.pi)).fit_transform(X).astype(np.float64), y.astype(int)


# ─────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────

def get_models(version: int, fast: bool):
    if version == 1:
        from models import (QSVMClassifier, VQCClassifier,
                            QKernelSVMClassifier, QNNClassifier)
        return {
            "QSVM":        QSVMClassifier(n_qubits=4),
            "VQC":         VQCClassifier (n_qubits=4, max_iter=30 if fast else 100),
            "QKernel+SVM": QKernelSVMClassifier(n_qubits=4),
            "QNN":         QNNClassifier (n_qubits=4, epochs=10 if fast else 30),
        }
    else:
        from models_v2 import (QSVMClassifier, VQCClassifier,
                               QKernelSVMClassifier, QNNClassifier)
        return {
            "QSVM":        QSVMClassifier(n_qubits=6),
            "VQC":         VQCClassifier (n_qubits=6, max_iter=30 if fast else 150),
            "QKernel+SVM": QKernelSVMClassifier(n_qubits=6),
            "QNN":         QNNClassifier (n_qubits=6, epochs=10 if fast else 40),
        }


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(model, X_tr, y_tr, X_te, y_te) -> dict:
    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    yp  = model.predict(X_te)
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
    return dict(acc=acc, f1=f1, auc=auc, time=elapsed,
                yp=yp, cm=confusion_matrix(y_te, yp))


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

MODEL_NAMES = ["QSVM", "VQC", "QKernel+SVM", "QNN"]
BG   = "#0f0f1a"
GRD  = "#2a2a40"
LC   = "#aaaacc"
CV1  = "#4C72B0"
CV2  = "#DD8452"
CPOS = "#55A868"
CNEG = "#C44E52"
CNA  = "#555577"


def _ax(ax, title, ylim=None):
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=7)
    ax.tick_params(colors=LC, labelsize=8)
    ax.yaxis.grid(True, color=GRD, linestyle="--", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor(GRD)
    if ylim: ax.set_ylim(*ylim)


def plot_results(v1: dict, v2: dict, save: Path,
                 dataset_label: str = "EBTC"):
    x = np.arange(len(MODEL_NAMES)); w = 0.35

    fig = plt.figure(figsize=(20, 15), facecolor=BG)
    fig.suptitle(
        f"QML Bladder Tissue Classification ({dataset_label})\n"
        "Option A: Cancer (HGC+LGC) vs Non-Cancer (NST+NTL)\n"
        "v1: 4 qubits · ZZFeatureMap · RealAmplitudes    "
        "v2: 6 qubits · PauliFeatureMap · EfficientSU2",
        color="white", fontsize=12, fontweight="bold", y=0.98,
    )

    outer = gridspec.GridSpec(3, 1, figure=fig,
                              hspace=0.55, top=0.93, bottom=0.05,
                              left=0.06, right=0.97)

    # ── Row 0: grouped bars (Acc / F1 / AUC / Time) ──────────────
    row0 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.38)
    metrics = [("acc","Accuracy"), ("f1","F1 Score"),
               ("auc","ROC-AUC"), ("time","Train Time (s)")]

    for col, (met, mlab) in enumerate(metrics):
        ax = fig.add_subplot(row0[col])
        v1v = [v1.get(n, {}).get(met, 0) or 0 for n in MODEL_NAMES]
        v2v = [v2.get(n, {}).get(met, 0) or 0 for n in MODEL_NAMES]
        na2 = [v2.get(n) is None for n in MODEL_NAMES]

        b1 = ax.bar(x - w/2, v1v, w, color=CV1, label="v1 (4q)", zorder=3)
        for i in range(len(MODEL_NAMES)):
            c = CNA if na2[i] else CV2
            ax.bar(x[i]+w/2, v2v[i], w, color=c,
                   label="v2 (6q)" if i == 0 else "", zorder=3)

        ylim = (0, 1.2) if met != "time" else None
        _ax(ax, mlab, ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_NAMES, rotation=22, ha="right", fontsize=8)

        for bar, v in zip(b1, v1v):
            if not np.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                        f"{v:.2f}", ha="center", color="white", fontsize=7)
        for i in range(len(MODEL_NAMES)):
            if not na2[i] and not np.isnan(v2v[i]):
                ax.text(x[i]+w/2, v2v[i]+0.01, f"{v2v[i]:.2f}",
                        ha="center", color="white", fontsize=7)
            elif na2[i]:
                ax.text(x[i]+w/2, 0.03, "N/A", ha="center",
                        color=CNA, fontsize=7, rotation=90)
        if col == 0:
            ax.legend(fontsize=8, labelcolor="white",
                      facecolor="#1a1a2e", edgecolor=GRD)

    # ── Row 1: delta accuracy bar ─────────────────────────────────
    ax_d = fig.add_subplot(outer[1])
    deltas = []
    for n in MODEL_NAMES:
        if v2.get(n) is not None:
            deltas.append(v2[n]["acc"] - v1[n]["acc"])
        else:
            deltas.append(None)
    dv     = [d if d is not None else 0 for d in deltas]
    dcols  = [CPOS if (d is not None and d >= 0) else (CNEG if d is not None else CNA)
              for d in deltas]
    bars   = ax_d.bar(MODEL_NAMES, dv, color=dcols, width=0.5, zorder=3)
    _ax(ax_d, "Accuracy Improvement  Δ = v2 − v1  (positive = better)")
    ax_d.axhline(0, color=LC, lw=0.8)
    for bar, d in zip(bars, deltas):
        if d is not None:
            yp = bar.get_height() + 0.004 if d >= 0 else bar.get_height() - 0.018
            ax_d.text(bar.get_x()+bar.get_width()/2, yp,
                      f"{d:+.3f}", ha="center", color="white",
                      fontsize=11, fontweight="bold")
        else:
            ax_d.text(bar.get_x()+bar.get_width()/2, 0.005, "timeout",
                      ha="center", color=CNA, fontsize=9)
    ax_d.set_xticklabels(MODEL_NAMES, color=LC, fontsize=9)

    # ── Row 2: confusion matrices (v2) ───────────────────────────
    row2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], wspace=0.42)
    for col, name in enumerate(MODEL_NAMES):
        ax = fig.add_subplot(row2[col])
        ax.set_facecolor(BG)
        r   = v2.get(name)
        cm  = r["cm"] if r else np.zeros((2, 2), int)
        acc = r["acc"] if r else 0
        disp = ConfusionMatrixDisplay(cm, display_labels=["non-cancer", "cancer"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(f"v2 · {name}\nacc={acc:.3f}",
                     color="white", fontsize=9, fontweight="bold", pad=5)
        ax.tick_params(colors=LC, labelsize=7)
        ax.xaxis.label.set_color(LC); ax.yaxis.label.set_color(LC)
        for txt in ax.texts: txt.set_color("white")

    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n[benchmark_ebtc] Plot saved → {save}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QML EBTC Benchmark")
    parser.add_argument("--data_dir",    default="data/EBTC")
    parser.add_argument("--max_samples", type=int,   default=300)
    parser.add_argument("--test_size",   type=float, default=0.25)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--synthetic",   action="store_true")
    parser.add_argument("--fast",        action="store_true",
                        help="Reduce iterations for quick testing")
    args = parser.parse_args()

    # ── Load features ──────────────────────────────────────
    if args.synthetic:
        print("[data] Synthetic mode (no dataset required).")
        X4, y = make_synthetic(args.max_samples, 4, args.seed)
        X6, _ = make_synthetic(args.max_samples, 6, args.seed)
    else:
        print("[data] Loading EBTC features with ResNet-18 + PCA...")
        from data_loader_ebtc import extract_features
        X4, y = extract_features(args.data_dir, n_features=4,
                                 max_samples=args.max_samples, seed=args.seed)
        X6, _ = extract_features(args.data_dir, n_features=6,
                                 max_samples=args.max_samples, seed=args.seed)

    def split(X, y):
        return train_test_split(X, y, test_size=args.test_size,
                                stratify=y, random_state=args.seed)

    X4tr, X4te, y4tr, y4te = split(X4, y)
    X6tr, X6te, y6tr, y6te = split(X6, y)

    print(f"\n[data] Train={len(y4tr)}, Test={len(y4te)}, "
          f"cancer ratio (train)={y4tr.mean():.3f}")

    dataset_label = "Synthetic" if args.synthetic else "EBTC (real)"

    # ── v1 ─────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  V1  4 qubits | ZZFeatureMap | RealAmplitudes")
    print("="*55)
    v1_res = {}
    for name, model in get_models(1, args.fast).items():
        print(f"\n[v1] {name} ...")
        try:
            r = evaluate(model, X4tr, y4tr, X4te, y4te)
            v1_res[name] = r
            print(f"  acc={r['acc']:.3f}  f1={r['f1']:.3f}  "
                  f"auc={r['auc']:.3f}  t={r['time']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            v1_res[name] = dict(acc=0, f1=0, auc=0, time=0,
                                yp=np.array([]), cm=np.zeros((2,2),int))

    # ── v2 ─────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  V2  6 qubits | PauliFeatureMap | EfficientSU2")
    print("="*55)
    v2_res = {}
    for name, model in get_models(2, args.fast).items():
        print(f"\n[v2] {name} ...")
        try:
            r = evaluate(model, X6tr, y6tr, X6te, y6te)
            v2_res[name] = r
            print(f"  acc={r['acc']:.3f}  f1={r['f1']:.3f}  "
                  f"auc={r['auc']:.3f}  t={r['time']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            v2_res[name] = None

    # ── Summary table ───────────────────────────────────────
    rows = []
    for n in MODEL_NAMES:
        r1 = v1_res.get(n, {})
        r2 = v2_res.get(n) or {}
        rows.append({
            "Model":    n,
            "v1_Acc":   round(r1.get("acc", 0), 4),
            "v2_Acc":   round(r2.get("acc", 0), 4) if r2 else "timeout",
            "Delta":    round(r2.get("acc",0)-r1.get("acc",0),4) if r2 else "N/A",
            "v1_F1":    round(r1.get("f1",  0), 4),
            "v2_F1":    round(r2.get("f1",  0), 4) if r2 else "timeout",
            "v2_AUC":   round(r2.get("auc", 0), 4) if r2 else "timeout",
            "v1_Time":  round(r1.get("time",0), 1),
            "v2_Time":  round(r2.get("time",0), 1) if r2 else "timeout",
        })
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "ebtc_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[results] Saved → {csv_path}")
    print("\n" + df.to_string(index=False))

    # ── Plot ────────────────────────────────────────────────
    plot_results(v1_res, v2_res,
                 RESULTS_DIR / "ebtc_comparison.png",
                 dataset_label=dataset_label)


if __name__ == "__main__":
    main()
