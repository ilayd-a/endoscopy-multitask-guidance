#!/usr/bin/env python3
"""
make_figures.py
===============
Collect every paper figure into one folder (default failure_detection/figures/),
using the CORRECTED RBF-SVM wherever a run has a rbf_corrected/ subfolder.
Nothing is re-trained: figures are drawn from the saved out-of-fold scores.

Per run (Dice < 0.5):  <tag>_roc.png, <tag>_auroc_bar.png, <tag>_selective_dice.png
Across runs:           entanglement_forest.png  (Δ AUROC entangled − product, 95% CI)
                       qubit_scaling.png        (AUROC vs qubits, both datasets)
Copied:                cvc_noisy_sim_kernel_scatter.png
Index:                 figures/README.md (one line per figure + caption notes)

Run from the repo root:
  python quantum-medical-research/endoscopy_guidance/make_figures.py --root failure_detection
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = [  # tag, folder, dataset, qubits, feature set
    ("cvc_8q", "results_hm", "CVC-ClinicDB", 8, "heatmap"),
    ("cvc_4q", "results_hm_q4", "CVC-ClinicDB", 4, "heatmap"),
    ("cvc_12q", "results_hm_q12", "CVC-ClinicDB", 12, "heatmap"),
    ("cvc_8q_allfeat", "results_all", "CVC-ClinicDB", 8, "all features"),
    ("polypgen_8q", "results_polypgen_hm", "PolypGen", 8, "heatmap"),
    ("polypgen_4q", "results_polypgen_q4", "PolypGen", 4, "heatmap"),
]
MODELS = ["logreg", "rbf_svm", "gbt", "mlp", "q_entangled", "q_product"]
LABEL = {"logreg": "Logistic reg.", "rbf_svm": "RBF-SVM", "gbt": "Grad. boosting", "mlp": "MLP",
         "q_entangled": "Quantum (entangled)", "q_product": "Quantum (no entanglement)",
         "u_mask_entropy": "U-Net mask entropy (no training)"}
COLOR = {"logreg": "#8c8c8c", "rbf_svm": "#3a78b5", "gbt": "#3d9a50", "mlp": "#8a63b3",
         "q_entangled": "#d1453b", "q_product": "#e8913a", "u_mask_entropy": "#222222"}
T = 0.5


def load_run(run_dir: Path):
    oof = pd.read_csv(run_dir / "oof_scores.csv")
    oof = oof[np.isclose(oof.threshold, T)].reset_index(drop=True)
    per = pd.read_csv(run_dir / "per_seed.csv")
    per = per[np.isclose(per.threshold, T)]
    stats = {m: (per[per.model == m].auroc.mean(), per[per.model == m].auroc.std(ddof=1)) for m in per.model.unique()}
    corrected = False
    cdir = run_dir / "rbf_corrected"
    f = cdir / f"rbf_corrected_oof_t{T}.csv"
    if f.exists():
        c = pd.read_csv(f).set_index("sample_id")
        oof["rbf_svm"] = c.loc[oof.sample_id, "rbf_svm_corrected"].to_numpy()
        md = (cdir / "RBF_CORRECTED.md").read_text(encoding="utf-8")
        sec = md.split(f"## Dice < {T}")[1]
        mm = re.search(r"AUROC ([0-9.]+) ± ([0-9.]+)", sec)
        stats["rbf_svm"] = (float(mm.group(1)), float(mm.group(2)))
        corrected = True
    return oof, stats, corrected


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("failure_detection"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    index, scaling, missing = [], [], []
    for tag, folder, ds, q, feat in RUNS:
        rd = args.root / folder
        if not (rd / "oof_scores.csv").exists():
            missing.append(folder)
            continue
        oof, stats, corrected = load_run(rd)
        y, dice = oof.failure.to_numpy(), oof.dice.to_numpy()
        title = f"{ds}, {q} qubits, {feat} (Dice < {T})"
        note = " RBF corrected." if corrected else ""

        fig, ax = plt.subplots(figsize=(4.4, 4.0))
        for m in MODELS + ["u_mask_entropy"]:
            fpr, tpr, _ = roc_curve(y, oof[m])
            ax.plot(fpr, tpr, color=COLOR[m], lw=2.2 if m.startswith("q_") else 1.3,
                    ls="--" if m == "u_mask_entropy" else "-", label=f"{LABEL[m]} ({auc(y, oof[m]):.3f})")
        ax.plot([0, 1], [0, 1], ":", color="#bbbbbb", lw=0.8)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
        ax.title.set_fontsize(8.5)
        ax.legend(fontsize=6.5, loc="lower right", frameon=False)
        fig.tight_layout(); fig.savefig(out / f"{tag}_roc.png", dpi=220); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        mu = [stats[m][0] for m in MODELS]; sd = [stats[m][1] for m in MODELS]
        ax.bar(range(len(MODELS)), mu, yerr=sd, color=[COLOR[m] for m in MODELS], capsize=3, width=0.7)
        ent = auc(y, oof["u_mask_entropy"])
        ax.axhline(ent, color=COLOR["u_mask_entropy"], ls="--", lw=1.2, label=f"{LABEL['u_mask_entropy']} = {ent:.3f}")
        ax.axhline(0.5, color="#bbbbbb", ls=":", lw=0.8)
        ax.set_xticks(range(len(MODELS)), [LABEL[m] for m in MODELS], rotation=25, ha="right", fontsize=7.5)
        ax.set(ylabel="AUROC (mean ± SD over seeds)", ylim=(0.5, 1.0), title=title)
        ax.title.set_fontsize(8.5)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        fig.tight_layout(); fig.savefig(out / f"{tag}_auroc_bar.png", dpi=220); plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.4, 3.5))
        rates = [0, 0.1, 0.2, 0.3]
        order = np.argsort(dice)
        for m in MODELS + ["u_mask_entropy", "oracle"]:
            vals = []
            for r in rates:
                k = int(round(r * len(y))); keep = np.ones(len(y), bool)
                keep[(order if m == "oracle" else np.argsort(-oof[m].to_numpy()))[:k]] = False
                vals.append(dice[keep].mean())
            ax.plot([r * 100 for r in rates], vals, marker="o", ms=3, color=COLOR.get(m, "#000000"),
                    ls="--" if m in ("oracle", "u_mask_entropy") else "-", lw=2 if m.startswith("q_") else 1.2,
                    label=LABEL.get(m, "Oracle (true worst frames)"))
        ax.set(xlabel="Frames rejected (%)", ylabel="Mean Dice of kept frames", title=title)
        ax.title.set_fontsize(8.5)
        ax.legend(fontsize=6.5, frameon=False)
        fig.tight_layout(); fig.savefig(out / f"{tag}_selective_dice.png", dpi=220); plt.close(fig)

        index.append(f"| `{tag}_roc.png`, `{tag}_auroc_bar.png`, `{tag}_selective_dice.png` | {title}.{note} |")
        if feat == "heatmap":
            row = {"dataset": ds, "qubits": q, **{m: stats[m][0] for m in MODELS}, "entropy": ent}
            scaling.append(row)

    # entanglement forest plot from each run's comparisons.csv
    rows = []
    for tag, folder, ds, q, feat in RUNS:
        f = args.root / folder / "comparisons.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f)
        c = c[c.question.str.startswith("Does entanglement help") & (c.metric == "auroc")]
        for _, r in c.iterrows():
            rows.append(dict(label=f"{ds}, {q}q{', all feat.' if feat != 'heatmap' else ''}", thr=r.threshold,
                             d=r.delta, lo=r.ci95_low, hi=r.ci95_high))
    if rows:
        fr = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
        labels = list(dict.fromkeys(fr.label))
        for ax, thr in zip(axes, (0.5, 0.2)):
            sub = fr[np.isclose(fr.thr, thr)].set_index("label").reindex(labels)
            yy = np.arange(len(labels))[::-1]
            ax.errorbar(sub.d, yy, xerr=[sub.d - sub.lo, sub.hi - sub.d], fmt="o", color="#d1453b", ecolor="#555555",
                        capsize=3, ms=5)
            ax.axvline(0, color="#999999", lw=0.8)
            ax.set_yticks(yy, labels, fontsize=7.5)
            ax.set(xlabel="Δ AUROC, entangled − no entanglement", title=f"Failure = Dice < {thr}")
        fig.suptitle("Entanglement ablation (95% cluster-bootstrap CI); > 0 favours entanglement", fontsize=9)
        fig.tight_layout(); fig.savefig(out / "entanglement_forest.png", dpi=220); plt.close(fig)
        index.append("| `entanglement_forest.png` | Δ AUROC entangled − product with 95% CI, all runs, both failure thresholds |")

    if scaling:
        sc = pd.DataFrame(scaling)
        fig, axes = plt.subplots(1, sc.dataset.nunique(), figsize=(3.8 * sc.dataset.nunique(), 3.3), sharey=True,
                                 squeeze=False)
        for ax, (ds, d) in zip(axes[0], sc.groupby("dataset", sort=False)):
            d = d.sort_values("qubits")
            for m in ["q_entangled", "q_product", "rbf_svm", "gbt"]:
                ax.plot(d.qubits, d[m], marker="o", color=COLOR[m], lw=2 if m.startswith("q_") else 1.2, label=LABEL[m])
            ax.axhline(d.entropy.iloc[0], color="#222222", ls="--", lw=1.1, label=LABEL["u_mask_entropy"])
            ax.set(xlabel="Qubits (= PCA dims)", title=ds, xticks=sorted(d.qubits))
        axes[0][0].set_ylabel("AUROC (Dice < 0.5)")
        handles, labs = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labs, loc="lower center", ncol=3, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(out / "qubit_scaling.png", dpi=220); plt.close(fig)
        index.append("| `qubit_scaling.png` | AUROC vs qubit count (heatmap features), quantum vs classical, with mask entropy |")

    ks = args.root / "hardware" / "kernel_scatter.png"
    if ks.exists():
        shutil.copy2(ks, out / "cvc_noisy_sim_kernel_scatter.png")
        index.append("| `cvc_noisy_sim_kernel_scatter.png` | Noisy simulation (FakeTorino) vs exact kernel, 20 CVC frames |")

    readme = ["# Figures (all in one place)", "",
              "Generated by `make_figures.py` from saved out-of-fold scores. Nothing was re-trained. RBF-SVM uses the "
              "corrected γ grid everywhere.", "",
              "Caption notes: ROC legends show AUROC of **seed-averaged** scores; bar charts show the **mean ± SD of "
              "per-seed** AUROCs (the numbers in the tables). The dashed black line is the training-free U-Net mask "
              "entropy.", "", "| File(s) | Content |", "|---|---|"] + index
    if missing:
        readme += ["", "Runs not found (skipped): " + ", ".join(missing)]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("\n".join(readme))


if __name__ == "__main__":
    main()
