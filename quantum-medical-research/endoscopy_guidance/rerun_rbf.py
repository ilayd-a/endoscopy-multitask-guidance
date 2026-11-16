#!/usr/bin/env python3
"""
rerun_rbf.py
============
Correction run for a baseline defect: the RBF-SVM gamma grid in the first runs was
gamma in {0.3, 3} / n_features, ignoring feature variance. After PCA on many raw
features (e.g. 2715), component variances are large, the RBF kernel collapses to
the identity and the classifier falls to chance. Corrected grid: gamma in
{0.3, 3} / (n_features * Var(X_train)), which is sklearn's 'scale' rule.

Only rbf_svm is re-run. Folds, PCA, inner CV and tuning budget (8 configurations)
are identical to failure_quantum_benchmark.py (same seeds). The new scores are
compared with the quantum and other columns of an existing oof_scores.csv using
the same seed-averaged z-scores and cluster bootstrap.

  python rerun_rbf.py --features failure_detection/features_cvc.csv --feature-prefix f_ \
      --oof failure_detection/results_all/oof_scores.csv --seeds 0 1 2 3 4 \
      --out failure_detection/results_all/rbf_corrected
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_quantum_benchmark import cluster_bootstrap_delta, safe_auc  # noqa: E402


def tuned_rbf(X, y, g, seed):
    g0 = 1.0 / (X.shape[1] * max(float(X.var()), 1e-12))
    grid = [(c, k * g0) for c in (0.1, 1, 10, 100) for k in (0.3, 3)]
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + 1000)
    splits = [(a, b) for a, b in inner.split(X, y, g) if 0 < y[a].sum() < len(a) and 0 < y[b].sum() < len(b)]
    best, best_auc = grid[0], -np.inf
    for c, gm in grid:
        aucs = [safe_auc(y[b], SVC(C=c, gamma=gm, class_weight="balanced").fit(X[a], y[a]).decision_function(X[b]))
                for a, b in splits]
        if np.nanmean(aucs) > best_auc:
            best, best_auc = (c, gm), np.nanmean(aucs)
    return SVC(C=best[0], gamma=best[1], class_weight="balanced").fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--feature-prefix", default="f_hm_")
    ap.add_argument("--oof", type=Path, required=True, help="oof_scores.csv of the run being corrected")
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features).dropna(subset=["dice"]).reset_index(drop=True)
    cols = [c for c in df.columns if c.startswith(args.feature_prefix)]
    Xraw = np.nan_to_num(df[cols].to_numpy(np.float64))
    groups, dice = df.sequence_id.to_numpy(), df.dice.to_numpy(np.float64)
    oof_old = pd.read_csv(args.oof)
    rng = np.random.default_rng(12345)
    lines = [f"# Corrected RBF-SVM ({len(cols)} raw features -> {args.n_qubits} PCA dims, seeds {args.seeds})", ""]
    for t in args.thresholds:
        y = (dice < t).astype(int)
        per_seed = []
        for seed in args.seeds:
            oof = np.full(len(y), np.nan)
            cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            for tr, te in cv.split(Xraw, y, groups):
                sc = StandardScaler().fit(Xraw[tr])
                pca = PCA(n_components=args.n_qubits, random_state=seed).fit(sc.transform(Xraw[tr]))
                Z = pca.transform(sc.transform(Xraw))
                oof[te] = tuned_rbf(Z[tr], y[tr], groups[tr], seed).decision_function(Z[te])
            per_seed.append(oof)
        aucs = [roc_auc_score(y, s) for s in per_seed]
        aps = [average_precision_score(y, s) for s in per_seed]
        avg = np.mean([(s - s.mean()) / (s.std() + 1e-12) for s in per_seed], axis=0)
        old = oof_old[np.isclose(oof_old.threshold, t)].set_index("sample_id").loc[df.sample_id]
        lines += [f"## Dice < {t} ({y.sum()} failures)", "",
                  f"rbf_svm (corrected): AUROC {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}, "
                  f"AUPRC {np.mean(aps):.3f} ± {np.std(aps, ddof=1):.3f}"
                  f" (previous run: AUROC {roc_auc_score(y, old.rbf_svm.to_numpy()):.3f} on seed-averaged scores)", "",
                  "| Comparison | Metric | Δ (A − B) | 95% CI | p |", "|---|---|---|---|---|"]
        for other in [c for c in ("q_entangled", "q_product", "logreg", "gbt", "mlp") if c in old.columns]:
            for name, met in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
                d, lo, hi, p = cluster_bootstrap_delta(y, old[other].to_numpy(), avg, groups, args.n_boot, rng, met)
                lines.append(f"| {other} vs rbf_svm (corrected) | {name} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |")
        lines.append("")
        pd.DataFrame({"sample_id": df.sample_id, "threshold": t, "rbf_svm_corrected": avg}).to_csv(
            args.out / f"rbf_corrected_oof_t{t}.csv", index=False)
    (args.out / "RBF_CORRECTED.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
