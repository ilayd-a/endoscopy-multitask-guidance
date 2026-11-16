# Corrected RBF-SVM (2715 raw features -> 8 PCA dims, seeds [0, 1, 2, 3, 4])

## Dice < 0.5 (78 failures)

rbf_svm (corrected): AUROC 0.825 ± 0.011, AUPRC 0.518 ± 0.027 (previous run: AUROC 0.556 on seed-averaged scores)

| Comparison | Metric | Δ (A − B) | 95% CI | p |
|---|---|---|---|---|
| q_entangled vs rbf_svm (corrected) | auroc | +0.015 | [-0.033, +0.058] | 0.536 |
| q_entangled vs rbf_svm (corrected) | auprc | -0.057 | [-0.158, +0.047] | 0.259 |
| q_product vs rbf_svm (corrected) | auroc | +0.002 | [-0.034, +0.040] | 0.954 |
| q_product vs rbf_svm (corrected) | auprc | -0.039 | [-0.137, +0.033] | 0.322 |
| logreg vs rbf_svm (corrected) | auroc | +0.019 | [-0.018, +0.056] | 0.350 |
| logreg vs rbf_svm (corrected) | auprc | -0.017 | [-0.115, +0.061] | 0.660 |
| gbt vs rbf_svm (corrected) | auroc | -0.002 | [-0.034, +0.030] | 0.816 |
| gbt vs rbf_svm (corrected) | auprc | -0.002 | [-0.088, +0.103] | 0.897 |
| mlp vs rbf_svm (corrected) | auroc | +0.025 | [-0.041, +0.088] | 0.496 |
| mlp vs rbf_svm (corrected) | auprc | -0.069 | [-0.192, +0.051] | 0.289 |

## Dice < 0.2 (36 failures)

rbf_svm (corrected): AUROC 0.896 ± 0.019, AUPRC 0.560 ± 0.046 (previous run: AUROC 0.448 on seed-averaged scores)

| Comparison | Metric | Δ (A − B) | 95% CI | p |
|---|---|---|---|---|
| q_entangled vs rbf_svm (corrected) | auroc | -0.012 | [-0.038, +0.014] | 0.303 |
| q_entangled vs rbf_svm (corrected) | auprc | -0.123 | [-0.201, -0.053] | 0.000 |
| q_product vs rbf_svm (corrected) | auroc | +0.000 | [-0.016, +0.022] | 0.966 |
| q_product vs rbf_svm (corrected) | auprc | -0.173 | [-0.255, -0.082] | 0.001 |
| logreg vs rbf_svm (corrected) | auroc | +0.006 | [-0.006, +0.018] | 0.349 |
| logreg vs rbf_svm (corrected) | auprc | -0.125 | [-0.230, -0.015] | 0.022 |
| gbt vs rbf_svm (corrected) | auroc | -0.031 | [-0.100, +0.033] | 0.375 |
| gbt vs rbf_svm (corrected) | auprc | -0.065 | [-0.170, +0.024] | 0.148 |
| mlp vs rbf_svm (corrected) | auroc | -0.022 | [-0.054, +0.010] | 0.176 |
| mlp vs rbf_svm (corrected) | auprc | -0.208 | [-0.293, -0.145] | 0.000 |
