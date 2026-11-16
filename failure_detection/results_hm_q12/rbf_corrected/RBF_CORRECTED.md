# Corrected RBF-SVM (27 raw features -> 12 PCA dims, seeds [0, 1, 2])

## Dice < 0.5 (78 failures)

rbf_svm (corrected): AUROC 0.838 ± 0.006, AUPRC 0.526 ± 0.024 (previous run: AUROC 0.850 on seed-averaged scores)

| Comparison | Metric | Δ (A − B) | 95% CI | p |
|---|---|---|---|---|
| q_entangled vs rbf_svm (corrected) | auroc | -0.056 | [-0.109, +0.001] | 0.055 |
| q_entangled vs rbf_svm (corrected) | auprc | -0.056 | [-0.151, +0.047] | 0.315 |
| q_product vs rbf_svm (corrected) | auroc | +0.022 | [-0.011, +0.057] | 0.194 |
| q_product vs rbf_svm (corrected) | auprc | +0.050 | [-0.017, +0.103] | 0.141 |
| logreg vs rbf_svm (corrected) | auroc | +0.028 | [-0.008, +0.071] | 0.137 |
| logreg vs rbf_svm (corrected) | auprc | +0.090 | [+0.007, +0.158] | 0.031 |
| gbt vs rbf_svm (corrected) | auroc | +0.024 | [-0.013, +0.065] | 0.212 |
| gbt vs rbf_svm (corrected) | auprc | +0.123 | [+0.027, +0.213] | 0.013 |
| mlp vs rbf_svm (corrected) | auroc | -0.018 | [-0.063, +0.024] | 0.428 |
| mlp vs rbf_svm (corrected) | auprc | +0.067 | [-0.014, +0.145] | 0.125 |

## Dice < 0.2 (36 failures)

rbf_svm (corrected): AUROC 0.875 ± 0.046, AUPRC 0.644 ± 0.060 (previous run: AUROC 0.882 on seed-averaged scores)

| Comparison | Metric | Δ (A − B) | 95% CI | p |
|---|---|---|---|---|
| q_entangled vs rbf_svm (corrected) | auroc | -0.030 | [-0.073, +0.010] | 0.156 |
| q_entangled vs rbf_svm (corrected) | auprc | -0.229 | [-0.322, -0.090] | 0.000 |
| q_product vs rbf_svm (corrected) | auroc | -0.015 | [-0.048, +0.020] | 0.399 |
| q_product vs rbf_svm (corrected) | auprc | -0.043 | [-0.117, +0.032] | 0.278 |
| logreg vs rbf_svm (corrected) | auroc | -0.026 | [-0.086, +0.022] | 0.326 |
| logreg vs rbf_svm (corrected) | auprc | +0.047 | [-0.013, +0.126] | 0.152 |
| gbt vs rbf_svm (corrected) | auroc | +0.016 | [-0.033, +0.065] | 0.512 |
| gbt vs rbf_svm (corrected) | auprc | +0.052 | [-0.013, +0.147] | 0.126 |
| mlp vs rbf_svm (corrected) | auroc | -0.054 | [-0.128, -0.002] | 0.035 |
| mlp vs rbf_svm (corrected) | auprc | -0.017 | [-0.068, +0.030] | 0.434 |
