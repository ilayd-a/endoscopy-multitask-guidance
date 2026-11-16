# Corrected RBF-SVM (27 raw features -> 4 PCA dims, seeds [0, 1, 2])

## Dice < 0.5 (78 failures)

rbf_svm (corrected): AUROC 0.827 ± 0.004, AUPRC 0.503 ± 0.017 (previous run: AUROC 0.854 on seed-averaged scores)

| Comparison | Metric | Δ (A − B) | 95% CI | p |
|---|---|---|---|---|
| q_entangled vs rbf_svm (corrected) | auroc | +0.007 | [-0.014, +0.027] | 0.571 |
| q_entangled vs rbf_svm (corrected) | auprc | +0.027 | [-0.013, +0.078] | 0.202 |
| q_product vs rbf_svm (corrected) | auroc | -0.011 | [-0.033, +0.010] | 0.302 |
| q_product vs rbf_svm (corrected) | auprc | -0.027 | [-0.089, +0.028] | 0.327 |
| logreg vs rbf_svm (corrected) | auroc | -0.050 | [-0.095, -0.011] | 0.012 |
| logreg vs rbf_svm (corrected) | auprc | -0.092 | [-0.229, +0.069] | 0.425 |
| gbt vs rbf_svm (corrected) | auroc | -0.009 | [-0.054, +0.032] | 0.750 |
| gbt vs rbf_svm (corrected) | auprc | +0.087 | [-0.000, +0.167] | 0.051 |
| mlp vs rbf_svm (corrected) | auroc | +0.020 | [-0.009, +0.047] | 0.184 |
| mlp vs rbf_svm (corrected) | auprc | +0.119 | [+0.036, +0.181] | 0.010 |

## Dice < 0.2 (36 failures)

rbf_svm (corrected): AUROC 0.852 ± 0.047, AUPRC 0.668 ± 0.082 (previous run: AUROC 0.878 on seed-averaged scores)

| Comparison | Metric | Δ (A − B) | 95% CI | p |
|---|---|---|---|---|
| q_entangled vs rbf_svm (corrected) | auroc | -0.028 | [-0.081, +0.020] | 0.244 |
| q_entangled vs rbf_svm (corrected) | auprc | +0.003 | [-0.083, +0.120] | 0.999 |
| q_product vs rbf_svm (corrected) | auroc | +0.008 | [-0.019, +0.034] | 0.604 |
| q_product vs rbf_svm (corrected) | auprc | +0.078 | [-0.027, +0.189] | 0.197 |
| logreg vs rbf_svm (corrected) | auroc | -0.005 | [-0.034, +0.024] | 0.728 |
| logreg vs rbf_svm (corrected) | auprc | +0.051 | [-0.102, +0.185] | 0.498 |
| gbt vs rbf_svm (corrected) | auroc | +0.040 | [-0.014, +0.091] | 0.147 |
| gbt vs rbf_svm (corrected) | auprc | +0.096 | [-0.025, +0.188] | 0.123 |
| mlp vs rbf_svm (corrected) | auroc | +0.026 | [-0.020, +0.076] | 0.377 |
| mlp vs rbf_svm (corrected) | auprc | +0.089 | [-0.005, +0.182] | 0.097 |
