# Corrected RBF-SVM (27 raw features -> 8 PCA dims, seeds [0, 1, 2, 3, 4])

## Dice < 0.5 (78 failures)

rbf_svm (corrected): AUROC 0.785 ± 0.007, AUPRC 0.404 ± 0.020 (previous run: AUROC 0.799 on seed-averaged scores)

| Comparison | Metric | Δ (A − B) | 95% CI | p |
|---|---|---|---|---|
| q_entangled vs rbf_svm (corrected) | auroc | -0.055 | [-0.103, -0.008] | 0.013 |
| q_entangled vs rbf_svm (corrected) | auprc | -0.044 | [-0.120, +0.048] | 0.321 |
| q_product vs rbf_svm (corrected) | auroc | -0.023 | [-0.057, +0.022] | 0.294 |
| q_product vs rbf_svm (corrected) | auprc | -0.004 | [-0.076, +0.115] | 0.942 |
| logreg vs rbf_svm (corrected) | auroc | -0.011 | [-0.069, +0.059] | 0.737 |
| logreg vs rbf_svm (corrected) | auprc | +0.058 | [-0.038, +0.195] | 0.233 |
| gbt vs rbf_svm (corrected) | auroc | +0.019 | [-0.024, +0.074] | 0.401 |
| gbt vs rbf_svm (corrected) | auprc | +0.182 | [+0.062, +0.302] | 0.000 |
| mlp vs rbf_svm (corrected) | auroc | -0.004 | [-0.053, +0.051] | 0.878 |
| mlp vs rbf_svm (corrected) | auprc | +0.152 | [+0.050, +0.239] | 0.001 |

## Dice < 0.2 (36 failures)

rbf_svm (corrected): AUROC 0.874 ± 0.045, AUPRC 0.646 ± 0.035 (previous run: AUROC 0.904 on seed-averaged scores)

| Comparison | Metric | Δ (A − B) | 95% CI | p |
|---|---|---|---|---|
| q_entangled vs rbf_svm (corrected) | auroc | -0.064 | [-0.097, -0.016] | 0.008 |
| q_entangled vs rbf_svm (corrected) | auprc | -0.089 | [-0.177, +0.010] | 0.080 |
| q_product vs rbf_svm (corrected) | auroc | -0.048 | [-0.084, -0.015] | 0.006 |
| q_product vs rbf_svm (corrected) | auprc | +0.015 | [-0.032, +0.103] | 0.545 |
| logreg vs rbf_svm (corrected) | auroc | -0.035 | [-0.109, +0.015] | 0.340 |
| logreg vs rbf_svm (corrected) | auprc | +0.030 | [-0.021, +0.107] | 0.255 |
| gbt vs rbf_svm (corrected) | auroc | +0.006 | [-0.026, +0.050] | 0.717 |
| gbt vs rbf_svm (corrected) | auprc | +0.035 | [-0.014, +0.122] | 0.182 |
| mlp vs rbf_svm (corrected) | auroc | -0.050 | [-0.106, +0.003] | 0.072 |
| mlp vs rbf_svm (corrected) | auprc | -0.005 | [-0.055, +0.064] | 0.791 |
