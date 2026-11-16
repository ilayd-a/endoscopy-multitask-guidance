# Failure-detection benchmark: what does the quantum circuit add?

612 frames, 29 sequences, 2715 raw features compressed to 8 PCA dims (= qubits). 5-fold CV grouped by sequence, seeds [0, 1, 2, 3, 4]. Circuit: ZZ feature map, reps=2, linear entanglement.

## Failure = Dice < 0.5  (78 failures, prevalence 12.7%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.850 ± 0.007 | 0.526 ± 0.013 |
| rbf_svm | 0.564 ± 0.014 | 0.224 ± 0.063 |
| gbt | 0.803 ± 0.038 | 0.520 ± 0.039 |
| mlp | 0.849 ± 0.014 | 0.432 ± 0.034 |
| q_entangled | 0.829 ± 0.026 | 0.421 ± 0.075 |
| q_product | 0.822 ± 0.013 | 0.490 ± 0.024 |
| logreg_full *(reference: all raw features)* | 0.849 ± 0.022 | 0.601 ± 0.015 |
| gbt_full *(reference: all raw features)* | 0.873 ± 0.022 | 0.647 ± 0.024 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.767 ± 0.000 | 0.447 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.922 ± 0.000 | 0.750 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.850 ± 0.000 | 0.648 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.924 ± 0.000 | 0.747 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.725 ± 0.000 | 0.281 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | +0.013 | [-0.010, +0.038] | 0.333 | 3/5 | no |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | -0.018 | [-0.088, +0.073] | 0.676 | 1/5 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs logreg | auroc | -0.004 | [-0.042, +0.030] | 0.742 | 2/5 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs logreg | auprc | -0.040 | [-0.128, +0.060] | 0.449 | 1/5 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | +0.292 | [+0.123, +0.448] | 0.000 | 5/5 | yes |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | +0.340 | [+0.160, +0.509] | 0.000 | 5/5 | yes |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | +0.279 | [+0.106, +0.452] | 0.001 | 5/5 | yes |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | +0.358 | [+0.155, +0.540] | 0.000 | 5/5 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.076 | [-0.131, -0.034] | 0.000 | 0/5 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.258 | [-0.362, -0.148] | 0.000 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: logreg vs u_mask_entropy | auroc | -0.073 | [-0.110, -0.035] | 0.000 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: logreg vs u_mask_entropy | auprc | -0.218 | [-0.338, -0.116] | 0.000 | 0/5 | yes |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.789 | 0.830 | 0.853 | 0.865 |
| rbf_svm | 0.789 | 0.789 | 0.802 | 0.800 |
| gbt | 0.789 | 0.839 | 0.850 | 0.862 |
| mlp | 0.789 | 0.828 | 0.858 | 0.869 |
| q_entangled | 0.789 | 0.827 | 0.852 | 0.868 |
| q_product | 0.789 | 0.831 | 0.848 | 0.860 |
| u_fragmentation | 0.789 | 0.826 | 0.846 | 0.851 |
| u_low_mask_confidence | 0.789 | 0.842 | 0.870 | 0.887 |
| u_low_max_confidence | 0.789 | 0.838 | 0.854 | 0.865 |
| u_mask_entropy | 0.789 | 0.843 | 0.870 | 0.887 |
| u_mean_entropy | 0.789 | 0.811 | 0.826 | 0.841 |
| oracle | 0.789 | 0.858 | 0.896 | 0.919 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.267 | 0.201 | 0.641 |
| q_product | 0.815 | 0.066 | 1.000 |

## Failure = Dice < 0.2  (36 failures, prevalence 5.9%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.905 ± 0.008 | 0.476 ± 0.023 |
| rbf_svm | 0.531 ± 0.036 | 0.117 ± 0.038 |
| gbt | 0.847 ± 0.019 | 0.510 ± 0.060 |
| mlp | 0.845 ± 0.041 | 0.326 ± 0.069 |
| q_entangled | 0.870 ± 0.020 | 0.378 ± 0.050 |
| q_product | 0.884 ± 0.034 | 0.433 ± 0.115 |
| logreg_full *(reference: all raw features)* | 0.924 ± 0.013 | 0.711 ± 0.027 |
| gbt_full *(reference: all raw features)* | 0.875 ± 0.023 | 0.634 ± 0.032 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.799 ± 0.000 | 0.497 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.954 ± 0.000 | 0.790 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.918 ± 0.000 | 0.771 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.955 ± 0.000 | 0.796 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.568 ± 0.000 | 0.088 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | -0.013 | [-0.031, +0.005] | 0.149 | 1/5 | no |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | +0.049 | [-0.066, +0.173] | 0.438 | 1/5 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs logreg | auroc | -0.018 | [-0.040, +0.004] | 0.099 | 0/5 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs logreg | auprc | +0.001 | [-0.121, +0.135] | 0.964 | 0/5 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | +0.443 | [+0.226, +0.632] | 0.000 | 5/5 | yes |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | +0.425 | [+0.286, +0.653] | 0.000 | 5/5 | yes |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | +0.456 | [+0.250, +0.638] | 0.000 | 5/5 | yes |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | +0.376 | [+0.253, +0.602] | 0.000 | 5/5 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.064 | [-0.098, -0.031] | 0.000 | 0/5 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.318 | [-0.463, -0.143] | 0.000 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: logreg vs u_mask_entropy | auroc | -0.046 | [-0.071, -0.023] | 0.000 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: logreg vs u_mask_entropy | auprc | -0.320 | [-0.442, -0.167] | 0.000 | 0/5 | yes |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.789 | 0.834 | 0.848 | 0.861 |
| rbf_svm | 0.789 | 0.776 | 0.772 | 0.789 |
| gbt | 0.789 | 0.833 | 0.853 | 0.860 |
| mlp | 0.789 | 0.821 | 0.840 | 0.856 |
| q_entangled | 0.789 | 0.826 | 0.849 | 0.861 |
| q_product | 0.789 | 0.831 | 0.847 | 0.858 |
| u_fragmentation | 0.789 | 0.826 | 0.846 | 0.851 |
| u_low_mask_confidence | 0.789 | 0.842 | 0.870 | 0.887 |
| u_low_max_confidence | 0.789 | 0.838 | 0.854 | 0.865 |
| u_mask_entropy | 0.789 | 0.843 | 0.870 | 0.887 |
| u_mean_entropy | 0.789 | 0.811 | 0.826 | 0.841 |
| oracle | 0.789 | 0.858 | 0.896 | 0.919 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.285 | 0.208 | 0.638 |
| q_product | 0.859 | 0.039 | 1.000 |

## How to read this

* The main question is the first comparison row: q_entangled vs q_product. Same circuit, same inputs, same tuning; the only difference is the entangling gates. If its CI includes 0, the data show no evidence that entanglement helps.
* 'Best classical' is chosen after seeing the results, which favours the classical side; that makes a positive quantum result more convincing, not less.
* The *_full rows use all raw features and are not matched; they show the cost of compressing to 8 dims.
* All quantum results are exact statevector simulations, not quantum hardware.
