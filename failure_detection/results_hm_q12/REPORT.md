# Failure-detection benchmark: what does the quantum circuit add?

612 frames, 29 sequences, 27 raw features compressed to 12 PCA dims (= qubits). 5-fold CV grouped by sequence, seeds [0, 1, 2]. Circuit: ZZ feature map, reps=2, linear entanglement.

## Failure = Dice < 0.5  (78 failures, prevalence 12.7%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.876 ± 0.007 | 0.637 ± 0.004 |
| rbf_svm | 0.834 ± 0.007 | 0.542 ± 0.009 |
| gbt | 0.854 ± 0.021 | 0.632 ± 0.022 |
| mlp | 0.817 ± 0.003 | 0.595 ± 0.021 |
| q_entangled | 0.785 ± 0.002 | 0.464 ± 0.040 |
| q_product | 0.859 ± 0.038 | 0.580 ± 0.088 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.767 ± 0.000 | 0.447 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.922 ± 0.000 | 0.750 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.850 ± 0.000 | 0.648 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.924 ± 0.000 | 0.747 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.725 ± 0.000 | 0.281 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | -0.079 | [-0.113, -0.029] | 0.000 | 0/3 | yes |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | -0.106 | [-0.169, -0.013] | 0.028 | 0/3 | yes |
| Quantum vs best classical on the same 8 features: q_entangled vs logreg | auroc | -0.084 | [-0.113, -0.044] | 0.000 | 0/3 | yes |
| Quantum vs best classical on the same 8 features: q_entangled vs logreg | auprc | -0.146 | [-0.186, -0.080] | 0.000 | 0/3 | yes |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | -0.055 | [-0.104, +0.001] | 0.059 | 0/3 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | -0.075 | [-0.140, -0.002] | 0.047 | 0/3 | yes |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | +0.024 | [-0.017, +0.063] | 0.232 | 2/3 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | +0.030 | [-0.045, +0.079] | 0.357 | 2/3 | no |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.129 | [-0.178, -0.066] | 0.000 | 0/3 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.249 | [-0.323, -0.155] | 0.000 | 0/3 | yes |
| Best classical vs best training-free uncertainty score: logreg vs u_mask_entropy | auroc | -0.045 | [-0.081, -0.011] | 0.010 | 0/3 | yes |
| Best classical vs best training-free uncertainty score: logreg vs u_mask_entropy | auprc | -0.103 | [-0.176, -0.034] | 0.002 | 0/3 | yes |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.789 | 0.836 | 0.859 | 0.881 |
| rbf_svm | 0.789 | 0.833 | 0.851 | 0.865 |
| gbt | 0.789 | 0.839 | 0.863 | 0.875 |
| mlp | 0.789 | 0.836 | 0.850 | 0.861 |
| q_entangled | 0.789 | 0.829 | 0.845 | 0.863 |
| q_product | 0.789 | 0.832 | 0.862 | 0.878 |
| u_fragmentation | 0.789 | 0.826 | 0.846 | 0.851 |
| u_low_mask_confidence | 0.789 | 0.842 | 0.870 | 0.887 |
| u_low_max_confidence | 0.789 | 0.838 | 0.854 | 0.865 |
| u_mask_entropy | 0.789 | 0.843 | 0.870 | 0.887 |
| u_mean_entropy | 0.789 | 0.811 | 0.826 | 0.841 |
| oracle | 0.789 | 0.858 | 0.896 | 0.919 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.287 | 0.183 | 0.649 |
| q_product | 0.838 | 0.064 | 1.000 |

## Failure = Dice < 0.2  (36 failures, prevalence 5.9%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.866 ± 0.030 | 0.676 ± 0.103 |
| rbf_svm | 0.863 ± 0.043 | 0.584 ± 0.058 |
| gbt | 0.882 ± 0.048 | 0.712 ± 0.051 |
| mlp | 0.824 ± 0.010 | 0.656 ± 0.031 |
| q_entangled | 0.843 ± 0.016 | 0.357 ± 0.103 |
| q_product | 0.863 ± 0.027 | 0.564 ± 0.058 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.799 ± 0.000 | 0.497 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.954 ± 0.000 | 0.790 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.918 ± 0.000 | 0.771 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.955 ± 0.000 | 0.796 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.568 ± 0.000 | 0.088 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | -0.015 | [-0.088, +0.047] | 0.695 | 1/3 | no |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | -0.186 | [-0.285, -0.049] | 0.000 | 0/3 | yes |
| Quantum vs best classical on the same 8 features: q_entangled vs gbt | auroc | -0.046 | [-0.132, +0.022] | 0.189 | 0/3 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs gbt | auprc | -0.280 | [-0.395, -0.140] | 0.000 | 0/3 | yes |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | -0.016 | [-0.071, +0.045] | 0.575 | 1/3 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | -0.183 | [-0.287, -0.049] | 0.004 | 0/3 | yes |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | -0.001 | [-0.029, +0.033] | 0.943 | 1/3 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | +0.003 | [-0.080, +0.091] | 0.902 | 1/3 | no |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.089 | [-0.146, -0.038] | 0.000 | 0/3 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.332 | [-0.454, -0.163] | 0.000 | 0/3 | yes |
| Best classical vs best training-free uncertainty score: gbt vs u_mask_entropy | auroc | -0.043 | [-0.095, -0.002] | 0.045 | 0/3 | yes |
| Best classical vs best training-free uncertainty score: gbt vs u_mask_entropy | auprc | -0.052 | [-0.124, +0.013] | 0.126 | 0/3 | no |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.789 | 0.836 | 0.839 | 0.842 |
| rbf_svm | 0.789 | 0.824 | 0.841 | 0.851 |
| gbt | 0.789 | 0.829 | 0.839 | 0.848 |
| mlp | 0.789 | 0.831 | 0.849 | 0.855 |
| q_entangled | 0.789 | 0.822 | 0.834 | 0.841 |
| q_product | 0.789 | 0.831 | 0.841 | 0.850 |
| u_fragmentation | 0.789 | 0.826 | 0.846 | 0.851 |
| u_low_mask_confidence | 0.789 | 0.842 | 0.870 | 0.887 |
| u_low_max_confidence | 0.789 | 0.838 | 0.854 | 0.865 |
| u_mask_entropy | 0.789 | 0.843 | 0.870 | 0.887 |
| u_mean_entropy | 0.789 | 0.811 | 0.826 | 0.841 |
| oracle | 0.789 | 0.858 | 0.896 | 0.919 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.142 | 0.130 | 0.687 |
| q_product | 0.339 | 0.121 | 1.000 |

## How to read this

* The main question is the first comparison row: q_entangled vs q_product. Same circuit, same inputs, same tuning; the only difference is the entangling gates. If its CI includes 0, the data show no evidence that entanglement helps.
* 'Best classical' is chosen after seeing the results, which favours the classical side; that makes a positive quantum result more convincing, not less.
* The *_full rows use all raw features and are not matched; they show the cost of compressing to 12 dims.
* All quantum results are exact statevector simulations, not quantum hardware.
