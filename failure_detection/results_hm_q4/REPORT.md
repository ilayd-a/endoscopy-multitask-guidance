# Failure-detection benchmark: what does the quantum circuit add?

612 frames, 29 sequences, 27 raw features compressed to 4 PCA dims (= qubits). 5-fold CV grouped by sequence, seeds [0, 1, 2]. Circuit: ZZ feature map, reps=2, linear entanglement.

## Failure = Dice < 0.5  (78 failures, prevalence 12.7%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.795 ± 0.004 | 0.448 ± 0.007 |
| rbf_svm | 0.832 ± 0.020 | 0.518 ± 0.066 |
| gbt | 0.816 ± 0.020 | 0.584 ± 0.067 |
| mlp | 0.849 ± 0.015 | 0.636 ± 0.026 |
| q_entangled | 0.843 ± 0.005 | 0.535 ± 0.030 |
| q_product | 0.828 ± 0.004 | 0.478 ± 0.015 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.767 ± 0.000 | 0.447 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.922 ± 0.000 | 0.750 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.850 ± 0.000 | 0.648 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.924 ± 0.000 | 0.747 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.725 ± 0.000 | 0.281 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | +0.017 | [-0.007, +0.044] | 0.177 | 3/3 | no |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | +0.054 | [-0.004, +0.124] | 0.067 | 3/3 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs mlp | auroc | -0.013 | [-0.041, +0.017] | 0.406 | 1/3 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs mlp | auprc | -0.092 | [-0.153, -0.015] | 0.015 | 0/3 | yes |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | -0.000 | [-0.023, +0.024] | 0.935 | 2/3 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | +0.001 | [-0.075, +0.077] | 0.993 | 2/3 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | -0.018 | [-0.049, +0.017] | 0.277 | 1/3 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | -0.053 | [-0.163, +0.045] | 0.273 | 1/3 | no |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.071 | [-0.115, -0.021] | 0.004 | 0/3 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.183 | [-0.269, -0.071] | 0.001 | 0/3 | yes |
| Best classical vs best training-free uncertainty score: mlp vs u_mask_entropy | auroc | -0.058 | [-0.104, -0.009] | 0.022 | 0/3 | yes |
| Best classical vs best training-free uncertainty score: mlp vs u_mask_entropy | auprc | -0.091 | [-0.167, -0.007] | 0.035 | 0/3 | yes |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.789 | 0.831 | 0.853 | 0.859 |
| rbf_svm | 0.789 | 0.837 | 0.850 | 0.866 |
| gbt | 0.789 | 0.837 | 0.854 | 0.869 |
| mlp | 0.789 | 0.837 | 0.858 | 0.868 |
| q_entangled | 0.789 | 0.831 | 0.859 | 0.867 |
| q_product | 0.789 | 0.828 | 0.852 | 0.865 |
| u_fragmentation | 0.789 | 0.826 | 0.846 | 0.851 |
| u_low_mask_confidence | 0.789 | 0.842 | 0.870 | 0.887 |
| u_low_max_confidence | 0.789 | 0.838 | 0.854 | 0.865 |
| u_mask_entropy | 0.789 | 0.843 | 0.870 | 0.887 |
| u_mean_entropy | 0.789 | 0.811 | 0.826 | 0.841 |
| oracle | 0.789 | 0.858 | 0.896 | 0.919 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.603 | 0.256 | 0.674 |
| q_product | 0.708 | 0.185 | 1.000 |

## Failure = Dice < 0.2  (36 failures, prevalence 5.9%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.851 ± 0.029 | 0.698 ± 0.047 |
| rbf_svm | 0.852 ± 0.017 | 0.626 ± 0.051 |
| gbt | 0.869 ± 0.014 | 0.700 ± 0.019 |
| mlp | 0.875 ± 0.020 | 0.729 ± 0.010 |
| q_entangled | 0.834 ± 0.028 | 0.615 ± 0.047 |
| q_product | 0.864 ± 0.029 | 0.707 ± 0.053 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.799 ± 0.000 | 0.497 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.954 ± 0.000 | 0.790 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.918 ± 0.000 | 0.771 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.955 ± 0.000 | 0.796 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.568 ± 0.000 | 0.088 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | -0.036 | [-0.102, +0.019] | 0.207 | 0/3 | no |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | -0.074 | [-0.171, +0.009] | 0.106 | 1/3 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs mlp | auroc | -0.054 | [-0.113, -0.006] | 0.014 | 0/3 | yes |
| Quantum vs best classical on the same 8 features: q_entangled vs mlp | auprc | -0.086 | [-0.155, -0.019] | 0.001 | 0/3 | yes |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | -0.053 | [-0.140, +0.017] | 0.160 | 0/3 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | +0.042 | [-0.038, +0.148] | 0.359 | 1/3 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | -0.017 | [-0.049, +0.017] | 0.319 | 2/3 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | +0.116 | [+0.011, +0.255] | 0.028 | 3/3 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.130 | [-0.213, -0.044] | 0.000 | 0/3 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.135 | [-0.230, -0.041] | 0.000 | 0/3 | yes |
| Best classical vs best training-free uncertainty score: mlp vs u_mask_entropy | auroc | -0.076 | [-0.139, -0.022] | 0.001 | 0/3 | yes |
| Best classical vs best training-free uncertainty score: mlp vs u_mask_entropy | auprc | -0.049 | [-0.115, +0.002] | 0.059 | 0/3 | no |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.789 | 0.835 | 0.844 | 0.842 |
| rbf_svm | 0.789 | 0.827 | 0.844 | 0.848 |
| gbt | 0.789 | 0.834 | 0.844 | 0.848 |
| mlp | 0.789 | 0.836 | 0.842 | 0.852 |
| q_entangled | 0.789 | 0.834 | 0.843 | 0.848 |
| q_product | 0.789 | 0.834 | 0.847 | 0.842 |
| u_fragmentation | 0.789 | 0.826 | 0.846 | 0.851 |
| u_low_mask_confidence | 0.789 | 0.842 | 0.870 | 0.887 |
| u_low_max_confidence | 0.789 | 0.838 | 0.854 | 0.865 |
| u_mask_entropy | 0.789 | 0.843 | 0.870 | 0.887 |
| u_mean_entropy | 0.789 | 0.811 | 0.826 | 0.841 |
| oracle | 0.789 | 0.858 | 0.896 | 0.919 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.571 | 0.258 | 0.687 |
| q_product | 0.846 | 0.112 | 1.000 |

## How to read this

* The main question is the first comparison row: q_entangled vs q_product. Same circuit, same inputs, same tuning; the only difference is the entangling gates. If its CI includes 0, the data show no evidence that entanglement helps.
* 'Best classical' is chosen after seeing the results, which favours the classical side; that makes a positive quantum result more convincing, not less.
* The *_full rows use all raw features and are not matched; they show the cost of compressing to 4 dims.
* All quantum results are exact statevector simulations, not quantum hardware.
