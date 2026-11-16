# Failure-detection benchmark: what does the quantum circuit add?

3119 frames, 27 sequences, 27 raw features compressed to 4 PCA dims (= qubits). 5-fold CV grouped by sequence, seeds [0, 1, 2, 3, 4]. Circuit: ZZ feature map, reps=2, linear entanglement.

## Failure = Dice < 0.5  (832 failures, prevalence 26.7%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.659 ± 0.002 | 0.554 ± 0.002 |
| rbf_svm | 0.775 ± 0.083 | 0.675 ± 0.057 |
| gbt | 0.769 ± 0.004 | 0.645 ± 0.005 |
| mlp | 0.784 ± 0.035 | 0.687 ± 0.026 |
| q_entangled | 0.849 ± 0.009 | 0.767 ± 0.011 |
| q_product | 0.775 ± 0.029 | 0.646 ± 0.008 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.774 ± 0.000 | 0.610 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.920 ± 0.000 | 0.828 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.866 ± 0.000 | 0.756 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.923 ± 0.000 | 0.831 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.762 ± 0.000 | 0.546 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | +0.064 | [-0.025, +0.153] | 0.422 | 5/5 | no |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | +0.115 | [-0.024, +0.247] | 0.387 | 5/5 | no |
| Quantum vs best classical on the same 4 features: q_entangled vs mlp | auroc | +0.051 | [-0.038, +0.138] | 0.519 | 5/5 | no |
| Quantum vs best classical on the same 4 features: q_entangled vs mlp | auprc | +0.073 | [-0.065, +0.187] | 0.629 | 5/5 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | +0.086 | [-0.015, +0.195] | 0.298 | 5/5 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | +0.105 | [-0.038, +0.214] | 0.358 | 5/5 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | +0.022 | [-0.018, +0.054] | 0.308 | 2/5 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | -0.010 | [-0.071, +0.034] | 0.773 | 2/5 | no |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.072 | [-0.130, -0.019] | 0.002 | 0/5 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.060 | [-0.194, +0.031] | 0.237 | 0/5 | no |
| Best classical vs best training-free uncertainty score: mlp vs u_mask_entropy | auroc | -0.123 | [-0.182, -0.069] | 0.000 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: mlp vs u_mask_entropy | auprc | -0.133 | [-0.194, -0.074] | 0.000 | 0/5 | yes |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.672 | 0.724 | 0.745 | 0.752 |
| rbf_svm | 0.672 | 0.732 | 0.758 | 0.785 |
| gbt | 0.672 | 0.730 | 0.752 | 0.767 |
| mlp | 0.672 | 0.733 | 0.761 | 0.790 |
| q_entangled | 0.672 | 0.737 | 0.779 | 0.809 |
| q_product | 0.672 | 0.728 | 0.752 | 0.771 |
| u_fragmentation | 0.672 | 0.723 | 0.752 | 0.777 |
| u_low_mask_confidence | 0.672 | 0.735 | 0.780 | 0.824 |
| u_low_max_confidence | 0.672 | 0.734 | 0.774 | 0.809 |
| u_mask_entropy | 0.672 | 0.735 | 0.781 | 0.824 |
| u_mean_entropy | 0.672 | 0.712 | 0.742 | 0.762 |
| oracle | 0.672 | 0.747 | 0.822 | 0.875 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.690 | 0.224 | 0.643 |
| q_product | 0.858 | 0.110 | 1.000 |

## Failure = Dice < 0.2  (524 failures, prevalence 16.8%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.844 ± 0.003 | 0.612 ± 0.004 |
| rbf_svm | 0.855 ± 0.001 | 0.668 ± 0.002 |
| gbt | 0.727 ± 0.022 | 0.597 ± 0.004 |
| mlp | 0.832 ± 0.033 | 0.665 ± 0.022 |
| q_entangled | 0.775 ± 0.001 | 0.629 ± 0.008 |
| q_product | 0.801 ± 0.006 | 0.633 ± 0.003 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.801 ± 0.000 | 0.562 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.901 ± 0.000 | 0.716 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.890 ± 0.000 | 0.707 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.907 ± 0.000 | 0.725 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.711 ± 0.000 | 0.370 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | -0.023 | [-0.043, -0.005] | 0.008 | 0/5 | yes |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | -0.001 | [-0.024, +0.018] | 0.879 | 0/5 | no |
| Quantum vs best classical on the same 4 features: q_entangled vs rbf_svm | auroc | -0.078 | [-0.180, +0.012] | 0.313 | 0/5 | no |
| Quantum vs best classical on the same 4 features: q_entangled vs rbf_svm | auprc | -0.040 | [-0.129, +0.033] | 0.645 | 0/5 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | -0.078 | [-0.181, +0.012] | 0.326 | 0/5 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | -0.040 | [-0.124, +0.034] | 0.682 | 0/5 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | -0.055 | [-0.141, +0.038] | 0.589 | 0/5 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | -0.040 | [-0.139, +0.051] | 0.654 | 0/5 | no |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.129 | [-0.192, -0.046] | 0.000 | 0/5 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.091 | [-0.147, -0.035] | 0.000 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: rbf_svm vs u_mask_entropy | auroc | -0.051 | [-0.115, +0.009] | 0.099 | 0/5 | no |
| Best classical vs best training-free uncertainty score: rbf_svm vs u_mask_entropy | auprc | -0.051 | [-0.113, +0.018] | 0.141 | 0/5 | no |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.672 | 0.725 | 0.749 | 0.776 |
| rbf_svm | 0.672 | 0.732 | 0.765 | 0.796 |
| gbt | 0.672 | 0.727 | 0.746 | 0.751 |
| mlp | 0.672 | 0.734 | 0.761 | 0.778 |
| q_entangled | 0.672 | 0.732 | 0.756 | 0.767 |
| q_product | 0.672 | 0.731 | 0.755 | 0.764 |
| u_fragmentation | 0.672 | 0.723 | 0.752 | 0.777 |
| u_low_mask_confidence | 0.672 | 0.735 | 0.780 | 0.824 |
| u_low_max_confidence | 0.672 | 0.734 | 0.774 | 0.809 |
| u_mask_entropy | 0.672 | 0.735 | 0.781 | 0.824 |
| u_mean_entropy | 0.672 | 0.712 | 0.742 | 0.762 |
| oracle | 0.672 | 0.747 | 0.822 | 0.875 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.709 | 0.216 | 0.645 |
| q_product | 0.925 | 0.068 | 1.000 |

## How to read this

* The main question is the first comparison row: q_entangled vs q_product. Same circuit, same inputs, same tuning; the only difference is the entangling gates. If its CI includes 0, the data show no evidence that entanglement helps.
* 'Best classical' is chosen after seeing the results, which favours the classical side; that makes a positive quantum result more convincing, not less.
* The *_full rows use all raw features and are not matched; they show the cost of compressing to 4 dims.
* All quantum results are exact statevector simulations, not quantum hardware.
