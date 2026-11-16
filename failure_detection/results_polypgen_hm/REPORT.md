# Failure-detection benchmark: what does the quantum circuit add?

3119 frames, 27 sequences, 27 raw features compressed to 8 PCA dims (= qubits). 5-fold CV grouped by sequence, seeds [0, 1, 2, 3, 4]. Circuit: ZZ feature map, reps=2, linear entanglement.

## Failure = Dice < 0.5  (832 failures, prevalence 26.7%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.878 ± 0.002 | 0.768 ± 0.003 |
| rbf_svm | 0.882 ± 0.003 | 0.741 ± 0.005 |
| gbt | 0.888 ± 0.003 | 0.766 ± 0.006 |
| mlp | 0.868 ± 0.033 | 0.747 ± 0.027 |
| q_entangled | 0.868 ± 0.018 | 0.726 ± 0.035 |
| q_product | 0.883 ± 0.009 | 0.765 ± 0.005 |
| logreg_full *(reference: all raw features)* | 0.911 ± 0.001 | 0.793 ± 0.001 |
| gbt_full *(reference: all raw features)* | 0.928 ± 0.001 | 0.837 ± 0.002 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.774 ± 0.000 | 0.610 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.920 ± 0.000 | 0.828 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.866 ± 0.000 | 0.756 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.923 ± 0.000 | 0.831 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.762 ± 0.000 | 0.546 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | -0.014 | [-0.042, +0.012] | 0.380 | 3/5 | no |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | -0.040 | [-0.096, +0.018] | 0.256 | 0/5 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs gbt | auroc | -0.018 | [-0.052, +0.016] | 0.430 | 0/5 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs gbt | auprc | -0.037 | [-0.090, +0.020] | 0.284 | 0/5 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | -0.011 | [-0.037, +0.007] | 0.279 | 1/5 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | -0.011 | [-0.070, +0.029] | 0.730 | 3/5 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | +0.003 | [-0.023, +0.026] | 0.890 | 2/5 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | +0.029 | [-0.023, +0.065] | 0.343 | 5/5 | no |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.051 | [-0.093, -0.020] | 0.000 | 0/5 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.098 | [-0.169, -0.039] | 0.000 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: gbt vs u_mask_entropy | auroc | -0.033 | [-0.062, -0.010] | 0.010 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: gbt vs u_mask_entropy | auprc | -0.061 | [-0.107, -0.021] | 0.002 | 0/5 | yes |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.672 | 0.734 | 0.776 | 0.810 |
| rbf_svm | 0.672 | 0.732 | 0.771 | 0.809 |
| gbt | 0.672 | 0.731 | 0.774 | 0.816 |
| mlp | 0.672 | 0.734 | 0.770 | 0.820 |
| q_entangled | 0.672 | 0.731 | 0.766 | 0.800 |
| q_product | 0.672 | 0.734 | 0.780 | 0.817 |
| u_fragmentation | 0.672 | 0.723 | 0.752 | 0.777 |
| u_low_mask_confidence | 0.672 | 0.735 | 0.780 | 0.824 |
| u_low_max_confidence | 0.672 | 0.734 | 0.774 | 0.809 |
| u_mask_entropy | 0.672 | 0.735 | 0.781 | 0.824 |
| u_mean_entropy | 0.672 | 0.712 | 0.742 | 0.762 |
| oracle | 0.672 | 0.747 | 0.822 | 0.875 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.640 | 0.206 | 0.617 |
| q_product | 0.614 | 0.144 | 1.000 |

## Failure = Dice < 0.2  (524 failures, prevalence 16.8%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.905 ± 0.000 | 0.741 ± 0.002 |
| rbf_svm | 0.882 ± 0.004 | 0.676 ± 0.004 |
| gbt | 0.855 ± 0.017 | 0.665 ± 0.007 |
| mlp | 0.898 ± 0.011 | 0.730 ± 0.030 |
| q_entangled | 0.827 ± 0.002 | 0.631 ± 0.004 |
| q_product | 0.898 ± 0.000 | 0.728 ± 0.001 |
| logreg_full *(reference: all raw features)* | 0.925 ± 0.002 | 0.779 ± 0.002 |
| gbt_full *(reference: all raw features)* | 0.915 ± 0.001 | 0.727 ± 0.005 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.801 ± 0.000 | 0.562 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.901 ± 0.000 | 0.716 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.890 ± 0.000 | 0.707 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.907 ± 0.000 | 0.725 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.711 ± 0.000 | 0.370 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | -0.071 | [-0.153, +0.002] | 0.075 | 0/5 | no |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | -0.091 | [-0.185, -0.007] | 0.008 | 0/5 | yes |
| Quantum vs best classical on the same 8 features: q_entangled vs logreg | auroc | -0.077 | [-0.169, +0.010] | 0.169 | 0/5 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs logreg | auprc | -0.104 | [-0.212, -0.002] | 0.036 | 0/5 | yes |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | -0.054 | [-0.128, +0.006] | 0.278 | 0/5 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | -0.041 | [-0.112, +0.023] | 0.443 | 0/5 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | +0.017 | [-0.007, +0.037] | 0.178 | 5/5 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | +0.050 | [-0.001, +0.075] | 0.061 | 5/5 | no |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.079 | [-0.124, -0.021] | 0.000 | 0/5 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.087 | [-0.138, -0.038] | 0.000 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: logreg vs u_mask_entropy | auroc | -0.001 | [-0.044, +0.047] | 0.882 | 0/5 | no |
| Best classical vs best training-free uncertainty score: logreg vs u_mask_entropy | auprc | +0.017 | [-0.061, +0.095] | 0.805 | 5/5 | no |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.672 | 0.735 | 0.780 | 0.814 |
| rbf_svm | 0.672 | 0.731 | 0.770 | 0.810 |
| gbt | 0.672 | 0.730 | 0.762 | 0.796 |
| mlp | 0.672 | 0.733 | 0.783 | 0.819 |
| q_entangled | 0.672 | 0.730 | 0.759 | 0.786 |
| q_product | 0.672 | 0.734 | 0.777 | 0.810 |
| u_fragmentation | 0.672 | 0.723 | 0.752 | 0.777 |
| u_low_mask_confidence | 0.672 | 0.735 | 0.780 | 0.824 |
| u_low_max_confidence | 0.672 | 0.734 | 0.774 | 0.809 |
| u_mask_entropy | 0.672 | 0.735 | 0.781 | 0.824 |
| u_mean_entropy | 0.672 | 0.712 | 0.742 | 0.762 |
| oracle | 0.672 | 0.747 | 0.822 | 0.875 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.603 | 0.218 | 0.654 |
| q_product | 0.960 | 0.032 | 1.000 |

## How to read this

* The main question is the first comparison row: q_entangled vs q_product. Same circuit, same inputs, same tuning; the only difference is the entangling gates. If its CI includes 0, the data show no evidence that entanglement helps.
* 'Best classical' is chosen after seeing the results, which favours the classical side; that makes a positive quantum result more convincing, not less.
* The *_full rows use all raw features and are not matched; they show the cost of compressing to 8 dims.
* All quantum results are exact statevector simulations, not quantum hardware.
