# Failure-detection benchmark: what does the quantum circuit add?

612 frames, 29 sequences, 27 raw features compressed to 8 PCA dims (= qubits). 5-fold CV grouped by sequence, seeds [0, 1, 2, 3, 4]. Circuit: ZZ feature map, reps=2, linear entanglement.

## Failure = Dice < 0.5  (78 failures, prevalence 12.7%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.780 ± 0.013 | 0.478 ± 0.017 |
| rbf_svm | 0.788 ± 0.021 | 0.479 ± 0.028 |
| gbt | 0.796 ± 0.024 | 0.562 ± 0.015 |
| mlp | 0.784 ± 0.013 | 0.561 ± 0.016 |
| q_entangled | 0.725 ± 0.027 | 0.353 ± 0.049 |
| q_product | 0.767 ± 0.017 | 0.410 ± 0.020 |
| logreg_full *(reference: all raw features)* | 0.874 ± 0.006 | 0.639 ± 0.010 |
| gbt_full *(reference: all raw features)* | 0.897 ± 0.005 | 0.689 ± 0.016 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.767 ± 0.000 | 0.447 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.922 ± 0.000 | 0.750 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.850 ± 0.000 | 0.648 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.924 ± 0.000 | 0.747 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.725 ± 0.000 | 0.281 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | -0.033 | [-0.081, +0.010] | 0.123 | 0/5 | no |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | -0.040 | [-0.153, +0.040] | 0.304 | 0/5 | no |
| Quantum vs best classical on the same 8 features: q_entangled vs gbt | auroc | -0.074 | [-0.118, -0.033] | 0.000 | 0/5 | yes |
| Quantum vs best classical on the same 8 features: q_entangled vs gbt | auprc | -0.226 | [-0.309, -0.128] | 0.000 | 0/5 | yes |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | -0.056 | [-0.120, +0.007] | 0.090 | 0/5 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | -0.127 | [-0.209, -0.041] | 0.004 | 0/5 | yes |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | -0.023 | [-0.067, +0.018] | 0.287 | 0/5 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | -0.087 | [-0.183, +0.035] | 0.258 | 0/5 | no |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.182 | [-0.241, -0.111] | 0.000 | 0/5 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.362 | [-0.456, -0.247] | 0.000 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: gbt vs u_mask_entropy | auroc | -0.108 | [-0.160, -0.045] | 0.001 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: gbt vs u_mask_entropy | auprc | -0.136 | [-0.193, -0.080] | 0.000 | 0/5 | yes |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.789 | 0.831 | 0.852 | 0.860 |
| rbf_svm | 0.789 | 0.830 | 0.851 | 0.863 |
| gbt | 0.789 | 0.835 | 0.853 | 0.861 |
| mlp | 0.789 | 0.835 | 0.844 | 0.853 |
| q_entangled | 0.789 | 0.822 | 0.837 | 0.847 |
| q_product | 0.789 | 0.826 | 0.849 | 0.859 |
| u_fragmentation | 0.789 | 0.826 | 0.846 | 0.851 |
| u_low_mask_confidence | 0.789 | 0.842 | 0.870 | 0.887 |
| u_low_max_confidence | 0.789 | 0.838 | 0.854 | 0.865 |
| u_mask_entropy | 0.789 | 0.843 | 0.870 | 0.887 |
| u_mean_entropy | 0.789 | 0.811 | 0.826 | 0.841 |
| oracle | 0.789 | 0.858 | 0.896 | 0.919 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.314 | 0.188 | 0.683 |
| q_product | 0.634 | 0.132 | 1.000 |

## Failure = Dice < 0.2  (36 failures, prevalence 5.9%)

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.856 ± 0.020 | 0.683 ± 0.076 |
| rbf_svm | 0.893 ± 0.017 | 0.661 ± 0.041 |
| gbt | 0.883 ± 0.035 | 0.689 ± 0.041 |
| mlp | 0.828 ± 0.029 | 0.673 ± 0.029 |
| q_entangled | 0.806 ± 0.026 | 0.433 ± 0.096 |
| q_product | 0.836 ± 0.014 | 0.643 ± 0.038 |
| logreg_full *(reference: all raw features)* | 0.860 ± 0.021 | 0.708 ± 0.049 |
| gbt_full *(reference: all raw features)* | 0.872 ± 0.019 | 0.676 ± 0.048 |
| u_fragmentation *(training-free U-Net uncertainty)* | 0.799 ± 0.000 | 0.497 ± 0.000 |
| u_low_mask_confidence *(training-free U-Net uncertainty)* | 0.954 ± 0.000 | 0.790 ± 0.000 |
| u_low_max_confidence *(training-free U-Net uncertainty)* | 0.918 ± 0.000 | 0.771 ± 0.000 |
| u_mask_entropy *(training-free U-Net uncertainty)* | 0.955 ± 0.000 | 0.796 ± 0.000 |
| u_mean_entropy *(training-free U-Net uncertainty)* | 0.568 ± 0.000 | 0.088 ± 0.000 |

AUPRC of a random guess equals the prevalence.

**Paired comparisons** (seed-averaged scores, 95% CI from bootstrap over sequences):

| Question | Metric | Δ (A − B) | 95% CI | p | seeds A better | CI excludes 0 |
|---|---|---|---|---|---|---|
| Does entanglement help? (main ablation): q_entangled vs q_product | auroc | -0.015 | [-0.051, +0.038] | 0.519 | 0/5 | no |
| Does entanglement help? (main ablation): q_entangled vs q_product | auprc | -0.104 | [-0.199, -0.038] | 0.001 | 0/5 | yes |
| Quantum vs best classical on the same 8 features: q_entangled vs rbf_svm | auroc | -0.072 | [-0.112, -0.018] | 0.005 | 0/5 | yes |
| Quantum vs best classical on the same 8 features: q_entangled vs rbf_svm | auprc | -0.089 | [-0.177, +0.022] | 0.112 | 0/5 | no |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auroc | -0.072 | [-0.114, -0.018] | 0.012 | 0/5 | yes |
| Quantum kernel vs classical kernel: q_entangled vs rbf_svm | auprc | -0.089 | [-0.176, +0.022] | 0.103 | 0/5 | no |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auroc | -0.057 | [-0.091, -0.021] | 0.003 | 0/5 | yes |
| Product-state circuit vs classical kernel: q_product vs rbf_svm | auprc | +0.015 | [-0.030, +0.109] | 0.616 | 2/5 | no |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auroc | -0.123 | [-0.169, -0.066] | 0.000 | 0/5 | yes |
| Quantum vs best training-free uncertainty score: q_entangled vs u_mask_entropy | auprc | -0.186 | [-0.274, -0.104] | 0.000 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: rbf_svm vs u_mask_entropy | auroc | -0.051 | [-0.095, -0.015] | 0.003 | 0/5 | yes |
| Best classical vs best training-free uncertainty score: rbf_svm vs u_mask_entropy | auprc | -0.098 | [-0.195, -0.032] | 0.004 | 0/5 | yes |

**Selective segmentation**: mean Dice of frames kept after rejecting the most suspicious fraction

| Model | reject 0% | reject 10% | reject 20% | reject 30% |
|---|---|---|---|---|
| logreg | 0.789 | 0.832 | 0.834 | 0.838 |
| rbf_svm | 0.789 | 0.825 | 0.841 | 0.846 |
| gbt | 0.789 | 0.834 | 0.838 | 0.841 |
| mlp | 0.789 | 0.830 | 0.835 | 0.842 |
| q_entangled | 0.789 | 0.826 | 0.831 | 0.832 |
| q_product | 0.789 | 0.832 | 0.835 | 0.840 |
| u_fragmentation | 0.789 | 0.826 | 0.846 | 0.851 |
| u_low_mask_confidence | 0.789 | 0.842 | 0.870 | 0.887 |
| u_low_max_confidence | 0.789 | 0.838 | 0.854 | 0.865 |
| u_mask_entropy | 0.789 | 0.843 | 0.870 | 0.887 |
| u_mean_entropy | 0.789 | 0.811 | 0.826 | 0.841 |
| oracle | 0.789 | 0.858 | 0.896 | 0.919 |

**Circuit diagnostics** (training folds, chosen configs): purity 1 = no entanglement; off-diagonal kernel mean near 0 with tiny SD = kernel concentration.

| Model | kernel off-diag mean | kernel off-diag SD | mean 1-qubit purity |
|---|---|---|---|
| q_entangled | 0.271 | 0.184 | 0.691 |
| q_product | 0.609 | 0.119 | 1.000 |

## How to read this

* The main question is the first comparison row: q_entangled vs q_product. Same circuit, same inputs, same tuning; the only difference is the entangling gates. If its CI includes 0, the data show no evidence that entanglement helps.
* 'Best classical' is chosen after seeing the results, which favours the classical side; that makes a positive quantum result more convincing, not less.
* The *_full rows use all raw features and are not matched; they show the cost of compressing to 8 dims.
* All quantum results are exact statevector simulations, not quantum hardware.
