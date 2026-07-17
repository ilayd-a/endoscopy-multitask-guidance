# EBTC HGC vs LGC Official-Split Results

Run date: 2026-07-17

Command:

```bash
TORCH_HOME=/private/tmp/torch_home python3 EndoscopicBladderTissue/experiments/publication_benchmark_ebtc.py \
  --data_dir EndoscopicBladderTissue/dataset/baldder_tissue_classification \
  --label_mode hgc_vs_lgc \
  --split_mode official \
  --official_train_parts train \
  --models classical pqk \
  --max_samples 0 \
  --train_sizes 40 80 160 240 \
  --max_test_samples 0 \
  --repeats 3 \
  --classical_grid \
  --qsvm_grid \
  --no_plot
```

Protocol:

- Dataset: EBTC HGC vs LGC only
- Split: official `annotations.csv` split
- Training subset: balanced low-label subsamples from the official train split
- Test set: full official HGC/LGC test split, 127 images
- Test class counts: 53 LGC, 74 HGC
- Repeats: 3 training subsamples per train size

## Best PQK vs Best Classical by AUC

| Train size | Best PQK | PQK AUC | PQK bal. acc. | PQK sensitivity | PQK specificity | PQK FNR | Best classical | Classical AUC | Classical bal. acc. | Classical sensitivity | Classical specificity | Classical FNR |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 40 | PQK reps1 | 0.764 | 0.607 | 0.842 | 0.371 | 0.158 | RBF SVM C1 gammaScale | 0.817 | 0.642 | 0.856 | 0.428 | 0.144 |
| 80 | PQK reps1 | 0.762 | 0.649 | 0.707 | 0.591 | 0.293 | Linear SVM C1 | 0.860 | 0.770 | 0.671 | 0.868 | 0.329 |
| 160 | PQK reps1 | 0.687 | 0.568 | 0.815 | 0.321 | 0.185 | RBF SVM C0.1 gammaScale | 0.822 | 0.743 | 0.631 | 0.855 | 0.369 |
| 240 | PQK reps1 | 0.750 | 0.648 | 0.779 | 0.516 | 0.221 | Linear SVM C10 | 0.856 | 0.778 | 0.644 | 0.912 | 0.356 |

## Interpretation

This run does not support a broad claim that the projected quantum kernel improves the overall
classifier beyond strong classical baselines on the official EBTC split. Classical models have
higher AUC and balanced accuracy at every tested training size.

The interesting signal is narrower: PQK often shifts the decision behavior toward higher HGC
sensitivity and lower false-negative rate than the best-AUC classical model, but this comes with
lower specificity and worse calibration. That may still be medically relevant if reframed as a
high-sensitivity second-reader or hard-case triage module, but it is not yet a trustworthy
standalone diagnostic improvement.

The next scientific step is threshold-aware evaluation: compare classical and PQK models at fixed
specificity or fixed sensitivity, rather than only using the default 0.5 decision threshold.

## Threshold-Aware Follow-Up

The benchmark now exports per-sample prediction scores to:

```text
EndoscopicBladderTissue/results/publication_benchmark/ebtc_publication_predictions.csv
```

Threshold analysis command:

```bash
python3 EndoscopicBladderTissue/experiments/threshold_analysis_ebtc.py
```

Outputs:

```text
EndoscopicBladderTissue/results/publication_benchmark/ebtc_threshold_metrics.csv
EndoscopicBladderTissue/results/publication_benchmark/ebtc_threshold_aggregate.csv
```

### Fixed Specificity

At fixed specificity, the question is: if we require few false positives, which model keeps the
highest HGC sensitivity?

| Train size | Constraint | Best PQK sensitivity | Best classical sensitivity | Best classical model |
|---:|---:|---:|---:|---|
| 40 | specificity >= 0.90 | 0.523 | 0.626 | MLP |
| 80 | specificity >= 0.90 | 0.568 | 0.680 | Linear SVM C0.1 |
| 160 | specificity >= 0.90 | 0.437 | 0.640 | RBF SVM C0.1 gammaScale |
| 240 | specificity >= 0.90 | 0.532 | 0.671 | Linear SVM C0.1 |
| 40 | specificity >= 0.95 | 0.473 | 0.577 | RBF SVM C1 gammaScale |
| 80 | specificity >= 0.95 | 0.532 | 0.613 | RBF SVM C0.1 gammaScale |
| 160 | specificity >= 0.95 | 0.374 | 0.595 | RBF SVM C0.1 gammaScale |
| 240 | specificity >= 0.95 | 0.446 | 0.644 | RBF SVM C0.1 gammaScale |

### Fixed Sensitivity

At fixed sensitivity, the question is: if we require few missed HGC cases, which model preserves
the highest specificity?

| Train size | Constraint | Best PQK specificity | Best classical specificity | Best classical model |
|---:|---:|---:|---:|---|
| 40 | sensitivity >= 0.90 | 0.327 | 0.447 | Logistic regression C0.1 |
| 80 | sensitivity >= 0.90 | 0.277 | 0.503 | MLP |
| 160 | sensitivity >= 0.90 | 0.176 | 0.333 | Linear SVM C1 |
| 240 | sensitivity >= 0.90 | 0.415 | 0.484 | Linear SVM C10 |
| 40 | sensitivity >= 0.95 | 0.214 | 0.377 | Logistic regression C0.1 |
| 80 | sensitivity >= 0.95 | 0.182 | 0.390 | MLP |
| 160 | sensitivity >= 0.95 | 0.145 | 0.220 | RBF SVM C0.1 gammaScale |
| 240 | sensitivity >= 0.95 | 0.302 | 0.365 | RBF SVM C0.1 gammaScale |

Conclusion: threshold-aware analysis does not reveal a hidden PQK advantage on this official
HGC/LGC split. Classical baselines still dominate the clinically constrained operating points.
The most honest next pivot is not "PQK improves diagnosis," but either:

- use PQK as a negative/limited benchmark with kernel diagnostics, or
- move the quantum component to a different task where it may have more room to help, such as
  hard-case candidate reranking, uncertainty triage, or multimodal WLI/NBI feature fusion.
