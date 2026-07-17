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
