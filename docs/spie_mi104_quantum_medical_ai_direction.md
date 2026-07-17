# SPIE MI104 Quantum Medical AI Direction

## Current Evidence From This Repo

The leakage-controlled EBTC benchmark and the kidney-stone Dataset4 runs show that a plain
"medical image classification + QSVM" story is not strong enough yet.

- Whole-frame kidney-stone classification is unstable under the temporal split. Classical models
  and PQK models both struggle with accuracy and balanced accuracy.
- Patch-level kidney-stone classification is more clinically meaningful, but it still shows poor
  calibrated detection on the held-out split. The best useful signal is ranking/AUC, not hard
  classification.
- Synthetic and EBTC runs suggest projected quantum kernels are more promising than vanilla
  fidelity QSVM, but only in a carefully controlled, low-data benchmark.

This is not a dead end. It means the publishable question should move away from broad image
classification and toward image-guided localization under limited annotation.

## Literature-Based Gap

Recent reviews of QML in healthcare converge on the same weaknesses in the field:

- Many papers report high accuracy on small curated classification datasets.
- Few use realistic validation, noisy/hardware-aware constraints, or clinically meaningful
  image-guidance endpoints.
- Data encoding, dimensionality reduction, and classical preprocessing are often under-analyzed.
- Strong classical baselines are frequently missing or not leakage-controlled.

For SPIE MI104, the gap is therefore not "can quantum beat CNNs on image classification?" A
better gap is:

> Can a shallow quantum or quantum-inspired kernel improve low-label target localization,
> candidate ranking, or uncertainty-aware guidance on endoscopic intervention images, when
> evaluated against strong classical baselines under leakage-controlled domain shift?

## Recommended Pivot

Use the endoscopy guidance pipeline as the clinical backbone and use QML as a small-data ranking
or refinement module.

Proposed title direction:

> Quantum-kernel candidate ranking for low-label endoscopic image guidance under domain shift

Core idea:

1. Train or reuse a classical segmentation/heatmap model from the endoscopy multitask guidance repo.
2. Generate candidate target regions from the heatmap, segmentation boundary, or sliding patches.
3. Extract compact descriptors for each candidate:
   - CNN/ResNet features
   - heatmap intensity statistics
   - distance-to-peak and local texture
   - optional temporal consistency features
4. Train shallow models to rank candidate regions as target vs background:
   - logistic regression
   - linear/RBF SVM
   - random forest
   - projected quantum kernel SVM
   - vanilla quantum kernel only as a baseline
5. Evaluate image-guidance outcomes, not only classification:
   - pointing-game accuracy
   - peak-to-center distance
   - top-k localization hit rate
   - Dice/IoU after heatmap refinement
   - calibration and uncertainty
   - low-label learning curves
   - domain-shift split, for example EBTC HGC to LGC or video-block split

## Why This Is Stronger

This direction fits MI104 better because it is about guidance behavior, not generic diagnosis.
It also gives QML a narrower and more plausible role: low-dimensional candidate ranking under
limited labels. That is where current QML claims are most defensible.

It also lets us write a more honest paper:

- We do not claim broad quantum advantage.
- We test whether quantum kernels help in a constrained, clinically motivated subproblem.
- We include strong classical baselines and leakage-safe splits.
- We report where quantum fails, where it ranks better, and whether it improves guidance metrics.

## Minimum Publishable Experiment

The minimum credible SPIE abstract should include:

1. Leakage-controlled EBTC classification benchmark:
   - classical vs QSVM vs PQK
   - repeated splits
   - low-label curves
   - kernel diagnostics

2. Endoscopy guidance candidate-ranking experiment:
   - candidate patches from masks or heatmaps
   - low-label training sizes
   - domain-shift validation
   - pointing-game, center-distance, top-k hit rate

3. Ablation:
   - raw CNN features vs PCA features
   - classical kernel vs projected quantum kernel
   - number of qubits/components
   - circuit reps

4. Failure analysis:
   - kernel-target alignment
   - kernel concentration
   - threshold sensitivity
   - examples where quantum ranking helps or hurts

## Practical Next Steps

1. Move the main paper story to endoscopic target localization.
2. Keep kidney-stone Dataset4 as a pilot or supplemental negative result unless labels are expanded.
3. Build a candidate-ranking script that reads masks/heatmaps from the endoscopy repo and writes:
   - per-candidate classification metrics
   - per-image top-k localization metrics
   - refined heatmap metrics
4. Add visual figures:
   - heatmap before/after quantum-kernel reranking
   - low-label performance curve
   - kernel diagnostics panel
5. Use the abstract to emphasize rigorous benchmarking and image-guided intervention relevance,
   not quantum hype.

## First Candidate-Ranking Smoke Test

The first implementation is in `endoscopy_guidance/candidate_ranking_benchmark.py`.
It reads the `eval/sample_data` exports from the endoscopy multitask guidance repo and performs
leave-one-frame-out candidate ranking.

Initial result on the current 10-frame sample export:

- 760 candidate points
- 180 target-positive candidates
- Baseline heatmap candidate AUC: about 0.9998
- Baseline top-1 and top-3 localization hit rate: 1.0
- Classical and projected quantum-kernel rerankers also achieve top-1 and top-3 hit rate of 1.0

Interpretation: this sample export is useful as a code smoke test, but it is too easy for a
publishable reranking comparison. The heatmap already localizes every sample correctly, so no
reranker has meaningful room to improve. The next dataset export should include domain-shifted
or failure-prone cases where the heatmap top candidates include plausible false positives.

## Held-Out CVC Test Export

The exporter in `endoscopy_guidance/export_cvc_predictions.py` reads the local
`endoscopy-multitask-guidance` checkpoint `models/unet_cvc.pth` and exports NumPy triplets for
the sequence-held-out CVC test split.

Export command:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdgcache \
python3 endoscopy_guidance/export_cvc_predictions.py \
  --split test \
  --output_dir endoscopy_guidance/exports/cvc_test \
  --device cpu
```

Exported test split:

- 66 held-out frames from CVC sequences 27-29
- Mean Dice: 0.0862
- Mean IoU: 0.0549
- Pointing-game accuracy: 0.1364

This is a much harder and more useful reranking setting than the original 10-frame sample export.

Candidate-ranking command:

```bash
python3 endoscopy_guidance/candidate_ranking_benchmark.py \
  --data_dir endoscopy_guidance/exports/cvc_test \
  --results_csv endoscopy_guidance/results/cvc_test_candidate_ranking_metrics_balanced.csv \
  --aggregate_csv endoscopy_guidance/results/cvc_test_candidate_ranking_aggregate_balanced.csv \
  --candidates_csv endoscopy_guidance/results/cvc_test_candidate_table_balanced.csv \
  --top_n 8 \
  --grid_stride 48 \
  --nms_dist 20 \
  --patch_radius 14 \
  --sample_folds 5
```

Held-out CVC candidate-ranking result without RGB patch features:

| Model | AUC | Balanced accuracy | F1 | Top-1 hit | Top-3 hit | Best positive rank |
|---|---:|---:|---:|---:|---:|---:|
| Heatmap score baseline | 0.640 | NA | NA | 0.135 | 0.212 | 11.655 |
| Logistic regression | 0.709 | 0.636 | 0.179 | 0.166 | 0.197 | 9.414 |
| RBF SVM | 0.744 | 0.695 | 0.218 | 0.105 | 0.334 | 9.058 |
| Random forest | 0.842 | 0.555 | 0.186 | 0.333 | 0.590 | 3.339 |
| Balanced PQK, reps 3, C=10 | 0.819 | 0.764 | 0.283 | 0.210 | 0.440 | 3.808 |

Interpretation:

- Random forest is currently the strongest classical reranker for top-k localization.
- Balanced projected quantum kernels substantially improve over the raw heatmap baseline:
  AUC improves from 0.640 to 0.819, top-3 hit from 0.212 to 0.440, and best-positive-rank
  from 11.655 to 3.808.
- The best PQK model has higher balanced accuracy and F1 than the listed classical baselines,
  but lower top-1/top-3 localization than random forest.
- This is now a viable SPIE-style result if framed carefully as quantum-kernel candidate
  reranking under domain shift, not as broad quantum superiority.

## RGB Patch Feature Upgrade

The exporter can also save resized RGB frames:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdgcache \
python3 endoscopy_guidance/export_cvc_predictions.py \
  --split test \
  --output_dir endoscopy_guidance/exports/cvc_test_rgb \
  --device cpu \
  --save_images
```

The candidate benchmark can then include local RGB statistics, patch contrast, and simple
gradient-texture descriptors:

```bash
python3 endoscopy_guidance/candidate_ranking_benchmark.py \
  --data_dir endoscopy_guidance/exports/cvc_test_rgb \
  --results_csv endoscopy_guidance/results/cvc_test_rgb_candidate_ranking_metrics.csv \
  --aggregate_csv endoscopy_guidance/results/cvc_test_rgb_candidate_ranking_aggregate.csv \
  --candidates_csv endoscopy_guidance/results/cvc_test_rgb_candidate_table.csv \
  --top_n 8 \
  --grid_stride 48 \
  --nms_dist 20 \
  --patch_radius 14 \
  --sample_folds 5 \
  --image_features
```

Held-out CVC result with RGB patch features:

| Model | AUC | Balanced accuracy | F1 | Top-1 hit | Top-3 hit | Top-5 hit | Best positive rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Heatmap score baseline | 0.640 | NA | NA | 0.135 | 0.212 | 0.256 | 11.655 |
| ExtraTrees | 0.902 | 0.558 | 0.200 | 0.437 | 0.590 | 0.652 | 2.517 |
| HistGradientBoosting | 0.871 | 0.653 | 0.364 | 0.379 | 0.529 | 0.651 | 3.107 |
| RBF SVM | 0.888 | 0.811 | 0.332 | 0.288 | 0.469 | 0.560 | 3.236 |
| Random forest | 0.891 | 0.534 | 0.123 | 0.424 | 0.592 | 0.637 | 2.560 |
| Balanced PQK, reps 2, C=1 | 0.898 | 0.822 | 0.377 | 0.318 | 0.545 | 0.682 | 2.647 |
| Balanced PQK, reps 3, C=10 | 0.882 | 0.753 | 0.407 | 0.364 | 0.546 | 0.697 | 2.565 |
| Balanced PQK, reps 3, C=1 | 0.895 | 0.818 | 0.402 | 0.319 | 0.591 | 0.621 | 2.854 |

Interpretation:

- RGB patch features make the reranking task much stronger across both classical and PQK models.
- ExtraTrees has the highest AUC and top-1 localization.
- Random forest and balanced PQK are essentially tied on top-3 localization.
- Balanced PQK gives the strongest F1 and top-5 hit among the tested models, while preserving
  high balanced accuracy.
- This is the most promising version for the SPIE abstract: it supports a careful claim that a
  projected quantum-kernel reranker is competitive with strong classical rerankers and improves
  clinically relevant top-k target recovery over the degraded heatmap baseline.

## Go/No-Go Criteria

This becomes abstract-worthy if at least one of these holds:

- PQK improves top-1 or top-k localization over classical baselines in the lowest-label setting.
- PQK gives better calibrated uncertainty or candidate ranking even when accuracy is similar.
- Kernel diagnostics identify when QML fails, giving a useful negative benchmark for medical QML.

If none of these hold, the publication should be reframed as a rigorous negative benchmark rather
than an intervention method paper.
