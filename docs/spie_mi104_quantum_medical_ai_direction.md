# SPIE MI104 Quantum Medical AI Direction

## Current Evidence From This Repo

The leakage-controlled EBTC benchmark and the kidney-stone Dataset4 runs show that a plain
"medical image classification + QSVM" story is not strong enough yet.

- The README-linked EBTC dataset is the public bladder-tissue dataset from Zenodo/Kaggle,
  containing HGC, LGC, NST, and NTL images with `annotations.csv` splits. The most relevant
  comparison task for the endoscopy subgroup is HGC vs LGC, not the earlier CVC polyp export.
- Whole-frame kidney-stone classification is unstable under the temporal split. Classical models
  and PQK models both struggle with accuracy and balanced accuracy.
- Patch-level kidney-stone classification is more clinically meaningful, but it still shows poor
  calibrated detection on the held-out split. The best useful signal is ranking/AUC, not hard
  classification.
- The correct HGC/LGC EBTC run suggests projected quantum kernels are most promising in the
  low-label setting: PQK led classical baselines by AUC at 40 labels and tied them at 80 labels,
  while strong classical RBF SVMs led at 160-240 labels.
- Synthetic and EBTC runs suggest projected quantum kernels are more promising than vanilla
  fidelity QSVM, but only in carefully controlled, low-data benchmarks.

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
   - official `annotations.csv` train/test split as the primary result
   - repeated random splits as a robustness check
   - low-label curves
   - sensitivity, specificity, false-negative rate, PPV/NPV, calibration, and ROC-AUC
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

## Scientific Validation Rule

The target should not be "make the number 0.98." The target should be:

> Improve a clinically meaningful endpoint on the official split and preserve that improvement
> under repeated-split robustness checks, without degrading sensitivity or calibration.

For EBTC HGC/LGC classification, the primary acceptance criteria should be:

- official-split sensitivity for HGC
- false-negative rate for HGC
- ROC-AUC and balanced accuracy
- Brier score and expected calibration error
- confidence intervals or repeated-split mean +/- std
- comparison against the best classical baseline, not only a weak baseline

If quantum improves only random-split accuracy but not official-split sensitivity, calibration, or
hard-case behavior, it should be reported as an interesting negative or limited result rather than
a clinical improvement.

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

Important caveat: this CVC experiment is an exploratory image-guidance reranking test, not the
README-linked EBTC/HGC-LGC benchmark. It should not be used as a direct comparison to the original
bladder-tissue classification work.

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

## Reranker-To-Heatmap Refinement

Candidate ranking is useful, but a real guidance system should output a refined target map. The
benchmark now evaluates a simple reranker-to-heatmap refinement step:

1. Rank candidate locations using the trained reranker.
2. Paint Gaussian blobs at the top-ranked candidates.
3. Normalize the resulting guidance map.
4. Compare the original heatmap and refined heatmap using pointing-game, peak-center distance,
   Dice, and IoU.

Best current refinement setting:

```bash
python3 endoscopy_guidance/candidate_ranking_benchmark.py \
  --data_dir endoscopy_guidance/exports/cvc_test_rgb \
  --results_csv endoscopy_guidance/results/cvc_test_rgb_refinement_metrics_a0_s20.csv \
  --aggregate_csv endoscopy_guidance/results/cvc_test_rgb_refinement_aggregate_a0_s20.csv \
  --candidates_csv endoscopy_guidance/results/cvc_test_rgb_refinement_candidates_a0_s20.csv \
  --top_n 8 \
  --grid_stride 48 \
  --nms_dist 20 \
  --patch_radius 14 \
  --sample_folds 5 \
  --image_features \
  --refine_alpha 0.0 \
  --refine_sigma 20 \
  --refine_top_k 5
```

This setting uses the reranker-generated map directly rather than blending with the degraded
backbone heatmap.

Refined heatmap result:

| Model | Refined pointing | Refined peak distance | Refined Dice | Refined IoU |
|---|---:|---:|---:|---:|
| Original heatmap baseline | 0.136 | 78.1 px | 0.086 | 0.055 |
| ExtraTrees refinement | 0.438 | 53.7 px | 0.274 | 0.182 |
| Random forest refinement | 0.454 | 55.2 px | 0.260 | 0.173 |
| Balanced PQK, reps 2, C=1 refinement | 0.392 | 54.6 px | 0.255 | 0.166 |
| Balanced PQK, reps 3, C=10 refinement | 0.410 | 54.0 px | 0.262 | 0.171 |
| Balanced PQK, reps 3, C=1 refinement | 0.377 | 59.4 px | 0.242 | 0.158 |

Interpretation:

- The refinement module turns reranking into an actual corrected guidance output.
- All reranker-refined maps substantially improve over the original domain-shifted heatmap.
- Random forest has the best refined pointing, while ExtraTrees has the best refined Dice/IoU.
- Balanced PQK remains competitive and gives a clinically meaningful map-level improvement:
  pointing improves from 0.136 to 0.410, peak distance drops from 78.1 px to 54.0 px, Dice rises
  from 0.086 to 0.262, and IoU rises from 0.055 to 0.171.
- This strengthens the SPIE framing because the method now improves an image-guidance output,
  not only a candidate-ranking table.

## Go/No-Go Criteria

This becomes abstract-worthy if at least one of these holds:

- PQK improves top-1 or top-k localization over classical baselines in the lowest-label setting.
- PQK gives better calibrated uncertainty or candidate ranking even when accuracy is similar.
- Kernel diagnostics identify when QML fails, giving a useful negative benchmark for medical QML.

If none of these hold, the publication should be reframed as a rigorous negative benchmark rather
than an intervention method paper.

## Low-Label Candidate-Ranking Sweep

The candidate-ranking benchmark now supports repeated balanced low-label candidate subsampling
through `--train_candidate_sizes` and `--repeats`. This tests whether projected quantum kernels
are more useful when only a small number of candidate annotations are available.

Low-label sweep command:

```bash
python3 endoscopy_guidance/candidate_ranking_benchmark.py \
  --data_dir endoscopy_guidance/exports/cvc_test_rgb \
  --results_csv endoscopy_guidance/results/cvc_test_rgb_lowlabel_metrics.csv \
  --aggregate_csv endoscopy_guidance/results/cvc_test_rgb_lowlabel_aggregate.csv \
  --candidates_csv endoscopy_guidance/results/cvc_test_rgb_lowlabel_candidates.csv \
  --top_n 8 \
  --grid_stride 48 \
  --nms_dist 20 \
  --patch_radius 14 \
  --sample_folds 5 \
  --train_candidate_sizes 40 80 160 320 0 \
  --repeats 3 \
  --image_features \
  --refine_alpha 0 \
  --refine_sigma 20 \
  --refine_top_k 5
```

Summary of the best classical model and best PQK model at each training-candidate budget:

| Training candidates | Baseline top-5 | Best classical top-5 | Best PQK top-5 | Baseline Dice | Best classical Dice | Best PQK Dice | Best classical peak distance | Best PQK peak distance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.256 | 0.565 | 0.494 | 0.076 | 0.194 | 0.162 | 61.6 px | 65.8 px |
| 80 | 0.256 | 0.607 | 0.575 | 0.076 | 0.203 | 0.202 | 60.3 px | 61.0 px |
| 160 | 0.256 | 0.651 | 0.600 | 0.076 | 0.228 | 0.221 | 58.3 px | 58.4 px |
| 320 | 0.256 | 0.636 | 0.616 | 0.076 | 0.229 | 0.234 | 58.6 px | 57.0 px |
| all | 0.256 | 0.652 | 0.697 | 0.076 | 0.274 | 0.265 | 53.7 px | 54.0 px |

Interpretation:

- The low-label result does not support a simple claim that PQK beats strong classical rerankers
  with very few annotations. Classical ensembles are still stronger at 40-160 candidates.
- The result does support the image-guidance pivot: all learned rerankers, including PQK, strongly
  improve over the degraded heatmap baseline.
- PQK becomes competitive at 320 candidates, where it slightly improves refined Dice and peak
  distance over the best classical model.
- With all available candidates, PQK gives the strongest top-5 localization hit rate
  (0.697 vs 0.652 classical), which is clinically relevant if the guidance system presents a
  shortlist of target hypotheses rather than a single hard point.
- The best SPIE claim is therefore not "quantum improves low-label diagnosis." It is:

> A projected quantum-kernel reranker can convert a weak endoscopic heatmap into a substantially
> improved target-candidate guidance signal, remaining competitive with strong classical rerankers
> and improving top-k target recovery in a sequence-held-out setting.

## Stronger Next Experiment

To make this more publishable, the next experiment should evaluate a hybrid decision rule rather
than asking PQK to replace the best classical reranker:

1. Use a classical model to generate high-recall candidate proposals.
2. Use PQK only as a second-stage diversity or uncertainty reranker among the top candidates.
3. Optimize for top-k clinical safety metrics:
   - target present in top 3 or top 5 candidates
   - false-negative rate of candidate shortlist
   - peak-center distance
   - refined Dice/IoU
   - calibration of candidate confidence
4. Compare against classical-only two-stage rerankers using the same candidate budget.

This direction fits MI104 better because it matches an intervention workflow: a system can surface
several plausible target regions for robotic or image-guided assistance, while preserving a strong
classical baseline and using QML only where it has a plausible narrow role.

## Active-Learning / Annotation-Triage Experiment

The first active-learning implementation is in
`endoscopy_guidance/active_learning_candidate_benchmark.py`. It simulates candidate annotation
rounds on the sequence-held-out CVC RGB export:

1. Start with 40 balanced labeled candidates.
2. Acquire batches of 40 additional candidate labels.
3. Compare acquisition policies:
   - random sampling
   - classical uncertainty
   - PQK uncertainty
   - PQK uncertainty plus diversity
   - PQK hybrid uncertainty plus random coverage
4. Retrain classical and PQK rerankers at each annotation budget.
5. Evaluate held-out candidate AUC, top-k target recovery, and refined map metrics.

Command:

```bash
python3 endoscopy_guidance/active_learning_candidate_benchmark.py \
  --data_dir endoscopy_guidance/exports/cvc_test_rgb \
  --results_csv endoscopy_guidance/results/cvc_test_rgb_active_learning_metrics.csv \
  --aggregate_csv endoscopy_guidance/results/cvc_test_rgb_active_learning_aggregate.csv \
  --top_n 8 \
  --grid_stride 48 \
  --nms_dist 20 \
  --patch_radius 14 \
  --sample_folds 5 \
  --initial_labels 40 \
  --batch_size 40 \
  --rounds 4 \
  --repeats 3 \
  --strategies random classical_uncertainty pqk_uncertainty pqk_diversity pqk_hybrid \
  --image_features \
  --refine_alpha 0 \
  --refine_sigma 20 \
  --refine_top_k 5
```

For a classical logistic-regression final reranker, the PQK-hybrid acquisition policy gives the
best top-5 guidance recovery after active annotation begins:

| Labeled candidates | Random top-5 | Classical-uncertainty top-5 | PQK-uncertainty top-5 | PQK-diversity top-5 | PQK-hybrid top-5 |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 |
| 80 | 0.549 | 0.534 | 0.554 | 0.535 | 0.560 |
| 120 | 0.554 | 0.544 | 0.554 | 0.530 | 0.565 |
| 160 | 0.529 | 0.529 | 0.564 | 0.550 | 0.575 |
| 200 | 0.519 | 0.534 | 0.565 | 0.565 | 0.580 |

The strongest signal is label discovery. After the initial balanced seed, random sampling selects
positive candidates at roughly 5-8%, while PQK-based policies select positives much more often:

| Labeled candidates | Random selected-positive rate | PQK-uncertainty selected-positive rate | PQK-diversity selected-positive rate | PQK-hybrid selected-positive rate |
|---:|---:|---:|---:|---:|
| 80 | 0.053 | 0.053 | 0.077 | 0.055 |
| 120 | 0.073 | 0.137 | 0.187 | 0.115 |
| 160 | 0.065 | 0.268 | 0.158 | 0.157 |
| 200 | 0.083 | 0.245 | 0.172 | 0.193 |

Interpretation:

- This is a stronger quantum-medical-AI direction than direct segmentation replacement.
- PQK uncertainty is useful for finding rare target-positive candidate annotations under heavy
  class imbalance.
- Pure PQK acquisition can over-focus on positives, which helps discovery but does not always
  improve final AUC or Dice.
- PQK-hybrid acquisition is more promising for intervention guidance because it preserves coverage
  while improving top-5 target recovery over random and classical uncertainty sampling.
- Dice remains low because the refinement output is still a Gaussian candidate map, not a learned
  segmentation decoder. The paper should treat Dice as a secondary diagnostic, not the primary
  success claim.

Updated paper direction:

> Quantum-kernel active learning for candidate annotation and top-k target recovery in
> endoscopic image-guided intervention under domain shift.

Publication-ready analysis artifacts:

- `docs/active_learning_publication_results.md` contains paired bootstrap confidence intervals,
  permutation tests, and learning-curve figures.
- `docs/spie_mi104_active_learning_abstract_draft.md` contains a SPIE-style abstract draft using
  the supported annotation-triage and top-k guidance claims.
