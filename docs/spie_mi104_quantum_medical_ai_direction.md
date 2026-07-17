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

## Go/No-Go Criteria

This becomes abstract-worthy if at least one of these holds:

- PQK improves top-1 or top-k localization over classical baselines in the lowest-label setting.
- PQK gives better calibrated uncertainty or candidate ranking even when accuracy is similar.
- Kernel diagnostics identify when QML fails, giving a useful negative benchmark for medical QML.

If none of these hold, the publication should be reframed as a rigorous negative benchmark rather
than an intervention method paper.
