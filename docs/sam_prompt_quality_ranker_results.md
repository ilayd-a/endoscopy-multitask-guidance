# SAM Prompt-Quality Ranker Results

Date: 2026-07-17

## Motivation

The earlier prompt-selection benchmark trained rankers to predict whether a
candidate point landed inside the target mask. That is not the same as the final
objective. A prompt can hit the target but still produce a poor SAM mask, and a
nearby point/box can sometimes produce a better mask.

This experiment labels candidate prompts by their actual SAM Dice, then trains
rankers to choose prompts that maximize downstream segmentation quality.

## Method

For each sampled frame:

1. Generate candidate points from the existing dense candidate table.
2. Pair each point with one or more box radii.
3. Run SAM point+box prompting.
4. Store:
   - candidate features
   - box radius
   - SAM score
   - actual SAM Dice and IoU
5. Train prompt-quality rankers.

Rankers tested:

- Classical regressors: ExtraTrees, RandomForest, HistGradientBoosting, Ridge.
- QML classifier: projected quantum kernel on high-quality prompt labels.
- QML regressors: PCA angle encoding followed by projected quantum features
  and regression.

## Subset-40 Cache

Prompt pool:

- 40 train frames, 40 validation frames, 40 test frames
- top 8 heatmap candidates per frame
- radii: 48, 64

Validation:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.673 |
| Classical HistGBReg | 0.612 |
| Classical RidgeReg | 0.580 |
| QML PQF ExtraTrees 8pc | 0.504 |
| QML PQK quality 16pc | 0.436 |

Test:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.295 |
| QML PQK quality 12pc | 0.211 |
| Classical ExtraTreesReg | 0.199 |
| Classical RandomForestReg | 0.197 |
| Heatmap score | 0.144 |

Interpretation: the subset-40 test slice was very hard, with a low prompt-pool
oracle. PQK was best among learned rankers on that slice, but the absolute Dice
was too low.

## Mixed-80 Cache

Prompt pool:

- 80 train frames, all 68 validation frames, all 66 test frames
- mixed candidate pool per frame:
  - top heatmap-score candidates
  - original NMS candidate order
  - grid/diverse candidates
- radius: 48

Validation:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.718 |
| Classical ExtraTreesReg | 0.531 |
| Classical RidgeReg | 0.505 |
| QML PQF RidgeReg 12pc | 0.497 |
| QML PQF RidgeReg 16pc | 0.486 |
| Heatmap score | 0.363 |

Test:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.421 |
| QML PQF ExtraTreesReg 10pc | 0.236 |
| QML PQF HistGBReg 16pc | 0.236 |
| QML PQF ExtraTreesReg 8pc | 0.231 |
| QML PQK quality 10pc | 0.222 |
| Classical RandomForestReg | 0.198 |
| Classical HistGBReg | 0.193 |
| Classical ExtraTreesReg | 0.186 |
| Heatmap score | 0.089 |

Interpretation: training on actual SAM Dice improves the scientific target.
Within the mixed prompt-quality pool, QML feature regressors outperform the
tested classical regressors on the held-out test subset. However, the mixed
prompt pool still has a much lower oracle ceiling than the previous full
candidate prompt oracle.

## What This Means

Promising:

- The task is now aligned with the metric we care about: SAM Dice.
- QML feature regressors can beat classical regressors on the held-out test
  split inside the same prompt pool.
- The method strongly improves over heatmap ranking.

Still not enough:

- Absolute test Dice is not high enough.
- The mixed prompt pool oracle is 0.421, while the earlier full-candidate oracle
  was 0.619. We are still losing too many good prompts before ranking.

## Next Step

Build a wider prompt-quality cache:

- all candidates per frame rather than mixed/top subsets
- radius 48 first, then add adaptive radii after confirming the wider pool
- train on train+val and evaluate once on test

The expected target is to recover the high oracle ceiling while preserving the
QML feature-regression advantage seen in the mixed-80 test.

## All-Candidate-80 Cache on MPS

After confirming that Apple MPS is available outside the sandbox, the wider
prompt-quality cache was generated on GPU.

Prompt pool:

- 80 train frames, all 68 validation frames, all 66 test frames
- all generated candidates per selected frame
- radius: 48

Validation:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.854 |
| QML PQF RidgeReg 10pc | 0.537 |
| QML PQF RidgeReg 12pc | 0.534 |
| QML PQF RidgeReg 8pc | 0.534 |
| Classical HistGBReg | 0.529 |
| Classical ExtraTreesReg | 0.522 |
| Heatmap score | 0.363 |

Test:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.664 |
| Classical RandomForestReg | 0.263 |
| QML PQF HistGBReg 16pc | 0.252 |
| QML PQK quality 16pc | 0.241 |
| QML PQF RidgeReg 12pc | 0.234 |
| Classical HistGBReg | 0.234 |
| Classical ExtraTreesReg | 0.210 |
| Heatmap score | 0.089 |

Interpretation: MPS makes the wider cache feasible and restores a high oracle
ceiling, but the validation-selected QML advantage did not transfer cleanly to
test. The best test ranker in this run was classical RandomForestReg. QML
rankers still strongly beat heatmap-score ranking but do not yet beat the best
classical prompt-quality regressor on the full held-out test.

## Calibrated Quality-Score Blends

The ranker was extended to test per-sample normalized blends of:

- learned prompt-quality prediction
- SAM's own mask confidence score
- original heatmap score

Validation:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.854 |
| QML PQF RidgeReg 10pc + 0.6/0.2/0.2 blend | 0.550 |
| Classical HistGBReg + 0.7/0.2/0.1 blend | 0.547 |
| QML PQF RidgeReg 10pc | 0.537 |
| Classical ExtraTreesReg | 0.533 |

Held-out test:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.664 |
| QML PQK quality 16pc | 0.315 |
| QML PQK quality 12pc | 0.305 |
| QML PQK quality 16pc + 0.7/0.2/0.1 blend | 0.288 |
| Classical RandomForestReg | 0.263 |
| QML PQF HistGBReg 16pc | 0.252 |
| Heatmap score | 0.089 |

Interpretation: score blending helps on validation but does not improve the
held-out test. The best automatic selector so far is the unblended
`QML_PQK_quality_16pc`, which reaches 0.315 Dice on the held-out test. This is a
small improvement over the previous candidate-ranking PQK prompt pipeline
(`0.303`) and a clearer quantum contribution than the multi-positive prompt
ablation, but it is still far below the oracle prompt-quality ceiling (`0.664`).

The next research bottleneck is prompt localization under domain shift, not SAM
mask generation once a good prompt is available.

## Full-Cache Training and Pairwise/Listwise Ablations

A full radius-48 prompt-quality cache was generated across all available frames:

- train: 478 frames / 40,138 prompt candidates
- validation: 68 frames / 5,710 prompt candidates
- test: 66 frames / 5,543 prompt candidates

Using the full cache substantially improved held-out test performance:

| Strategy | Held-out Dice |
|---|---:|
| Oracle prompt quality | 0.664 |
| QML PQK quality 12pc | 0.397 |
| Classical HistGBReg | 0.378 |
| QML PQK quality 16pc | 0.374 |
| Previous QML PQK quality 16pc, 80-train cache | 0.315 |
| Heatmap score | 0.089 |

Interpretation: the strongest improvement came from scaling prompt-quality
supervision, not from adding more prompt points or score blending. The best
current automatic selector is `QML_PQK_quality_12pc` trained on the full cache.

Two ranking-specific ablations were then added:

1. Pairwise ranking: learn whether prompt A beats prompt B within the same
   frame, then score candidates by pairwise wins.
2. Listwise ranking: label the best prompts within each frame as local winners.

Validation results:

| Ablation | Best QML Dice | Best classical Dice | Oracle Dice |
|---|---:|---:|---:|
| Pairwise, 800 pairs, 45-candidate cap | 0.430 | 0.510 | 0.849 |
| Pairwise, 1600 pairs, 60-candidate cap | 0.211 | 0.510 | 0.853 |
| Listwise, top-2 winners | 0.354 | 0.428 | 0.854 |
| Listwise, top-1 winner | 0.382 | 0.455 | 0.854 |

Interpretation: pairwise/listwise reformulations are useful ablations, but they
do not improve on the full-cache pointwise PQK prompt-quality classifier. The
likely reason is that the present feature representation is still too weak for
fine prompt localization under sequence/domain shift; reframing the objective
alone does not solve that bottleneck.

Next publishable improvement should therefore target the representation:

- extract SAM/MedSAM image-embedding features at each candidate prompt location
- compare vanilla SAM with a medical/domain-adapted SAM or SAM2 variant
- test label efficiency curves for the full-cache PQK quality selector
- add cross-dataset validation after the selector is stable

## SAM-Embedding Prompt Features

The next representation-level experiment appended local frozen-SAM image
embedding descriptors to each candidate prompt feature vector. For each
candidate point, the cached SAM ViT-B image embedding was sampled at the prompt
location using:

- center embedding vector
- local 3x3 mean
- local 3x3 standard deviation

This adds prompt-local semantic features without rerunning SAM masks.

Validation with full-cache radius-48 labels:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.854 |
| Classical RidgeReg + SAM embedding features | 0.708 |
| Classical HistGBReg + SAM embedding features | 0.687 |
| QML PQF HistGBReg 12pc + SAM embedding features | 0.671 |
| QML PQK quality 12pc + SAM embedding features | 0.584 |
| Heatmap score | 0.363 |

Held-out test:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.664 |
| QML PQF HistGBReg 12pc + SAM embedding features | 0.439 |
| Classical HistGBReg + SAM embedding features | 0.438 |
| Classical RidgeReg + SAM embedding features | 0.423 |
| QML PQK quality 16pc + SAM embedding features | 0.405 |
| Previous best QML PQK quality 12pc, no SAM embeddings | 0.397 |
| Heatmap score | 0.089 |

Interpretation: adding SAM image-embedding descriptors is the largest
representation improvement so far. The best held-out test result improved from
0.397 to 0.439 Dice, and the best QML-derived projected-feature ranker is
competitive with the strongest classical prompt-quality regressor. The result
also shows that richer medical/domain features matter more than objective
reformulation alone.

This strengthens the paper direction:

> Quantum-feature prompt-quality ranking for promptable endoscopic
> segmentation, using frozen SAM representations to improve domain-shifted
> prompt localization.

## Candidate-Context Feature Augmentation

A follow-up experiment appended inference-available per-frame candidate context
features to the SAM-embedding prompt descriptors:

- within-frame normalized heatmap score
- within-frame normalized SAM score
- inverse center distance
- normalized prompt location
- heatmap/SAM/center percentile ranks
- combined candidate prior

These features do not use ground-truth masks or SAM Dice labels.

Validation:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.854 |
| Classical HistGBReg + SAM embedding + context | 0.801 |
| Classical RidgeReg + SAM embedding + context | 0.736 |
| QML PQF HistGBReg 12pc + SAM embedding + context | 0.653 |
| QML PQK quality 12pc + SAM embedding + context | 0.606 |
| Heatmap score | 0.363 |

Held-out test:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.664 |
| Classical HistGBReg + SAM embedding + context | 0.595 |
| Classical RidgeReg + SAM embedding + context | 0.526 |
| QML PQF HistGBReg 12pc + SAM embedding only | 0.439 |
| QML PQF HistGBReg 12pc + SAM embedding + context | 0.396 |
| QML PQK quality 12pc + SAM embedding + context | 0.376 |
| Heatmap score | 0.089 |

Interpretation: candidate-context features are the strongest absolute
performance improvement so far, reducing the gap to the oracle prompt-quality
ceiling. However, the improvement is classical rather than quantum: the QML
models degrade when the raw context features are appended before PCA. This
suggests that future quantum experiments should not simply concatenate all
features before projection. A more defensible next QML design is a two-branch
selector: use context features for a classical candidate prior, then apply
quantum/projected quantum features only to SAM-embedding semantics or to the
residual ambiguity among plausible prompts.

## Two-Branch and Confidence-Gated Selection

A two-branch selector was evaluated after the context-feature result:

- branch A: classical context-augmented prompt-quality prior
- branch B: projected-quantum-feature HistGB trained on SAM-embedding semantic
  features only
- final score: per-frame normalized blend of the two branches

Held-out test:

| Strategy | Dice |
|---|---:|
| Oracle prompt quality | 0.664 |
| Two-branch 90% prior / 10% QML semantic | 0.596 |
| Classical context-augmented prior | 0.595 |
| Two-branch 95% prior / 5% QML semantic | 0.595 |
| QML semantic branch alone | 0.439 |

Interpretation: two-branch blending gives only a negligible improvement over the
classical context prior. The QML semantic branch is not harmful at low weight,
but it is not yet a strong standalone improvement.

A more clinically relevant improvement is confidence gating. Using the
context-augmented selector's predicted prompt quality as a confidence score, the
system can auto-accept high-confidence cases and route uncertain cases to review.

Held-out test confidence-gated performance:

| Accepted coverage | Review coverage | Dice | IoU | Dice >= 0.50 | Dice >= 0.70 |
|---:|---:|---:|---:|---:|---:|
| 25% | 75% | 0.732 | 0.619 | 0.875 | 0.562 |
| 50% | 50% | 0.680 | 0.549 | 0.848 | 0.394 |
| 75% | 25% | 0.645 | 0.509 | 0.800 | 0.360 |
| 100% | 0% | 0.595 | 0.462 | 0.682 | 0.318 |

Interpretation: confidence gating turns the method from a forced fully
automatic system into a review-aware clinical workflow. This is likely more
publishable and realistic than claiming full automation on every frame. The
high-confidence subset exceeds the average prompt-pool oracle over all test
cases because the system identifies easier, reliable cases and abstains on
harder frames.

## Validation-Calibrated Confidence Gating

The confidence-gated result above ranks held-out cases by predicted quality
after evaluation. A stricter analysis was added to avoid choosing operating
points from test labels:

1. Train the context-augmented selector on training sequences only.
2. Choose predicted-quality thresholds on validation sequences only.
3. Freeze each threshold.
4. Apply the frozen threshold once to held-out test sequences.

Held-out test with validation-calibrated thresholds:

| Validation target Dice | Test auto coverage | Test Dice | Test IoU | Dice >= 0.50 | Dice >= 0.70 |
|---:|---:|---:|---:|---:|---:|
| all automatic | 100.0% | 0.586 | 0.455 | NA | NA |
| 0.75 | 65.2% | 0.641 | 0.510 | 0.767 | 0.326 |
| 0.82 | 53.0% | 0.655 | 0.525 | 0.829 | 0.314 |
| 0.84 | 42.4% | 0.697 | 0.571 | 0.857 | 0.393 |
| 0.86 | 30.3% | 0.722 | 0.611 | 0.850 | 0.500 |
| 0.88 | 24.2% | 0.764 | 0.652 | 0.938 | 0.562 |

Interpretation: validation-calibrated abstention gives a scientifically cleaner
clinical workflow. At the strictest validated operating point, the selector
automatically accepts roughly one quarter of held-out frames with 0.764 mean
Dice, while routing the rest to human review. This avoids overclaiming full
automation and gives a concrete safety/coverage tradeoff for SPIE MI104.

## Quantum-Agreement Confidence Gating

The next experiment gave the quantum branch a more specific role: trust
estimation rather than direct prompt selection. The classical context model
still selects the candidate prompt, while the projected-quantum semantic branch
estimates whether the selected prompt is semantically consistent with the
SAM-embedding representation. Confidence is then:

> classical predicted prompt quality + quantum semantic agreement

Thresholds are still chosen only on validation and then frozen for held-out test.

Held-out test, validation-calibrated confidence:

| Validation target Dice | Quantum agreement weight | Test coverage | Test Dice | Test IoU | Dice >= 0.50 | Dice >= 0.70 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.82 | 0.0 | 53.0% | 0.655 | 0.525 | 0.829 | 0.314 |
| 0.82 | 0.4 | 39.4% | 0.715 | 0.590 | 0.885 | 0.423 |
| 0.84 | 0.0 | 42.4% | 0.697 | 0.571 | 0.857 | 0.393 |
| 0.84 | 0.2 | 34.8% | 0.737 | 0.618 | 0.913 | 0.478 |
| 0.86 | 0.0 | 30.3% | 0.722 | 0.611 | 0.850 | 0.500 |
| 0.86 | 0.2 | 25.8% | 0.774 | 0.667 | 0.941 | 0.588 |
| 0.88 | 0.0 | 24.2% | 0.764 | 0.652 | 0.938 | 0.562 |
| 0.88 | 0.1 | 22.7% | 0.788 | 0.678 | 1.000 | 0.600 |

Interpretation: this is the clearest quantum contribution so far. QML does not
beat the classical context model as the main prompt selector, but projected
quantum semantic agreement improves validation-calibrated confidence gating at
strict operating points. In a clinical workflow, this means the quantum branch
can act as a trust/triage signal: auto-accept fewer cases, but with higher
expected segmentation quality and fewer low-Dice failures.

Paired held-out statistical analysis was then added using per-sample
accepted/rejected decisions. The strongest statistically supported operating
point was:

| Validation target | Quantum agreement weight | Coverage delta | Dice delta | 95% bootstrap CI | Permutation p |
|---:|---:|---:|---:|---:|---:|
| 0.82 | 0.4 | -13.6 pp | +0.060 | [0.007, 0.122] | 0.037 |

Other strict operating points had positive Dice deltas but wider confidence
intervals:

| Validation target | Quantum agreement weight | Dice delta | 95% bootstrap CI | Permutation p |
|---:|---:|---:|---:|---:|
| 0.84 | 0.2 | +0.040 | [-0.015, 0.102] | 0.201 |
| 0.86 | 0.2 | +0.052 | [0.000, 0.127] | 0.247 |
| 0.88 | 0.1 | +0.024 | [0.000, 0.083] | 1.000 |

Interpretation: the quantum-agreement result is promising but should be framed
as preliminary. It is strongest at moderate review coverage, where the quantum
agreement signal improves accepted-case quality with statistical support. At
very strict coverage levels, the accepted sample count is small, so confidence
intervals widen and the result should be treated as an operating-point
exploration rather than definitive evidence.

This changes the preferred paper framing from "quantum improves segmentation
accuracy" to:

> Quantum-assisted confidence gating for promptable endoscopic segmentation:
> a hybrid classical/quantum selector that uses classical context for prompt
> choice and projected quantum semantic agreement for review-aware trust
> calibration.

## External PolypGen Validation

External validation paths were added using PolypGen and Kvasir-SEG. Kvasir-SEG
is an open-access polyp segmentation dataset with 1,000 images and masks; any
publication using it must cite the Kvasir-SEG paper.

### PolypGen

A locally available PolypGen positive subset was exported:

- Dataset source: `PolypGen2021_MultiCenterData_v3/positive`
- Exported frames: 80 valid positive frames
- Skipped during export: 20 near-empty/problem masks after resizing
- Candidate prompts: 6,720 total, 84 per frame
- Candidate-generation inputs: image-derived saliency/center prior only; no
  ground-truth leakage
- External data usage: evaluation only. Models are trained on CVC train and
  thresholds are calibrated on CVC validation.

External candidate ceiling:

| Metric | Value |
|---|---:|
| Samples | 80 |
| Mean best-candidate Dice | 0.765 |
| Frames with any point-hit candidate | 97.5% |
| Frames with best-candidate Dice >= 0.50 | 83.8% |
| Frames with best-candidate Dice >= 0.70 | 67.5% |

External PolypGen results with CVC-frozen calibration:

| Policy | Quantum agreement weight | Validation target Dice | External coverage | External Dice | External IoU | Dice >= 0.50 | Dice >= 0.70 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All automatic prior | - | - | 100.0% | 0.645 | 0.542 | 0.650 | 0.537 |
| Confidence only | 0.0 | 0.82 | 63.7% | 0.726 | 0.635 | 0.745 | 0.686 |
| Quantum agreement | 0.4 | 0.82 | 45.0% | 0.759 | 0.664 | 0.806 | 0.722 |
| Confidence only | 0.0 | 0.86 | 46.3% | 0.732 | 0.638 | 0.757 | 0.703 |
| Quantum agreement | 0.2 | 0.86 | 38.8% | 0.779 | 0.686 | 0.839 | 0.774 |
| Confidence only | 0.0 | 0.88 | 40.0% | 0.743 | 0.651 | 0.781 | 0.719 |
| Quantum agreement | 0.4 | 0.88 | 32.5% | 0.784 | 0.689 | 0.846 | 0.769 |

Interpretation: the external result supports the same review-aware framing. A
CVC-trained selector transfers to PolypGen at 0.645 Dice when it accepts every
case automatically. Frozen confidence gates improve accepted-case quality, and
adding the projected-quantum agreement signal gives a better high-confidence
external subset at strict operating points. The strongest external setting so
far reaches 0.784 Dice on 32.5% automatic coverage, close to the 0.765
best-candidate ceiling over all external frames because it selectively accepts
easier/high-agreement cases and routes the rest to review.

### Kvasir-SEG

A second external validation subset was exported from the official Kvasir-SEG
archive:

- Dataset source: `Kvasir-SEG`
- Official archive: `https://datasets.simula.no/downloads/kvasir-seg.zip`
- Exported frames: 120 valid positive frames
- Candidate prompts: 10,078 total, 83.98 per frame on average
- Candidate-generation inputs: image-derived saliency/center prior only; no
  ground-truth leakage
- External data usage: evaluation only. Models are trained on CVC train and
  thresholds are calibrated on CVC validation.

External candidate ceiling:

| Metric | Value |
|---|---:|
| Samples | 120 |
| Mean best-candidate Dice | 0.810 |
| Frames with any point-hit candidate | 100.0% |
| Frames with best-candidate Dice >= 0.50 | 93.3% |
| Frames with best-candidate Dice >= 0.70 | 78.3% |

External Kvasir-SEG results with CVC-frozen calibration:

| Policy | Quantum agreement weight | Validation target Dice | External coverage | External Dice | External IoU | Dice >= 0.50 | Dice >= 0.70 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All automatic prior | - | - | 100.0% | 0.737 | 0.621 | 0.867 | 0.683 |
| Best confidence-gated subset | 0.4 | 0.82 | 51.7% | 0.817 | 0.714 | 0.952 | 0.839 |
| Best confidence-gated subset | 0.3 | 0.84 | 43.3% | 0.845 | 0.750 | 0.981 | 0.904 |
| Best confidence-gated subset | 0.3 | 0.86 | 40.0% | 0.842 | 0.747 | 0.979 | 0.896 |
| Best confidence-gated subset | 0.0 | 0.88 | 46.7% | 0.849 | 0.754 | 0.982 | 0.929 |

Interpretation: Kvasir-SEG provides a larger and commonly used external polyp
segmentation validation set. The frozen CVC-trained selector transfers better
to Kvasir than to PolypGen, reaching 0.737 Dice with full automatic coverage.
Validation-calibrated confidence gating raises accepted-case quality to roughly
0.82-0.85 Dice on 40-52% of cases. Unlike PolypGen, the strongest Kvasir
operating point at the strictest target does not require the quantum-agreement
term, so the quantum contribution should be presented as dataset-dependent and
most useful for selective trust calibration rather than as a universal
performance booster.

## Residual QML and Label Efficiency

The two-branch model was extended with a residual QML branch:

1. Train the context-augmented classical prior.
2. Predict training-set prompt quality with that prior.
3. Train the QML semantic branch on the residual error.
4. Add the predicted residual back to the classical prior.

Validation improved slightly:

| Strategy | Validation Dice |
|---|---:|
| Oracle prompt quality | 0.854 |
| Residual QML, weight 1.0 | 0.806 |
| Classical context prior | 0.801 |

Held-out test did not improve:

| Strategy | Held-out Dice |
|---|---:|
| Oracle prompt quality | 0.664 |
| Two-branch 90% prior / 10% semantic | 0.596 |
| Classical context prior | 0.595 |
| Residual QML, weight 0.25 | 0.595 |
| Residual QML, weight 1.0 | 0.589 |

Interpretation: residual QML is a useful ablation but not the current best
held-out selector. The small validation gain does not transfer reliably.

A label-efficiency study then trained the main classical context selector and
the QML semantic selector with fewer prompt-quality labeled training frames.

Held-out test:

| Training frames | Classical context Dice | QML semantic Dice |
|---:|---:|---:|
| 25 | 0.503 | 0.392 |
| 50 | 0.531 | 0.439 |
| 100 | 0.546 | 0.374 |
| 200 | 0.605 | 0.437 |
| 546 | 0.595 | 0.439 |

Interpretation: the current data do not support a label-efficiency quantum
advantage. Classical context features are stronger across all tested training
sizes. This is still useful for a rigorous paper: it shows that the strongest
practical contribution is the prompt-quality framework and confidence-gated
workflow, while the quantum branch remains an exploratory semantic-ranking
component rather than the source of the best absolute performance.
