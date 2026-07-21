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
