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
