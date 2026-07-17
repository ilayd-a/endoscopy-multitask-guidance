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

