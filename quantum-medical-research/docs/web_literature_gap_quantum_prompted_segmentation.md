# Web Literature Gap: Quantum-Prompted Endoscopic Segmentation

Date: 2026-07-17

## What Changed After Searching

The better publication direction is not another small QSVM classification
benchmark. The stronger opening is:

> Quantum-guided prompt selection for foundation-model or SAM-style endoscopic
> segmentation.

This connects the useful result we already have, PQK candidate ranking, to a
modern segmentation mechanism that can actually produce masks. The current
Gaussian candidate refinement loses Dice; a promptable segmentation model is a
more realistic way to convert high-quality candidate points or boxes into a
clinical segmentation output.

## Why This Is a Real Gap

Recent work has moved the bar upward:

1. QPolypNet reports a hybrid quantum-classical polyp segmentation architecture
   with Dice around 0.942 on CVC-ClinicDB and 0.892 on Kvasir-SEG. That means a
   direct quantum segmentation claim needs a strong segmentation backbone.

2. PSF-SAM and related SAM adaptation papers report CVC/Kvasir Dice near the
   0.94 range with efficient fine-tuning, LoRA, adapters, or multi-scale prompt
   strategies.

3. Weakly supervised and point-supervised polyp segmentation papers show that
   point/box prompts are clinically relevant because pixel-wise medical
   annotation is expensive.

4. Active-learning medical segmentation papers show that annotation selection is
   still an important unsolved problem, especially when uncertainty estimates
   are unreliable.

Together, these suggest a gap:

> Can quantum kernels improve which points/boxes are selected as prompts for a
> promptable medical segmentation model, reducing annotation burden or improving
> failure recovery under limited labels?

## Proposed Paper Direction

Title-style framing:

> Quantum-Kernel Prompt Selection for Label-Efficient Endoscopic Segmentation

Core pipeline:

1. A classical segmentation backbone produces an initial heatmap.
2. Dense candidate points or boxes are generated from heatmap peaks, grid
   coverage, uncertainty regions, and image features.
3. A projected quantum kernel ranks candidate prompts using limited labeled
   candidates.
4. The top-ranked prompt is passed to a SAM-style or lightweight prompt decoder.
5. Evaluation measures:
   - prompt hit rate
   - Dice / IoU after prompt refinement
   - annotation budget needed to reach a target Dice
   - robustness under sequence/domain shift

## Why This Is Better Than the Current Gaussian Refinement

Current local result:

- PQK dense RGB candidate ranking reaches about 0.75 top-5 target recovery.
- But Gaussian candidate maps reduce Dice because they are not object-aware.

The web literature suggests that promptable segmentation is exactly the missing
bridge: use PQK to choose prompts, then use SAM-like object-aware segmentation to
turn prompts into masks.

## Concrete Next Experiment

Minimum viable run:

1. Install or add a promptable model:
   - SAM / MedSAM / SAM2 if weights can be used locally, or
   - a small learned prompt decoder trained on CVC if SAM weights are too heavy.

2. Generate PQK-ranked candidate prompts from:
   - stride-32 dense CVC candidates
   - RGB patch features
   - heatmap statistics

3. Compare prompt sources:
   - heatmap peak
   - random candidate
   - classical uncertainty/diversity
   - PQK top-1 / top-3 / top-5
   - oracle prompt upper bound

4. Validation rule:
   - tune prompt/refinement parameters on validation sequences 24-26
   - report once on test sequences 27-29

5. Success threshold for a publishable result:
   - PQK prompt selection improves prompt-refined Dice over classical prompt
     selection under the same annotation budget, or
   - PQK reaches the same Dice with substantially fewer labeled candidates.

## Current Blocker

The local environment does not currently have `segment_anything`, `sam2`,
`transformers`, or SAM/MedSAM weights cached. To run this direction properly, we
need to install/download a promptable segmentation model or implement a small
local prompt decoder.

