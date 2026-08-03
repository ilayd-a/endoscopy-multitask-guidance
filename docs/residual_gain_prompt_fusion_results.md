# Residual-Gain Prompt Fusion Results

This experiment changes the prompt-selection target from absolute SAM mask Dice
to residual gain over a fixed strong UNet:

```text
target = SAM_Dice(candidate prompt) - UNet_Dice(frame)
```

The motivation is clinical: the alternate quantum/SAM pathway should not merely
find good-looking SAM masks; it should find masks that are likely to improve the
classical baseline on frames where the baseline is weak.

## Kvasir-120 Oracle Portfolio

On the 120 Kvasir frames with cached prompt-quality candidates:

| Quantity | Value |
| --- | ---: |
| Mean UNet Dice | 0.8916 |
| Mean best-SAM Dice | 0.8102 |
| Oracle best-of-UNet/SAM Dice | 0.9165 |
| Oracle gain vs UNet | +0.0249 |
| Frames where best SAM beats UNet | 24 / 120, 20.0% |

For hard UNet frames, defined as UNet Dice < 0.80:

| Quantity | Value |
| --- | ---: |
| Hard frames | 18 / 120 |
| Mean hard-frame UNet Dice | 0.6135 |
| Mean hard-frame best-SAM Dice | 0.6845 |
| Hard-frame oracle best-of-UNet/SAM Dice | 0.7590 |
| Hard frames where best SAM beats UNet | 12 / 18, 66.7% |

This supports the core project idea: SAM is not a universal replacement for the
classical model, but it can be a useful second-opinion pathway on difficult
frames.

## Multi-Radius Candidate Expansion

The strongest improvement came from expanding the prompt family instead of
changing only the downstream switch. The original external cache used one box
radius, 48 pixels. The expanded cache evaluates four radii, 32, 48, 64, and 96
pixels, for the same candidate points.

| Quantity | Single radius | Multi-radius |
| --- | ---: | ---: |
| Candidate rows | 10,078 | 40,312 |
| Mean UNet Dice | 0.8916 | 0.8916 |
| Mean best-SAM Dice | 0.8102 | 0.9145 |
| Oracle best-of-UNet/SAM Dice | 0.9165 | 0.9331 |
| Oracle gain vs UNet | +0.0249 | +0.0414 |
| Frames where best SAM beats UNet | 24 / 120, 20.0% | 43 / 120, 35.8% |

For hard UNet frames:

| Quantity | Single radius | Multi-radius |
| --- | ---: | ---: |
| Hard frames | 18 / 120 | 18 / 120 |
| Mean hard-frame UNet Dice | 0.6135 | 0.6135 |
| Mean hard-frame best-SAM Dice | 0.6845 | 0.8368 |
| Hard-frame oracle best-of-UNet/SAM Dice | 0.7590 | 0.8410 |
| Hard-frame oracle gain vs UNet | +0.1454 | +0.2275 |
| Hard frames where best SAM beats UNet | 12 / 18, 66.7% | 17 / 18, 94.4% |

This is the clearest publication signal so far. Prompt diversity turns the
alternate quantum/SAM branch from a niche rescue option into a broad hard-frame
second opinion. It also gives a concrete research hypothesis: quantum-guided
candidate generation may be valuable when it increases the diversity of
high-quality prompts available to a frozen foundation segmenter.

The oracle portfolio is saved locally by:

```bash
python endoscopy_guidance/export_oracle_portfolio.py \
  --prompt_quality_csv endoscopy_guidance/results/kvasir_external_120_prompt_quality.csv \
  --unet_metrics_csv endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test/baseline_metrics.csv \
  --output_csv endoscopy_guidance/results/kvasir120_unet_sam_oracle_portfolio.csv \
  --summary_csv endoscopy_guidance/results/kvasir120_unet_sam_oracle_summary.csv
```

## Ten-Split Residual-Fusion Benchmark

The residual-gain model was evaluated with the same 10 random source-level
splits used in the TTA switch-detector benchmark. Each split used 40% train,
40% validation/calibration, and 20% held-out test. Features included the cached
SAM/context prompt features plus UNet confidence and TTA uncertainty features.

| Policy | Mean selected Dice | Mean delta vs UNet | Mean SAM rate | Hard-frame selected Dice | Hard-frame delta vs UNet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical absolute-SAM oracle best-of-two | 0.9093 | +0.0186 | 0.0875 | 0.7156 | +0.0957 |
| Classical residual-gain oracle best-of-two | 0.9051 | +0.0145 | 0.0917 | 0.6885 | +0.0687 |
| Quantum absolute-SAM oracle best-of-two | 0.8990 | +0.0084 | 0.0500 | 0.6586 | +0.0388 |
| Quantum residual-gain oracle best-of-two | 0.8988 | +0.0081 | 0.0458 | 0.6577 | +0.0378 |
| Always UNet | 0.8907 | +0.0000 | 0.0000 | 0.6198 | +0.0000 |
| Classical residual-gain validation switch | 0.8910 | +0.0004 | 0.0375 | 0.6317 | +0.0119 |
| Classical absolute-SAM validation switch | 0.8897 | -0.0009 | 0.0333 | 0.6198 | +0.0000 |
| Quantum residual-gain validation switch | 0.8856 | -0.0051 | 0.0208 | 0.6198 | +0.0000 |

## Two-Stage Switch Attempt

A stronger two-stage version was also evaluated. It first ranks SAM candidates,
then trains a separate frame-level gain regressor using predicted candidate
gain, candidate-score distribution features, SAM confidence, UNet confidence,
and TTA uncertainty features. This did not improve the mean result on the
current 120-frame cache:

| Policy | Mean selected Dice | Mean delta vs UNet | Mean SAM rate | Hard-frame selected Dice | Hard-frame delta vs UNet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always UNet | 0.8907 | +0.0000 | 0.0000 | 0.6198 | +0.0000 |
| Classical residual-gain validation switch | 0.8910 | +0.0004 | 0.0375 | 0.6317 | +0.0119 |
| Classical residual-gain meta-RF switch | 0.8876 | -0.0030 | 0.0375 | 0.6255 | +0.0057 |
| Classical absolute-SAM meta-RF switch | 0.8892 | -0.0014 | 0.0417 | 0.6124 | -0.0075 |

The meta-switch occasionally captures high-value hard-frame switches, but it
also over-calls SAM on some splits. The simpler residual-gain validation switch
remains the best deployable policy in this small-cache benchmark.

## Multi-Radius Learned Switches

The multi-radius cache was then evaluated with the same 10 source-level splits.
Two feature settings were tested: the compact 49-dimensional prompt features and
an expanded 829-dimensional feature matrix that appends local frozen-SAM image
embedding descriptors plus within-frame candidate context/rank features.

| Setting / Policy | Mean selected Dice | Mean delta vs UNet | Mean SAM rate | Hard-frame selected Dice | Hard-frame delta vs UNet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multi-radius selected-candidate oracle | 0.9025 | +0.0119 | 0.0958 | 0.6786 | +0.0588 |
| Multi-radius residual-gain meta-RF switch | 0.8955 | +0.0048 | 0.0292 | 0.6437 | +0.0239 |
| Multi-radius residual-gain validation switch | 0.8919 | +0.0012 | 0.0125 | 0.6263 | +0.0065 |
| Multi-radius + SAM-embedding selected-candidate oracle | 0.9116 | +0.0210 | 0.1458 | 0.7366 | +0.1167 |
| Multi-radius + SAM-embedding residual-gain meta-RF switch | 0.8943 | +0.0037 | 0.0292 | 0.6431 | +0.0233 |
| Multi-radius + SAM-embedding residual-gain validation switch | 0.8929 | +0.0022 | 0.0250 | 0.6351 | +0.0153 |
| Always UNet | 0.8907 | +0.0000 | 0.0000 | 0.6198 | +0.0000 |

The richer features improved the learned selected-candidate oracle, meaning the
prompt ranker has more recoverable signal, but the deployable switch still
captures only a small part of that headroom. This is consistent with a
small-calibration-data problem: positive-gain SAM frames are uncommon overall
even though they are concentrated among hard frames.

## Hard-Weighted Calibration

A hard-frame-weighted calibration option was added:

```bash
--hard_weight 1.0
```

This tunes the validation threshold using mean Dice plus hard-frame Dice. It is
implemented for follow-up experiments, but it is not the recommended headline
setting on the current 120-frame cache. Across 10 splits with the SAM-embedding
features, the best hard-weighted deployable policy had mean Dice 0.8916
(+0.0009 vs UNet) and hard-frame Dice 0.6351 (+0.0153 vs UNet). It switched more
often, but also over-switched on some splits.

## Interpretation

Residual-gain targeting produced the first validation-calibrated non-oracle
switch with a positive mean gain over UNet. The gain is still small, but it is
directionally important because it improves hard-frame Dice without broadly
switching away from the strong UNet. A more complex two-stage switch did not
improve the aggregate result, suggesting that the current limitation is not a
missing switch model alone.

The remaining bottleneck is calibration, not the existence of a better alternate
mask. The full multi-radius oracle shows large recoverable headroom, especially
on hard frames, but only a small number of held-out frames per split have
positive-gain SAM candidates. With only 48 calibration frames per split,
threshold selection is unstable and usually abstains.

## Next Scientific Step

The most publication-relevant next step is to increase the candidate/evaluation
coverage, not to claim a large result from the 120-frame cache. A stronger study
should:

1. Expand cached SAM candidate quality to the full Kvasir validation/test pool.
2. Keep multiple prompt radii and add richer prompt families, because prompt
   diversity is now the strongest observed source of improvement.
3. Evaluate residual-gain switching with confidence intervals and paired
   source-level tests.
4. Keep UNet as the default output and frame the quantum/SAM branch as
   hard-case second-opinion routing.
