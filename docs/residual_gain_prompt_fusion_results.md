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

## Interpretation

Residual-gain targeting produced the first validation-calibrated non-oracle
switch with a positive mean gain over UNet. The gain is still small, but it is
directionally important because it improves hard-frame Dice without broadly
switching away from the strong UNet.

The remaining bottleneck is calibration and alternate-expert strength. The
oracle result shows recoverable headroom, especially on hard frames, but only a
small number of held-out frames per split have positive-gain SAM candidates.
With only 48 calibration frames per split, threshold selection is unstable and
usually abstains.

## Next Scientific Step

The most publication-relevant next step is to increase the candidate/evaluation
coverage, not to claim a large result from the 120-frame cache. A stronger study
should:

1. Expand cached SAM candidate quality to the full Kvasir validation/test pool.
2. Add multiple prompt radii or richer prompt families, because the current
   Kvasir cache only uses radius 48.
3. Evaluate residual-gain switching with confidence intervals and paired
   source-level tests.
4. Keep UNet as the default output and frame the quantum/SAM branch as
   hard-case second-opinion routing.
