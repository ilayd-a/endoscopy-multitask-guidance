# Kvasir TTA Switch-Detector Results

This experiment adds inference-time UNet test-time augmentation (TTA)
uncertainty features to the side-by-side UNet plus quantum/SAM fusion benchmark.
The goal is to detect frames where the strong UNet is unreliable and route only
those frames to an alternate quantum-assisted SAM prompt-selection path.

## Setup

- Dataset slice: 120 Kvasir frames with cached SAM prompt-quality candidates.
- Baseline: fixed pretrained ResNet34-UNet predictions from
  `strong_unet_pretrained_kvasir_train_val_test`.
- Alternate expert: SAM candidate selected by either a classical HistGB prompt
  ranker or the quantum feature ridge ranker.
- Split protocol: 10 random source-level splits, each using 40% train, 40%
  calibration/validation, and 20% held-out test.
- TTA features: horizontal, vertical, and horizontal+vertical flip consistency
  statistics, including probability standard deviation, vote entropy, area
  instability, and mean-probability shift.

## Aggregate Result

| Policy | Mean selected Dice | Mean delta vs UNet | Mean expert rate | Hard-frame selected Dice | Hard-frame delta vs UNet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical SAM oracle best-of-two | 0.9075 | +0.0168 | 0.0708 | 0.7084 | +0.0886 |
| Quantum-ridge SAM oracle best-of-two | 0.8986 | +0.0079 | 0.0500 | 0.6562 | +0.0364 |
| Always UNet | 0.8907 | +0.0000 | 0.0000 | 0.6198 | +0.0000 |
| Quantum expected-gain HistGB switch | 0.8904 | -0.0003 | 0.0083 | 0.6198 | +0.0000 |
| Classical expected-gain RF switch | 0.8901 | -0.0006 | 0.0250 | 0.6198 | +0.0000 |
| Classical+quantum expected-gain RF switch | 0.8896 | -0.0011 | 0.0250 | 0.6198 | +0.0000 |
| Classical confidence switch | 0.8881 | -0.0025 | 0.0292 | 0.6024 | -0.0174 |
| Classical+quantum confidence switch | 0.8850 | -0.0056 | 0.0500 | 0.5961 | -0.0238 |

## Interpretation

The TTA features are scientifically useful as an uncertainty signal: higher
TTA instability is negatively correlated with UNet Dice on the cached Kvasir
subset. The strongest correlations were:

| TTA feature | Correlation with UNet Dice |
| --- | ---: |
| `tta_prob_std_p95` | -0.3949 |
| `tta_mean_prob_shift` | -0.3587 |
| `tta_area_std` | -0.3396 |
| `tta_area_range` | -0.3391 |
| `tta_prob_std_mean` | -0.3373 |
| `tta_vote_entropy` | -0.3242 |

However, this did not produce a reliable Dice improvement in the current
switching setup. The oracle best-of-two result shows there is some recoverable
headroom, especially on hard frames, but the learned switch usually abstains
because the SAM alternate expert is often weaker than the strong UNet. In other
words, the current bottleneck is not detecting hard UNet cases; it is producing
an alternate mask that is consistently better on those cases.

## Publication Implication

This result supports a more rigorous direction for the paper: quantum-assisted
uncertainty triage or selective second-opinion routing, rather than claiming
that the present quantum/SAM path already improves a strong segmentation model.
To make the system publishable as a performance-improving intervention, the next
step should strengthen the alternate expert, for example with prompt expansion,
SAM2/MedSAM-style prompting, or a learned correction/refinement head trained
only on hard frames.
