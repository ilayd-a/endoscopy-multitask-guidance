# Quantum Residual Hard-Case Segmentation Plan

The SAM prompt-selection experiments are useful for selective confidence, but
they are not the main clinical target. A real-time surgical guidance tool must
produce guidance for every frame, including hard cases. The publishable target
should therefore use a strong dense classical segmentation model as the baseline
and test whether a compact quantum module improves difficult-frame performance.

## Proposed Claim

Hybrid quantum-classical residual refinement improves difficult-frame
endoscopic segmentation over a strong real-time classical baseline.

## Baseline

Use the UNet + ResNet34 encoder from `endoscopy-multitask-guidance` as the
classical baseline. The quantum repo should consume exported baseline
predictions, probability maps, and per-frame/per-patch error labels without
modifying the source segmentation repo.

The weak local CVC-only checkpoint at:

`/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance/models/unet_cvc.pth`

does not reproduce a strong held-out CVC baseline in the export diagnostic:

| Split | Dice at threshold 0.5 | Best swept threshold Dice |
|---|---:|---:|
| CVC val, seq 24-26 | 0.377 | 0.403 at threshold 0.05 |
| CVC test, seq 27-29 | 0.086 | 0.092 at threshold 0.05 |

This suggests the local checkpoint is either not the intended strong baseline,
was trained under a different preprocessing/data pairing, or is not suitable for
the sequence-held-out CVC evaluation. Do not build the final quantum claim on
this checkpoint.

The intended stronger checkpoint has been recovered and pushed to the
endoscopy repo branch:

`/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance/models/unet_pretrained.pth`

Older local evaluation artifacts for this checkpoint report Kvasir validation
Dice around 0.917, Kvasir test Dice around 0.867, and all-CVC external Dice
around 0.789. The quantum residual work should use this checkpoint as the
classical baseline.

## Quantum Roles To Test

1. Quantum residual patch classifier
   - Input: local RGB/probability/boundary features around predicted-mask
     boundary and uncertain pixels.
   - Target: false positive, false negative, or correct classical prediction.
   - Output: residual correction map applied to the classical probability map.

2. Quantum hard-case error predictor
   - Input: frame-level features from probability maps, uncertainty, morphology,
     and encoder/SAM embeddings.
   - Target: whether the baseline Dice falls below a clinically relevant
     threshold.
   - Use: not abstention as the final product, but trigger a stronger refinement
     pass for hard frames.

3. Quantum mask-hypothesis selector
   - Generate multiple classical masks using thresholds, test-time augmentation,
     CRF/morphology, or lightweight decoder variants.
   - Use a projected quantum kernel to select or blend the best hypothesis.
   - This keeps full-frame output while giving quantum a direct segmentation
     improvement role.

## Required Validation

- Full-frame Dice/IoU on all test frames.
- Hard-case Dice/IoU on bottom-quartile baseline frames.
- Boundary F1 or boundary Dice.
- External validation on CVC, Kvasir-SEG, and PolypGen.
- Paired bootstrap/permutation tests against the classical baseline.
- Latency/FPS measurement with and without the quantum module.

## Immediate Next Steps

1. Export Kvasir validation/test probability maps from `unet_pretrained.pth`.
2. Add paired bootstrap/permutation tests for full-mask residual refinement.
3. Build a patch-level residual dataset from baseline false positives and false
   negatives.
4. Compare classical residual models against projected quantum-kernel residual
   models under the same train/val/test split.
5. Improve the quantum residual module beyond parity by adding richer
   texture/shape features, sequence-aware validation, and quantum-kernel
   hyperparameter search.

## Implemented Scaffold

- `export_classical_cvc_baseline.py`
  - Loads the sibling repo's UNet/ResNet34 checkpoint.
  - Exports images, masks, probability maps, predicted masks, and per-frame
    Dice/IoU/error features into the quantum repo.
  - Supports threshold sweeps for calibration diagnostics.

- `export_unet_baseline_dataset.py`
  - Exports the same checkpoint on split-folder or split-CSV datasets.
  - This is the preferred exporter for the Kvasir train/val/test headline
    experiment.

- `build_residual_patch_dataset.py`
  - Samples false-positive, false-negative, boundary, uncertain, and correct
    patches from baseline predictions.
  - Produces a compact patch-feature matrix for residual correction.

- `residual_patch_quantum_benchmark.py`
  - Compares classical residual classifiers with a projected-quantum feature
    residual classifier.
  - Supports explicit `train_split` and `test_split` settings.

- `apply_residual_mask_refinement.py`
  - Trains a residual classifier on a designated training split.
  - Selects add/remove thresholds on validation frames only.
  - Applies residual corrections back to every held-out test mask and reports
    full-frame Dice/IoU changes.

Smoke-test residual classifier results on the weak local CVC checkpoint:

| Model | Test patch accuracy | Balanced accuracy | Macro F1 | Error F1 |
|---|---:|---:|---:|---:|
| Classical logistic | 0.818 | 0.696 | 0.683 | 0.876 |
| Classical HistGB | 0.806 | 0.744 | 0.699 | 0.874 |
| Classical random forest | 0.798 | 0.734 | 0.691 | 0.866 |
| Projected quantum HistGB | 0.769 | 0.758 | 0.675 | 0.859 |

Interpretation: the residual benchmark is operational, but the final study
requires the stronger classical checkpoint. With the current weak checkpoint,
the benchmark mostly proves that the pipeline can identify segmentation errors;
it should not be used for the headline result.

## Strong-Checkpoint CVC Residual Results

Using `unet_pretrained.pth`, CVC frames were exported into
`strong_unet_pretrained_cvc_all`. The sequence split is intentionally harsh:
546 earlier frames are used for residual training/validation and 66 late frames
are held out for final testing.

Patch-level residual benchmark on held-out CVC test patches:

| Model | Accuracy | Balanced accuracy | Macro F1 | Error F1 | Error AUC |
|---|---:|---:|---:|---:|---:|
| Projected quantum HistGB | 0.670 | 0.685 | 0.643 | 0.865 | 0.943 |
| Classical random forest | 0.671 | 0.688 | 0.646 | 0.863 | 0.940 |
| Classical HistGB | 0.670 | 0.703 | 0.650 | 0.862 | 0.946 |
| Classical logistic | 0.666 | 0.697 | 0.647 | 0.861 | 0.942 |

Full-mask residual refinement after validation-only threshold tuning:

| Model | Split | Baseline Dice | Refined Dice | Delta Dice | Hard baseline Dice | Hard refined Dice | Hard delta Dice |
|---|---|---:|---:|---:|---:|---:|---:|
| Projected quantum HistGB | val | 0.8316 | 0.8394 | +0.0078 | 0.5031 | 0.5437 | +0.0406 |
| Projected quantum HistGB | test | 0.6356 | 0.6496 | +0.0140 | 0.5285 | 0.5596 | +0.0311 |
| Classical HistGB | val | 0.8316 | 0.8447 | +0.0132 | 0.5031 | 0.5524 | +0.0493 |
| Classical HistGB | test | 0.6356 | 0.6489 | +0.0133 | 0.5285 | 0.5603 | +0.0318 |

Interpretation: the residual-refinement direction is real and improves
full-frame segmentation, especially hard frames. The quantum feature map is
currently competitive with the matched classical residual model, but not yet
clearly superior. The next publishability step is to strengthen the
quantum-specific module and test it across Kvasir/CVC/PolypGen with paired
statistics.

## Corrected Kvasir Train/Val/Test Results

The proper headline protocol should use all Kvasir splits:

- Train residual patch models on Kvasir train.
- Select mask add/remove thresholds on Kvasir validation.
- Report full-frame Dice once on Kvasir test.

Regenerated strong UNet baseline at threshold 0.5:

| Split | Frames | Baseline Dice | Baseline IoU |
|---|---:|---:|---:|
| Kvasir train | 800 | 0.8971 | 0.8351 |
| Kvasir val | 100 | 0.8917 | 0.8301 |
| Kvasir test | 100 | 0.8576 | 0.7852 |

Patch-level residual benchmark, train on Kvasir train and test on Kvasir test:

| Model | Accuracy | Balanced accuracy | Macro F1 | Error F1 | Error AUC |
|---|---:|---:|---:|---:|---:|
| Classical HistGB | 0.724 | 0.722 | 0.714 | 0.878 | 0.939 |
| Classical random forest | 0.724 | 0.723 | 0.716 | 0.875 | 0.937 |
| Projected quantum HistGB | 0.715 | 0.710 | 0.705 | 0.871 | 0.932 |
| Classical logistic | 0.717 | 0.712 | 0.707 | 0.869 | 0.932 |

Full-mask residual refinement, train on Kvasir train, tune on Kvasir val, test
on Kvasir test:

| Model | Split | Baseline Dice | Refined Dice | Delta Dice | Hard baseline Dice | Hard refined Dice | Hard delta Dice |
|---|---|---:|---:|---:|---:|---:|---:|
| Projected quantum HistGB | val | 0.8917 | 0.8920 | +0.0003 | 0.5694 | 0.5715 | +0.0021 |
| Projected quantum HistGB | test | 0.8576 | 0.8582 | +0.0006 | 0.5781 | 0.5828 | +0.0047 |
| Classical HistGB | val | 0.8917 | 0.8919 | +0.0002 | 0.5694 | 0.5699 | +0.0005 |
| Classical HistGB | test | 0.8576 | 0.8581 | +0.0005 | 0.5781 | 0.5826 | +0.0045 |

Interpretation: with the scientifically correct split, residual refinement is
safe but conservative on a strong Kvasir baseline. The projected quantum model
slightly exceeds the matched classical full-mask Dice improvement, but the
margin is tiny. This should be treated as a baseline result to improve, not as
the final publication claim.

## Current Best Direction: Triggered Quantum Residual Refinement

The stronger clinical framing is not to modify every frame. Instead:

1. Train an inference-safe hard-frame trigger from probability-map and
   predicted-mask morphology features.
2. Train the residual correction model on Kvasir train patches.
3. Tune trigger/action thresholds on Kvasir validation hard-frame Dice.
4. Apply residual refinement only to triggered Kvasir test frames.

This preserves easy frames while targeting the failure modes that matter for a
real-time surgical guidance assistant.

Kvasir train/val/test results with the hard-frame trigger:

| Model | Split | Triggered frames | Baseline Dice | Refined Dice | Delta Dice | Hard baseline Dice | Hard refined Dice | Hard delta Dice |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Projected quantum HistGB | val | 7/100 | 0.8917 | 0.8921 | +0.0004 | 0.5694 | 0.5784 | +0.0090 |
| Projected quantum HistGB | test | 4/100 | 0.8576 | 0.8594 | +0.0018 | 0.5781 | 0.5864 | +0.0083 |
| Classical HistGB | val | 7/100 | 0.8917 | 0.8918 | +0.0001 | 0.5694 | 0.5734 | +0.0040 |
| Classical HistGB | test | 4/100 | 0.8576 | 0.8584 | +0.0008 | 0.5781 | 0.5817 | +0.0036 |
| Hybrid quantum HistGB | val | 7/100 | 0.8917 | 0.8923 | +0.0006 | 0.5694 | 0.5748 | +0.0055 |
| Hybrid quantum HistGB | test | 4/100 | 0.8576 | 0.8592 | +0.0016 | 0.5781 | 0.5855 | +0.0075 |

Interpretation: the hard-triggered projected quantum residual is currently the
best variant. It improves Kvasir test Dice more than the matched classical
triggered residual while touching only four test frames. The hard-frame gain is
also larger (+0.0083 vs +0.0036). This is a much more plausible publication
direction than full-frame residual correction applied indiscriminately, but it
still needs repeated-seed validation, external CVC/PolypGen confirmation, and
paired significance testing.

## Five-Seed Triggered Refinement Stability

Repeated runs with different training subsamples/seeds preserve the same
direction. All runs use Kvasir train for residual/trigger training, Kvasir val
for threshold tuning, and Kvasir test for final reporting.

| Model | Seeds | Mean test Dice delta | Mean hard Dice delta | Mean triggered frames |
|---|---:|---:|---:|---:|
| Projected quantum triggered | 5 | +0.00197 | +0.00875 | 4.4/100 |
| Classical triggered | 5 | +0.00121 | +0.00538 | 4.2/100 |

Paired quantum-minus-classical seed-level differences:

| Endpoint | Mean advantage | Paired t-test p | Wilcoxon p | One-sided sign-test p |
|---|---:|---:|---:|---:|
| All-frame Dice delta | +0.00076 | 0.0023 | 0.0625 | 0.0313 |
| Hard-frame Dice delta | +0.00337 | 0.0018 | 0.0625 | 0.0313 |

Interpretation: the effect is consistent across seeds and reaches a one-sided
sign-test threshold because all five paired seeds favor the quantum residual.
However, the nonparametric two-sided Wilcoxon test remains just above 0.05 with
only five paired seeds. The result is promising, but the paper should still add
more external frames and paired bootstrap/permutation tests before claiming a
definitive improvement.

## Larger-Gain Route: Mask-Hypothesis Selection

The residual gains are consistent but numerically small because the baseline
UNet is already strong and the trigger intentionally edits only a few frames.
A higher-leverage route is to choose among multiple mask hypotheses per frame
before residual cleanup.

Initial Kvasir test threshold-selection results:

| Method | Test Dice | Delta Dice | Hard Dice | Hard delta Dice |
|---|---:|---:|---:|---:|
| Fixed threshold 0.50 | 0.8576 | +0.0000 | 0.5781 | +0.0000 |
| Classical logistic selector | 0.8601 | +0.0025 | 0.6010 | +0.0229 |
| Hybrid quantum logistic selector | 0.8563 | -0.0013 | 0.5880 | +0.0100 |
| Oracle threshold selector | 0.8778 | +0.0202 | 0.6343 | +0.0562 |

Interpretation: threshold/hypothesis selection has much more headroom than
pixel residual cleanup. The first quantum selectors underperform the classical
logistic selector, so the next scientific target should be a stacked system:
adaptive mask-hypothesis selection to get the larger hard-case gain, followed
by triggered quantum residual refinement on the remaining high-risk frames.
This keeps the quantum contribution clinically meaningful without pretending
the current quantum threshold selector is already best.

Follow-up stack test: materializing the classical-logistic selected thresholds
as a new baseline improved Kvasir test Dice to 0.8601, but residual refinement
on top did not compound the gain.

| Stacked method | Test Dice | Delta vs selected baseline | Hard Dice | Hard delta vs selected baseline |
|---|---:|---:|---:|---:|
| Selected-threshold baseline | 0.8601 | +0.0000 | 0.5693 | +0.0000 |
| Selected + triggered quantum residual | 0.8593 | -0.0008 | 0.5693 | -0.0000 |
| Selected + triggered classical residual | 0.8595 | -0.0006 | 0.5703 | +0.0009 |

Interpretation: naive residual cleanup after threshold selection is unstable.
The larger-gain path should focus on better mask-hypothesis selection and a
better inference-safe gate, likely using richer encoder/SAM embeddings,
augmentation-consistency features, or temporal/video consistency features. The
current stack should not be used as the headline.

## Quantum-Kernel Selector Status

A more explicitly quantum selector was added using a pairwise projected quantum
kernel SVC: each training example asks whether a candidate threshold improves
the frame Dice over the fixed 0.50 mask, and the model ranks candidate
thresholds at inference.

Kvasir test results:

| Selector | Test Dice | Delta Dice | Hard Dice | Hard delta Dice |
|---|---:|---:|---:|---:|
| Fixed threshold 0.50 | 0.8576 | +0.0000 | 0.5781 | +0.0000 |
| Best projected quantum kernel sweep | 0.8588 | +0.0012 | 0.5808 | +0.0027 |
| Best hard-frame quantum kernel setting | 0.8578 | +0.0002 | 0.5847 | +0.0066 |
| Classical pairwise HistGB | 0.8624 | +0.0048 | 0.5967 | +0.0187 |
| Oracle threshold selector | 0.8778 | +0.0202 | 0.6343 | +0.0562 |

Interpretation: this makes the threshold-selector route genuinely quantum, but
the current quantum kernel still underperforms the best classical selector.
The next quantum-specific improvement should therefore add richer features
before the quantum map: encoder embeddings, SAM embeddings, test-time
augmentation consistency, or temporal consistency. Simple probability-map
morphology alone does not give the quantum kernel enough signal.

## Updated Quantum Selector With Candidate-Specific Morphology

The pairwise quantum-kernel selector improves substantially when each threshold
candidate is represented by its own inference-safe mask morphology features
(candidate area, boundary fraction, connected components, uncertainty summaries)
plus the difference from the fixed 0.50 mask. This gives the quantum kernel a
direct representation of the mask hypothesis it is ranking.

Kvasir test, train on Kvasir train and tune score threshold on Kvasir val:

| Selector | Test Dice | Delta Dice | Hard Dice | Hard delta Dice | Changed frames |
|---|---:|---:|---:|---:|---:|
| Fixed threshold 0.50 | 0.8576 | +0.0000 | 0.5781 | +0.0000 | 0 |
| Earlier classical logistic selector | 0.8601 | +0.0025 | 0.6010 | +0.0229 | 96 |
| Projected quantum kernel selector | 0.8622 | +0.0046 | 0.5949 | +0.0169 | 73 |
| Matched classical HistGB pairwise selector | 0.8599 | +0.0023 | 0.5839 | +0.0059 | 19 |
| Oracle threshold selector | 0.8778 | +0.0202 | 0.6343 | +0.0562 | 97 |

Interpretation: this is the strongest quantum-specific result so far. The
projected quantum kernel selector now beats the matched classical pairwise
controls and improves overall Dice more than the earlier classical logistic
threshold selector. It does not yet beat the earlier classical logistic selector
on hard-frame Dice, so the next step is repeated-seed validation and external
CVC/PolypGen testing for the candidate-specific quantum selector.

UNet encoder embeddings were also exported and tested, but naive concatenation
of high-dimensional encoder features made the quantum selector too
conservative. Learned embedding features may still help, but they likely need a
separate low-dimensional reduction or consistency-derived summaries before
entering the quantum kernel.

## Selector Improvement and Statistical Check

Additional selector variants were tested after the candidate-specific quantum
result:

| Variant | Test Dice | Delta Dice | Hard Dice | Hard delta Dice | Interpretation |
|---|---:|---:|---:|---:|---|
| Projected quantum kernel, legacy candidate features, c3/r3 | 0.8622 | +0.0046 | 0.5949 | +0.0169 | Best current quantum-specific result |
| Projected quantum kernel, threshold-curve features | 0.8606 | +0.0030 | 0.5899 | +0.0118 | Richer curve features helped classical controls more than quantum |
| Projected quantum kernel ensemble, configs 3x2/3x3/4x3 | 0.8611 | +0.0036 | 0.5921 | +0.0140 | Stabilized predictions but did not beat c3/r3 |
| Three-threshold quantum selector, thresholds 0.30/0.50/0.90 | 0.8600 | +0.0024 | 0.5856 | +0.0075 | Compact hypothesis set underused by quantum |
| Three-threshold classical random forest | 0.8642 | +0.0066 | 0.6065 | +0.0284 | Strong non-quantum control, useful as a benchmark |

The oracle distribution shows why threshold selection has headroom: on the
held-out Kvasir test split, the best threshold is often extreme (45/100 frames
prefer 0.30 and 21/100 prefer 0.90). However, the compact three-threshold task
currently favors a random forest rather than the projected quantum kernel, so
the quantum claim should stay with the 13-threshold candidate-specific ranker.

Paired statistics for the best quantum ranker (`c3/r3`, legacy candidate
features, hard-frame validation tuning):

| Comparison | Subset | Mean delta | 95% bootstrap CI | Sign-flip p | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| Quantum selector vs fixed 0.50 | all | +0.00458 | [+0.00128, +0.00835] | 0.0108 | 0.0594 |
| Quantum selector vs fixed 0.50 | hard | +0.01689 | [+0.00482, +0.03076] | 0.0180 | 0.0193 |
| Quantum selector - classical random forest | all | +0.00343 | [+0.00050, +0.00672] | 0.0323 | 0.2156 |
| Quantum selector - classical random forest | hard | +0.01160 | [+0.00299, +0.02104] | 0.0216 | 0.1117 |
| Quantum selector - classical HistGB | hard | +0.01104 | [+0.00045, +0.02383] | 0.0844 | 0.0909 |

Interpretation: the best current quantum result is modest but defensible. It
improves the strong UNet baseline most on hard frames and beats matched
pairwise classical controls in the same 13-threshold ranking protocol. The
stronger three-threshold random-forest result should be treated as a classical
upper control that motivates better quantum gating, not as the quantum headline.

## Hard-Case Routed Compact Quantum Selector

A more clinically targeted variant was added after the quantum-gate experiment:
use a compact random-forest threshold proposer as the default policy, then use
a validation-tuned hard-case router to send only likely hard frames to a compact
projected-quantum selector. The compact thresholds are intentionally
interpretable: `0.30` for under-segmentation rescue, `0.50` for the standard
UNet mask, and `0.90` for over-segmentation correction.

Held-out Kvasir test results:

| Method | Test Dice | Delta Dice | Hard Dice | Hard delta Dice | Changed frames |
|---|---:|---:|---:|---:|---:|
| Fixed threshold 0.50 | 0.8576 | +0.0000 | 0.5781 | +0.0000 | 0 |
| Compact RF proposer | 0.8642 | +0.0066 | 0.6065 | +0.0284 | 66 |
| Routed classical logistic selector | 0.8636 | +0.0060 | 0.6038 | +0.0258 | 67 |
| Routed classical HistGB selector | 0.8642 | +0.0066 | 0.6065 | +0.0284 | 66 |
| Routed classical RF selector | 0.8621 | +0.0045 | 0.6045 | +0.0264 | 67 |
| Hard-routed projected quantum logistic selector | 0.8664 | +0.0088 | 0.6164 | +0.0384 | 67 |
| Compact oracle threshold | 0.8765 | +0.0189 | 0.6312 | +0.0532 | 79 |

Paired statistics for the routed quantum selector versus fixed 0.50:

| Subset | Mean delta | 95% bootstrap CI | Sign-flip p | Wilcoxon p |
|---|---:|---:|---:|---:|
| All frames | +0.00877 | [+0.00122, +0.01746] | 0.0375 | 0.0785 |
| Hard frames | +0.03835 | [+0.00821, +0.07205] | 0.0391 | 0.1187 |

Interpretation: this is the best current performance and the most clinically
plausible quantum role so far: a quantum compact selector acts as a rare
hard-case rescue module on top of a strong classical proposer. The improvement
over the RF proposer is driven by one routed test frame, so this should not yet
be claimed as statistically stronger than RF. The next improvement target is
to raise hard-router coverage on validation/external data while preserving
easy-frame Dice.

## External and Out-of-Fold Significance Check

Kvasir-trained threshold policies were also evaluated on the exported CVC
baseline. When applied to all 612 CVC frames without CVC-specific calibration,
the compact threshold policies over-changed masks and reduced overall Dice. This
is an important negative result: Kvasir-tuned calibration does not transfer
directly to the full CVC distribution.

To test whether the idea works when calibrated to the target endoscopy
distribution, a 5-fold out-of-fold CVC experiment was added. The fixed UNet
probability maps are unchanged; only the compact threshold selector is trained
on four folds and evaluated once on the held-out fold.

Out-of-fold CVC results, 612 frames:

| Method | Dice | Delta Dice | Hard Dice | Hard delta Dice | Changed frames |
|---|---:|---:|---:|---:|---:|
| Fixed threshold 0.50 | 0.8105 | +0.0000 | 0.5102 | +0.0000 | 0 |
| Projected quantum logistic selector | 0.8119 | +0.0014 | 0.5347 | +0.0245 | 389 |
| Classical HistGB selector | 0.8130 | +0.0026 | 0.5350 | +0.0248 | 485 |
| Classical RF selector | 0.8131 | +0.0026 | 0.5355 | +0.0253 | 480 |
| Oracle threshold | 0.8353 | +0.0248 | 0.5710 | +0.0608 | 453 |

Paired statistics for projected quantum logistic versus fixed 0.50 on OOF CVC:

| Subset | Mean delta | 95% bootstrap CI | Sign-flip p | Wilcoxon p |
|---|---:|---:|---:|---:|
| All frames | +0.00140 | [-0.00359, +0.00630] | 0.5813 | 0.2820 |
| Hard frames | +0.02453 | [+0.01154, +0.03765] | 0.0002 | 0.000084 |

Interpretation: this gives a statistically strong hard-case improvement for
quantum calibration, but it still does not show superiority over the best
classical calibrators. A publishable claim should therefore focus on hard-case
calibration significance and rigorous comparison, or the project needs a new
quantum mechanism that beats RF/HistGB rather than matching them.

## Stronger Direction: Quantum Active Learning for Candidate Triage

The segmentation-threshold calibration work gives significant hard-case gains
versus a fixed threshold, but not a clear quantum advantage over RF/HistGB. A
stronger quantum-specific direction is active learning for candidate guidance:
use a projected quantum-kernel uncertainty policy to choose which candidate
annotations should be labeled next under class imbalance.

A focused rerun was added with 5 grouped folds, 10 annotation-sampling repeats,
40-to-200 labeled candidates, four acquisition policies, and a classical
logistic final reranker. The final top-5 guidance endpoint was mixed, but the
annotation-triage endpoint was strong.

Selected-positive rate in each newly acquired batch:

| Labeled candidates | Random | Classical uncertainty | PQK uncertainty | PQK hybrid |
|---:|---:|---:|---:|---:|
| 80 | 0.062 | 0.060 | 0.046 | 0.049 |
| 120 | 0.053 | 0.053 | 0.114 | 0.102 |
| 160 | 0.058 | 0.037 | 0.134 | 0.112 |
| 200 | 0.066 | 0.043 | 0.119 | 0.114 |

Paired PQK uncertainty versus classical uncertainty:

| Labeled candidates | Difference | 95% CI | Permutation p |
|---:|---:|:---|:---|
| 120 | +0.061 | [0.039, 0.083] | <0.001 |
| 160 | +0.097 | [0.069, 0.127] | <0.001 |
| 200 | +0.076 | [0.045, 0.106] | <0.001 |

Interpretation: this is currently the clearest statistically significant
quantum-specific result. It does not claim better final segmentation Dice.
Instead, it supports a publication direction around quantum active learning for
rare positive candidate discovery in image-guided intervention workflows. The
next technical improvement should connect this triage gain to a stronger
candidate-conditioned mask or instrument-guidance module.

## Implemented Hard-Case Quantum Switch

The hard-case routing idea is now implemented as an inference-style policy in
`endoscopy_guidance/hard_case_quantum_switch.py`.

Training command:

```bash
python endoscopy_guidance/hard_case_quantum_switch.py train \
  --baseline_dir endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test \
  --policy_path models/hard_case_quantum_switch_kvasir.joblib \
  --selection_objective overall \
  --pqk_components 8 \
  --pqk_reps 3
```

Application command:

```bash
python endoscopy_guidance/hard_case_quantum_switch.py apply \
  --baseline_dir endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test \
  --policy_path models/hard_case_quantum_switch_kvasir.joblib \
  --split test \
  --per_frame_csv endoscopy_guidance/results/hard_case_quantum_switch_kvasir_test_per_frame.csv \
  --summary_json endoscopy_guidance/results/hard_case_quantum_switch_kvasir_test_summary.json
```

Held-out Kvasir test behavior:

| Frames | Routed to quantum | Route rate | Test Dice | Delta Dice | Hard Dice | Hard delta Dice |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1 | 0.01 | 0.8664 | +0.0088 | 0.6164 | +0.0384 |

The routed frame was `kvasir_test_0032_cju8bj2ssrmlm0871gc2ug2rs`: baseline
Dice was 0.630, the router assigned hard-case score 0.886, and the quantum
selector chose threshold 0.90, matching the oracle and raising Dice to 0.848.

Interpretation: this is the concrete system behavior requested for a
real-time-support concept: the classical policy handles routine frames, and the
quantum selector is invoked only for high-risk frames. The current limitation
is low route coverage; the next improvement is a stronger hard-case detector
that routes more genuinely difficult frames without harming easy frames.
