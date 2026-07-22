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
