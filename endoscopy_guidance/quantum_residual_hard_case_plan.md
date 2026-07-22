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

The current local CVC checkpoint at:

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
- External validation on Kvasir-SEG and PolypGen.
- Paired bootstrap/permutation tests against the classical baseline.
- Latency/FPS measurement with and without the quantum module.

## Immediate Next Steps

1. Locate or retrain the strong baseline checkpoint that achieves approximately
   0.85-0.90 Dice on the intended validation/test split.
2. Export predictions/logits/features with `export_classical_cvc_baseline.py`.
3. Build a patch-level residual dataset from baseline false positives and false
   negatives.
4. Compare classical residual models against projected quantum-kernel residual
   models under the same train/val/test split.
5. Apply the best residual correction to probability maps and evaluate full
   segmentation Dice, hard-case Dice, and boundary metrics.

## Implemented Scaffold

- `export_classical_cvc_baseline.py`
  - Loads the sibling repo's UNet/ResNet34 checkpoint.
  - Exports images, masks, probability maps, predicted masks, and per-frame
    Dice/IoU/error features into the quantum repo.
  - Supports threshold sweeps for calibration diagnostics.

- `build_residual_patch_dataset.py`
  - Samples false-positive, false-negative, boundary, uncertain, and correct
    patches from baseline predictions.
  - Produces a compact patch-feature matrix for residual correction.

- `residual_patch_quantum_benchmark.py`
  - Compares classical residual classifiers with a projected-quantum feature
    residual classifier.
  - Current diagnostic run is only a smoke test because the available CVC
    checkpoint is not a strong baseline.

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
