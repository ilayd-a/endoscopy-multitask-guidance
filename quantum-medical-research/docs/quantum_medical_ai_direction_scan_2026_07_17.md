# Quantum Medical AI Direction Scan

Date: 2026-07-17  
Branch: `spie-mi104-qml-benchmark-improvements`

## Bottom Line

The current repos do not yet contain a "super impressive" quantum result suitable
for a strong SPIE MI104 full-paper claim. The best honest result remains a
label-efficient guidance result: projected quantum-kernel acquisition improves
rare positive candidate discovery under limited labels. That is real, but it is
not enough by itself for an oral-presentation-level paper.

The strongest next direction is:

> Quantum-assisted label-efficient endoscopic guidance: use quantum kernels or
> compact quantum feature maps to prioritize expert annotation and reliability
> review for difficult surgical/endoscopic frames, while a stronger classical
> segmentation backbone handles the dense mask prediction.

This avoids overclaiming clinical segmentation performance and targets a real
gap: medical image annotation cost, noisy labels, and deployment reliability.

## Experiments Tried

### 1. EBTC HGC/LGC Quantum Triage

Outcome: not strong enough.

- Classical baselines were competitive or stronger.
- Validation-tuned quantum triage did not hold under clean testing.
- Useful as a negative result, not as the main paper direction.

### 2. CVC Candidate Reranking, Sparse Candidate Pool

Outcome: modest quantum signal, but limited by candidate coverage.

- Stride-48 candidate pool oracle top-k ceiling: about 0.742.
- PQK improved top-5 target recovery over several classical baselines in the
  full-label setting, but refined Dice stayed low.
- Active learning gave the cleanest quantum-specific result:
  - PQK hybrid top-5 recovery at 200 labels: 0.580 vs random 0.519.
  - PQK uncertainty selected positives much more efficiently at 160-200 labels.

Interpretation: scientifically valid, but not yet "wow."

### 3. Dense CVC Candidate Pool

Outcome: coverage improved, ranking got harder.

| Candidate stride | Candidates | Positive rate | Oracle sample hit |
|---:|---:|---:|---:|
| 48 | 2,178 | 0.064 | 0.742 |
| 32 | 4,751 | 0.053 | 0.955 |
| 24 | 8,511 | 0.047 | 1.000 |
| 16 | 17,423 | 0.049 | 0.985 |

Dense stride-24 without richer image features did not produce strong top-5
results, so candidate density alone is not sufficient.

### 4. CVC Reliability Monitor with Frozen Encoder Features

Outcome: better safety-monitor framing, but classical remains too strong.

Using U-Net encoder features plus heatmap/image statistics, with train sequences
1-23, validation 24-26, and test 27-29:

| Failure label | Best held-out test AUC | Caveat |
|---|---:|---|
| Dice < 0.1 | PQK 0.716 | validation-selected classical stronger/stabler |
| Dice < 0.2 | PQK 0.832 | good test result, but val+test aggregate favors ExtraTrees |
| Dice < 0.3 | PQK 0.891 | test positives are 0.939, so this is imbalanced/fragile |

Interpretation: this is promising as a reliability sub-experiment, not yet a
main contribution.

### 5. CVC Dense RGB Candidate Reranking

Outcome: ranking improves, but current mask refinement still hurts Dice.

Using all 612 CVC frames, stride-32 candidates, RGB patch features, five
grouped folds:

| Train candidates | Best model | Top-5 hit | Candidate AUC | Base Dice | Refined Dice |
|---:|---|---:|---:|---:|---:|
| 300 | PQK reps2 C1 | 0.752 | 0.850 | 0.298 | 0.236 |
| 600 | PQK reps3 C10 | 0.747 | 0.847 | 0.298 | 0.238 |

The dense candidate oracle sample hit was 0.979. The ranker can find good
candidate points much better than the heatmap baseline, but the current
Gaussian-point refinement is not a good segmentation mechanism.

### 6. CVC Threshold and Component Refinement

Outcome: not enough.

The original CVC heatmap script used threshold 0.3, while the export default was
0.5. Sweeping thresholds did not rescue held-out test Dice:

| Threshold | Train Dice | Val Dice | Test Dice |
|---:|---:|---:|---:|
| 0.1 | 0.339 | 0.397 | 0.090 |
| 0.3 | 0.324 | 0.385 | 0.087 |
| 0.5 | 0.316 | 0.377 | 0.086 |

Connected-component refinement around the heatmap peak improved all-frame Dice
only slightly, from about 0.326 to 0.330 at threshold 0.05.

Interpretation: the segmentation checkpoint is the bottleneck on test sequences.

### 7. Kidney-Stone Dataset

Outcome: not a strong substitute.

The uploaded zip contains only COCO annotations, not the image frames. Existing
repo results on the temporal split are not strong:

- Whole-frame dataset4 classification: tiny test set, weak classical and quantum
  accuracy.
- Patch benchmark: models mostly predicted negatives; not publishable as-is.

## Literature-Guided Research Gap

Recent work points to three relevant facts:

1. Hybrid quantum-classical segmentation is emerging. QPolypNet reports very
   high Dice on CVC-ClinicDB and Kvasir-SEG, suggesting reviewers may now expect
   strong classical-grade segmentation performance from any direct quantum
   segmentation claim.
2. Systematic reviews warn that QML rarely shows consistent advantage under
   realistic health-data conditions, so a credible paper needs strong validation
   and restrained claims.
3. Active learning in medical image analysis is a real, clinically relevant
   problem because expert labels are expensive.

Therefore, the most defensible gap is not "quantum magically improves Dice from
0.90 to 0.98." It is:

> Can compact quantum kernels improve label efficiency and failure discovery in
> image-guided endoscopic AI, especially when annotations are scarce or model
> failures cluster under temporal domain shift?

## What Would Make This Publishable

1. Replace the weak CVC checkpoint with a competent segmentation backbone.
   - Target: baseline Dice in the 0.85-0.92 range on a standard validation
     protocol.
   - Then test whether quantum-assisted acquisition reduces annotation burden.

2. Keep the quantum contribution focused.
   - Primary claim: fewer labels needed to reach the same guidance/reliability
     performance.
   - Secondary claim: quantum kernel diagnostics explain when the acquisition
     policy helps.

3. Use validation-locked analysis.
   - Select hyperparameters on validation sequences only.
   - Report final test once.
   - Include classical active-learning baselines, random sampling, uncertainty,
     diversity, and ablations.

4. Add a better point-to-mask refinement module.
   - Current Gaussian blobs hurt Dice.
   - Next attempt should use a learned lightweight decoder or SAM-style prompt
     refinement from top-ranked candidates.

5. Expand beyond one dataset.
   - At minimum: CVC-ClinicDB plus Kvasir-SEG.
   - Better: train on one dataset, test on another to make the reliability and
     active-learning claim clinically relevant.

## Current Recommendation

Do not submit the current numbers as a high-performance segmentation paper.

Build the paper around quantum-assisted active learning/reliability, but only
after we either:

- train or import a stronger segmentation backbone, or
- add a prompt/refinement module that turns the improved candidate ranking into
  a real Dice improvement.

The current branch now contains the tooling needed to continue this search
cleanly:

- `endoscopy_guidance/active_learning_candidate_benchmark.py`
- `endoscopy_guidance/candidate_ranking_benchmark.py`
- `endoscopy_guidance/encoder_embedding_failure_benchmark.py`

