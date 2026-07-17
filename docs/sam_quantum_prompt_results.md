# SAM Quantum Prompt Benchmark Results

Date: 2026-07-17  
Branch: `spie-mi104-qml-benchmark-improvements`

## Question

Can projected quantum-kernel candidate ranking choose better point/box prompts
for SAM than a heatmap peak or a classical candidate ranker?

## Setup

- Dataset: CVC-ClinicDB export from the local endoscopy guidance pipeline.
- Train candidates: sequences 1-26, capped to 600 balanced candidates.
- Held-out test: sequences 27-29.
- Promptable model: vanilla Segment Anything ViT-B.
- Prompt format: positive point plus a fixed 48 px box around the point.
- Candidate features: stride-32 dense candidates with RGB patch features.
- Quantum model: projected quantum kernel SVC, usually reps=2, C=1.
- Classical comparator: ExtraTrees candidate ranker.

## Key Results

### 12-frame held-out smoke test, top-1 prompt

| Strategy | Prompt hit | Dice |
|---|---:|---:|
| Oracle candidate | 0.750 | 0.425 |
| PQK top-1 | 0.250 | 0.215 |
| Heatmap peak | 0.250 | 0.199 |
| Classical ExtraTrees top-1 | 0.250 | 0.180 |
| Base heatmap threshold 0.3 | NA | 0.194 |
| Random candidate | 0.000 | 0.116 |

### 12-frame held-out smoke test, top-3 prompts selected by SAM score

| Strategy | Prompt hit | Dice |
|---|---:|---:|
| Oracle candidate | 0.917 | 0.543 |
| PQK top-3 | 0.333 | 0.406 |
| Classical ExtraTrees top-3 | 0.333 | 0.390 |
| Heatmap peak | 0.167 | 0.179 |
| Base heatmap threshold 0.3 | NA | 0.147 |
| Random candidate | 0.000 | 0.035 |

### 66-frame full held-out test, top-1 prompt

| Strategy | Prompt hit | Dice |
|---|---:|---:|
| Oracle candidate | 0.955 | 0.619 |
| Classical ExtraTrees top-1 | 0.273 | 0.269 |
| PQK top-1 | 0.152 | 0.208 |
| Heatmap peak | 0.136 | 0.089 |
| Base heatmap threshold 0.3 | NA | 0.087 |
| Random candidate | 0.045 | 0.056 |

### 66-frame full held-out test, top-3 prompts selected by SAM score

| Strategy | Prompt hit | Dice |
|---|---:|---:|
| Oracle candidate | 0.955 | 0.619 |
| Classical ExtraTrees top-3 | 0.318 | 0.299 |
| PQK top-3 | 0.227 | 0.248 |
| Heatmap peak | 0.136 | 0.089 |
| Base heatmap threshold 0.3 | NA | 0.087 |
| Random candidate | 0.045 | 0.047 |

### 12-frame held-out smoke test, top-5 prompts selected by SAM score

| Strategy | Prompt hit | Dice |
|---|---:|---:|
| Oracle candidate | 0.917 | 0.543 |
| PQK top-5 | 0.333 | 0.392 |
| Classical ExtraTrees top-5 | 0.250 | 0.319 |
| Heatmap peak | 0.167 | 0.179 |
| Base heatmap threshold 0.3 | NA | 0.147 |
| Random candidate | 0.000 | 0.046 |

## Interpretation

This is the first genuinely new direction that converts candidate ranking into a
mask-quality experiment. The promptable segmentation bridge works: SAM point-box
prompts can substantially improve over the weak heatmap mask, and the oracle
candidate result shows large remaining headroom.

However, the full held-out test does not yet support a broad quantum-over-
classical claim. PQK beats the heatmap and random baselines by a large margin,
and it beats ExtraTrees on the 12-frame top-3/top-5 slices, but ExtraTrees is
stronger on the full 66-frame test.

## Current Best Claim

Defensible:

> Quantum-kernel prompt selection is a promising mechanism for converting
> endoscopic candidate ranking into promptable segmentation, improving far above
> raw heatmap prompts and exposing a large oracle gap.

Not yet defensible:

> Quantum prompt selection is consistently better than strong classical
> candidate rankers.

## Next Improvements

1. Cache SAM image embeddings so prompt sweeps are fast.
2. Tune PQK prompt selection on validation sequences instead of using one fixed
   configuration.
3. Add more classical comparators and paired statistical tests.
4. Test MedSAM/SAM2 or an endoscopy-adapted SAM variant; vanilla SAM is not
   optimized for endoscopic frames.
5. Learn a quantum/classical prompt-set selector that optimizes SAM score and
   candidate probability jointly, not candidate probability alone.

