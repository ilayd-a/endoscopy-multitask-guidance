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

### Validation sweep after caching

After adding candidate and SAM image-embedding caching, a validation sweep was
run on sequences 24-26 while training prompt rankers on sequences 1-23.

Best validation setting for PQK:

- prompt mode: point + box
- box radius: 48 px
- SAM selection: best SAM score among top-3 ranked prompts
- PQK: reps=2, C=1, PCA components=10

Validation summary:

| Strategy | Prompt hit | Dice |
|---|---:|---:|
| Oracle candidate | 1.000 | 0.799 |
| Classical ExtraTrees | 0.574 | 0.485 |
| PQK, 10 components | 0.603 | 0.477 |
| Base heatmap threshold 0.3 | NA | 0.384 |
| Heatmap peak prompt | 0.544 | 0.363 |
| Random candidate | 0.176 | 0.142 |

This setting was then frozen and run once on the held-out test sequences 27-29.

### Final held-out test with validation-selected PQK setting

| Strategy | Prompt hit | Dice | IoU |
|---|---:|---:|---:|
| Oracle candidate | 0.955 | 0.619 | 0.490 |
| PQK, validation-selected | 0.348 | 0.303 | 0.228 |
| Classical ExtraTrees | 0.318 | 0.299 | 0.221 |
| Heatmap peak prompt | 0.136 | 0.089 | 0.057 |
| Base heatmap threshold 0.3 | NA | 0.087 | 0.055 |
| Random candidate | 0.045 | 0.047 | 0.031 |

Paired differences on the 66 held-out test frames:

| Comparison | Mean Dice difference | 95% bootstrap CI | Permutation p |
|---|---:|---:|---:|
| PQK - Classical ExtraTrees | +0.004 | [-0.059, 0.069] | 0.908 |
| PQK - Heatmap peak prompt | +0.214 | [0.140, 0.293] | <0.001 |
| PQK - Base heatmap threshold | +0.216 | [0.147, 0.288] | <0.001 |

The quantum-vs-classical margin is small and not statistically significant, but
the validation-selected PQK prompt pipeline is competitive with the best
classical prompt ranker and strongly improves over the raw heatmap baselines.

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

After validation tuning, PQK is slightly ahead of ExtraTrees on the full held-out
test, but the paired difference is tiny and not statistically significant. The
main result should therefore be framed as competitive quantum prompt selection
with strong heatmap-baseline improvement, not a definitive quantum advantage.

## Current Best Claim

Defensible:

> Validation-selected quantum-kernel prompt selection is a promising mechanism
> for converting endoscopic candidate ranking into promptable segmentation. It is
> competitive with a strong classical prompt ranker, improves far above raw
> heatmap prompts, and exposes a large oracle gap.

Not yet defensible:

> Quantum prompt selection is significantly better than strong classical
> candidate rankers.

## Multi-Positive Prompt Ablation

To test whether the bottleneck was single-prompt brittleness, the benchmark was
extended with `--prompt_aggregation multi`, which passes the top-K selected
positive points to SAM together. For `point_box`, the selected prompts share one
union box.

Validation improved, but held-out test did not:

| Setting | Split | PQK Dice | Classical Dice | Oracle Dice |
|---|---|---:|---:|---:|
| Top-3 point+union-box multi-prompt | Val | 0.509 | 0.513 | 0.803 |
| Top-3 point+union-box multi-prompt | Test | 0.242 | 0.237 | 0.502 |
| Top-5 point+union-box multi-prompt | Test | 0.220 | 0.200 | 0.465 |
| Top-3 point-only multi-prompt | Test | 0.176 | 0.173 | 0.371 |
| Top-5 point-only multi-prompt | Test | 0.182 | 0.150 | 0.377 |

Interpretation: giving SAM more positive prompts raises prompt-hit rate, but the
combined prompt often makes the predicted mask less specific. The stronger path
is calibrated prompt-quality selection, not multi-positive prompting.

## Next Improvements

1. Add more classical comparators and paired statistical tests.
2. Test MedSAM/SAM2 or an endoscopy-adapted SAM variant; vanilla SAM is not
   optimized for endoscopic frames.
3. Learn a quantum/classical prompt-set selector that optimizes SAM score and
   candidate probability jointly, not candidate probability alone.
4. Expand to Kvasir-SEG and cross-dataset validation.
