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

## Hard-Enriched Kvasir-300 Validation

To address the main weakness of the 120-frame random cache, a larger
hard-enriched Kvasir validation subset was built from the strong UNet baseline.
The selected set contains 300 frames, including 135 frames with UNet Dice < 0.80
and 264 frames with UNet Dice < 0.90. The subset is reproducible with:

```bash
python endoscopy_guidance/select_kvasir_hard_enriched_subset.py \
  --baseline_csv endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test/baseline_metrics.csv \
  --output_dir endoscopy_guidance/exports/kvasir_hard_enriched_300

python endoscopy_guidance/export_kvasir_external.py \
  --kvasir_root /Users/ilaydadilek/Downloads/Kvasir-SEG \
  --output_dir endoscopy_guidance/exports/kvasir_hard_enriched_300 \
  --source_list endoscopy_guidance/exports/kvasir_hard_enriched_300/source_files.txt \
  --max_samples 300
```

The multi-radius prompt-quality cache uses radii 32, 48, 64, and 96 pixels and
contains 100,780 candidate masks.

| Quantity | Value |
| --- | ---: |
| Frames | 300 |
| Candidate masks | 100,780 |
| Mean UNet Dice | 0.7476 |
| Mean best-SAM Dice | 0.8647 |
| Oracle best-of-UNet/SAM Dice | 0.8796 |
| Oracle gain vs UNet | +0.1320 |
| Frames where best SAM beats UNet | 231 / 300, 77.0% |

For hard UNet frames:

| Quantity | Value |
| --- | ---: |
| Hard frames | 135 / 300 |
| Mean hard-frame UNet Dice | 0.5940 |
| Mean hard-frame best-SAM Dice | 0.8423 |
| Hard-frame oracle best-of-UNet/SAM Dice | 0.8457 |
| Hard-frame oracle gain vs UNet | +0.2518 |
| Hard frames where best SAM beats UNet | 128 / 135, 94.8% |

This validates the core hard-case claim on a substantially larger and more
clinically relevant set: the alternate promptable branch is usually better when
the classical UNet is weak.

### Learned Switch on Kvasir-300

The deployable residual switch was evaluated across 5 source-level splits, each
using 50% train, 25% validation/calibration, and 25% held-out test. Features
included multi-radius prompt features, local frozen-SAM embedding descriptors,
within-frame candidate context/rank features, UNet confidence features, and TTA
uncertainty features.

| Policy | Mean selected Dice | Mean delta vs UNet | Mean SAM rate | Hard-frame selected Dice | Hard-frame delta vs UNet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical absolute-SAM selected-candidate oracle | 0.8277 | +0.0680 | 0.4747 | 0.7565 | +0.1467 |
| Classical residual-gain selected-candidate oracle | 0.8259 | +0.0662 | 0.4773 | 0.7501 | +0.1403 |
| Classical absolute-SAM meta-RF switch | 0.7775 | +0.0178 | 0.2267 | 0.6820 | +0.0721 |
| Classical residual-gain meta-RF switch | 0.7774 | +0.0177 | 0.2773 | 0.6752 | +0.0654 |
| Classical residual-gain validation switch | 0.7740 | +0.0143 | 0.3013 | 0.6768 | +0.0670 |
| Always UNet | 0.7597 | +0.0000 | 0.0000 | 0.6098 | +0.0000 |

This is the first result in the project that looks like a credible deployable
improvement rather than only an oracle story. It is not yet the full oracle, but
it moves the held-out hard-frame Dice by about +0.07 while preserving an overall
positive Dice gain.

Paired bootstrap confidence intervals and sign-flip permutation tests were then
computed over the held-out per-frame decisions:

| Policy | Overall delta 95% CI | Overall p | Hard-frame delta 95% CI | Hard-frame p |
| --- | ---: | ---: | ---: | ---: |
| Classical absolute-SAM meta-RF switch | +0.0178 [0.0044, 0.0319] | 0.0106 | +0.0686 [0.0425, 0.0971] | 0.0002 |
| Classical residual-gain meta-RF switch | +0.0177 [0.0032, 0.0324] | 0.0166 | +0.0631 [0.0342, 0.0938] | 0.0002 |
| Classical residual-gain validation switch | +0.0143 [-0.0006, 0.0294] | 0.0618 | +0.0645 [0.0360, 0.0946] | 0.0004 |

## CVC External Validation

The same residual-switch pipeline was also evaluated on the existing CVC prompt
quality cache and CVC UNet baseline. This is a useful external-domain check
because the Kvasir hard-enriched subset was selected from the Kvasir UNet
baseline, while CVC has a different filename/export path and baseline metric
distribution. This CVC run used the available radius-48 prompt-quality cache.

| Policy | Mean selected Dice | Mean delta vs UNet | Mean SAM rate | Hard-frame selected Dice | Hard-frame delta vs UNet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical absolute-SAM selected-candidate oracle | 0.8614 | +0.0523 | 0.3216 | 0.6828 | +0.1769 |
| Classical residual-gain selected-candidate oracle | 0.8607 | +0.0516 | 0.3268 | 0.6814 | +0.1755 |
| Classical residual-gain validation switch | 0.8387 | +0.0296 | 0.1856 | 0.6236 | +0.1177 |
| Classical residual-gain meta-RF switch | 0.8367 | +0.0276 | 0.2536 | 0.6279 | +0.1220 |
| Always UNet | 0.8091 | +0.0000 | 0.0000 | 0.5059 | +0.0000 |

Paired statistics on CVC were strong:

| Policy | Overall delta 95% CI | Overall p | Hard-frame delta 95% CI | Hard-frame p |
| --- | ---: | ---: | ---: | ---: |
| Classical residual-gain validation switch | +0.0296 [0.0208, 0.0390] | 0.0002 | +0.1169 [0.0869, 0.1486] | 0.0002 |
| Classical residual-gain meta-RF switch | +0.0276 [0.0182, 0.0374] | 0.0002 | +0.1210 [0.0899, 0.1537] | 0.0002 |

This external validation substantially strengthens the paper argument: the
second-opinion switch improves hard frames on both Kvasir and CVC, and the CVC
effect is statistically clearer than the Kvasir hard-enriched result.

## Qualitative Rescue Examples

Qualitative panels were exported for held-out Kvasir hard frames where the
learned switch selected SAM and improved over UNet. The generated figures are in
`docs/figures/kvasir_hard_enriched_switch_examples/`, with a manifest CSV
recording the source file, UNet Dice, switched SAM Dice, Dice gain, selected
radius, and policy.

One representative rescue case improves from UNet Dice 0.097 to switched SAM
Dice 0.816, illustrating the intended clinical behavior: keep the classical UNet
as the default, but route difficult frames to the promptable second-opinion
branch when the switch predicts a likely gain.

## Interpretation

Residual-gain targeting produced the first validation-calibrated non-oracle
switch with a positive mean gain over UNet. The gain is still small, but it is
directionally important because it improves hard-frame Dice without broadly
switching away from the strong UNet. A more complex two-stage switch did not
improve the aggregate result, suggesting that the current limitation is not a
missing switch model alone.

The remaining bottleneck is calibration, not the existence of a better alternate
mask. The full multi-radius oracle shows large recoverable headroom, especially
on hard frames. The hard-enriched 300-frame validation improves calibration
enough to produce a meaningful learned-switch gain, but a gap remains between
the deployable switch and the selected-candidate oracle.

## Next Scientific Step

The most publication-relevant next step is to increase the candidate/evaluation
coverage, not to claim a large result from the 120-frame cache. A stronger study
should:

1. Expand cached SAM candidate quality to the full Kvasir validation/test pool
   after validating the hard-enriched 300-frame result.
2. Add multi-radius CVC/PolypGen caches, because the CVC external validation
   already works with radius 48 and may improve further with prompt diversity.
3. Keep multiple prompt radii and add richer prompt families, because prompt
   diversity is now the strongest observed source of improvement.
4. Keep UNet as the default output and frame the quantum/SAM branch as
   hard-case second-opinion routing.
