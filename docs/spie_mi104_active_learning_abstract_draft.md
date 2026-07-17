# SPIE MI104 Abstract Draft

## Working Title

Projected quantum-kernel active learning for candidate annotation in endoscopic image-guided intervention

## Draft Abstract

Reliable image-guided and robotic endoscopic intervention depends on target localization under
domain shift, but dense expert annotation of candidate regions is costly. We evaluate whether a
projected quantum-kernel (PQK) uncertainty module can improve candidate annotation efficiency and
top-k target recovery when used around a classical endoscopic guidance backbone. A sequence-held-out
CVC endoscopy export was used to generate heatmap-derived and grid-based candidate regions. Each
candidate was represented by compact heatmap, geometric, mask, RGB, and local texture descriptors.
Starting from 40 balanced candidate annotations, we simulated active-learning rounds that acquired
batches of 40 additional labels up to 200 labels. We compared random sampling, classical
uncertainty sampling, PQK uncertainty sampling, PQK uncertainty with diversity, and a hybrid policy
combining PQK uncertainty with coverage-preserving random sampling. Policies were evaluated using
5 grouped folds and 3 repeated annotation simulations.

PQK uncertainty substantially enriched rare target-positive candidate annotations compared with
random sampling after 120 labels. The selected-positive-rate improvement was 0.063 at 120 labels
(95% CI 0.028 to 0.095, permutation p=0.004), 0.203 at 160 labels (95% CI 0.153 to 0.250,
p<0.001), and 0.162 at 200 labels (95% CI 0.120 to 0.205, p<0.001). For a classical
logistic-regression final reranker, the PQK-hybrid policy improved top-5 target recovery over
random sampling at later annotation budgets: +0.046 at 160 labels (95% CI 0.021 to 0.077,
p=0.017) and +0.061 at 200 labels (95% CI 0.030 to 0.092, p=0.007). PQK-hybrid also improved
top-5 recovery over classical uncertainty at 160 labels (+0.045, 95% CI 0.015 to 0.082,
p=0.032) and 200 labels (+0.046, 95% CI 0.011 to 0.084, p=0.032).

These results suggest that projected quantum-kernel uncertainty can serve as an annotation-triage
module for endoscopic image-guided intervention, particularly for discovering rare positive target
candidates under domain shift. The method is not proposed as a replacement for classical
segmentation; rather, it is a lightweight quantum-assisted acquisition layer around a classical
guidance backbone. Future work will evaluate stronger segmentation backbones, candidate-conditioned
mask refinement, calibration, and broader domain-shift validation.

## Recommended Claims

- PQK uncertainty improves rare positive candidate discovery under class imbalance.
- PQK-hybrid acquisition improves top-5 target recovery at 160 and 200 candidate-label budgets.
- The contribution is annotation triage and top-k guidance, not clinical-grade segmentation Dice.

## Claims To Avoid

- Quantum directly improves segmentation quality to clinical levels.
- Quantum replaces the classical endoscopy guidance backbone.
- Broad quantum advantage over all classical approaches.

## Missing Before Full Paper

1. Run at least 10 annotation repeats for tighter confidence intervals.
2. Repeat on a stronger endoscopy-multiguidance export and preferably a second dataset.
3. Add qualitative panels showing candidates selected by random, classical uncertainty, and PQK.
4. Replace Gaussian candidate maps with candidate-conditioned segmentation refinement.
5. Add calibration, abstention, and failure-detection analysis.
