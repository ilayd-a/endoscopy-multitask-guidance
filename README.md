# Endoscopy Multitask Guidance

Endoscopic polyp segmentation and localization-guidance evaluation under
cross-dataset domain shift.

This repository compares a baseline U-Net and a ResNet34 encoder-pretrained
U-Net for polyp segmentation on Kvasir-SEG. The pretrained model is also
evaluated on external CVC-ClinicDB without retraining to probe domain-shift
robustness. Pixel-wise probability outputs are used as guidance heatmaps and
evaluated with localization metrics including pointing-game accuracy (PG) and
peak-to-center distance (PCD).

## Current Framing

The current project should be framed as:

- segmentation with U-Net-based models,
- guidance/localization heatmaps derived from pixel-wise probability outputs,
- in-domain Kvasir-SEG evaluation,
- external CVC-ClinicDB evaluation without retraining.

It should not be described as a fully validated multitask architecture unless a
separate trained task head and corresponding results are added.

## Datasets

### Kvasir-SEG

Kvasir-SEG is prepared into:

```text
dataset/data/train/images
dataset/data/train/masks
dataset/data/val/images
dataset/data/val/masks
dataset/data/test/images
dataset/data/test/masks
dataset/data/splits.csv
```

The current split contains 800 train, 100 validation, and 100 test images.

To download and prepare Kvasir-SEG:

```bash
bash dataset/dataset_a/prepare_kvasir.sh
```

### CVC-ClinicDB

CVC-ClinicDB is used as an external evaluation set for domain-shift analysis.
The preparation script expects the Kaggle archive at
`dataset/dataset_b/archive.zip`:

```bash
bash dataset/dataset_b/prepare_cvc.sh
```

Raw dataset folders and archives are intentionally ignored by Git.

## Headline Results

All results below use masks thresholded at 0.5. PG and PCD are reported for
probability heatmaps from the pretrained model. PCD is measured in pixels after
resizing to 256 x 256.

| Model | Test set | Dice | IoU | Precision | Recall | F2 | PG | PCD (px) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla U-Net | Kvasir | 0.461 | 0.338 | 0.435 | 0.645 | 0.528 | -- | -- |
| Pretrained U-Net | Kvasir | 0.867 | 0.801 | 0.871 | 0.909 | 0.885 | 0.980 | 28.3 |
| Pretrained U-Net | CVC external | 0.789 | 0.704 | 0.795 | 0.853 | 0.815 | 0.962 | 28.2 |

A center-point localization baseline achieved lower PG and larger PCD:

| Baseline | Test set | PG | PCD (px) |
| --- | --- | ---: | ---: |
| Center point | Kvasir | 0.470 | 51.3 |
| Center point | CVC external | 0.289 | 52.3 |

The main preliminary finding is that segmentation quality decreases under
external CVC evaluation, while coarse localization remains comparatively stable.
These are single-run results and should be treated as a preliminary baseline.

## Repository Structure

```text
dataset/
  dataset_a/prepare_kvasir.sh      # Kvasir download/preparation
  dataset_b/prepare_cvc.sh         # CVC archive preparation
  requirements.txt                 # Python dependencies

models/
  train_pretrained.py              # ResNet34 encoder-pretrained U-Net training
  train.py                         # Baseline U-Net training
  heatmap_pretrained.py            # Export masks and probability heatmaps

eval/
  evaluate_outputs.py              # Per-sample and summary evaluation
  metrics.py                       # Segmentation and localization metrics
  generate_panels.py               # Qualitative panels
  plot_metric_distributions.py     # Metric distribution plots
  compare_runs.py                  # Summary comparison plots
  results/                         # Committed evaluation CSV/JSON artifacts

results/
  bhi_figures/                     # Tables and figures used for abstract drafting
```

## Reproducing Main Outputs

Install dependencies:

```bash
python3 -m pip install -r dataset/requirements.txt
```

Prepare Kvasir:

```bash
bash dataset/dataset_a/prepare_kvasir.sh
```

Train the pretrained U-Net:

```bash
PYTHONPATH=. python3 models/train_pretrained.py
```

Export Kvasir test predictions:

```bash
PYTHONPATH=. python3 models/heatmap_pretrained.py \
  --split test \
  --num-samples 100 \
  --output-dir output/pretrained_kvasir_test
```

Evaluate Kvasir test predictions:

```bash
PYTHONPATH=eval python3 eval/evaluate_outputs.py \
  --dataset kvasir \
  --dataset-root dataset/data \
  --split test \
  --outputs-dir output/pretrained_kvasir_test \
  --run-name pretrained_kvasir_test \
  --results-dir eval/results \
  --allow-index-fallback
```

Export pretrained Kvasir-trained predictions on CVC:

```bash
PYTHONPATH=. python3 models/heatmap_pretrained.py \
  --image-dir dataset/dataset_b/PNG/Original \
  --num-samples 612 \
  --output-dir output/domain_shift_kvasir_to_cvc
```

Evaluate CVC external predictions:

```bash
PYTHONPATH=eval python3 eval/evaluate_outputs.py \
  --dataset cvc \
  --dataset-root dataset/dataset_b \
  --metadata-path dataset/dataset_b/metadata.csv \
  --split test \
  --split-manifest results/bhi_figures/cvc_all_manifest.csv \
  --outputs-dir output/domain_shift_kvasir_to_cvc \
  --run-name kvasir_to_cvc \
  --results-dir eval/results
```

## Abstract Figures and Tables

The most useful abstract artifacts are:

```text
results/bhi_figures/domain_shift_results_table.csv
results/bhi_figures/domain_shift_compact_metrics.png
results/bhi_figures/kvasir_test_pretrained_kvasir_test_median_dice_panel.png
results/bhi_figures/cvc_test_kvasir_to_cvc_median_dice_panel.png
results/bhi_figures/center_baseline_localization.csv
```

## Limitations

- Results are currently single-run.
- The vanilla U-Net baseline should be strengthened or retrained for a more
  rigorous full-paper comparison.
- Reverse domain shift (CVC to Kvasir) has not yet been evaluated.
- The current heatmaps are probability outputs, not a separately trained
  localization head.
