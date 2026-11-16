# Endoscopy Multitask Guidance

Endoscopic polyp segmentation and localization-guidance evaluation under
cross-dataset domain shift.

This repository compares a baseline U-Net and a ResNet34 encoder-pretrained
U-Net for polyp segmentation on Kvasir-SEG. The pretrained model is also
evaluated on external CVC-ClinicDB without retraining to probe domain-shift
robustness. Pixel-wise probability outputs are used as guidance heatmaps and
evaluated with localization metrics including pointing-game accuracy (PG) and
peak-to-center distance (PCD).

## Integrated research code

The latest classical training and evaluation pipeline lives in the top-level
`models/`, `dataset/`, and `eval/` directories. The quantum-assisted research
is included in [`quantum-medical-research/`](quantum-medical-research/README.md):

- `endoscopy_guidance/`: quantum prompt selection, residual refinement, active
  learning, and external-validation experiments.
- `EndoscopicBladderTissue/`: bladder tissue classification benchmarks.
- `docs/`: research reports, publication drafts, and qualitative figures.
- `scripts/make_iros_figures.py`: publication figure generation (requires Pillow
  and ReportLab, with the paired-statistics inputs included).
- `classical_ml/`: the classical baseline snapshot used by those research
  experiments; use the top-level pipeline for current classical development.

Install research dependencies from `quantum-medical-research/requirements.txt`
in a suitable research environment. Dataset downloads, generated embedding
caches, and local training outputs are not bundled; follow the relevant module
instructions to prepare them. Experimental quantum results are separate from
the classical evaluation described below.

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

### PolypGen (external, optional)

PolypGen 2021 (multi-center, `PolypGen2021_MultiCenterData_v3`) is used as a
second external evaluation set. The release stores single frames per center in
`data_C1` to `data_C6` (`images_Cn/` and `masks_Cn/`, masks named
`<image>_mask.jpg`). The preparation script gathers them into
`dataset/PolypGen/positive/{images,masks}` with matching file names and writes
`manifest.csv` (frame, center, size, mask pixels):

```bash
python3 dataset/dataset_c/prepare_polypgen.py \
  --source /path/to/PolypGen2021_MultiCenterData_v3
```

Of the 1,537 single frames, 1,412 have a non-empty mask and are kept. The 125
frames with empty masks are dropped because Dice/IoU are undefined for them.
The sequence frames in `sequenceData/` and `imagesAll_positive/` have no masks
and are not used.

```bash
PYTHONPATH=. python3 models/heatmap_pretrained.py \
  --image-dir dataset/PolypGen/positive/images \
  --num-samples 1412 \
  --output-dir output/domain_shift_kvasir_to_polypgen

PYTHONPATH=eval python3 eval/evaluate_outputs.py \
  --dataset polypgen \
  --dataset-root dataset/PolypGen/positive \
  --outputs-dir output/domain_shift_kvasir_to_polypgen \
  --run-name kvasir_to_polypgen \
  --results-dir eval/results
```

`--num-samples` must cover every image in the folder. The output files are named
`polypgen_val_*` only because `--split` defaults to `val`; PolypGen has no
splits and the argument is ignored.

Single-run results for the Kvasir-trained pretrained U-Net on PolypGen (1,412
frames, threshold 0.5, PCD in pixels at 256 x 256):

| Center | Frames | Dice | IoU | Recall | PG | PCD (px) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 251 | 0.768 | 0.687 | 0.835 | 0.944 | 30.9 |
| C2 | 270 | 0.765 | 0.695 | 0.845 | 0.907 | 32.1 |
| C3 | 456 | 0.837 | 0.761 | 0.876 | 0.978 | 24.5 |
| C4 | 146 | 0.501 | 0.420 | 0.732 | 0.664 | 50.6 |
| C5 | 206 | 0.486 | 0.396 | 0.633 | 0.655 | 62.6 |
| C6 | 83 | 0.743 | 0.669 | 0.796 | 0.880 | 33.1 |
| All | 1412 | 0.719 | 0.641 | 0.808 | 0.873 | 35.8 |

Performance varies strongly by center, with C4 and C5 well below the others.
The model produced an empty mask for 64 of the 1,412 frames. The overall PCD
excludes one frame whose predicted heatmap gives an infinite distance.

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

## Multi-Seed Pretrained Results

The pretrained U-Net experiment was repeated across 10 random seeds. The table
reports mean +/- SD across seed-level test metrics. Each seed used the fixed
Kvasir train/val/test split and the best validation checkpoint from 20 epochs.

| Model | Test set | Seeds | Dice | IoU | Precision | Recall | F2 | PG | PCD (px) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained U-Net | Kvasir | 10 | 0.8543 +/- 0.0147 | 0.7838 +/- 0.0174 | 0.8830 +/- 0.0185 | 0.8717 +/- 0.0153 | 0.8602 +/- 0.0134 | 0.9780 +/- 0.0063 | 40.2948 +/- 4.1677 |
| Pretrained U-Net | CVC external | 10 | 0.7462 +/- 0.0326 | 0.6635 +/- 0.0319 | 0.7891 +/- 0.0303 | 0.7817 +/- 0.0433 | 0.7591 +/- 0.0369 | 0.9232 +/- 0.0217 | 39.9351 +/- 5.5626 |

Run the repeated evaluation with:

```bash
PYTHONPATH=. python3 -u models/run_pretrained_multiseed.py \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --epochs 20 \
  --batch-size 8 \
  --output-dir results/bhi_multiseed \
  --resume
```

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
results/bhi_multiseed/pretrained_multiseed_summary.csv
results/bhi_multiseed/pretrained_multiseed_per_seed.csv
```

## Limitations

- Vanilla U-Net results are currently single-run.
- The multi-seed results currently cover the pretrained U-Net only.
- The vanilla U-Net baseline should be strengthened or retrained for a more
  rigorous full-paper comparison.
- Reverse domain shift (CVC to Kvasir) has not yet been evaluated.
- The current heatmaps are probability outputs, not a separately trained
  localization head.
