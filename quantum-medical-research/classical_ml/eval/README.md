# Evaluation Pipeline

This folder contains the reusable evaluation pipeline for saved segmentation and
guidance outputs. It does not train models or run model inference.

Use it after the models team has exported predictions for a fixed dataset split.

## What This Evaluates

The pipeline evaluates:

- binary segmentation masks
- probability / guidance heatmaps
- per-sample failures
- threshold and two-stage decision settings
- run-to-run quantitative and qualitative comparisons

Generated CSV / PNG outputs are written under `eval/results/` and should not be
committed.

## File Index

- `evaluate_outputs.py`: main entry point for evaluating saved predictions
  against ground-truth masks.
- `threshold_sweep.py`: evaluates different pixel thresholds and optional
  image-level gating for the two-stage pipeline.
- `compare_runs.py`: combines summary files from multiple runs into comparison
  tables and a metric bar chart.
- `plot_metric_distributions.py`: creates boxplots and histograms from
  per-sample metric CSVs.
- `generate_panels.py`: creates single-run qualitative panels for strong,
  weak, median, or random cases.
- `compare_qualitative.py`: creates side-by-side qualitative figures across
  multiple runs for the same selected samples.
- `failure_analysis.py`: ranks worst cases and tags common failure patterns.
- `plot_training_curves.py`: plots train / validation loss and optional
  validation Dice from logs or CSVs.
- `datasets.py`: shared dataset loading utilities for CVC, split CSV, Kvasir,
  and generic split-folder layouts.
- `predictions.py`: shared prediction discovery, loading, and sample-matching
  utilities.
- `metrics.py`: shared metric implementations used by the eval scripts.
- `README.md`: this pipeline guide.

## Required Inputs

### Dataset

Evaluation needs ground-truth image / mask pairs and a reproducible split.

Supported dataset loaders:

- `cvc`: CVC-ClinicDB layout with `metadata.csv` and an official split artifact.
- `split_csv`: generic `splits.csv` layout from the data team.
- `kvasir`: Kvasir split-folder or `Kvasir-SEG/splits.csv` layout.
- `split_folder`: generic `train/val/test/images` and `train/val/test/masks`.

For `cvc`, eval will not define the split itself. Provide one of:

- `--split-manifest path/to/val.txt`
- `--split-column split`
- files such as `dataset/CVC-ClinicDB/splits/val.txt`

For `split_csv`, the dataset root should contain `splits.csv` with:

- `split`
- image path column: `image_path`, `image`, `image_file`, or `png_image_path`
- mask path column: `mask_path`, `mask`, or `mask_file`

Example:

```text
dataset/CVC-ClinicDB/PNG/augmented/
  splits.csv
  images/
  masks/
```

### Predictions

Each evaluated run should export one folder per sample:

```text
outputs/
  sample_id/
    pred_mask.png
    pred_heatmap.npy
```

`pred_mask.png` should be a binary mask. `pred_heatmap.npy` should be the raw
float probability / confidence map.

For reliable matching, use one of these:

- prediction folder name matches the sample id or filename
- `source_image.txt` inside each prediction folder
- `original_<filename>` inside each prediction folder

Legacy numbered folders can be evaluated with `--allow-index-fallback`, but
exact sample matching is preferred for final results.

## Metrics

Segmentation metrics:

- Dice
- IoU
- Precision
- Recall
- F2
- MAE
- Boundary F1

Heatmap / localization metrics:

- Pointing Game
- Peak-to-center distance
- Coherence MER

Extra per-sample fields include pixel counts, area ratios, component counts,
false-positive pixels, false-negative pixels, and heatmap summary statistics.

## How To Interpret Outputs

Use `*_summary.csv` or `*_summary.json` for the high-level result of one run.
These files report mean, standard deviation, minimum, and maximum values for
the main metrics.

Use `*_per_sample.csv` when you need to inspect individual images. This is the
best file for finding weak cases, checking false-positive / false-negative
pixels, and understanding whether a run fails on only a few samples or across
the whole split.

Higher is better:

- Dice
- IoU
- Precision
- Recall
- F2
- Boundary F1
- Pointing Game
- Coherence MER

Lower is better:

- MAE
- Peak-to-center distance
- false-positive pixels
- false-negative pixels

Important trade-offs:

- Higher recall with much lower precision means the model finds more target
  pixels but also predicts more background as target.
- Higher precision with much lower recall means the model is conservative and
  may miss target regions.
- Lower threshold settings should improve recall only if precision, MAE, and
  qualitative examples remain acceptable.
- Pointing Game and peak-to-center distance evaluate the heatmap location, not
  just the segmentation mask.
- Boundary F1 is useful when Dice / IoU look reasonable but the predicted edge
  is visibly poor.

Use qualitative panels and failure-analysis figures to confirm whether metric
changes are visually meaningful.

## Recommended Workflow

Run these from the repository root or from `eval/`. The examples below use
commands from `eval/`.

1. Evaluate each saved run.
2. Compare runs with summary tables.
3. Generate metric distributions.
4. Generate qualitative strong / weak cases.
5. Run failure analysis.
6. Run threshold sweep if comparing lower thresholds or two-stage gating.

## Evaluate Saved Outputs

```bash
cd eval
python3 evaluate_outputs.py \
  --dataset cvc \
  --split val \
  --split-manifest ../dataset/CVC-ClinicDB/splits/val.txt \
  --outputs-dir ../outputs/baseline \
  --run-name baseline
```

For a generic data-team `splits.csv`:

```bash
python3 evaluate_outputs.py \
  --dataset split_csv \
  --dataset-root ../dataset/CVC-ClinicDB/PNG/augmented \
  --split val \
  --outputs-dir ../outputs/baseline \
  --run-name baseline
```

Outputs:

- `results/<dataset>_<split>_<run>_per_sample.csv`
- `results/<dataset>_<split>_<run>_summary.csv`
- `results/<dataset>_<split>_<run>_summary.json`

## Compare Runs

```bash
python3 compare_runs.py \
  results/cvc_val_baseline_summary.json \
  results/cvc_val_deeplabv3_summary.json \
  --labels baseline deeplabv3 \
  --output-prefix results/cvc_model_comparison
```

Outputs:

- combined long-format CSV
- wide summary CSV
- metric bar chart

## Plot Metric Distributions

```bash
python3 plot_metric_distributions.py \
  results/cvc_val_baseline_per_sample.csv \
  results/cvc_val_deeplabv3_per_sample.csv \
  --labels baseline deeplabv3 \
  --output-prefix results/cvc_metric_distributions
```

For threshold-sweep outputs:

```bash
python3 plot_metric_distributions.py \
  results/cvc_val_thresholds_threshold_per_sample.csv \
  --group-column setting \
  --output-prefix results/cvc_threshold_distributions
```

Outputs:

- distribution summary CSV
- boxplots
- histograms

## Generate Qualitative Panels

```bash
python3 generate_panels.py \
  --dataset cvc \
  --split val \
  --split-manifest ../dataset/CVC-ClinicDB/splits/val.txt \
  --outputs-dir ../outputs/baseline \
  --run-name baseline \
  --mode weak \
  --metric dice \
  --samples 12
```

Modes:

- `strong` / `best`
- `weak` / `worst`
- `median`
- `random`

Each run saves a panel image and a CSV listing the selected samples.

## Compare Qualitative Cases Across Runs

```bash
python3 compare_qualitative.py \
  --dataset cvc \
  --split val \
  --split-manifest ../dataset/CVC-ClinicDB/splits/val.txt \
  --outputs-dirs ../outputs/baseline ../outputs/deeplabv3 \
  --labels baseline deeplabv3 \
  --eval-csvs results/cvc_val_baseline_per_sample.csv results/cvc_val_deeplabv3_per_sample.csv \
  --run-name baseline_vs_deeplabv3 \
  --mode weak \
  --metric dice \
  --samples 8
```

This generates a side-by-side figure for the same selected samples across runs.

## Analyze Failure Cases

```bash
python3 failure_analysis.py \
  --dataset cvc \
  --split val \
  --split-manifest ../dataset/CVC-ClinicDB/splits/val.txt \
  --outputs-dir ../outputs/baseline \
  --run-name baseline \
  --metric dice \
  --worst-k 12
```

Outputs:

- failure-analysis CSV
- worst-case CSV
- individual worst-case figures
- markdown and JSON failure summary

Failure tags are heuristic and intended for analysis, not as ground-truth
clinical labels.

## Threshold Sweep / Two-Stage Evaluation

Use this to compare fixed pixel thresholds with lower thresholds and image-level
gating.

```bash
python3 threshold_sweep.py \
  --dataset cvc \
  --split val \
  --split-manifest ../dataset/CVC-ClinicDB/splits/val.txt \
  --outputs-dir ../outputs/baseline \
  --run-name thresholds \
  --pixel-thresholds 0.2 0.3 0.4 0.5 \
  --image-score-methods mean topk_mean \
  --image-thresholds 0.01 0.02 0.05
```

Outputs:

- per-sample CSV for every threshold setting
- summary CSV / JSON
- threshold curves
- precision-recall trade-off plot

Sample-level false-positive counts are only meaningful when the split contains
images without target regions. Pixel-level FP/FN counts are still reported for
every sample.

## Plot Training Curves

```bash
python3 plot_training_curves.py \
  ../logs/baseline.txt \
  ../logs/deeplabv3.txt \
  --labels baseline deeplabv3 \
  --output results/training_curves.png
```

The parser supports raw logs with:

```text
Epoch 01: Train Loss ... | Val Loss ... | Val Dice ...
```

It also supports CSV files with `epoch`, `train_loss`, `val_loss`, and optional
`val_dice`.

## Dependencies

```bash
pip install numpy pandas pillow matplotlib scikit-image
```

If Matplotlib cannot write to the default config directory, run plotting
commands with:

```bash
MPLCONFIGDIR=/tmp python3 <script>.py ...
```

## Notes

- Use the same split, resolution, and prediction format for all compared runs.
- `pred_heatmap.npy` is treated as a normalized probability / confidence map.
- Evaluation does not generate model predictions.
