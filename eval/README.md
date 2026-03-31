# Evaluation Pipeline

This directory now contains only the active evaluation pipeline for the endoscopy project.
It intentionally excludes:

- generated figures and result artifacts
- old `sample_data/` sanity-check arrays
- bladder-classification evaluation code
- model-specific inference code

The active scope is evaluation of saved segmentation / guidance outputs for the current segmentation-guidance track, with `CVC-ClinicDB` as the primary dataset and split-folder datasets such as legacy Kvasir kept as optional support.

## Expected Prediction Format

The preferred evaluation interface is saved model outputs, not ad hoc local notebooks.

Each sample folder should contain:

```text
outputs/
  sample_id_or_number/
    pred_mask.png
    pred_heatmap.npy
```

Recommended additions for reliable matching:

```text
outputs/
  000_CVCClinicDB/
    original_612.png
    pred_mask.png
    pred_heatmap.npy
```

`pred_heatmap.npy` is preferred over `pred_heatmap.png` because it preserves the raw float heatmap.

## Supported Datasets

### `cvc`

Primary active dataset.

Expected layout:

```text
dataset/CVC-ClinicDB/
  metadata.csv
  PNG/
    Original/
    Ground Truth/
```

The loader uses the current sequence split from the models branch:

- `train`: sequence `1-23`
- `val`: sequence `24-26`
- `test`: sequence `27-29`

### `kvasir` / `split_folder`

Legacy split-folder loader.

Expected layout:

```text
dataset/data/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

## Scripts

### 1. Evaluate Saved Outputs

```bash
cd eval
python3 evaluate_outputs.py --dataset cvc --split val --run-name baseline_cvc --outputs-dir ../output
```

Useful flags:

```bash
python3 evaluate_outputs.py --dataset cvc --split test --outputs-dir ../outputs
python3 evaluate_outputs.py --dataset cvc --split val --allow-index-fallback
python3 evaluate_outputs.py --dataset kvasir --split test --dataset-root ../dataset/data
```

Outputs:

- `eval/results/<dataset>_<split>_<run>_per_sample.csv`
- `eval/results/<dataset>_<split>_<run>_summary.csv`
- `eval/results/<dataset>_<split>_<run>_summary.json`

Metrics:

- Dice
- IoU
- Pointing Game
- Peak-to-center distance
- Coherence MER (`mask_energy_ratio`)

### 2. Generate Qualitative Panels

```bash
cd eval
python3 generate_panels.py --dataset cvc --split val --run-name baseline_cvc --mode worst --metric dice
```

Modes:

- `best`
- `median`
- `worst`
- `random`

This is intended for the eval-team deliverable of best / median / failure examples.

### 3. Plot Training Curves

The parser supports either:

- raw training logs with lines like `Epoch 01: Train Loss ... | Val Loss ... | Val Dice ...`
- CSV files with `epoch`, `train_loss`, `val_loss` and optional `val_dice`

```bash
cd eval
python3 plot_training_curves.py ../logs/cvc_pretrained.txt --output results/cvc_pretrained_curves.png
python3 plot_training_curves.py ../logs/run_a.txt ../logs/run_b.txt --labels baseline multitask --output results/cvc_compare_curves.png
```

### 4. Compare Evaluated Runs

```bash
cd eval
python3 compare_runs.py \
  results/cvc_val_baseline_summary.json \
  results/cvc_val_multitask_summary.json \
  --labels baseline multitask \
  --output-prefix results/cvc_model_comparison
```

This generates:

- a combined CSV
- a bar chart with mean ± std for the selected metrics

## Dependencies

```bash
pip install numpy pandas pillow matplotlib scikit-image
```

## PR Scope

If this directory is prepared for merge, the PR should contain:

- reusable evaluation code
- README / documentation
- ignore rules for generated outputs

It should not contain:

- generated CSV / PNG results
- sample arrays
- outdated side tracks that are not part of the current paper direction
