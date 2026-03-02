# Evaluation Scripts

## Dependencies

```
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
scikit-image
scipy
Pillow
monai
segmentation-models-pytorch
```

Install all at once:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn scikit-image scipy Pillow monai segmentation-models-pytorch
```

## Dataset Paths

Scripts expect the following datasets under `dataset/` (project root):

```
endoscopy-multitask-guidance/
├── dataset/
│   ├── data/                          ← train/val/test split (800/100/100)
│   │   ├── train/images/ masks/
│   │   ├── val/images/ masks/
│   │   └── test/images/ masks/
│   └── EndoscopicBladderTissue/       ← 4 class folders: HGC/ LGC/ NST/ NTL/
```

Both datasets are gitignored. See `dataset/DATASET.md` for download instructions.
To prepare the Kvasir-SEG split, run `dataset/prepare_kvasir.sh` then split with seed 42.

## Running

All scripts should be run from the `eval/` directory:

```bash
cd eval
python eval_bladder_model.py
python eval_kvasir_model.py
```

### Prerequisites

- `eval_bladder_model.py` requires `models/bladder_model.pth`:
  ```bash
  PYTHONPATH=. python models/train.py
  ```

- `eval_kvasir_model.py` requires `models/unet_pretrained.pth`:
  ```bash
  PYTHONPATH=. python models/train_pretrained.py
  ```

### Optional Arguments

```bash
# Bladder eval
python eval_bladder_model.py --model_path ../models/bladder_model.pth \
    --data_dir ../dataset/EndoscopicBladderTissue --batch_size 16

# Kvasir model eval
python eval_kvasir_model.py --split test --threshold 0.5
python eval_kvasir_model.py --split val          # evaluate on val set
python eval_kvasir_model.py --threshold 0.4      # custom binarization threshold
```

## Output Files

All outputs are written to `eval/results/` (gitignored, not committed):

| Script | File | Description |
|--------|------|-------------|
| `eval_bladder_model.py` | `bladder_per_sample.csv` | Per-image predictions (1754 rows) |
| | `bladder_summary.csv` | Per-class accuracy and confidence |
| | `bladder_confusion.png` | Confusion matrix |
| | `bladder_confidence.png` | Confidence distribution plot |
| `eval_kvasir_model.py` | `kvasir_model_test.csv` | Per-image Dice/IoU/PG/PCD (100 rows) |
| | `kvasir_model_test_summary.csv` | Mean ± std for each metric |
| | `kvasir_model_test_table.png` | Summary table image |
| | `kvasir_model_test_localization.png` | Localization quality plot |
