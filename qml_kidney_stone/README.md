# QML Kidney Stone — QSVM Benchmark (Stone Present / Absent)

QML classification experiment using kidney stone endoscopy frames annotated in CVAT (COCO instance segmentation export). This follows the same design philosophy as the previous Kvasir-SEG / EBTC QML benchmarks (frozen ResNet-18 feature extraction → PCA → angle encoding → quantum model). For now, only QSVM (the fastest model) has been implemented.

## ⚠️ Important: This pipeline requires network access

`torchvision` will download ImageNet pretrained weights for ResNet-18 from `download.pytorch.org` on first run. Since the Anthropic sandbox environment blocks access to this domain, **please run this on Koki's local environment (no network restrictions)**. The `qml_endo` conda env used previously for EBTC should work as-is.

## Dataset overview

| Item | Value |
|---|---|
| Number of frames | 294 (extracted from a single endoscopy video) |
| Annotation format | COCO instance segmentation (CVAT export, `instances_default.json`) |
| Category | "Kidney Stone" — single class only |
| Annotated (stone present) | 250 frames |
| Unannotated (stone absent) | 44 frames |

### ⚠️ On the temporal clustering of labels

The 44 "absent" (unannotated) frames are not randomly scattered throughout the video — they form **contiguous temporal blocks** (e.g., frames 071–077 appear consecutively as absent). This happens because the periods where "the stone leaves the field of view / the camera moves" tend to occur in clusters.

Because of this, **splitting train/test randomly causes nearly identical adjacent frames to end up in both train and test**, making the task artificially easy (leakage).

By default, this pipeline uses `--split video_block`, which cuts out contiguous blocks in frame-number order as the test set, and excludes `--gap` frames around that block from the training set. You can also run the traditional random split for comparison using `--split random` (to see how much this overestimates performance).

## Setup (local / qml_endo environment)

```bash
conda activate qml_endo   # The environment used previously for EBTC. If it doesn't exist, create it with:
# conda create -n qml_endo python=3.10 -y && conda activate qml_endo

pip install qiskit qiskit-machine-learning qiskit-aer \
            torch torchvision scikit-learn \
            numpy pandas pillow
```

## Data preparation

This folder already includes the following:
qml_kidney_stone/
├── instances_default.json                      # COCO annotation
├── images/kidney_stone/kidney video.jpg/       # 294 frames (jpg)
├── data_loader.py
├── models.py
├── run_qsvm.py
└── README.md  (you are here now)
```

Note that the image folder path contains a space (the directory is literally named `kidney video.jpg` — this is preserved from the original zip structure). Be sure to quote it when passing it on the command line.

## How to run

```bash
cd qml_kidney_stone

# 1. First, a smoke test that doesn't require network access (verify pipeline works with synthetic data)
python run_qsvm.py --synthetic --n_qubits 4 --max_samples 40

# 2. Run on real data (video-block split, AerSimulator, 80 samples, 4 qubits)
python run_qsvm.py \
    --coco_json instances_default.json \
    --image_dir "images/kidney_stone/kidney video.jpg" \
    --n_qubits 4 --max_samples 80 \
    --backend aer --shots 1024 \
    --split video_block

# 3. For comparison: check how much accuracy is "inflated" with a random split
python run_qsvm.py \
    --coco_json instances_default.json \
    --image_dir "images/kidney_stone/kidney video.jpg" \
    --n_qubits 4 --max_samples 80 \
    --backend aer --shots 1024 \
    --split random
```

On the first run, `torchvision.models.resnet18(weights=...)` will download the ImageNet weights (~45MB). Subsequent runs will read from the cache (`~/.cache/torch/hub/checkpoints/`), so they'll be faster.

## Overall pipeline

```
294 kidney stone video frames (COCO annotations)
      │
      ▼
Labeling: binarize based on presence/absence of annotation
            present (n=250) / absent (n=44)
      │
      ▼
balanced_sample(): use all "absent" samples, subsample "present" to roughly match
→ approximately max_samples worth of balanced data
      │
      ▼
ResNet-18 (frozen, ImageNet pretrained)   512-dimensional feature extraction
      │
      ▼
StandardScaler + PCA                       → n_qubits dimensions (default 4)
      │
      ▼
MinMaxScaler [-π, π]                       normalize to a range suitable for angle encoding
      │
      ▼
video_block_split()                        train/test split that avoids temporal leakage
      │
      ▼
ZZFeatureMap → FidelityQuantumKernel        quantum kernel K(x_i,x_j) = |<ψ(x_i)|ψ(x_j)>|²
      │
      ▼
SVC(kernel='precomputed')                  classical SVM
      │
      ▼
results/qsvm_metrics.csv                   Accuracy / F1 / ROC-AUC / compute time
```

## Main parameters

| Argument | Description | Default |
|---|---|---|
| `--n_qubits` | Number of qubits (dimensionality after PCA) | 4 |
| `--reps` | Number of ZZFeatureMap repetitions | 2 |
| `--max_samples` | Total number of samples used (present + absent combined). Since QSVM is O(N²), 60–100 is the practical upper limit | 80 |
| `--backend` | `aer` (no noise but has shot noise) or `statevector` (fully exact) | aer |
| `--shots` | Number of shots for AerSimulator | 1024 |
| `--split` | `video_block` (recommended) or `random` (for comparison) | video_block |
| `--gap` | Number of frames to exclude around the test block during video_block split | 5 |

## Known limitations

- Since the absent class only has 44 frames, even increasing `max_samples` caps the balanced set at a maximum of 88 frames (44+44). Beyond that, only the present side grows, causing imbalance.
- Because QSVM kernel matrix computation requires `O(N²)` quantum circuit evaluations, increasing `max_samples` beyond 100 causes a sharp slowdown (same constraint as the previous EBTC work).
- Keep in mind that the "stone present/absent" classification is likely, in practice, detecting "whether the camera is capturing the stone" rather than "whether a stone is present" per se — this is closer to an endoscopy frame composition recognition task than a clinical disease diagnosis task.
