# Quantum Machine Learning for Medical Image Classification

> Summer research project applying quantum machine learning (QML) to MRI/CT image classification using NISQ-era quantum devices.

## Overview

This project investigates whether quantum machine learning models can offer advantages over classical approaches in medical image analysis. The focus is on encoding high-dimensional MRI/CT data into quantum states and classifying them using variational quantum circuits (VQC).

## Research Questions

- Can amplitude encoding efficiently represent MRI/CT image data on near-term quantum hardware?
- Do quantum neural networks (QNN) achieve competitive accuracy against classical CNNs on medical datasets?
- How does noise from NISQ devices affect classification performance?

## Approach

```
MRI/CT Data → Preprocessing → Amplitude Encoding → VQC → Classification
```

1. **Data Preprocessing** — Normalize and dimensionality-reduce image data (PCA) to fit qubit constraints
2. **Amplitude Encoding** — Encode pixel intensities into quantum state amplitudes
3. **Variational Quantum Circuit (VQC)** — Parameterized ansatz trained via gradient descent
4. **Measurement & Classification** — Map expectation values to class labels

## File Structure

```
quantum-medical-research/
│
├── README.md
├── requirements.txt
│
├── medical_imaging/
│   ├── preprocessing/
│   │   ├── normalize.py          # Image normalization & resizing
│   │   └── pca_reduction.py      # Dimensionality reduction for encoding
│   │
│   ├── encoding/
│   │   ├── amplitude_encoding.py # Amplitude encoding circuits (Qiskit)
│   │   └── angle_encoding.py     # Angle encoding baseline
│   │
│   ├── models/
│   │   ├── vqc.py                # Variational Quantum Classifier
│   │   ├── qnn.py                # Quantum Neural Network
│   │   └── classical_baseline.py # CNN baseline for comparison
│   │
│   └── evaluation/
│       ├── metrics.py            # Accuracy, AUC, confusion matrix
│       └── noise_analysis.py     # NISQ noise effect analysis
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_encoding_experiments.ipynb
│   └── 03_vqc_training.ipynb
│
├── results/
│   ├── figures/
│   └── benchmarks/
│
└── docs/
    ├── references.md             # Papers & resources
    └── notes/                    # Weekly research notes
```

## Tech Stack

| Area | Tools |
|------|-------|
| Quantum Framework | Qiskit, Qiskit Machine Learning |
| Classical ML | PyTorch, scikit-learn |
| Medical Imaging | SimpleITK, nibabel, pydicom |
| Data | TCIA, MedMNIST |

## Getting Started

```bash
git clone https://github.com/<your-username>/quantum-medical-research.git
cd quantum-medical-research
pip install -r requirements.txt
```

## References

- Biamonte et al. (2017) — Quantum Machine Learning
- LaRose & Coyle (2020) — Robust data encodings for quantum classifiers
- MedMNIST — [medmnist.com](https://medmnist.com)

## Author

Koki — NYU Tandon, ECE & Applied Physics  
Summer Research 2025
