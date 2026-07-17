# QML Endoscopy - EBTC Publication Benchmark

This folder now contains a leakage-controlled benchmark path for the
Endoscopic Bladder Tissue Classification (EBTC) dataset, intended for
SPIE-style abstract/manuscript preparation.

## Publication-Grade EBTC Workflow

Use `experiments/publication_benchmark_ebtc.py` for results that are suitable
for reporting. It fixes the main evaluation risks in the exploratory scripts:

- train/test split happens before model preprocessing decisions
- `StandardScaler`, `PCA`, and `MinMaxScaler` are fit on the training split only
- optional class balancing is applied to the training split only
- classical baselines are evaluated on the same low-dimensional feature space
- quantum kernels are scored with kernel-target alignment and concentration
  diagnostics
- a projected-kernel SVM (`pqk`) is included as a lightweight test of whether
  local-observable/projected kernels avoid plain fidelity-kernel failure modes
- outputs include CSV metrics, JSON metadata with split/sample IDs, and an
  optional comparison figure

### Smoke Test

```bash
python experiments/publication_benchmark_ebtc.py \
  --synthetic \
  --models classical \
  --fast \
  --no_plot
```

### Real EBTC Run

Expected data layout:

```text
data/EBTC/
  HGC/*.png
  LGC/*.png
  NST/*.png
  NTL/*.png
```

The public dataset can be downloaded from the Zenodo record cited by the
Kaggle/README dataset page:

```bash
mkdir -p EndoscopicBladderTissue/dataset
curl -L 'https://zenodo.org/records/7741476/files/baldder_tissue_classification.zip?download=1' \
  -o EndoscopicBladderTissue/dataset/baldder_tissue_classification.zip
unzip EndoscopicBladderTissue/dataset/baldder_tissue_classification.zip \
  -d EndoscopicBladderTissue/dataset
```

This extracts:

```text
EndoscopicBladderTissue/dataset/baldder_tissue_classification/
  HGC/*.png
  LGC/*.png
  NST/*.png
  NTL/*.png
  annotations.csv
```

Run classical baselines plus the 4-qubit and 6-qubit QSVM variants:

```bash
python experiments/publication_benchmark_ebtc.py \
  --data_dir data/EBTC \
  --models classical qsvm_v1 qsvm_v2 \
  --max_samples 300 \
  --max_train_samples 120 \
  --max_test_samples 80
```

For faster early checks:

```bash
python experiments/publication_benchmark_ebtc.py \
  --data_dir data/EBTC \
  --models classical \
  --fast \
  --no_plot
```

Outputs are written to:

- `results/publication_benchmark/ebtc_publication_metrics.csv`
- `results/publication_benchmark/ebtc_publication_aggregate.csv`
- `results/publication_benchmark/ebtc_kernel_diagnostics.csv`
- `results/publication_benchmark/ebtc_publication_metadata.json`
- `results/publication_benchmark/ebtc_publication_comparison.png`

### Recommended SPIE Experiments

For publishable claims, prefer repeated splits and low-data curves over a
single train/test split. This reports mean and standard deviation by model and
training-set size:

```bash
python experiments/publication_benchmark_ebtc.py \
  --data_dir data/EBTC \
  --models classical qsvm_v1 qsvm_v2 pqk \
  --max_samples 300 \
  --max_test_samples 80 \
  --train_sizes 20 40 80 120 \
  --repeats 5 \
  --classical_grid \
  --qsvm_grid \
  --no_plot
```

Use the aggregate CSV to identify the strongest setting:

```text
results/publication_benchmark/ebtc_publication_aggregate.csv
```

Suggested reporting table columns:

- model
- training-set size
- accuracy mean +/- std
- balanced accuracy mean +/- std
- F1 mean +/- std
- ROC-AUC mean +/- std
- training time mean +/- std

Suggested kernel-diagnostics table columns:

- model
- training-set size
- kernel-target alignment mean +/- std
- off-diagonal kernel mean +/- std
- off-diagonal kernel standard deviation mean +/- std

Interpretation:

- Higher kernel-target alignment usually indicates a kernel more compatible
  with the class labels.
- Very high off-diagonal mean with very low off-diagonal standard deviation can
  indicate kernel concentration, where most samples look too similar.
- Very low off-diagonal mean can indicate near-identity behavior, where the
  model may memorize but generalize poorly.

If Qiskit is not installed in the active Python environment, the script records
the missing dependency in the metrics CSV instead of failing the whole run.
Use the team's `qml_endo` environment, or install:

```bash
pip install qiskit qiskit-machine-learning qiskit-aer
```

### README-Focused HGC vs LGC Run

The endoscopy README task is high-grade vs low-grade cancer tissue, so use
`--label_mode hgc_vs_lgc` when comparing against that project direction:

```bash
TORCH_HOME=/private/tmp/torch_home python3 EndoscopicBladderTissue/experiments/publication_benchmark_ebtc.py \
  --data_dir EndoscopicBladderTissue/dataset/baldder_tissue_classification \
  --label_mode hgc_vs_lgc \
  --models classical pqk \
  --max_samples 500 \
  --max_test_samples 120 \
  --train_sizes 40 80 160 240 \
  --repeats 3 \
  --classical_grid \
  --qsvm_grid \
  --no_plot
```

Best observed AUC by training-set size:

| Train size | Best PQK AUC | Best classical AUC | Interpretation |
|---:|---:|---:|---|
| 40 | 0.852 | 0.835 | PQK leads in the lowest-label regime. |
| 80 | 0.858 | 0.857 | PQK and RBF SVM are essentially tied. |
| 160 | 0.907 | 0.918 | Classical RBF SVM leads with more labels. |
| 240 | 0.900 | 0.921 | Classical RBF SVM leads with more labels. |

This supports a cautious SPIE claim: projected quantum kernels are competitive
and may be most useful for low-label HGC/LGC discrimination, but they do not
currently beat strong classical kernels once more labeled data is available.

## Legacy Exploratory Notes

# QML Endoscopy — Kvasir-SEG Benchmark

Quantum Machine Learning による内視鏡画像（ポリープ）分類の比較実験フレームワーク。

## 実装モデル（4種）

| # | モデル | 特徴 |
|---|--------|------|
| 1 | **QSVM** | ZZFeatureMap の量子カーネルを SVC に適用。パラメータなし。 |
| 2 | **VQC** | ZZFeatureMap + RealAmplitudes ansatz。COBYLA 最適化。 |
| 3 | **QKernel+SVM** | TrainableFidelityQuantumKernel（KTA で訓練可能）+ sklearn SVC。 |
| 4 | **QNN** | EstimatorQNN → TorchConnector → PyTorch Linear head。Adam 最適化。 |

## パイプライン全体像

```
Kvasir-SEG images
      │
      ▼
ResNet-18 (frozen)        512-dim feature extraction
      │
      ▼
StandardScaler + PCA      → n_qubits 次元 (e.g. 4)
      │
      ▼
MinMaxScaler [-π, π]      angle encoding に適した範囲に正規化
      │
      ├──▶ QSVM
      ├──▶ VQC
      ├──▶ QKernel+SVM
      └──▶ QNN
```

> **なぜPCAで次元削減するのか？**  
> 現状の量子ハードウェア／シミュレータでは qubit 数が限られており (4〜8 qubits が現実的)、
> 512次元をそのまま量子回路に入力できない。
> クラシカル CNN で特徴抽出してから PCA で圧縮するのが、現在の QML 研究の標準的アプローチ。

## セットアップ

```bash
pip install qiskit qiskit-machine-learning qiskit-aer \
            torch torchvision scikit-learn \
            numpy pandas matplotlib pillow tqdm
```

## データセット準備

```
# Kvasir-SEG を https://datasets.simula.no/kvasir-seg/ からダウンロード
unzip Kvasir-SEG.zip -d data/
# 以下の構造になるように:
# data/Kvasir-SEG/images/*.jpg
# data/Kvasir-SEG/masks/*.jpg
```

## 実行

```bash
# 実データで実行 (200サンプル, 4 qubits)
python benchmark.py --data_dir data/Kvasir-SEG --n_qubits 4 --max_samples 200

# Kvasir-SEGなしで合成データを使ったスモークテスト
python benchmark.py --synthetic --n_qubits 4 --max_samples 80

# 高速テスト (反復回数を減らす)
python benchmark.py --synthetic --fast
```

## 出力

- `results/metrics.csv`            — 全モデルの Accuracy / F1 / AUC / 学習時間
- `results/benchmark_results.png`  — バーチャートと混同行列の比較図

## ファイル構成

```
qml_endoscopy/
├── data_loader.py   # Kvasir-SEG 読み込み + CNN特徴抽出 + PCA
├── models.py        # QSVM / VQC / QKernel+SVM / QNN の実装
├── benchmark.py     # 全モデル比較・可視化
└── README.md
```

## 設計上のポイント

### ラベル設計
マスク画像のピクセル被覆率でバイナリ分類:
- **Class 1 (polyp-rich)**:  マスク比率 ≥ 2%
- **Class 0 (polyp-sparse)**: マスク比率 < 2%

### 量子回路設計 (QNN)
```
Ry(x_0) Ry(x_1) ... Ry(x_n)     ← angle encoding
Rz(w_0) Rz(w_1) ... Rz(w_n)     ┐
CX(0,1) CX(1,2) ... CX(n-1,n)   ┘ × reps (variational layer)
    ↓
⟨Z⊗I⊗...⟩  (observable)  →  TorchConnector  →  Linear(1→2)  →  CrossEntropy
```

### スケーリング
`--max_samples` を減らすと速く動くが精度は下がる。
実研究では量子優位性の主張には慎重であること
（現状では量子カーネル法でも classical SVM と同等程度）。

## 既知の制限

- `StatevectorSampler` / `StatevectorEstimator` は完全なシミュレータなので、
  実機ノイズは含まれない。ノイズ実験には `AerSimulator` + noise model を使用。
- n_qubits=4 での `QSVM` はカーネル行列計算に O(N²) の量子回路実行が必要なため
  サンプル数が増えると非常に遅くなる。
- QML for classical data における量子優位性は未確立。本コードはあくまで
  実験・比較のためのフレームワーク。
