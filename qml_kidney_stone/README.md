# QML Kidney Stone — QSVM Benchmark (Stone Present / Absent)

CVATでアノテーションされたkidney stone内視鏡フレーム（COCO instance segmentation export）を使った
QML分類実験。前回のKvasir-SEG / EBTC QMLベンチマークと同じ設計思想（frozen ResNet-18特徴抽出 → PCA → angle encoding → 量子モデル）を踏襲しています。今回はまずQSVM（一番速いモデル）のみを実装しています。

## ⚠️ 重要：このパイプラインはネットワークが必要です

ResNet-18のImageNet事前学習済み重みを `torchvision` が初回実行時に
`download.pytorch.org` からダウンロードします。Anthropicのサンドボックス環境はこのドメインへの
アクセスがブロックされているため、**Kokiさんのローカル環境（ネットワーク制限なし）で実行してください**。
前回EBTCで使った conda env `qml_endo` がそのまま使えるはずです。

## データセット概要

| 項目 | 値 |
|---|---|
| フレーム数 | 294枚（1本の内視鏡動画から抽出） |
| アノテーション形式 | COCO instance segmentation（CVAT export, `instances_default.json`） |
| カテゴリ | "Kidney Stone" 1クラスのみ |
| アノテーションあり（stone present） | 250枚 |
| アノテーションなし（stone absent） | 44枚 |

### ⚠️ ラベルの時間的クラスタリングについて

absent（アノテーションなし）の44枚は動画内でランダムに散らばっているわけではなく、
**連続した時間ブロック**を形成しています（例：frame 071〜077が連続してabsent）。
これは「結石が視野外に出る/カメラが動く」区間がまとまって発生するためです。

このため、**ランダムにtrain/testを分割すると、ほぼ同じ見た目の隣接フレームが
trainとtestの両方に入ってしまい、タスクが不自然に簡単になります**（リーケージ）。

このパイプラインはデフォルトで `--split video_block` を使い、フレーム番号順に
連続したブロックをtestセットとして切り出し、その周囲を `--gap` フレームぶん
trainから除外します。`--split random` で比較用に従来のランダム分割も実行できます
（過大評価がどの程度起きるか確認する目的で）。

## セットアップ（ローカル / qml_endo環境）

```bash
conda activate qml_endo   # 前回EBTCで使った環境。無ければ下記で新規作成:
# conda create -n qml_endo python=3.10 -y && conda activate qml_endo

pip install qiskit qiskit-machine-learning qiskit-aer \
            torch torchvision scikit-learn \
            numpy pandas pillow
```

## データの準備

このフォルダには既に以下が含まれています：

```
qml_kidney_stone/
├── instances_default.json                      # COCOアノテーション
├── images/kidney_stone/kidney video.jpg/       # 294枚のフレーム (jpg)
├── data_loader.py
├── models.py
├── run_qsvm.py
└── README.md  (このファイル)
```

画像フォルダのパスにスペースが含まれている点に注意してください
（`kidney video.jpg` というディレクトリ名。元のzip構造のままです）。
コマンドラインで渡す際は必ずクォートしてください。

## 実行方法

```bash
cd qml_kidney_stone

# 1. まずネットワーク不要のスモークテスト（合成データでパイプラインの動作確認）
python run_qsvm.py --synthetic --n_qubits 4 --max_samples 40

# 2. 実データで実行（video-block split、AerSimulator、80サンプル、4 qubits）
python run_qsvm.py \
    --coco_json instances_default.json \
    --image_dir "images/kidney_stone/kidney video.jpg" \
    --n_qubits 4 --max_samples 80 \
    --backend aer --shots 1024 \
    --split video_block

# 3. 比較用：ランダムsplitでどれだけ精度が"盛られる"か確認
python run_qsvm.py \
    --coco_json instances_default.json \
    --image_dir "images/kidney_stone/kidney video.jpg" \
    --n_qubits 4 --max_samples 80 \
    --backend aer --shots 1024 \
    --split random
```

初回実行時、`torchvision.models.resnet18(weights=...)` がImageNet重み（~45MB）を
ダウンロードします。2回目以降はキャッシュ（`~/.cache/torch/hub/checkpoints/`）から読まれるので速いです。

## パイプライン全体像

```
294 kidney stone video frames (COCO annotations)
      │
      ▼
ラベル付け: アノテーション有無で2値化
   present (n=250) / absent (n=44)
      │
      ▼
balanced_sample(): absentを全部使い、presentを同数程度サブサンプル
   → 約 max_samples 件のバランスデータ
      │
      ▼
ResNet-18 (frozen, ImageNet pretrained)   512次元特徴抽出
      │
      ▼
StandardScaler + PCA                       → n_qubits 次元 (デフォルト4)
      │
      ▼
MinMaxScaler [-π, π]                       angle encoding に適した範囲に正規化
      │
      ▼
video_block_split()                        時間的リーケージを避けるtrain/test分割
      │
      ▼
ZZFeatureMap → FidelityQuantumKernel        量子カーネル K(x_i,x_j) = |<ψ(x_i)|ψ(x_j)>|²
      │
      ▼
SVC(kernel='precomputed')                  classical SVM
      │
      ▼
results/qsvm_metrics.csv                   Accuracy / F1 / ROC-AUC / 計算時間
```

## 主なパラメータ

| 引数 | 説明 | デフォルト |
|---|---|---|
| `--n_qubits` | qubit数（PCA後の次元数） | 4 |
| `--reps` | ZZFeatureMapの繰り返し回数 | 2 |
| `--max_samples` | 使用する合計サンプル数（present+absent合計）。QSVMはO(N²)なので60〜100が実用上限 | 80 |
| `--backend` | `aer`（ノイズ無しだがshotノイズあり）or `statevector`（完全厳密） | aer |
| `--shots` | AerSimulatorのショット数 | 1024 |
| `--split` | `video_block`（推奨）or `random`（比較用） | video_block |
| `--gap` | video_block split時にtestブロック周辺で除外するフレーム数 | 5 |

## 既知の制限

- absentクラスが44枚しかないため、`max_samples` を増やしてもバランスは
  最大88枚（44+44）が上限です。それ以上はpresent側のみ増えて不均衡になります。
- QSVMのカーネル行列計算は `O(N²)` の量子回路評価が必要なため、
  `max_samples` を100超に増やすと急激に遅くなります（前回のEBTCと同じ制約）。
- 「stone present/absent」の分類は「結石の有無」というよりも実質的に
  「カメラが結石を写しているフレームか否か」を見ている可能性が高く、
  臨床的な疾患診断タスクというよりは内視鏡フレームの構成認識タスクに近い点に
  留意してください。
