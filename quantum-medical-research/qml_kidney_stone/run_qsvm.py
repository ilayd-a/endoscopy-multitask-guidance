"""
run_qsvm.py — Kidney Stone QSVM benchmark, end-to-end.

Usage:
    python run_qsvm.py \
        --merged_json ../training_data/merged_instances.json \
        --image_dir ../images \
        --n_qubits 4 --max_train_samples 100 --max_test_samples 40 \
        --backend aer --shots 1024

    # quick smoke test with synthetic data, no images/network needed:
    python run_qsvm.py --synthetic --n_qubits 4 --max_train_samples 60 --max_test_samples 24

Notes:
    - ResNet-18 feature extraction requires internet access (downloads ImageNet
      pretrained weights from download.pytorch.org on first run). Run this on a
      machine with normal network access (e.g. your local qml_endo conda env).
    - QSVM kernel computation is O(N^2) circuit evaluations. Keep sample counts
      in the 60-100 range on a laptop, as in the EBTC benchmark.
    - The train/test split is now precomputed per-source-video at merge time
      (see merge_and_split.py) and stored in merged_instances.json as a "split"
      field on each image. This avoids leaking near-duplicate adjacent video
      frames across train/test, since each source video's "absent" frames tend
      to cluster temporally. This script simply reads that split rather than
      recomputing one.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import load_kidney_stone_dataset
from models import run_qsvm


def make_synthetic(n_qubits, n_train, n_test, seed=42):
    """Synthetic separable data for a quick pipeline smoke test (no images needed)."""
    rng = np.random.RandomState(seed)

    def make_split(n):
        n0, n1 = n // 2, n - n // 2
        X0 = rng.uniform(-np.pi, -0.5, size=(n0, n_qubits))
        X1 = rng.uniform(0.5, np.pi, size=(n1, n_qubits))
        X = np.vstack([X0, X1])
        y = np.array([0] * n0 + [1] * n1)
        idx = rng.permutation(n)
        return X[idx], y[idx]

    X_train, y_train = make_split(n_train)
    X_test, y_test = make_split(n_test)
    return X_train, y_train, X_test, y_test


def main():
    p = argparse.ArgumentParser(description="Kidney Stone QSVM benchmark")
    p.add_argument("--merged_json", type=str, default="../training_data/merged_instances.json")
    p.add_argument("--image_dir", type=str, default="../images")
    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--reps", type=int, default=2, help="ZZFeatureMap repetitions")
    p.add_argument("--max_train_samples", type=int, default=100,
                    help="Balanced train samples (present/absent). QSVM is O(N^2), keep <=100.")
    p.add_argument("--max_test_samples", type=int, default=40,
                    help="Balanced test samples (present/absent).")
    p.add_argument("--backend", type=str, default="aer", choices=["aer", "statevector"])
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0, help="SVC regularization strength")
    p.add_argument("--synthetic", action="store_true",
                    help="Skip image loading entirely; run on synthetic separable data as a smoke test.")
    p.add_argument("--results_dir", type=str, default="results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 1. Load data (train/test split is already baked into merged_json)
    # ---------------------------------------------------------------- #
    if args.synthetic:
        print("[main] Using SYNTHETIC data (smoke test mode, no images/network needed)")
        X_train, y_train, X_test, y_test = make_synthetic(
            args.n_qubits, args.max_train_samples, args.max_test_samples, seed=args.seed
        )
    else:
        print("[main] Loading kidney stone dataset (this downloads ResNet-18 "
              "ImageNet weights on first run — requires network access)")
        data = load_kidney_stone_dataset(
            merged_json_path=args.merged_json,
            image_dir=args.image_dir,
            n_qubits=args.n_qubits,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            seed=args.seed,
        )
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

    print(f"[main] train={len(y_train)} test={len(y_test)}")
    print(f"[main] train class balance: {np.bincount(y_train)}  "
          f"test class balance: {np.bincount(y_test)}")

    # ---------------------------------------------------------------- #
    # 2. Run QSVM
    # ---------------------------------------------------------------- #
    result = run_qsvm(
        X_train, y_train, X_test, y_test,
        n_qubits=args.n_qubits, reps=args.reps,
        backend=args.backend, shots=args.shots,
        seed=args.seed, C=args.C,
    )

    # ---------------------------------------------------------------- #
    # 3. Save metrics
    # ---------------------------------------------------------------- #
    df = pd.DataFrame([{
        "Model": result["name"],
        "Accuracy": round(result["accuracy"], 4),
        "F1": round(result["f1"], 4),
        "ROC-AUC": round(result["auc"], 4) if not np.isnan(result["auc"]) else "N/A",
        "Train+Kernel(sec)": round(result["train_time"], 2),
        "n_qubits": args.n_qubits,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "backend": args.backend,
    }])
    csv_path = results_dir / "qsvm_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[main] Metrics saved -> {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())