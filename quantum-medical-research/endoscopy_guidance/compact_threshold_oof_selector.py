"""
compact_threshold_oof_selector.py
=================================
Out-of-fold compact threshold calibration for a fixed segmentation model.

This evaluates whether a post-hoc selector can improve a fixed UNet probability
map without training and testing the selector on the same frames. The default
compact threshold set has an interpretable role in surgical guidance:
0.30 rescues likely under-segmentation, 0.50 preserves the standard mask, and
0.90 corrects likely over-segmentation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from quantum_mask_hypothesis_selector import QuantumClassifier, load_frame_table, selected_dice
from quantum_threshold_pairwise_ranker import evaluate


def make_selector(name: str, components: int, reps: int, seed: int):
    if name == "classical_logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    if name == "classical_histgb":
        return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, random_state=seed)
    if name == "classical_rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    if name == "projected_quantum_logistic":
        return QuantumClassifier(components, reps, seed, hybrid=False, head="logistic")
    if name == "hybrid_quantum_logistic":
        return QuantumClassifier(components, reps, seed, hybrid=True, head="logistic")
    raise ValueError(f"Unknown selector={name}")


def frame_rows(name: str, frame: pd.DataFrame, selected: np.ndarray, thresholds: list[float], fold: int) -> pd.DataFrame:
    dice = selected_dice(frame, selected, thresholds)
    baseline = frame["baseline_dice"].to_numpy(dtype=np.float32)
    rows = frame[["sample_id", "split", "source_file", "baseline_dice", "oracle_dice", "oracle_threshold"]].copy()
    rows["fold"] = fold
    rows["model"] = name
    rows["selected_threshold"] = [thresholds[int(idx)] for idx in selected]
    rows["selected_dice"] = dice
    rows["delta_dice"] = dice - baseline
    return rows


def main():
    parser = argparse.ArgumentParser(description="Out-of-fold compact threshold selector")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_cvc_all")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/compact_threshold_oof_selector.csv")
    parser.add_argument("--per_frame_csv", default="")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.50, 0.90])
    parser.add_argument("--models", nargs="+", default=[
        "classical_logistic",
        "classical_histgb",
        "classical_rf",
        "projected_quantum_logistic",
        "hybrid_quantum_logistic",
    ])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hard_dice_threshold", type=float, default=0.80)
    parser.add_argument("--pqk_components", type=int, default=8)
    parser.add_argument("--pqk_reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds)
    frame = load_frame_table(Path(args.baseline_dir), thresholds).reset_index(drop=True)
    X = np.asarray(frame["features"].tolist(), dtype=np.float32)
    y = frame["best_threshold_index"].to_numpy(dtype=np.int64)
    hard = (frame["baseline_dice"].to_numpy(dtype=float) < args.hard_dice_threshold).astype(np.int64)

    fold_rows = []
    frame_outputs = []
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, hard)):
        train_X = X[train_idx]
        train_y = y[train_idx]
        test_X = X[test_idx]
        test_df = frame.iloc[test_idx].copy().reset_index(drop=True)

        fixed = np.full(len(test_df), thresholds.index(0.5), dtype=np.int64)
        fold_rows.append({**evaluate("fixed_threshold_0.50", test_df, fixed, thresholds, args.hard_dice_threshold), "fold": fold})
        frame_outputs.append(frame_rows("fixed_threshold_0.50", test_df, fixed, thresholds, fold))

        oracle = test_df["best_threshold_index"].to_numpy(dtype=np.int64)
        fold_rows.append({**evaluate("oracle_threshold", test_df, oracle, thresholds, args.hard_dice_threshold), "fold": fold})
        frame_outputs.append(frame_rows("oracle_threshold", test_df, oracle, thresholds, fold))

        for model_name in args.models:
            model = make_selector(model_name, args.pqk_components, args.pqk_reps, args.seed + fold)
            model.fit(train_X, train_y)
            selected = model.predict(test_X).astype(np.int64)
            fold_rows.append({**evaluate(model_name, test_df, selected, thresholds, args.hard_dice_threshold), "fold": fold})
            frame_outputs.append(frame_rows(model_name, test_df, selected, thresholds, fold))

    fold_summary = pd.DataFrame(fold_rows)
    summary = (
        fold_summary.groupby("model", as_index=False)
        .agg(
            selected_dice=("selected_dice", "mean"),
            baseline_dice=("baseline_dice", "mean"),
            delta_dice=("delta_dice", "mean"),
            hard_selected_dice=("hard_selected_dice", "mean"),
            hard_baseline_dice=("hard_baseline_dice", "mean"),
            hard_delta_dice=("hard_delta_dice", "mean"),
            changed_frames=("changed_frames", "sum"),
        )
        .sort_values(["selected_dice", "hard_selected_dice"], ascending=False)
    )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    per_frame = Path(args.per_frame_csv) if args.per_frame_csv else output.with_name(output.stem + "_per_frame.csv")
    pd.concat(frame_outputs, ignore_index=True).to_csv(per_frame, index=False)
    print(f"[saved] {output}")
    print(f"[saved] {per_frame}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
