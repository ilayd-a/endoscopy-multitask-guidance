"""
quantum_threshold_gated_selector.py
===================================
Two-stage threshold selector with a projected quantum accept/reject gate.

Stage 1 proposes a clinically interpretable threshold from a compact hypothesis
set (default: under-segment rescue 0.30, fixed 0.50, over-segment correction
0.90). Stage 2 learns whether to accept the proposed non-default change or
fall back to the fixed 0.50 mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from quantum_mask_hypothesis_selector import load_frame_table, selected_dice
from quantum_threshold_pairwise_ranker import (
    LowDimProjectedQuantumSVC,
    candidate_features,
    evaluate,
    selector_score,
)


def positive_proba(model, X: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(X)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "model") and hasattr(model.model, "model"):
        classes = list(model.model.model.classes_)
    else:
        classes = list(model[-1].classes_)
    return probs[:, classes.index(1)]


def choose_with_scores(
    model,
    X_eval: np.ndarray,
    rows: list[dict],
    n_frames: int,
    thresholds: list[float],
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = positive_proba(model, X_eval)
    fixed_idx = thresholds.index(0.5)
    selected = np.full(n_frames, fixed_idx, dtype=np.int64)
    gated_scores = np.full(n_frames, min_score, dtype=np.float32)
    raw_best_scores = np.full(n_frames, -1.0, dtype=np.float32)
    for row, score in zip(rows, scores):
        frame_idx = int(row["frame_idx"])
        threshold_idx = int(row["threshold_idx"])
        if threshold_idx == fixed_idx:
            continue
        raw_best_scores[frame_idx] = max(raw_best_scores[frame_idx], float(score))
        if score > gated_scores[frame_idx]:
            gated_scores[frame_idx] = float(score)
            selected[frame_idx] = threshold_idx
    return selected, raw_best_scores


def gate_features(frame: pd.DataFrame, selected: np.ndarray, proposer_scores: np.ndarray, thresholds: list[float]) -> np.ndarray:
    rows = []
    for row_idx, record in frame.iterrows():
        threshold = thresholds[int(selected[row_idx])]
        base_features = np.asarray(record["features"], dtype=np.float32)
        candidate = np.asarray(record[f"features_t{threshold:.2f}"], dtype=np.float32)
        threshold_features = np.asarray(
            [
                threshold,
                abs(threshold - 0.5),
                threshold < 0.5,
                threshold > 0.5,
                proposer_scores[row_idx],
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate([base_features, candidate, candidate - base_features[: len(candidate)], threshold_features]))
    return np.asarray(rows, dtype=np.float32)


def gate_labels(frame: pd.DataFrame, selected: np.ndarray, thresholds: list[float]) -> np.ndarray:
    proposed = selected_dice(frame, selected, thresholds)
    baseline = frame["baseline_dice"].to_numpy(dtype=np.float32)
    return (proposed > baseline + 1e-6).astype(np.int64)


def make_gate_model(name: str, components: int, reps: int, seed: int):
    if name == "projected_quantum_gate":
        return LowDimProjectedQuantumSVC(components, reps, seed)
    if name == "classical_logistic_gate":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    if name == "classical_histgb_gate":
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, random_state=seed)
    if name == "classical_rf_gate":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown gate model={name}")


def gate_selected(selected: np.ndarray, gate_scores: np.ndarray, gate_score_threshold: float, fixed_idx: int) -> np.ndarray:
    gated = selected.copy()
    gated[gate_scores < gate_score_threshold] = fixed_idx
    return gated


def frame_rows(name: str, frame: pd.DataFrame, selected: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    dice = selected_dice(frame, selected, thresholds)
    baseline = frame["baseline_dice"].to_numpy(dtype=np.float32)
    rows = frame[["sample_id", "split", "source_file", "baseline_dice", "oracle_dice", "oracle_threshold"]].copy()
    rows["model"] = name
    rows["selected_threshold"] = [thresholds[int(idx)] for idx in selected]
    rows["selected_dice"] = dice
    rows["delta_dice"] = dice - baseline
    return rows


def main():
    parser = argparse.ArgumentParser(description="Quantum-gated compact threshold selector")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/quantum_threshold_gated_selector_kvasir.csv")
    parser.add_argument("--per_frame_csv", default="")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.50, 0.90])
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--tune_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--proposer_min_score_grid", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--gate_score_grid", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    parser.add_argument("--selection_objective", choices=["overall", "hard", "combined"], default="hard")
    parser.add_argument("--gate_models", nargs="+", default=["projected_quantum_gate", "classical_logistic_gate", "classical_histgb_gate", "classical_rf_gate"])
    parser.add_argument("--pqk_components", type=int, default=8)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--hard_dice_threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds)
    frame = load_frame_table(Path(args.baseline_dir), thresholds)
    train_df = frame.loc[frame["split"].eq(args.train_split)].copy().reset_index(drop=True)
    tune_df = frame.loc[frame["split"].eq(args.tune_split)].copy().reset_index(drop=True)
    test_df = frame.loc[frame["split"].eq(args.test_split)].copy().reset_index(drop=True)

    X_train, y_train, train_rows = candidate_features(train_df, thresholds, "legacy_candidate")
    X_tune, _, tune_rows = candidate_features(tune_df, thresholds, "legacy_candidate")
    X_test, _, test_rows = candidate_features(test_df, thresholds, "legacy_candidate")

    proposer = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    proposer.fit(X_train, y_train)

    proposer_best = None
    for min_score in args.proposer_min_score_grid:
        tune_selected, _ = choose_with_scores(proposer, X_tune, tune_rows, len(tune_df), thresholds, min_score)
        row = evaluate("compact_rf_proposer", tune_df, tune_selected, thresholds, args.hard_dice_threshold)
        score = selector_score(row, args.selection_objective)
        if proposer_best is None or score > proposer_best[0]:
            proposer_best = (score, min_score)
    proposer_min_score = float(proposer_best[1])

    train_selected, train_scores = choose_with_scores(proposer, X_train, train_rows, len(train_df), thresholds, proposer_min_score)
    tune_selected, tune_scores = choose_with_scores(proposer, X_tune, tune_rows, len(tune_df), thresholds, proposer_min_score)
    test_selected, test_scores = choose_with_scores(proposer, X_test, test_rows, len(test_df), thresholds, proposer_min_score)

    gate_X_train = gate_features(train_df, train_selected, train_scores, thresholds)
    gate_y_train = gate_labels(train_df, train_selected, thresholds)
    gate_X_tune = gate_features(tune_df, tune_selected, tune_scores, thresholds)
    gate_X_test = gate_features(test_df, test_selected, test_scores, thresholds)

    results = []
    frame_outputs = []
    fixed = np.full(len(test_df), thresholds.index(0.5), dtype=np.int64)
    results.append(evaluate("fixed_threshold_0.50", test_df, fixed, thresholds, args.hard_dice_threshold))
    frame_outputs.append(frame_rows("fixed_threshold_0.50", test_df, fixed, thresholds))
    oracle = test_df["best_threshold_index"].to_numpy(dtype=np.int64)
    results.append(evaluate("compact_oracle_threshold", test_df, oracle, thresholds, args.hard_dice_threshold))
    frame_outputs.append(frame_rows("compact_oracle_threshold", test_df, oracle, thresholds))
    proposer_row = evaluate("compact_rf_proposer", test_df, test_selected, thresholds, args.hard_dice_threshold)
    proposer_row["proposer_min_score"] = proposer_min_score
    results.append(proposer_row)
    frame_outputs.append(frame_rows("compact_rf_proposer", test_df, test_selected, thresholds))

    for model_name in args.gate_models:
        gate = make_gate_model(model_name, args.pqk_components, args.pqk_reps, args.seed)
        gate.fit(gate_X_train, gate_y_train)
        tune_gate_scores = positive_proba(gate, gate_X_tune)
        test_gate_scores = positive_proba(gate, gate_X_test)
        best = None
        for gate_score_threshold in args.gate_score_grid:
            tune_gated = gate_selected(tune_selected, tune_gate_scores, gate_score_threshold, thresholds.index(0.5))
            row = evaluate(model_name, tune_df, tune_gated, thresholds, args.hard_dice_threshold)
            score = selector_score(row, args.selection_objective)
            if best is None or score > best[0]:
                best = (score, gate_score_threshold)
        gate_score_threshold = float(best[1])
        test_gated = gate_selected(test_selected, test_gate_scores, gate_score_threshold, thresholds.index(0.5))
        row = evaluate(model_name, test_df, test_gated, thresholds, args.hard_dice_threshold)
        row["proposer_min_score"] = proposer_min_score
        row["gate_score_threshold"] = gate_score_threshold
        results.append(row)
        frame_outputs.append(frame_rows(model_name, test_df, test_gated, thresholds))

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(results).sort_values(["selected_dice", "hard_selected_dice"], ascending=False)
    summary.to_csv(output, index=False)
    per_frame = Path(args.per_frame_csv) if args.per_frame_csv else output.with_name(output.stem + "_per_frame.csv")
    pd.concat(frame_outputs, ignore_index=True).to_csv(per_frame, index=False)
    print(f"[saved] {output}")
    print(f"[saved] {per_frame}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
