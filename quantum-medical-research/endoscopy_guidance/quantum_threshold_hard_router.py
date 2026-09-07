"""
quantum_threshold_hard_router.py
================================
Hard-case routed compact threshold selection.

This experiment keeps the high-precision compact random-forest selector as the
default policy, then routes likely hard frames to a compact quantum threshold
selector. The route decision is learned from inference-safe probability-map
features on the training split and tuned on validation.
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

from quantum_mask_hypothesis_selector import QuantumClassifier, load_frame_table, selected_dice, split_arrays
from quantum_threshold_pairwise_ranker import candidate_features, evaluate, selector_score


def positive_proba(model, X: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(X)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = list(model[-1].classes_)
    return probs[:, classes.index(1)]


def choose_pairwise_thresholds(model, X_eval: np.ndarray, rows: list[dict], n_frames: int, thresholds: list[float], min_score: float) -> np.ndarray:
    scores = positive_proba(model, X_eval)
    fixed_idx = thresholds.index(0.5)
    selected = np.full(n_frames, fixed_idx, dtype=np.int64)
    best_scores = np.full(n_frames, min_score, dtype=np.float32)
    for row, score in zip(rows, scores):
        frame_idx = int(row["frame_idx"])
        threshold_idx = int(row["threshold_idx"])
        if threshold_idx == fixed_idx:
            continue
        if score > best_scores[frame_idx]:
            best_scores[frame_idx] = float(score)
            selected[frame_idx] = threshold_idx
    return selected


def make_selector(name: str, components: int, reps: int, seed: int):
    if name == "classical_logistic_selector":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    if name == "classical_histgb_selector":
        return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, random_state=seed)
    if name == "classical_rf_selector":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    if name == "projected_quantum_logistic":
        return QuantumClassifier(components, reps, seed, hybrid=False, head="logistic")
    if name == "hybrid_quantum_logistic":
        return QuantumClassifier(components, reps, seed, hybrid=True, head="logistic")
    raise ValueError(f"Unknown selector={name}")


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
    parser = argparse.ArgumentParser(description="Hard-routed compact quantum threshold selector")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test")
    parser.add_argument("--external_baseline_dir", default="")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/quantum_threshold_hard_router_kvasir.csv")
    parser.add_argument("--per_frame_csv", default="")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.50, 0.90])
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--tune_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--external_test_split", default="all")
    parser.add_argument("--proposer_min_score_grid", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    parser.add_argument("--router_score_grid", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--selection_objective", choices=["overall", "hard", "combined"], default="overall")
    parser.add_argument("--selectors", nargs="+", default=[
        "classical_logistic_selector",
        "classical_histgb_selector",
        "classical_rf_selector",
        "projected_quantum_logistic",
        "hybrid_quantum_logistic",
    ])
    parser.add_argument("--pqk_components", type=int, default=8)
    parser.add_argument("--pqk_reps", type=int, default=3)
    parser.add_argument("--hard_dice_threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds)
    frame = load_frame_table(Path(args.baseline_dir), thresholds)
    X_train, y_train, train_df = split_arrays(frame, args.train_split)
    X_tune, _, tune_df = split_arrays(frame, args.tune_split)
    if args.external_baseline_dir:
        external_frame = load_frame_table(Path(args.external_baseline_dir), thresholds)
        if args.external_test_split == "all":
            test_df = external_frame.copy().reset_index(drop=True)
            X_test = np.asarray(test_df["features"].tolist(), dtype=np.float32)
        elif args.external_test_split in set(external_frame["split"]):
            X_test, _, test_df = split_arrays(external_frame, args.external_test_split)
        else:
            raise ValueError(f"external_test_split={args.external_test_split!r} not found")
    else:
        X_test, _, test_df = split_arrays(frame, args.test_split)

    pair_X_train, pair_y_train, train_rows = candidate_features(train_df, thresholds, "legacy_candidate")
    pair_X_tune, _, tune_rows = candidate_features(tune_df, thresholds, "legacy_candidate")
    pair_X_test, _, test_rows = candidate_features(test_df, thresholds, "legacy_candidate")
    proposer = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    proposer.fit(pair_X_train, pair_y_train)

    proposer_best = None
    for min_score in args.proposer_min_score_grid:
        tune_selected = choose_pairwise_thresholds(proposer, pair_X_tune, tune_rows, len(tune_df), thresholds, min_score)
        row = evaluate("compact_rf_proposer", tune_df, tune_selected, thresholds, args.hard_dice_threshold)
        score = selector_score(row, args.selection_objective)
        if proposer_best is None or score > proposer_best[0]:
            proposer_best = (score, min_score)
    proposer_min_score = float(proposer_best[1])
    tune_proposer = choose_pairwise_thresholds(proposer, pair_X_tune, tune_rows, len(tune_df), thresholds, proposer_min_score)
    test_proposer = choose_pairwise_thresholds(proposer, pair_X_test, test_rows, len(test_df), thresholds, proposer_min_score)

    hard_train = (train_df["baseline_dice"].to_numpy(dtype=float) < args.hard_dice_threshold).astype(np.int64)
    router = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, random_state=args.seed)
    router.fit(X_train, hard_train)
    tune_router_scores = positive_proba(router, X_tune)
    test_router_scores = positive_proba(router, X_test)

    results = []
    frame_outputs = []
    fixed = np.full(len(test_df), thresholds.index(0.5), dtype=np.int64)
    results.append(evaluate("fixed_threshold_0.50", test_df, fixed, thresholds, args.hard_dice_threshold))
    frame_outputs.append(frame_rows("fixed_threshold_0.50", test_df, fixed, thresholds))
    oracle = test_df["best_threshold_index"].to_numpy(dtype=np.int64)
    results.append(evaluate("compact_oracle_threshold", test_df, oracle, thresholds, args.hard_dice_threshold))
    frame_outputs.append(frame_rows("compact_oracle_threshold", test_df, oracle, thresholds))
    proposer_row = evaluate("compact_rf_proposer", test_df, test_proposer, thresholds, args.hard_dice_threshold)
    proposer_row["proposer_min_score"] = proposer_min_score
    results.append(proposer_row)
    frame_outputs.append(frame_rows("compact_rf_proposer", test_df, test_proposer, thresholds))

    for selector_name in args.selectors:
        selector = make_selector(selector_name, args.pqk_components, args.pqk_reps, args.seed)
        selector.fit(X_train, y_train)
        tune_selector = selector.predict(X_tune)
        test_selector = selector.predict(X_test)
        best = None
        for router_threshold in args.router_score_grid:
            tune_routed = tune_proposer.copy()
            route_mask = tune_router_scores >= router_threshold
            tune_routed[route_mask] = tune_selector[route_mask]
            row = evaluate(selector_name, tune_df, tune_routed, thresholds, args.hard_dice_threshold)
            score = selector_score(row, args.selection_objective)
            if best is None or score > best[0]:
                best = (score, router_threshold)
        router_threshold = float(best[1])
        test_routed = test_proposer.copy()
        route_mask = test_router_scores >= router_threshold
        test_routed[route_mask] = test_selector[route_mask]
        model_name = f"hard_router_{selector_name}"
        row = evaluate(model_name, test_df, test_routed, thresholds, args.hard_dice_threshold)
        row["proposer_min_score"] = proposer_min_score
        row["router_score_threshold"] = router_threshold
        row["routed_frames"] = int(route_mask.sum())
        results.append(row)
        frame_outputs.append(frame_rows(model_name, test_df, test_routed, thresholds))

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
