"""
hard_case_quantum_switch_oof.py
===============================
Out-of-fold evaluation for the hard-case quantum switching policy.

Each Kvasir frame is evaluated by a switch trained without that frame. The
fixed UNet probability map is always the first-stage model; the learned router
decides when to replace the compact classical threshold proposer with a
projected-quantum selector for likely hard cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from hard_case_quantum_switch import choose_pairwise_thresholds, positive_proba
from quantum_mask_hypothesis_selector import QuantumClassifier, load_frame_table, selected_dice
from quantum_threshold_pairwise_ranker import candidate_features, evaluate, selector_score


def frame_feature_matrix(frame: pd.DataFrame, thresholds: list[float], feature_mode: str = "base") -> np.ndarray:
    base = np.asarray(frame["features"].tolist(), dtype=np.float32)
    if feature_mode == "base":
        return base
    if feature_mode != "threshold_stack":
        raise ValueError(f"Unknown selector_feature_mode={feature_mode}")
    stacks = []
    for _, record in frame.iterrows():
        threshold_stack = np.asarray(
            [record[f"features_t{threshold:.2f}"] for threshold in thresholds],
            dtype=np.float32,
        )
        stacks.append(np.concatenate([
            threshold_stack.reshape(-1),
            threshold_stack.mean(axis=0),
            threshold_stack.std(axis=0),
            threshold_stack[0] - threshold_stack[-1],
        ]))
    return np.asarray(stacks, dtype=np.float32)


def safe_positive_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = list(model[-1].classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=np.float32)
    return positive_proba(model, X)


def router_scores(model, X: np.ndarray, route_model: str) -> np.ndarray:
    if route_model == "regressor":
        return model.predict(X).astype(np.float32)
    return safe_positive_proba(model, X)


def fixed_default(frame: pd.DataFrame, thresholds: list[float]) -> tuple[np.ndarray, np.ndarray]:
    selected = np.full(len(frame), thresholds.index(0.5), dtype=np.int64)
    scores = np.full(len(frame), -1.0, dtype=np.float32)
    return selected, scores


def router_feature_matrix(
    frame: pd.DataFrame,
    default_selected: np.ndarray,
    quantum_selected: np.ndarray,
    default_scores: np.ndarray,
    thresholds: list[float],
    feature_mode: str,
) -> np.ndarray:
    base = frame_feature_matrix(frame, thresholds, feature_mode)
    default_threshold = np.asarray([thresholds[int(idx)] for idx in default_selected], dtype=np.float32)
    quantum_threshold = np.asarray([thresholds[int(idx)] for idx in quantum_selected], dtype=np.float32)
    meta = np.column_stack([
        default_threshold,
        quantum_threshold,
        np.abs(default_threshold - quantum_threshold),
        (default_selected != quantum_selected).astype(np.float32),
        default_scores.astype(np.float32),
    ])
    return np.concatenate([base, meta], axis=1).astype(np.float32)


def stratified_inner_split(
    trainval_df: pd.DataFrame,
    hard_labels: np.ndarray,
    tune_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=tune_fraction, random_state=seed)
    train_idx, tune_idx = next(splitter.split(np.zeros(len(trainval_df)), hard_labels))
    train_df = trainval_df.iloc[train_idx].copy().reset_index(drop=True)
    tune_df = trainval_df.iloc[tune_idx].copy().reset_index(drop=True)
    return train_df, tune_df


def select_with_proposer(
    proposer,
    frame: pd.DataFrame,
    thresholds: list[float],
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    X, _, rows = candidate_features(frame, thresholds, "legacy_candidate")
    return choose_pairwise_thresholds(proposer, X, rows, len(frame), thresholds, min_score)


def tune_proposer_min_score(
    proposer,
    tune_df: pd.DataFrame,
    thresholds: list[float],
    min_score_grid: list[float],
    objective: str,
    hard_dice_threshold: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    X_tune, _, tune_rows = candidate_features(tune_df, thresholds, "legacy_candidate")
    best = None
    for min_score in min_score_grid:
        selected, scores = choose_pairwise_thresholds(proposer, X_tune, tune_rows, len(tune_df), thresholds, min_score)
        row = evaluate("compact_rf_proposer", tune_df, selected, thresholds, hard_dice_threshold)
        score = selector_score(row, objective)
        if best is None or score > best[0]:
            best = (score, float(min_score), selected, scores)
    return best[1], best[2], best[3]


def tune_router_threshold(
    tune_df: pd.DataFrame,
    tune_proposer: np.ndarray,
    tune_router_scores: np.ndarray,
    tune_quantum: np.ndarray,
    thresholds: list[float],
    router_score_grid: list[float],
    objective: str,
    hard_dice_threshold: float,
    max_route_rate: float,
    allowed_quantum_idx: set[int] | None = None,
) -> tuple[float, dict]:
    best = None
    no_route_threshold = float(np.max(tune_router_scores) + 1e-6) if len(tune_router_scores) else 1.01
    for router_threshold in list(router_score_grid) + [no_route_threshold]:
        route_mask = tune_router_scores >= router_threshold
        if allowed_quantum_idx is not None:
            route_mask &= np.isin(tune_quantum, list(allowed_quantum_idx))
        if max_route_rate < 1.0 and float(route_mask.mean()) > max_route_rate:
            continue
        selected = tune_proposer.copy()
        selected[route_mask] = tune_quantum[route_mask]
        row = evaluate("hard_case_quantum_switch", tune_df, selected, thresholds, hard_dice_threshold)
        row["routed_frames"] = int(route_mask.sum())
        row["route_rate"] = float(route_mask.mean())
        score = selector_score(row, objective)
        if best is None or score > best[0]:
            best = (score, float(router_threshold), row)
    if best is None:
        raise ValueError("No router threshold satisfied --max_route_rate")
    return best[1], best[2]


def apply_fold(
    fold: int,
    eval_df: pd.DataFrame,
    router,
    router_score_threshold: float,
    quantum_selector,
    thresholds: list[float],
    default_selector: str,
    selector_feature_mode: str,
    route_model: str,
    allowed_quantum_idx: set[int] | None,
    proposer=None,
    proposer_min_score: float | None = None,
) -> pd.DataFrame:
    X_eval = frame_feature_matrix(eval_df, thresholds, selector_feature_mode)
    quantum_selected = quantum_selector.predict(X_eval).astype(np.int64)
    if default_selector == "proposer":
        proposer_selected, proposer_scores = select_with_proposer(proposer, eval_df, thresholds, proposer_min_score)
    else:
        proposer_selected, proposer_scores = fixed_default(eval_df, thresholds)
    X_router = router_feature_matrix(eval_df, proposer_selected, quantum_selected, proposer_scores, thresholds, selector_feature_mode)
    hard_scores = router_scores(router, X_router, route_model)
    route_mask = hard_scores >= router_score_threshold
    if allowed_quantum_idx is not None:
        route_mask &= np.isin(quantum_selected, list(allowed_quantum_idx))
    selected = proposer_selected.copy()
    selected[route_mask] = quantum_selected[route_mask]
    dice = selected_dice(eval_df, selected, thresholds)
    baseline = eval_df["baseline_dice"].to_numpy(dtype=np.float32)

    out = eval_df[["sample_id", "split", "source_file"]].copy()
    out["fold"] = int(fold)
    out["baseline_dice"] = baseline
    out["router_hard_score"] = hard_scores
    out["route_to_quantum"] = route_mask
    out["proposer_score"] = proposer_scores
    out["proposer_threshold"] = [thresholds[int(idx)] for idx in proposer_selected]
    out["quantum_threshold"] = [thresholds[int(idx)] for idx in quantum_selected]
    out["selected_threshold"] = [thresholds[int(idx)] for idx in selected]
    out["selected_dice"] = dice
    out["delta_dice"] = dice - baseline
    out["oracle_dice"] = eval_df["oracle_dice"].to_numpy(dtype=float)
    out["oracle_threshold"] = eval_df["oracle_threshold"].to_numpy(dtype=float)
    return out


def tune_allowed_quantum_thresholds(
    tune_df: pd.DataFrame,
    tune_default: np.ndarray,
    tune_quantum: np.ndarray,
    thresholds: list[float],
    min_count: int,
) -> set[int]:
    default_dice = selected_dice(tune_df, tune_default, thresholds)
    quantum_dice = selected_dice(tune_df, tune_quantum, thresholds)
    gains = quantum_dice - default_dice
    allowed = set()
    for threshold_idx in sorted(set(tune_quantum.tolist())):
        mask = tune_quantum == threshold_idx
        if int(mask.sum()) >= min_count and float(gains[mask].mean()) > 0.0:
            allowed.add(int(threshold_idx))
    return allowed


def summarize(per_frame: pd.DataFrame, hard_dice_threshold: float) -> dict:
    hard = per_frame["baseline_dice"].to_numpy(dtype=float) < hard_dice_threshold
    routed = per_frame["route_to_quantum"].to_numpy(dtype=bool)
    summary = {
        "frames": int(len(per_frame)),
        "baseline_dice": float(per_frame["baseline_dice"].mean()),
        "selected_dice": float(per_frame["selected_dice"].mean()),
        "delta_dice": float(per_frame["delta_dice"].mean()),
        "oracle_dice": float(per_frame["oracle_dice"].mean()),
        "hard_frames": int(hard.sum()),
        "hard_baseline_dice": float(per_frame.loc[hard, "baseline_dice"].mean()) if hard.any() else None,
        "hard_selected_dice": float(per_frame.loc[hard, "selected_dice"].mean()) if hard.any() else None,
        "hard_delta_dice": float(per_frame.loc[hard, "delta_dice"].mean()) if hard.any() else None,
        "routed_frames": int(routed.sum()),
        "route_rate": float(routed.mean()),
        "routed_hard_frames": int(np.logical_and(routed, hard).sum()),
        "routed_easy_frames": int(np.logical_and(routed, ~hard).sum()),
        "threshold_counts": {str(k): int(v) for k, v in per_frame["selected_threshold"].value_counts().sort_index().items()},
    }
    if routed.any():
        summary["routed_baseline_dice"] = float(per_frame.loc[routed, "baseline_dice"].mean())
        summary["routed_selected_dice"] = float(per_frame.loc[routed, "selected_dice"].mean())
        summary["routed_delta_dice"] = float(per_frame.loc[routed, "delta_dice"].mean())
    return summary


def main():
    parser = argparse.ArgumentParser(description="Full-Kvasir out-of-fold hard-case quantum switch evaluation")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/hard_case_quantum_switch_oof_kvasir.csv")
    parser.add_argument("--per_frame_csv", default="endoscopy_guidance/results/hard_case_quantum_switch_oof_kvasir_per_frame.csv")
    parser.add_argument("--summary_json", default="endoscopy_guidance/results/hard_case_quantum_switch_oof_kvasir_summary.json")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.50, 0.90])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tune_fraction", type=float, default=0.20)
    parser.add_argument("--proposer_min_score_grid", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    parser.add_argument("--router_score_grid", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--selection_objective", choices=["overall", "hard", "combined"], default="overall")
    parser.add_argument("--default_selector", choices=["fixed", "proposer"], default="fixed")
    parser.add_argument("--route_target", choices=["quantum_gain", "hard"], default="quantum_gain")
    parser.add_argument("--route_model", choices=["classifier", "regressor"], default="classifier")
    parser.add_argument("--selector_feature_mode", choices=["base", "threshold_stack"], default="base")
    parser.add_argument("--quantum_threshold_gate", choices=["none", "positive_tune"], default="none")
    parser.add_argument("--threshold_gate_min_count", type=int, default=2)
    parser.add_argument("--min_quantum_gain", type=float, default=0.0)
    parser.add_argument("--hard_dice_threshold", type=float, default=0.80)
    parser.add_argument("--max_route_rate", type=float, default=1.0)
    parser.add_argument("--proposer_trees", type=int, default=300)
    parser.add_argument("--proposer_min_leaf", type=int, default=3)
    parser.add_argument("--router_iter", type=int, default=120)
    parser.add_argument("--router_lr", type=float, default=0.04)
    parser.add_argument("--pqk_components", type=int, default=8)
    parser.add_argument("--pqk_reps", type=int, default=3)
    parser.add_argument("--hybrid_quantum", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds)
    frame = load_frame_table(Path(args.baseline_dir), thresholds).reset_index(drop=True)
    hard_labels = (frame["baseline_dice"].to_numpy(dtype=float) < args.hard_dice_threshold).astype(np.int64)
    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_outputs = []
    fold_summaries = []

    for fold, (trainval_idx, eval_idx) in enumerate(outer.split(np.zeros(len(frame)), hard_labels), start=1):
        trainval_df = frame.iloc[trainval_idx].copy().reset_index(drop=True)
        eval_df = frame.iloc[eval_idx].copy().reset_index(drop=True)
        train_hard = (trainval_df["baseline_dice"].to_numpy(dtype=float) < args.hard_dice_threshold).astype(np.int64)
        train_df, tune_df = stratified_inner_split(trainval_df, train_hard, args.tune_fraction, args.seed + fold)

        pair_X_train, pair_y_train, _ = candidate_features(train_df, thresholds, "legacy_candidate")
        proposer = RandomForestClassifier(
            n_estimators=args.proposer_trees,
            min_samples_leaf=args.proposer_min_leaf,
            class_weight="balanced_subsample",
            random_state=args.seed + fold,
            n_jobs=-1,
        )
        proposer.fit(pair_X_train, pair_y_train)
        proposer_min_score, tune_proposer_selected, tune_proposer_scores = tune_proposer_min_score(
            proposer,
            tune_df,
            thresholds,
            args.proposer_min_score_grid,
            args.selection_objective,
            args.hard_dice_threshold,
        )

        train_proposer_selected, train_proposer_scores = select_with_proposer(proposer, train_df, thresholds, proposer_min_score)
        if args.default_selector == "fixed":
            train_default, train_default_scores = fixed_default(train_df, thresholds)
            tune_default, tune_default_scores = fixed_default(tune_df, thresholds)
        else:
            train_default, train_default_scores = train_proposer_selected, train_proposer_scores
            tune_default, tune_default_scores = tune_proposer_selected, tune_proposer_scores

        X_train = frame_feature_matrix(train_df, thresholds, args.selector_feature_mode)
        y_train = train_df["best_threshold_index"].to_numpy(dtype=np.int64)
        X_tune = frame_feature_matrix(tune_df, thresholds, args.selector_feature_mode)

        quantum_selector = QuantumClassifier(
            args.pqk_components,
            args.pqk_reps,
            args.seed + fold,
            hybrid=args.hybrid_quantum,
            head="logistic",
        )
        quantum_selector.fit(X_train, y_train)
        train_quantum = quantum_selector.predict(X_train).astype(np.int64)
        tune_quantum = quantum_selector.predict(X_tune).astype(np.int64)
        allowed_quantum_idx = None
        if args.quantum_threshold_gate == "positive_tune":
            allowed_quantum_idx = tune_allowed_quantum_thresholds(
                tune_df,
                tune_default,
                tune_quantum,
                thresholds,
                args.threshold_gate_min_count,
            )

        train_default_dice = selected_dice(train_df, train_default, thresholds)
        train_quantum_dice = selected_dice(train_df, train_quantum, thresholds)
        if args.route_model == "regressor":
            route_train = train_quantum_dice - train_default_dice
        elif args.route_target == "hard":
            route_train = (train_df["baseline_dice"].to_numpy(dtype=float) < args.hard_dice_threshold).astype(np.int64)
        else:
            route_train = (train_quantum_dice > train_default_dice + args.min_quantum_gain).astype(np.int64)
        X_router_train = router_feature_matrix(train_df, train_default, train_quantum, train_default_scores, thresholds, args.selector_feature_mode)
        X_router_tune = router_feature_matrix(tune_df, tune_default, tune_quantum, tune_default_scores, thresholds, args.selector_feature_mode)
        if args.route_model == "regressor":
            router = HistGradientBoostingRegressor(max_iter=args.router_iter, learning_rate=args.router_lr, random_state=args.seed + fold)
        elif len(np.unique(route_train)) < 2:
            router = DummyClassifier(strategy="constant", constant=int(route_train[0]))
        else:
            router = HistGradientBoostingClassifier(max_iter=args.router_iter, learning_rate=args.router_lr, random_state=args.seed + fold)
        router.fit(X_router_train, route_train)
        tune_router_scores = router_scores(router, X_router_tune, args.route_model)
        router_score_grid = args.router_score_grid
        if args.route_model == "regressor" and args.router_score_grid == [
            0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
        ]:
            router_score_grid = [-0.005, 0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]

        router_threshold, tune_summary = tune_router_threshold(
            tune_df,
            tune_default,
            tune_router_scores,
            tune_quantum,
            thresholds,
            router_score_grid,
            args.selection_objective,
            args.hard_dice_threshold,
            args.max_route_rate,
            allowed_quantum_idx,
        )
        fold_out = apply_fold(
            fold,
            eval_df,
            router,
            router_threshold,
            quantum_selector,
            thresholds,
            args.default_selector,
            args.selector_feature_mode,
            args.route_model,
            allowed_quantum_idx,
            proposer=proposer,
            proposer_min_score=proposer_min_score,
        )
        fold_outputs.append(fold_out)
        fold_summary = summarize(fold_out, args.hard_dice_threshold)
        fold_summary.update({
            "fold": int(fold),
            "proposer_min_score": float(proposer_min_score),
            "router_score_threshold": float(router_threshold),
            "allowed_quantum_thresholds": [float(thresholds[idx]) for idx in sorted(allowed_quantum_idx)] if allowed_quantum_idx is not None else "all",
            "tune_selected_dice": float(tune_summary["selected_dice"]),
            "tune_hard_selected_dice": float(tune_summary["hard_selected_dice"]),
            "tune_route_rate": float(tune_summary["route_rate"]),
        })
        fold_summaries.append(fold_summary)
        print(
            f"[fold {fold}/{args.folds}] "
            f"dice {fold_summary['baseline_dice']:.4f}->{fold_summary['selected_dice']:.4f}, "
            f"hard {fold_summary['hard_baseline_dice']:.4f}->{fold_summary['hard_selected_dice']:.4f}, "
            f"routed {fold_summary['routed_frames']}/{fold_summary['frames']}"
        )

    per_frame = pd.concat(fold_outputs, ignore_index=True).sort_values(["split", "sample_id"]).reset_index(drop=True)
    summary = summarize(per_frame, args.hard_dice_threshold)
    summary.update({
        "baseline_dir": args.baseline_dir,
        "thresholds": thresholds,
        "folds": args.folds,
        "tune_fraction": args.tune_fraction,
        "selection_objective": args.selection_objective,
        "default_selector": args.default_selector,
        "route_target": args.route_target,
        "route_model": args.route_model,
        "selector_feature_mode": args.selector_feature_mode,
        "quantum_threshold_gate": args.quantum_threshold_gate,
        "threshold_gate_min_count": args.threshold_gate_min_count,
        "min_quantum_gain": args.min_quantum_gain,
        "pqk_components": args.pqk_components,
        "pqk_reps": args.pqk_reps,
        "hybrid_quantum": bool(args.hybrid_quantum),
        "fold_summaries": fold_summaries,
    })

    output_csv = Path(args.output_csv)
    per_frame_csv = Path(args.per_frame_csv)
    summary_json = Path(args.summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    per_frame_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).drop(columns=["fold_summaries"]).to_csv(output_csv, index=False)
    per_frame.to_csv(per_frame_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[saved] {output_csv}")
    print(f"[saved] {per_frame_csv}")
    print(f"[saved] {summary_json}")
    print(json.dumps({k: v for k, v in summary.items() if k != "fold_summaries"}, indent=2))


if __name__ == "__main__":
    main()
