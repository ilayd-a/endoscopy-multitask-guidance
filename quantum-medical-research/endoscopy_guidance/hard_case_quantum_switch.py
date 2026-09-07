"""
hard_case_quantum_switch.py
===========================
Train and apply a hard-case quantum switching policy.

The intended deployment behavior is:

1. Use the fixed classical UNet probability map for every frame.
2. Use a compact classical proposer as the default threshold selector.
3. Estimate whether the frame is a hard case from inference-safe probability
   and morphology features.
4. Switch to a compact projected-quantum selector only when the hard-case
   score exceeds a validation-tuned threshold.

This separates the scientific claim from implementation details: quantum is not
used everywhere. It is a conditional expert for risky frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from export_classical_cvc_baseline import dice_iou, sample_features
from quantum_mask_hypothesis_selector import QuantumClassifier, load_frame_table, risk_features, selected_dice, split_arrays
from quantum_threshold_pairwise_ranker import candidate_features, evaluate, selector_score


def positive_proba(model, X: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(X)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = list(model[-1].classes_)
    return probs[:, classes.index(1)]


def choose_pairwise_thresholds(
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
    best_scores = np.full(n_frames, min_score, dtype=np.float32)
    raw_scores = np.full(n_frames, -1.0, dtype=np.float32)
    for row, score in zip(rows, scores):
        frame_idx = int(row["frame_idx"])
        threshold_idx = int(row["threshold_idx"])
        if threshold_idx == fixed_idx:
            continue
        raw_scores[frame_idx] = max(raw_scores[frame_idx], float(score))
        if score > best_scores[frame_idx]:
            best_scores[frame_idx] = float(score)
            selected[frame_idx] = threshold_idx
    return selected, raw_scores


def inference_frame_table(base: Path, thresholds: list[float], split: str) -> pd.DataFrame:
    metrics = pd.read_csv(base / "baseline_metrics.csv")
    if split != "all":
        metrics = metrics.loc[metrics["split"].eq(split)].copy()
    rows = []
    for record in metrics.itertuples(index=False):
        sample_id = str(record.sample_id)
        prob = np.load(base / "prob_maps" / f"{sample_id}.npy").astype(np.float32)
        row = {
            "sample_id": sample_id,
            "split": getattr(record, "split", split),
            "source_file": getattr(record, "source_file", sample_id),
            "baseline_dice": float(getattr(record, "dice", np.nan)),
            "features": risk_features(prob, threshold=0.5),
        }
        for threshold in thresholds:
            row[f"features_t{threshold:.2f}"] = risk_features(prob, threshold=threshold)
        gt_path = base / "gt_masks" / f"{sample_id}.npy"
        if gt_path.exists():
            gt = np.load(gt_path).astype(np.uint8)
            threshold_dice = []
            for threshold in thresholds:
                pred = (prob >= threshold).astype(np.uint8)
                dice, iou = dice_iou(pred, gt)
                row[f"dice_t{threshold:.2f}"] = float(dice)
                row[f"iou_t{threshold:.2f}"] = float(iou)
                threshold_dice.append(float(dice))
            best_idx = int(np.argmax(threshold_dice))
            row["oracle_dice"] = threshold_dice[best_idx]
            row["oracle_threshold"] = float(thresholds[best_idx])
            row["best_threshold_index"] = best_idx
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def inference_candidate_features(frame: pd.DataFrame, thresholds: list[float]) -> tuple[np.ndarray, list[dict]]:
    X = []
    rows = []
    for frame_idx, record in frame.iterrows():
        base_features = np.asarray(record["features"], dtype=np.float32)
        for threshold_idx, threshold in enumerate(thresholds):
            candidate_morphology = np.asarray(record[f"features_t{threshold:.2f}"], dtype=np.float32)
            threshold_features = np.asarray([
                threshold,
                abs(threshold - 0.5),
                threshold < 0.5,
                threshold > 0.5,
            ], dtype=np.float32)
            X.append(np.concatenate([
                base_features,
                candidate_morphology,
                candidate_morphology - base_features[: len(candidate_morphology)],
                threshold_features,
            ]))
            rows.append({
                "frame_idx": frame_idx,
                "sample_id": record["sample_id"],
                "threshold_idx": threshold_idx,
                "threshold": threshold,
            })
    return np.asarray(X, dtype=np.float32), rows


def train_policy(args) -> dict:
    thresholds = sorted(args.thresholds)
    frame = load_frame_table(Path(args.baseline_dir), thresholds)
    X_train, y_train, train_df = split_arrays(frame, args.train_split)
    X_tune, _, tune_df = split_arrays(frame, args.tune_split)

    pair_X_train, pair_y_train, _ = candidate_features(train_df, thresholds, "legacy_candidate")
    pair_X_tune, _, tune_rows = candidate_features(tune_df, thresholds, "legacy_candidate")
    proposer = RandomForestClassifier(
        n_estimators=args.proposer_trees,
        min_samples_leaf=args.proposer_min_leaf,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    proposer.fit(pair_X_train, pair_y_train)

    proposer_best = None
    for min_score in args.proposer_min_score_grid:
        tune_proposer, _ = choose_pairwise_thresholds(proposer, pair_X_tune, tune_rows, len(tune_df), thresholds, min_score)
        row = evaluate("compact_rf_proposer", tune_df, tune_proposer, thresholds, args.hard_dice_threshold)
        score = selector_score(row, args.selection_objective)
        if proposer_best is None or score > proposer_best[0]:
            proposer_best = (score, float(min_score))
    proposer_min_score = proposer_best[1]
    tune_proposer, _ = choose_pairwise_thresholds(proposer, pair_X_tune, tune_rows, len(tune_df), thresholds, proposer_min_score)

    hard_train = (train_df["baseline_dice"].to_numpy(dtype=float) < args.hard_dice_threshold).astype(np.int64)
    router = HistGradientBoostingClassifier(max_iter=args.router_iter, learning_rate=args.router_lr, random_state=args.seed)
    router.fit(X_train, hard_train)
    tune_router_scores = positive_proba(router, X_tune)

    quantum_selector = QuantumClassifier(args.pqk_components, args.pqk_reps, args.seed, hybrid=args.hybrid_quantum, head="logistic")
    quantum_selector.fit(X_train, y_train)
    tune_quantum = quantum_selector.predict(X_tune).astype(np.int64)

    router_best = None
    fixed_idx = thresholds.index(0.5)
    for router_threshold in args.router_score_grid:
        tune_selected = tune_proposer.copy()
        route_mask = tune_router_scores >= router_threshold
        tune_selected[route_mask] = tune_quantum[route_mask]
        row = evaluate("hard_case_quantum_switch", tune_df, tune_selected, thresholds, args.hard_dice_threshold)
        row["routed_frames"] = int(route_mask.sum())
        row["route_rate"] = float(route_mask.mean())
        if args.max_route_rate < 1.0 and row["route_rate"] > args.max_route_rate:
            continue
        score = selector_score(row, args.selection_objective)
        if router_best is None or score > router_best[0]:
            router_best = (score, float(router_threshold), row)
    if router_best is None:
        raise ValueError("No router threshold satisfied --max_route_rate")

    policy = {
        "thresholds": thresholds,
        "fixed_idx": fixed_idx,
        "proposer": proposer,
        "proposer_min_score": proposer_min_score,
        "router": router,
        "router_score_threshold": router_best[1],
        "quantum_selector": quantum_selector,
        "metadata": {
            "baseline_dir": args.baseline_dir,
            "train_split": args.train_split,
            "tune_split": args.tune_split,
            "selection_objective": args.selection_objective,
            "hard_dice_threshold": args.hard_dice_threshold,
            "pqk_components": args.pqk_components,
            "pqk_reps": args.pqk_reps,
            "hybrid_quantum": bool(args.hybrid_quantum),
            "seed": args.seed,
            "tune_summary": router_best[2],
        },
    }
    return policy


def apply_policy(policy: dict, base: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    thresholds = policy["thresholds"]
    frame = inference_frame_table(base, thresholds, split)
    X = np.asarray(frame["features"].tolist(), dtype=np.float32)
    pair_X, pair_rows = inference_candidate_features(frame, thresholds)
    proposer_selected, proposer_scores = choose_pairwise_thresholds(
        policy["proposer"],
        pair_X,
        pair_rows,
        len(frame),
        thresholds,
        float(policy["proposer_min_score"]),
    )
    router_scores = positive_proba(policy["router"], X)
    quantum_selected = policy["quantum_selector"].predict(X).astype(np.int64)
    route_mask = router_scores >= float(policy["router_score_threshold"])
    selected = proposer_selected.copy()
    selected[route_mask] = quantum_selected[route_mask]

    out = frame[["sample_id", "split", "source_file"]].copy()
    out["router_hard_score"] = router_scores
    out["route_to_quantum"] = route_mask
    out["proposer_score"] = proposer_scores
    out["proposer_threshold"] = [thresholds[int(idx)] for idx in proposer_selected]
    out["quantum_threshold"] = [thresholds[int(idx)] for idx in quantum_selected]
    out["selected_threshold"] = [thresholds[int(idx)] for idx in selected]
    if "baseline_dice" in frame and not frame["baseline_dice"].isna().all():
        out["baseline_dice"] = frame["baseline_dice"].to_numpy(dtype=float)
    if f"dice_t{thresholds[0]:.2f}" in frame:
        dice = selected_dice(frame, selected, thresholds)
        baseline = frame["baseline_dice"].to_numpy(dtype=np.float32)
        out["selected_dice"] = dice
        out["delta_dice"] = dice - baseline
        out["oracle_dice"] = frame["oracle_dice"].to_numpy(dtype=float)
        out["oracle_threshold"] = frame["oracle_threshold"].to_numpy(dtype=float)
    return out, selected


def export_masks(base: Path, output_dir: Path, selected: np.ndarray, per_frame: pd.DataFrame, thresholds: list[float]):
    for subdir in ["images", "gt_masks", "prob_maps", "pred_masks"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    rows = []
    source_metrics = pd.read_csv(base / "baseline_metrics.csv")
    source_lookup = {str(row.sample_id): row for row in source_metrics.itertuples(index=False)}
    for idx, row in per_frame.reset_index(drop=True).iterrows():
        sample_id = str(row.sample_id)
        threshold = thresholds[int(selected[idx])]
        prob = np.load(base / "prob_maps" / f"{sample_id}.npy").astype(np.float32)
        pred = (prob >= threshold).astype(np.uint8)
        for subdir in ["images", "prob_maps"]:
            src = base / subdir / f"{sample_id}.npy"
            dst = output_dir / subdir / f"{sample_id}.npy"
            if src.exists() and not dst.exists():
                shutil.copyfile(src, dst)
        gt_path = base / "gt_masks" / f"{sample_id}.npy"
        metric = source_lookup[sample_id]
        output_row = {
            "sample_id": sample_id,
            "source_file": getattr(metric, "source_file", sample_id),
            "sequence_id": getattr(metric, "sequence_id", -1),
            "frame_id": getattr(metric, "frame_id", -1),
            "split": getattr(metric, "split", row.split),
            "selected_threshold": float(threshold),
            "route_to_quantum": bool(row.route_to_quantum),
            "router_hard_score": float(row.router_hard_score),
            "proposer_threshold": float(row.proposer_threshold),
            "quantum_threshold": float(row.quantum_threshold),
        }
        if gt_path.exists():
            gt = np.load(gt_path).astype(np.uint8)
            shutil.copyfile(gt_path, output_dir / "gt_masks" / f"{sample_id}.npy")
            dice, iou = dice_iou(pred, gt)
            output_row.update({
                "dice": float(dice),
                "iou": float(iou),
                "source_baseline_dice": float(getattr(metric, "dice", np.nan)),
                "source_delta_dice": float(dice - float(getattr(metric, "dice", np.nan))),
            })
            output_row.update(sample_features(prob, pred, gt))
        np.save(output_dir / "pred_masks" / f"{sample_id}.npy", pred)
        rows.append(output_row)
    if rows:
        with (output_dir / "baseline_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def write_summary(per_frame: pd.DataFrame, output_json: Path, hard_dice_threshold: float = 0.80):
    summary = {
        "frames": int(len(per_frame)),
        "routed_frames": int(per_frame["route_to_quantum"].sum()),
        "route_rate": float(per_frame["route_to_quantum"].mean()),
        "threshold_counts": {str(k): int(v) for k, v in per_frame["selected_threshold"].value_counts().sort_index().items()},
    }
    if "selected_dice" in per_frame:
        hard = per_frame["baseline_dice"].to_numpy(dtype=float) < hard_dice_threshold
        summary.update({
            "baseline_dice": float(per_frame["baseline_dice"].mean()),
            "selected_dice": float(per_frame["selected_dice"].mean()),
            "delta_dice": float(per_frame["delta_dice"].mean()),
            "hard_frames": int(hard.sum()),
            "hard_baseline_dice": float(per_frame.loc[hard, "baseline_dice"].mean()) if hard.any() else None,
            "hard_selected_dice": float(per_frame.loc[hard, "selected_dice"].mean()) if hard.any() else None,
            "hard_delta_dice": float(per_frame.loc[hard, "delta_dice"].mean()) if hard.any() else None,
        })
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train/apply hard-case quantum switching policy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and save a hard-case quantum switch policy")
    train.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test")
    train.add_argument("--policy_path", default="models/hard_case_quantum_switch.joblib")
    train.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.50, 0.90])
    train.add_argument("--train_split", default="train")
    train.add_argument("--tune_split", default="val")
    train.add_argument("--proposer_min_score_grid", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    train.add_argument("--router_score_grid", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    train.add_argument("--selection_objective", choices=["overall", "hard", "combined"], default="overall")
    train.add_argument("--hard_dice_threshold", type=float, default=0.80)
    train.add_argument("--max_route_rate", type=float, default=1.0)
    train.add_argument("--proposer_trees", type=int, default=300)
    train.add_argument("--proposer_min_leaf", type=int, default=3)
    train.add_argument("--router_iter", type=int, default=120)
    train.add_argument("--router_lr", type=float, default=0.04)
    train.add_argument("--pqk_components", type=int, default=8)
    train.add_argument("--pqk_reps", type=int, default=3)
    train.add_argument("--hybrid_quantum", action="store_true")
    train.add_argument("--seed", type=int, default=123)

    apply = subparsers.add_parser("apply", help="Apply a saved switch policy to a baseline export")
    apply.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test")
    apply.add_argument("--policy_path", default="models/hard_case_quantum_switch.joblib")
    apply.add_argument("--split", default="test")
    apply.add_argument("--per_frame_csv", default="endoscopy_guidance/results/hard_case_quantum_switch_per_frame.csv")
    apply.add_argument("--summary_json", default="endoscopy_guidance/results/hard_case_quantum_switch_summary.json")
    apply.add_argument("--output_dir", default="")

    args = parser.parse_args()
    if args.command == "train":
        policy = train_policy(args)
        path = Path(args.policy_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(policy, path)
        print(f"[saved] {path}")
        print(json.dumps(policy["metadata"], indent=2))
    else:
        policy = joblib.load(args.policy_path)
        per_frame, selected = apply_policy(policy, Path(args.baseline_dir), args.split)
        per_frame_path = Path(args.per_frame_csv)
        per_frame_path.parent.mkdir(parents=True, exist_ok=True)
        per_frame.to_csv(per_frame_path, index=False)
        print(f"[saved] {per_frame_path}")
        if args.output_dir:
            export_masks(Path(args.baseline_dir), Path(args.output_dir), selected, per_frame, policy["thresholds"])
            print(f"[saved] {args.output_dir}")
        hard_threshold = float(policy.get("metadata", {}).get("hard_dice_threshold", 0.80))
        write_summary(per_frame, Path(args.summary_json), hard_threshold)


if __name__ == "__main__":
    main()
