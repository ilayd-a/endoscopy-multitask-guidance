"""
side_by_side_unet_quantum_fusion.py
===================================
Confidence fusion between a fixed UNet mask and a quantum-assisted SAM prompt
selector.

The deployment idea is side-by-side rather than replacement:

1. UNet produces the default segmentation mask.
2. A SAM prompt selector proposes an alternate mask.
3. A validation-calibrated confidence rule chooses the output expected to be
   more reliable for each frame.

This script uses cached prompt-quality labels to evaluate that policy without
recomputing SAM masks. Reported Dice is therefore the known Dice of the selected
candidate mask versus the known Dice of the fixed UNet export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from prompt_selector_confidence_analysis import selected_with_confidence
from sam_prompt_active_learning import QuantumFeatureRidgeRegressor
from sam_prompt_quality_ranker import per_sample_normalize


UNET_CONFIDENCE_COLUMNS = [
    "mean_prob",
    "mean_uncertainty",
    "boundary_uncertainty",
    "mask_area_frac",
]


def load_inputs(prompt_quality_csv: Path, features_path: Path, unet_metrics_csv: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    qdf = pd.read_csv(prompt_quality_csv)
    X = np.load(features_path).astype(np.float32)
    if len(qdf) != len(X):
        raise ValueError(f"Prompt cache mismatch: {len(qdf)} rows but {len(X)} feature rows")
    unet = pd.read_csv(unet_metrics_csv)
    keep = qdf["source_file"].astype(str).isin(set(unet["source_file"].astype(str)))
    qdf = qdf.loc[keep].copy().reset_index(drop=True)
    X = X[keep.to_numpy()]
    if qdf.empty:
        raise ValueError("No prompt-quality rows match the UNet metrics by source_file")
    return qdf, X, unet


def split_masks(qdf: pd.DataFrame, mode: str, seed: int, train_fraction: float, val_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits = set(qdf["split"].astype(str).unique())
    if mode == "auto":
        mode = "existing" if {"train", "val", "test"}.issubset(splits) else "random_source"
    if mode == "existing":
        return (
            qdf["sequence_id"].le(23).to_numpy(),
            qdf["split"].eq("val").to_numpy(),
            qdf["split"].eq("test").to_numpy(),
        )
    if mode != "random_source":
        raise ValueError(f"Unknown split_mode={mode}")
    rng = np.random.default_rng(seed)
    sources = np.asarray(sorted(qdf["source_file"].astype(str).unique()), dtype=object)
    rng.shuffle(sources)
    n_train = max(1, int(round(len(sources) * train_fraction)))
    n_val = max(1, int(round(len(sources) * val_fraction)))
    if n_train + n_val >= len(sources):
        raise ValueError("train_fraction + val_fraction leaves no test sources")
    train_sources = set(sources[:n_train].tolist())
    val_sources = set(sources[n_train:n_train + n_val].tolist())
    test_sources = set(sources[n_train + n_val:].tolist())
    source_values = qdf["source_file"].astype(str)
    return (
        source_values.isin(train_sources).to_numpy(),
        source_values.isin(val_sources).to_numpy(),
        source_values.isin(test_sources).to_numpy(),
    )


def attach_unet(chosen: pd.DataFrame, unet: pd.DataFrame, unet_scores: np.ndarray | None = None) -> pd.DataFrame:
    lookup = unet.set_index("source_file")
    rows = chosen.copy().reset_index(drop=True)
    rows["unet_dice"] = [float(lookup.loc[source_file, "dice"]) for source_file in rows["source_file"].astype(str)]
    rows["unet_iou"] = [float(lookup.loc[source_file, "iou"]) for source_file in rows["source_file"].astype(str)]
    if unet_scores is not None:
        rows["predicted_unet_quality"] = unet_scores
    return rows


def fit_unet_quality_model(unet: pd.DataFrame, train_sources: set[str], seed: int):
    train = unet[unet["source_file"].astype(str).isin(train_sources)].copy()
    model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, random_state=seed)
    model.fit(train[UNET_CONFIDENCE_COLUMNS].to_numpy(dtype=np.float32), train["dice"].to_numpy(dtype=float))
    return model


def predict_unet_quality(model, unet: pd.DataFrame, sources: pd.Series) -> np.ndarray:
    lookup = unet.set_index("source_file")
    rows = lookup.loc[sources.astype(str), UNET_CONFIDENCE_COLUMNS]
    return model.predict(rows.to_numpy(dtype=np.float32))


def fit_prompt_selector(model_name: str, X_train: np.ndarray, y_train: np.ndarray, components: int, reps: int, seed: int):
    if model_name == "classical_histgb":
        model = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=seed)
    elif model_name == "quantum_feature_ridge":
        model = QuantumFeatureRidgeRegressor(components, reps, seed)
    else:
        raise ValueError(f"Unknown prompt selector={model_name}")
    model.fit(X_train, y_train)
    return model


def selected_prompt_rows(qdf: pd.DataFrame, scores: np.ndarray, expert_name: str) -> pd.DataFrame:
    chosen = selected_with_confidence(qdf, scores)
    chosen = chosen.rename(columns={
        "sam_dice": "expert_dice",
        "sam_iou": "expert_iou",
        "predicted_quality": "predicted_expert_quality",
    })
    chosen["expert"] = expert_name
    return chosen


def selected_prompt_rows_with_quantum_agreement(
    eval_df: pd.DataFrame,
    primary_scores: np.ndarray,
    quantum_scores: np.ndarray,
) -> pd.DataFrame:
    rows = eval_df.copy().reset_index(drop=True)
    sample_ids = rows["sample_id"].to_numpy()
    rows["_primary_score"] = primary_scores
    rows["_quantum_score"] = quantum_scores
    rows["_primary_norm"] = per_sample_normalize(primary_scores, sample_ids)
    rows["_quantum_norm"] = per_sample_normalize(quantum_scores, sample_ids)
    out = []
    for sid, group in rows.groupby("sample_id", sort=False):
        ranked = group.sort_values("_primary_score", ascending=False)
        top = ranked.iloc[0].copy()
        top_score = float(top["_primary_score"])
        second_score = float(ranked.iloc[1]["_primary_score"]) if len(ranked) > 1 else top_score
        score_values = group["_primary_score"].to_numpy(dtype=float)
        spread = float(np.nanmax(score_values) - np.nanmin(score_values))
        top["score_margin"] = top_score - second_score
        top["score_margin_norm"] = 0.0 if spread <= 0 else (top_score - second_score) / spread
        top["predicted_expert_quality"] = top_score
        top["quantum_candidate_quality"] = float(top["_quantum_score"])
        top["quantum_agreement"] = float(1.0 - abs(top["_primary_norm"] - top["_quantum_norm"]))
        top["expert"] = "classical_histgb_quantum_confidence"
        top["sample_id"] = sid
        out.append(top)
    chosen = pd.DataFrame(out)
    return chosen.rename(columns={"sam_dice": "expert_dice", "sam_iou": "expert_iou"})


def policy_metrics(name: str, rows: pd.DataFrame, use_expert: np.ndarray) -> dict:
    selected_dice = np.where(use_expert, rows["expert_dice"].to_numpy(dtype=float), rows["unet_dice"].to_numpy(dtype=float))
    selected_iou = np.where(use_expert, rows["expert_iou"].to_numpy(dtype=float), rows["unet_iou"].to_numpy(dtype=float))
    hard = rows["unet_dice"].to_numpy(dtype=float) < 0.80
    return {
        "policy": name,
        "frames": int(len(rows)),
        "selected_dice": float(selected_dice.mean()),
        "selected_iou": float(selected_iou.mean()),
        "unet_dice": float(rows["unet_dice"].mean()),
        "expert_dice": float(rows["expert_dice"].mean()),
        "delta_vs_unet": float((selected_dice - rows["unet_dice"].to_numpy(dtype=float)).mean()),
        "expert_rate": float(np.mean(use_expert)),
        "expert_frames": int(np.sum(use_expert)),
        "hard_frames": int(hard.sum()),
        "hard_selected_dice": float(selected_dice[hard].mean()) if hard.any() else np.nan,
        "hard_unet_dice": float(rows.loc[hard, "unet_dice"].mean()) if hard.any() else np.nan,
        "hard_delta_vs_unet": float((selected_dice[hard] - rows.loc[hard, "unet_dice"].to_numpy(dtype=float)).mean()) if hard.any() else np.nan,
    }


def switch_feature_matrix(rows: pd.DataFrame, include_quantum: bool) -> np.ndarray:
    pieces = [
        rows["predicted_expert_quality"].to_numpy(dtype=float),
        rows["predicted_unet_quality"].to_numpy(dtype=float),
        rows["predicted_expert_quality"].to_numpy(dtype=float) - rows["predicted_unet_quality"].to_numpy(dtype=float),
        rows["score_margin"].to_numpy(dtype=float),
        rows["score_margin_norm"].to_numpy(dtype=float),
    ]
    if include_quantum and "quantum_candidate_quality" in rows:
        pieces.extend([
            rows["quantum_candidate_quality"].to_numpy(dtype=float),
            rows["quantum_agreement"].to_numpy(dtype=float),
            rows["quantum_candidate_quality"].to_numpy(dtype=float) - rows["predicted_unet_quality"].to_numpy(dtype=float),
        ])
    return np.column_stack(pieces).astype(np.float32)


def fit_learned_switch(train_rows: pd.DataFrame, include_quantum: bool, seed: int):
    X = switch_feature_matrix(train_rows, include_quantum)
    y = (train_rows["expert_dice"].to_numpy(dtype=float) > train_rows["unet_dice"].to_numpy(dtype=float)).astype(np.int64)
    if len(np.unique(y)) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y[0]))
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    model.fit(X, y)
    return model


def positive_probability(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = list(model[-1].classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=np.float32)
    return model.predict_proba(X)[:, classes.index(1)]


def tune_switch_threshold(val_rows: pd.DataFrame, score: np.ndarray, min_expert_rate: float, max_expert_rate: float) -> tuple[float, dict]:
    thresholds = np.unique(np.quantile(score, np.linspace(0, 1, 101)))
    thresholds = np.concatenate([thresholds, [float(np.max(score) + 1e-6)]])
    best = None
    for threshold in thresholds:
        use_expert = score >= threshold
        rate = float(use_expert.mean())
        if rate < min_expert_rate or rate > max_expert_rate:
            continue
        row = policy_metrics("val_tuned_switch", val_rows, use_expert)
        if best is None or (row["selected_dice"], row["hard_selected_dice"]) > (best[1]["selected_dice"], best[1]["hard_selected_dice"]):
            best = (float(threshold), row)
    if best is None:
        use_expert = np.zeros(len(val_rows), dtype=bool)
        return float(np.max(score) + 1e-6), policy_metrics("val_tuned_switch", val_rows, use_expert)
    return best


def tune_agreement_switch(
    val_rows: pd.DataFrame,
    min_expert_rate: float,
    max_expert_rate: float,
    agreement_weights: list[float],
    quantum_weights: list[float],
) -> tuple[float, float, float, dict]:
    base = val_rows["predicted_expert_quality"].to_numpy(dtype=float) - val_rows["predicted_unet_quality"].to_numpy(dtype=float)
    agreement = val_rows["quantum_agreement"].to_numpy(dtype=float)
    quantum_delta = val_rows["quantum_candidate_quality"].to_numpy(dtype=float) - val_rows["predicted_unet_quality"].to_numpy(dtype=float)
    best = None
    for agreement_weight in agreement_weights:
        for quantum_weight in quantum_weights:
            score = base + agreement_weight * (agreement - 0.5) + quantum_weight * quantum_delta
            threshold, row = tune_switch_threshold(val_rows, score, min_expert_rate, max_expert_rate)
            candidate = (row["selected_dice"], row["hard_selected_dice"], row["expert_rate"])
            if best is None or candidate > best[0]:
                best = (candidate, float(threshold), float(agreement_weight), float(quantum_weight), row)
    return best[1], best[2], best[3], best[4]


def evaluate_expert(
    expert_name: str,
    prompt_model,
    qdf: pd.DataFrame,
    X: np.ndarray,
    unet: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    unet_quality_model,
    args,
) -> tuple[list[dict], pd.DataFrame]:
    train_df = qdf.loc[train_mask].copy().reset_index(drop=True)
    val_df = qdf.loc[val_mask].copy().reset_index(drop=True)
    test_df = qdf.loc[test_mask].copy().reset_index(drop=True)
    train_scores = prompt_model.predict(X[train_mask])
    val_scores = prompt_model.predict(X[val_mask])
    test_scores = prompt_model.predict(X[test_mask])
    train_rows = selected_prompt_rows(train_df, train_scores, expert_name)
    val_rows = selected_prompt_rows(val_df, val_scores, expert_name)
    test_rows = selected_prompt_rows(test_df, test_scores, expert_name)
    train_rows = attach_unet(train_rows, unet, predict_unet_quality(unet_quality_model, unet, train_rows["source_file"]))
    val_rows = attach_unet(val_rows, unet, predict_unet_quality(unet_quality_model, unet, val_rows["source_file"]))
    test_rows = attach_unet(test_rows, unet, predict_unet_quality(unet_quality_model, unet, test_rows["source_file"]))

    val_switch_score = val_rows["predicted_expert_quality"].to_numpy(dtype=float) - val_rows["predicted_unet_quality"].to_numpy(dtype=float)
    test_switch_score = test_rows["predicted_expert_quality"].to_numpy(dtype=float) - test_rows["predicted_unet_quality"].to_numpy(dtype=float)
    threshold, val_tune = tune_switch_threshold(val_rows, val_switch_score, args.min_expert_rate, args.max_expert_rate)

    rows = []
    rows.append({**policy_metrics(f"{expert_name}:always_unet", test_rows, np.zeros(len(test_rows), dtype=bool)), "expert": expert_name, "threshold": np.inf})
    rows.append({**policy_metrics(f"{expert_name}:always_expert", test_rows, np.ones(len(test_rows), dtype=bool)), "expert": expert_name, "threshold": -np.inf})
    rows.append({**policy_metrics(f"{expert_name}:oracle_best_of_two", test_rows, test_rows["expert_dice"].to_numpy(dtype=float) > test_rows["unet_dice"].to_numpy(dtype=float)), "expert": expert_name, "threshold": np.nan})
    rows.append({**policy_metrics(f"{expert_name}:val_confidence_switch", test_rows, test_switch_score >= threshold), "expert": expert_name, "threshold": threshold, "val_tuned_dice": val_tune["selected_dice"], "val_expert_rate": val_tune["expert_rate"]})
    switch = fit_learned_switch(train_rows, include_quantum=False, seed=args.seed)
    val_learned_score = positive_probability(switch, switch_feature_matrix(val_rows, include_quantum=False))
    test_learned_score = positive_probability(switch, switch_feature_matrix(test_rows, include_quantum=False))
    learned_threshold, learned_val = tune_switch_threshold(val_rows, val_learned_score, args.min_expert_rate, args.max_expert_rate)
    rows.append({**policy_metrics(f"{expert_name}:learned_best_of_two_switch", test_rows, test_learned_score >= learned_threshold), "expert": expert_name, "threshold": learned_threshold, "val_tuned_dice": learned_val["selected_dice"], "val_expert_rate": learned_val["expert_rate"]})

    per = test_rows[[
        "sample_id",
        "source_file",
        "sequence_id",
        "split",
        "expert",
        "expert_dice",
        "expert_iou",
        "predicted_expert_quality",
        "score_margin",
        "score_margin_norm",
        "unet_dice",
        "unet_iou",
        "predicted_unet_quality",
    ]].copy()
    per["switch_score"] = test_switch_score
    per["val_threshold"] = threshold
    per["use_expert"] = test_switch_score >= threshold
    per["selected_dice"] = np.where(per["use_expert"], per["expert_dice"], per["unet_dice"])
    per["delta_vs_unet"] = per["selected_dice"] - per["unet_dice"]
    return rows, per


def evaluate_quantum_confidence_fusion(
    classical_model,
    quantum_model,
    qdf: pd.DataFrame,
    X: np.ndarray,
    unet: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    unet_quality_model,
    args,
) -> tuple[list[dict], pd.DataFrame]:
    train_df = qdf.loc[train_mask].copy().reset_index(drop=True)
    val_df = qdf.loc[val_mask].copy().reset_index(drop=True)
    test_df = qdf.loc[test_mask].copy().reset_index(drop=True)
    train_primary = classical_model.predict(X[train_mask])
    val_primary = classical_model.predict(X[val_mask])
    test_primary = classical_model.predict(X[test_mask])
    train_quantum = quantum_model.predict(X[train_mask])
    val_quantum = quantum_model.predict(X[val_mask])
    test_quantum = quantum_model.predict(X[test_mask])
    train_rows = selected_prompt_rows_with_quantum_agreement(train_df, train_primary, train_quantum)
    val_rows = selected_prompt_rows_with_quantum_agreement(val_df, val_primary, val_quantum)
    test_rows = selected_prompt_rows_with_quantum_agreement(test_df, test_primary, test_quantum)
    train_rows = attach_unet(train_rows, unet, predict_unet_quality(unet_quality_model, unet, train_rows["source_file"]))
    val_rows = attach_unet(val_rows, unet, predict_unet_quality(unet_quality_model, unet, val_rows["source_file"]))
    test_rows = attach_unet(test_rows, unet, predict_unet_quality(unet_quality_model, unet, test_rows["source_file"]))

    threshold, agreement_weight, quantum_weight, val_tune = tune_agreement_switch(
        val_rows,
        args.min_expert_rate,
        args.max_expert_rate,
        args.agreement_weights,
        args.quantum_weights,
    )
    base = test_rows["predicted_expert_quality"].to_numpy(dtype=float) - test_rows["predicted_unet_quality"].to_numpy(dtype=float)
    agreement = test_rows["quantum_agreement"].to_numpy(dtype=float)
    quantum_delta = test_rows["quantum_candidate_quality"].to_numpy(dtype=float) - test_rows["predicted_unet_quality"].to_numpy(dtype=float)
    switch_score = base + agreement_weight * (agreement - 0.5) + quantum_weight * quantum_delta
    use_expert = switch_score >= threshold
    row = policy_metrics("classical_sam:quantum_confidence_switch", test_rows, use_expert)
    row.update({
        "expert": "classical_histgb_quantum_confidence",
        "threshold": threshold,
        "agreement_weight": agreement_weight,
        "quantum_weight": quantum_weight,
        "val_tuned_dice": val_tune["selected_dice"],
        "val_expert_rate": val_tune["expert_rate"],
    })
    switch = fit_learned_switch(train_rows, include_quantum=True, seed=args.seed)
    val_learned = positive_probability(switch, switch_feature_matrix(val_rows, include_quantum=True))
    test_learned = positive_probability(switch, switch_feature_matrix(test_rows, include_quantum=True))
    learned_threshold, learned_val = tune_switch_threshold(val_rows, val_learned, args.min_expert_rate, args.max_expert_rate)
    learned_row = policy_metrics("classical_sam:quantum_learned_best_of_two_switch", test_rows, test_learned >= learned_threshold)
    learned_row.update({
        "expert": "classical_histgb_quantum_confidence",
        "threshold": learned_threshold,
        "agreement_weight": np.nan,
        "quantum_weight": np.nan,
        "val_tuned_dice": learned_val["selected_dice"],
        "val_expert_rate": learned_val["expert_rate"],
    })
    per = test_rows[[
        "sample_id",
        "source_file",
        "sequence_id",
        "split",
        "expert",
        "expert_dice",
        "expert_iou",
        "predicted_expert_quality",
        "quantum_candidate_quality",
        "quantum_agreement",
        "score_margin",
        "score_margin_norm",
        "unet_dice",
        "unet_iou",
        "predicted_unet_quality",
    ]].copy()
    per["switch_score"] = switch_score
    per["val_threshold"] = threshold
    per["agreement_weight"] = agreement_weight
    per["quantum_weight"] = quantum_weight
    per["use_expert"] = use_expert
    per["selected_dice"] = np.where(per["use_expert"], per["expert_dice"], per["unet_dice"])
    per["delta_vs_unet"] = per["selected_dice"] - per["unet_dice"]
    return [row, learned_row], per


def main():
    parser = argparse.ArgumentParser(description="Side-by-side confidence fusion of UNet and quantum-assisted SAM masks")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset_full_r48_mps.csv")
    parser.add_argument("--features", default="endoscopy_guidance/results/sam_prompt_quality_features_full_r48_samembed_context.npy")
    parser.add_argument("--unet_metrics_csv", default="endoscopy_guidance/results/strong_unet_pretrained_cvc_all/baseline_metrics.csv")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/side_by_side_unet_quantum_fusion.csv")
    parser.add_argument("--per_sample_csv", default="endoscopy_guidance/results/side_by_side_unet_quantum_fusion_per_sample.csv")
    parser.add_argument("--experts", nargs="+", default=["classical_histgb", "quantum_feature_ridge"])
    parser.add_argument("--split_mode", choices=["auto", "existing", "random_source"], default="auto")
    parser.add_argument("--train_fraction", type=float, default=0.60)
    parser.add_argument("--val_fraction", type=float, default=0.20)
    parser.add_argument("--pqk_components", type=int, default=8)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--min_expert_rate", type=float, default=0.0)
    parser.add_argument("--max_expert_rate", type=float, default=0.50)
    parser.add_argument("--agreement_weights", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--quantum_weights", type=float, nargs="+", default=[0.0, 0.10, 0.25, 0.50])
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    qdf, X, unet = load_inputs(Path(args.prompt_quality_csv), Path(args.features), Path(args.unet_metrics_csv))
    train_mask, val_mask, test_mask = split_masks(qdf, args.split_mode, args.seed, args.train_fraction, args.val_fraction)
    y_train = qdf.loc[train_mask, "sam_dice"].to_numpy(dtype=float)
    train_sources = set(qdf.loc[train_mask, "source_file"].astype(str))
    unet_quality_model = fit_unet_quality_model(unet, train_sources, args.seed)

    summary_rows = []
    per_sample_rows = []
    fitted_models = {}
    for expert in args.experts:
        prompt_model = fit_prompt_selector(expert, X[train_mask], y_train, args.pqk_components, args.pqk_reps, args.seed)
        fitted_models[expert] = prompt_model
        rows, per = evaluate_expert(
            expert,
            prompt_model,
            qdf,
            X,
            unet,
            train_mask,
            val_mask,
            test_mask,
            unet_quality_model,
            args,
        )
        summary_rows.extend(rows)
        per_sample_rows.append(per)
    if {"classical_histgb", "quantum_feature_ridge"}.issubset(fitted_models):
        rows, per = evaluate_quantum_confidence_fusion(
            fitted_models["classical_histgb"],
            fitted_models["quantum_feature_ridge"],
            qdf,
            X,
            unet,
            train_mask,
            val_mask,
            test_mask,
            unet_quality_model,
            args,
        )
        summary_rows.extend(rows)
        per_sample_rows.append(per)

    summary = pd.DataFrame(summary_rows).sort_values(["selected_dice", "hard_selected_dice"], ascending=False)
    output = Path(args.output_csv)
    per_output = Path(args.per_sample_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    pd.concat(per_sample_rows, ignore_index=True).to_csv(per_output, index=False)
    print(f"[saved] {output}")
    print(f"[saved] {per_output}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
