"""
residual_gain_prompt_fusion.py
==============================
Train prompt selectors to predict candidate-mask gain over a fixed UNet.

Earlier side-by-side fusion ranked SAM prompts by their absolute Dice and then
decided whether to switch from UNet to SAM. This script targets the clinical
question more directly: for each candidate prompt, predict

    SAM_Dice(candidate) - UNet_Dice(frame)

and choose the prompt with the largest predicted gain. A validation-calibrated
threshold then decides whether the frame should use UNet or the alternate mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sam_prompt_active_learning import QuantumFeatureRidgeRegressor


UNET_FEATURE_COLUMNS = [
    "mean_prob",
    "mean_uncertainty",
    "boundary_uncertainty",
    "mask_area_frac",
]


def split_masks(qdf: pd.DataFrame, seed: int, train_fraction: float, val_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sources = np.asarray(sorted(qdf["source_file"].astype(str).unique()), dtype=object)
    rng.shuffle(sources)
    n_train = max(1, int(round(len(sources) * train_fraction)))
    n_val = max(1, int(round(len(sources) * val_fraction)))
    if n_train + n_val >= len(sources):
        raise ValueError("train_fraction + val_fraction leaves no held-out test sources")
    train_sources = set(sources[:n_train])
    val_sources = set(sources[n_train : n_train + n_val])
    test_sources = set(sources[n_train + n_val :])
    source_values = qdf["source_file"].astype(str)
    return (
        source_values.isin(train_sources).to_numpy(),
        source_values.isin(val_sources).to_numpy(),
        source_values.isin(test_sources).to_numpy(),
    )


def load_augmented_features(args) -> tuple[pd.DataFrame, np.ndarray]:
    qdf = pd.read_csv(args.prompt_quality_csv)
    X = np.load(args.features).astype(np.float32)
    if len(qdf) != len(X):
        raise ValueError(f"Prompt cache mismatch: {len(qdf)} rows but {len(X)} feature rows")

    unet = pd.read_csv(args.unet_metrics_csv)
    keep = qdf["source_file"].astype(str).isin(set(unet["source_file"].astype(str)))
    qdf = qdf.loc[keep].copy().reset_index(drop=True)
    X = X[keep.to_numpy()]
    unet_cols = ["source_file", "dice", "iou", *UNET_FEATURE_COLUMNS]
    qdf = qdf.merge(unet[unet_cols], on="source_file", how="left", suffixes=("", "_unet"))
    qdf = qdf.rename(columns={"dice": "unet_dice", "iou": "unet_iou"})

    extra = [qdf[UNET_FEATURE_COLUMNS].to_numpy(dtype=np.float32)]
    if args.unet_tta_csv:
        tta = pd.read_csv(args.unet_tta_csv)
        tta_columns = sorted(column for column in tta.columns if column.startswith("tta_"))
        qdf = qdf.merge(tta[["source_file", *tta_columns]], on="source_file", how="left")
        qdf[tta_columns] = qdf[tta_columns].fillna(0.0)
        extra.append(qdf[tta_columns].to_numpy(dtype=np.float32))

    X_aug = np.concatenate([X, *extra], axis=1).astype(np.float32)
    qdf["residual_gain"] = qdf["sam_dice"].to_numpy(dtype=float) - qdf["unet_dice"].to_numpy(dtype=float)
    qdf["beats_unet"] = qdf["residual_gain"] > 0
    return qdf, X_aug


def fit_model(name: str, X: np.ndarray, y: np.ndarray, components: int, reps: int, seed: int):
    if name == "classical_histgb":
        model = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=seed)
    elif name == "quantum_feature_ridge":
        model = QuantumFeatureRidgeRegressor(components, reps, seed)
    else:
        raise ValueError(f"Unknown model={name}")
    return model.fit(X, y)


def selected_rows(eval_df: pd.DataFrame, scores: np.ndarray, model_name: str, target_name: str) -> pd.DataFrame:
    rows = eval_df.copy().reset_index(drop=True)
    rows["_score"] = scores
    out = []
    for sid, group in rows.groupby("sample_id", sort=False):
        ranked = group.sort_values("_score", ascending=False).reset_index(drop=True)
        top = ranked.iloc[0].copy()
        second = float(ranked.iloc[1]["_score"]) if len(ranked) > 1 else float(top["_score"])
        score_values = group["_score"].to_numpy(dtype=float)
        spread = float(np.nanmax(score_values) - np.nanmin(score_values))
        top["predicted_gain"] = float(top["_score"])
        top["score_margin"] = float(top["_score"]) - second
        top["score_margin_norm"] = 0.0 if spread <= 0 else (float(top["_score"]) - second) / spread
        top["candidate_pred_mean"] = float(np.nanmean(score_values))
        top["candidate_pred_std"] = float(np.nanstd(score_values))
        top["candidate_pred_max"] = float(np.nanmax(score_values))
        top["candidate_pred_p90"] = float(np.nanpercentile(score_values, 90))
        top["candidate_count"] = int(len(group))
        top["model"] = model_name
        top["target"] = target_name
        out.append(top)
    return pd.DataFrame(out)


def policy_metrics(name: str, rows: pd.DataFrame, use_sam: np.ndarray) -> dict:
    selected_dice = np.where(use_sam, rows["sam_dice"].to_numpy(dtype=float), rows["unet_dice"].to_numpy(dtype=float))
    hard = rows["unet_dice"].to_numpy(dtype=float) < 0.80
    return {
        "policy": name,
        "frames": int(len(rows)),
        "selected_dice": float(selected_dice.mean()),
        "unet_dice": float(rows["unet_dice"].mean()),
        "sam_dice": float(rows["sam_dice"].mean()),
        "delta_vs_unet": float((selected_dice - rows["unet_dice"].to_numpy(dtype=float)).mean()),
        "sam_rate": float(use_sam.mean()),
        "sam_frames": int(use_sam.sum()),
        "hard_frames": int(hard.sum()),
        "hard_selected_dice": float(selected_dice[hard].mean()) if hard.any() else np.nan,
        "hard_unet_dice": float(rows.loc[hard, "unet_dice"].mean()) if hard.any() else np.nan,
        "hard_delta_vs_unet": float((selected_dice[hard] - rows.loc[hard, "unet_dice"].to_numpy(dtype=float)).mean()) if hard.any() else np.nan,
    }


def per_policy_rows(name: str, rows: pd.DataFrame, use_sam: np.ndarray, threshold: float) -> pd.DataFrame:
    per = rows[[
        "sample_id",
        "source_file",
        "model",
        "target",
        "sam_dice",
        "sam_iou",
        "unet_dice",
        "unet_iou",
        "residual_gain",
        "predicted_gain",
        "score_margin",
        "score_margin_norm",
        "y",
        "x",
        "radius",
        "heatmap_score",
        "sam_score",
    ]].copy()
    per["policy"] = name
    per["threshold"] = threshold
    per["use_sam"] = use_sam
    per["selected_dice"] = np.where(per["use_sam"], per["sam_dice"], per["unet_dice"])
    per["delta_vs_unet"] = per["selected_dice"] - per["unet_dice"]
    per["hard_unet"] = per["unet_dice"] < 0.80
    return per


def tune_threshold(val_rows: pd.DataFrame, score_col: str, min_rate: float, max_rate: float, hard_weight: float) -> tuple[float, dict]:
    scores = val_rows[score_col].to_numpy(dtype=float)
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))
    thresholds = np.concatenate([thresholds, [float(scores.max() + 1e-6)]])
    best = None
    for threshold in thresholds:
        use_sam = scores >= threshold
        rate = float(use_sam.mean())
        if rate < min_rate or rate > max_rate:
            continue
        row = policy_metrics("val_tuned", val_rows, use_sam)
        hard_selected = row["hard_selected_dice"]
        if np.isnan(hard_selected):
            hard_selected = row["selected_dice"]
        objective = row["selected_dice"] + hard_weight * hard_selected
        candidate = (objective, row["selected_dice"], hard_selected, -row["sam_rate"])
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold), row)
    if best is None:
        threshold = float(scores.max() + 1e-6)
        return threshold, policy_metrics("val_tuned", val_rows, scores >= threshold)
    return best[1], best[2]


def switch_feature_columns(rows: pd.DataFrame) -> list[str]:
    base = [
        "predicted_gain",
        "score_margin",
        "score_margin_norm",
        "candidate_pred_mean",
        "candidate_pred_std",
        "candidate_pred_max",
        "candidate_pred_p90",
        "candidate_count",
        "radius",
        "heatmap_score",
        "sam_score",
        "center_dist",
    ]
    columns = [column for column in base if column in rows.columns]
    columns.extend(column for column in UNET_FEATURE_COLUMNS if column in rows.columns)
    columns.extend(sorted(column for column in rows.columns if column.startswith("tta_")))
    return columns


def fit_switch_model(name: str, train_rows: pd.DataFrame, seed: int):
    columns = switch_feature_columns(train_rows)
    X = train_rows[columns].to_numpy(dtype=np.float32)
    y = train_rows["residual_gain"].to_numpy(dtype=float)
    if name == "histgb":
        model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.04, random_state=seed)
    elif name == "rf":
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown switch model={name}")
    model.fit(X, y)
    return model, columns


def evaluate_one_model(model_name: str, qdf: pd.DataFrame, X: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray, args) -> tuple[list[dict], pd.DataFrame]:
    rows = []
    per_outputs = []
    for target_name, target_col in [("residual_gain", "residual_gain"), ("absolute_sam_dice", "sam_dice")]:
        model = fit_model(
            model_name,
            X[train_mask],
            qdf.loc[train_mask, target_col].to_numpy(dtype=float),
            args.pqk_components,
            args.pqk_reps,
            args.seed,
        )
        train_rows = selected_rows(qdf.loc[train_mask].copy(), model.predict(X[train_mask]), model_name, target_name)
        val_rows = selected_rows(qdf.loc[val_mask].copy(), model.predict(X[val_mask]), model_name, target_name)
        test_rows = selected_rows(qdf.loc[test_mask].copy(), model.predict(X[test_mask]), model_name, target_name)
        threshold, val_tune = tune_threshold(val_rows, "predicted_gain", args.min_sam_rate, args.max_sam_rate, args.hard_weight)
        use_test = test_rows["predicted_gain"].to_numpy(dtype=float) >= threshold
        policy_defs = [
            (f"{model_name}:{target_name}:always_unet", np.zeros(len(test_rows), dtype=bool), np.inf, {}),
            (f"{model_name}:{target_name}:always_sam", np.ones(len(test_rows), dtype=bool), -np.inf, {}),
            (f"{model_name}:{target_name}:oracle_best_of_two", test_rows["sam_dice"].to_numpy(dtype=float) > test_rows["unet_dice"].to_numpy(dtype=float), np.nan, {}),
            (f"{model_name}:{target_name}:val_gain_switch", use_test, threshold, {"val_tuned_dice": val_tune["selected_dice"], "val_sam_rate": val_tune["sam_rate"]}),
        ]
        for policy_name, policy_use_sam, policy_threshold, extra in policy_defs:
            rows.append({
                **policy_metrics(policy_name, test_rows, policy_use_sam),
                "model": model_name,
                "target": target_name,
                "threshold": policy_threshold,
                **extra,
            })
            per_outputs.append(per_policy_rows(policy_name, test_rows, policy_use_sam, policy_threshold))
        for switch_name in args.switch_models:
            switch_model, switch_columns = fit_switch_model(switch_name, train_rows, args.seed)
            val_rows[f"switch_gain_{switch_name}"] = switch_model.predict(val_rows[switch_columns].to_numpy(dtype=np.float32))
            test_rows[f"switch_gain_{switch_name}"] = switch_model.predict(test_rows[switch_columns].to_numpy(dtype=np.float32))
            switch_threshold, switch_val = tune_threshold(val_rows, f"switch_gain_{switch_name}", args.min_sam_rate, args.max_sam_rate, args.hard_weight)
            use_switch = test_rows[f"switch_gain_{switch_name}"].to_numpy(dtype=float) >= switch_threshold
            policy_name = f"{model_name}:{target_name}:meta_{switch_name}_gain_switch"
            rows.append({
                **policy_metrics(policy_name, test_rows, use_switch),
                "model": model_name,
                "target": target_name,
                "threshold": switch_threshold,
                "val_tuned_dice": switch_val["selected_dice"],
                "val_sam_rate": switch_val["sam_rate"],
            })
            per_outputs.append(per_policy_rows(policy_name, test_rows, use_switch, switch_threshold))
    return rows, pd.concat(per_outputs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Residual-gain prompt fusion over a fixed UNet")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/kvasir_external_120_prompt_quality.csv")
    parser.add_argument("--features", default="endoscopy_guidance/results/kvasir_external_120_prompt_quality_features_samembed_context.npy")
    parser.add_argument("--unet_metrics_csv", default="endoscopy_guidance/results/strong_unet_pretrained_kvasir_train_val_test/baseline_metrics.csv")
    parser.add_argument("--unet_tta_csv", default="")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/residual_gain_prompt_fusion.csv")
    parser.add_argument("--per_sample_csv", default="endoscopy_guidance/results/residual_gain_prompt_fusion_per_sample.csv")
    parser.add_argument("--models", nargs="+", default=["classical_histgb", "quantum_feature_ridge"])
    parser.add_argument("--train_fraction", type=float, default=0.40)
    parser.add_argument("--val_fraction", type=float, default=0.40)
    parser.add_argument("--min_sam_rate", type=float, default=0.0)
    parser.add_argument("--max_sam_rate", type=float, default=0.50)
    parser.add_argument("--hard_weight", type=float, default=0.0, help="Validation objective weight for hard-frame Dice, where hard means UNet Dice < 0.80.")
    parser.add_argument("--switch_models", nargs="+", choices=["histgb", "rf"], default=["histgb", "rf"])
    parser.add_argument("--pqk_components", type=int, default=8)
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    qdf, X = load_augmented_features(args)
    train_mask, val_mask, test_mask = split_masks(qdf, args.seed, args.train_fraction, args.val_fraction)
    summary_rows = []
    per_rows = []
    for model_name in args.models:
        rows, per = evaluate_one_model(model_name, qdf, X, train_mask, val_mask, test_mask, args)
        summary_rows.extend(rows)
        per_rows.append(per)

    summary = pd.DataFrame(summary_rows).sort_values(["selected_dice", "hard_selected_dice"], ascending=False)
    output = Path(args.output_csv)
    per_output = Path(args.per_sample_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    pd.concat(per_rows, ignore_index=True).to_csv(per_output, index=False)
    print(f"[saved] {output}")
    print(f"[saved] {per_output}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
