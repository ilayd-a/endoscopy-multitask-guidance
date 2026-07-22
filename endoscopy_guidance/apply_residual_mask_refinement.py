"""
apply_residual_mask_refinement.py
=================================
Apply residual patch classifiers back to full segmentation masks.

The training labels come from sampled residual patches, but evaluation happens
at the frame level: every test frame receives a refined mask and is scored with
Dice/IoU. Thresholds for add/remove actions are selected on the validation
split only, then applied once to the held-out test split.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_residual_patch_dataset import build_features, boundary, mask_centroid
from export_classical_cvc_baseline import dice_iou
from residual_patch_quantum_benchmark import limit_training
from sam_prompt_quality_ranker import projected_quantum_features


@dataclass
class QuantumProjector:
    scaler: StandardScaler
    pca: PCA
    angle_scaler: MinMaxScaler
    reps: int

    @classmethod
    def fit(cls, X: np.ndarray, components: int, reps: int, seed: int) -> "QuantumProjector":
        max_components = max(1, min(components, X.shape[1], X.shape[0] - 1))
        scaler = StandardScaler()
        pca = PCA(n_components=max_components, random_state=seed)
        angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        train_std = scaler.fit_transform(X)
        train_pca = pca.fit_transform(train_std)
        angle_scaler.fit(train_pca)
        return cls(scaler=scaler, pca=pca, angle_scaler=angle_scaler, reps=reps)

    def transform(self, X: np.ndarray) -> np.ndarray:
        low_dim = self.angle_scaler.transform(self.pca.transform(self.scaler.transform(X)))
        return projected_quantum_features(low_dim, self.reps)

    def hybrid_transform(self, X: np.ndarray) -> np.ndarray:
        return np.concatenate([X, self.transform(X)], axis=1)


def candidate_mask(prob: np.ndarray, pred: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    uncertainty = 1.0 - np.abs(prob - 0.5) * 2.0
    mask = boundary(pred, args.boundary_radius)
    mask |= uncertainty >= np.quantile(uncertainty, args.uncertainty_quantile)
    mask |= prob >= np.quantile(prob, args.prob_quantile)
    mask |= (prob >= args.low_prob) & (prob <= args.high_prob)
    if pred.mean() <= args.empty_pred_area:
        mask |= prob >= np.quantile(prob, args.empty_prob_quantile)
    return mask


def frame_risk_features(prob: np.ndarray, pred: np.ndarray) -> list[float]:
    uncertainty = 1.0 - np.abs(prob - 0.5) * 2.0
    pred_u8 = pred.astype(np.uint8)
    n_components, _, stats, _ = cv2.connectedComponentsWithStats(pred_u8, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if n_components > 1 else np.asarray([], dtype=np.float32)
    pred_boundary = boundary(pred, radius=3)
    return [
        float(prob.mean()),
        float(prob.std()),
        float(prob.max()),
        float(np.quantile(prob, 0.90)),
        float(np.quantile(prob, 0.95)),
        float(np.quantile(prob, 0.99)),
        float(pred.mean()),
        float(pred_boundary.mean()),
        float(uncertainty.mean()),
        float(uncertainty.std()),
        float(np.quantile(uncertainty, 0.95)),
        float(n_components - 1),
        float(component_areas.max() / pred.size) if len(component_areas) else 0.0,
        float(component_areas.mean() / pred.size) if len(component_areas) else 0.0,
    ]


def frame_features(base: Path, row, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sid = row.sample_id
    image = np.load(base / "images" / f"{sid}.npy")
    image_float = image.astype(np.float32) / 255.0
    gt = np.load(base / "gt_masks" / f"{sid}.npy").astype(np.uint8)
    prob = np.load(base / "prob_maps" / f"{sid}.npy").astype(np.float32)
    pred = np.load(base / "pred_masks" / f"{sid}.npy").astype(np.uint8)
    mask = candidate_mask(prob, pred, args)
    ys, xs = np.where(mask)
    if len(ys) > args.max_candidates_per_frame:
        priority = np.maximum(prob[ys, xs], 1.0 - np.abs(prob[ys, xs] - 0.5) * 2.0)
        keep = np.argsort(priority)[-args.max_candidates_per_frame :]
        ys, xs = ys[keep], xs[keep]
    centroid = mask_centroid(gt)
    feats = [
        build_features(image_float, prob, pred, centroid, int(y), int(x), args.patch_radius)
        for y, x in zip(ys, xs)
    ]
    return np.asarray(feats, dtype=np.float32), ys, xs, pred, gt


def refine_mask(
    probs: np.ndarray,
    classes: list[int],
    ys: np.ndarray,
    xs: np.ndarray,
    pred: np.ndarray,
    remove_threshold: float,
    add_threshold: float,
) -> np.ndarray:
    refined = pred.copy()
    if len(probs) == 0:
        return refined
    remove_idx = classes.index(1) if 1 in classes else None
    add_idx = classes.index(2) if 2 in classes else None
    if remove_idx is not None:
        remove = (pred[ys, xs] == 1) & (probs[:, remove_idx] >= remove_threshold)
        refined[ys[remove], xs[remove]] = 0
    if add_idx is not None:
        add = (pred[ys, xs] == 0) & (probs[:, add_idx] >= add_threshold)
        refined[ys[add], xs[add]] = 1
    return refined


def attach_probabilities(cached_frames: list[dict], model, split: str) -> list[dict]:
    classes = list(model.classes_)
    for count, frame in enumerate(cached_frames, start=1):
        X = frame["X"]
        frame["probs"] = model.predict_proba(X) if len(X) else np.empty((0, len(classes)), dtype=np.float32)
        frame["classes"] = classes
        if count % 50 == 0 or count == len(cached_frames):
            print(f"[score:{split}] {count}/{len(cached_frames)} frames", flush=True)
    return cached_frames


def build_split_cache(
    name: str,
    base: Path,
    metrics_df: pd.DataFrame,
    args: argparse.Namespace,
) -> list[dict]:
    cached = []
    split_df = metrics_df.loc[metrics_df["split"].eq(name)].copy()
    for count, row in enumerate(split_df.itertuples(index=False), start=1):
        X, ys, xs, pred, gt = frame_features(base, row, args)
        cached.append(
            {
                "sample_id": row.sample_id,
                "source_file": row.source_file,
                "risk_features": frame_risk_features(np.load(base / "prob_maps" / f"{row.sample_id}.npy").astype(np.float32), np.load(base / "pred_masks" / f"{row.sample_id}.npy").astype(np.uint8)),
                "X": X,
                "ys": ys,
                "xs": xs,
                "pred": pred,
                "gt": gt,
            }
        )
        if count % 50 == 0 or count == len(split_df):
            print(f"[cache:{name}] {count}/{len(split_df)} frames", flush=True)
    return cached


def train_hard_trigger(base: Path, metrics_df: pd.DataFrame, args: argparse.Namespace):
    train_df = metrics_df.loc[metrics_df["split"].eq(args.train_split)].copy()
    X = []
    y = []
    for row in train_df.itertuples(index=False):
        prob = np.load(base / "prob_maps" / f"{row.sample_id}.npy").astype(np.float32)
        pred = np.load(base / "pred_masks" / f"{row.sample_id}.npy").astype(np.uint8)
        X.append(frame_risk_features(prob, pred))
        y.append(float(row.dice) < args.hard_dice_threshold)
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    if len(np.unique(y_arr)) < 2:
        return None
    trigger = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, random_state=args.seed)
    trigger.fit(X_arr, y_arr)
    return trigger


def attach_trigger_scores(cached_frames: list[dict], trigger) -> list[dict]:
    for frame in cached_frames:
        if trigger is None:
            frame["trigger_score"] = 1.0
        else:
            frame["trigger_score"] = float(trigger.predict_proba(np.asarray([frame["risk_features"]], dtype=np.float32))[0, 1])
    return cached_frames


def evaluate_cached_split(
    name: str,
    cached_frames: list[dict],
    args: argparse.Namespace,
    remove_threshold: float,
    add_threshold: float,
    trigger_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for frame in cached_frames:
        triggered = frame.get("trigger_score", 1.0) >= trigger_threshold
        if triggered:
            refined = refine_mask(
                frame["probs"],
                frame["classes"],
                frame["ys"],
                frame["xs"],
                frame["pred"],
                remove_threshold,
                add_threshold,
            )
        else:
            refined = frame["pred"].copy()
        base_dice, base_iou = dice_iou(frame["pred"], frame["gt"])
        refined_dice, refined_iou = dice_iou(refined, frame["gt"])
        rows.append(
            {
                "sample_id": frame["sample_id"],
                "source_file": frame["source_file"],
                "split": name,
                "baseline_dice": base_dice,
                "refined_dice": refined_dice,
                "delta_dice": refined_dice - base_dice,
                "baseline_iou": base_iou,
                "refined_iou": refined_iou,
                "delta_iou": refined_iou - base_iou,
                "candidate_pixels": int(len(frame["ys"])),
                "trigger_score": float(frame.get("trigger_score", 1.0)),
                "triggered": bool(triggered),
            }
        )
    frame = pd.DataFrame(rows)
    hard = frame["baseline_dice"] < args.hard_dice_threshold
    summary = {
        "split": name,
        "frames": int(len(frame)),
        "baseline_dice": float(frame["baseline_dice"].mean()),
        "refined_dice": float(frame["refined_dice"].mean()),
        "delta_dice": float(frame["delta_dice"].mean()),
        "improved_frames": int((frame["delta_dice"] > 1e-6).sum()),
        "worsened_frames": int((frame["delta_dice"] < -1e-6).sum()),
        "hard_frames": int(hard.sum()),
        "hard_baseline_dice": float(frame.loc[hard, "baseline_dice"].mean()) if hard.any() else np.nan,
        "hard_refined_dice": float(frame.loc[hard, "refined_dice"].mean()) if hard.any() else np.nan,
        "hard_delta_dice": float(frame.loc[hard, "delta_dice"].mean()) if hard.any() else np.nan,
        "mean_candidate_pixels": float(frame["candidate_pixels"].mean()),
        "triggered_frames": int(frame["triggered"].sum()),
    }
    return frame, summary


def tune_thresholds(cached_val: list[dict], args: argparse.Namespace) -> tuple[float, float, float, pd.DataFrame]:
    rows = []
    remove_grid = args.remove_threshold_grid or args.threshold_grid
    add_grid = args.add_threshold_grid or args.threshold_grid
    for remove_threshold in remove_grid:
        for add_threshold in add_grid:
            for trigger_threshold in args.trigger_threshold_grid:
                _, summary = evaluate_cached_split("val", cached_val, args, remove_threshold, add_threshold, trigger_threshold)
                summary["remove_threshold"] = remove_threshold
                summary["add_threshold"] = add_threshold
                summary["trigger_threshold"] = trigger_threshold
                rows.append(summary)
    table = pd.DataFrame(rows)
    if args.tune_objective == "hard":
        sort_cols = ["hard_refined_dice", "refined_dice", "delta_dice"]
    elif args.tune_objective == "delta":
        sort_cols = ["delta_dice", "hard_delta_dice", "refined_dice"]
    else:
        sort_cols = ["refined_dice", "hard_refined_dice", "delta_dice"]
    best = table.sort_values(sort_cols, ascending=False).iloc[0]
    return float(best.remove_threshold), float(best.add_threshold), float(best.trigger_threshold), table


def main():
    parser = argparse.ArgumentParser(description="Apply residual model predictions back to masks")
    parser.add_argument("--baseline_dir", default="endoscopy_guidance/results/strong_unet_pretrained_cvc_all")
    parser.add_argument("--dataset_npz", default="endoscopy_guidance/results/residual_patch_dataset_strong_cvc_all.npz")
    parser.add_argument("--rows_csv", default="endoscopy_guidance/results/residual_patch_dataset_strong_cvc_all.csv")
    parser.add_argument("--output_dir", default="endoscopy_guidance/results/residual_mask_refinement_strong_cvc_all")
    parser.add_argument(
        "--model",
        choices=["classical_histgb", "projected_quantum_histgb", "hybrid_quantum_histgb"],
        default="projected_quantum_histgb",
    )
    parser.add_argument("--max_train", type=int, default=50000)
    parser.add_argument("--pqk_components", type=int, default=12)
    parser.add_argument("--pqk_reps", type=int, default=3)
    parser.add_argument("--patch_radius", type=int, default=7)
    parser.add_argument("--boundary_radius", type=int, default=3)
    parser.add_argument("--max_candidates_per_frame", type=int, default=6000)
    parser.add_argument("--uncertainty_quantile", type=float, default=0.96)
    parser.add_argument("--prob_quantile", type=float, default=0.96)
    parser.add_argument("--empty_prob_quantile", type=float, default=0.90)
    parser.add_argument("--low_prob", type=float, default=0.03)
    parser.add_argument("--high_prob", type=float, default=0.97)
    parser.add_argument("--empty_pred_area", type=float, default=0.002)
    parser.add_argument("--hard_dice_threshold", type=float, default=0.80)
    parser.add_argument("--threshold_grid", type=float, nargs="*", default=[0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    parser.add_argument("--remove_threshold_grid", type=float, nargs="*", default=[])
    parser.add_argument("--add_threshold_grid", type=float, nargs="*", default=[])
    parser.add_argument("--trigger_threshold_grid", type=float, nargs="*", default=[0.0])
    parser.add_argument("--tune_objective", choices=["overall", "hard", "delta"], default="overall")
    parser.add_argument("--hard_trigger", action="store_true")
    parser.add_argument("--train_split", default="val")
    parser.add_argument("--tune_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    base = Path(args.baseline_dir)
    metrics_df = pd.read_csv(base / "baseline_metrics.csv")
    payload = np.load(args.dataset_npz, allow_pickle=True)
    X = payload["X"].astype(np.float32)
    y = payload["y"].astype(np.int64)
    patch_rows = pd.read_csv(args.rows_csv)
    train_mask = patch_rows["split"].eq(args.train_split).to_numpy()
    if not train_mask.any():
        raise ValueError(f"No patch rows found for train_split={args.train_split}")
    X_train, y_train = limit_training(X[train_mask], y[train_mask], args.max_train, args.seed)

    if args.model in {"projected_quantum_histgb", "hybrid_quantum_histgb"}:
        projector = QuantumProjector.fit(X_train, args.pqk_components, args.pqk_reps, args.seed)
        if args.model == "hybrid_quantum_histgb":
            X_fit = projector.hybrid_transform(X_train)
        else:
            X_fit = projector.transform(X_train)
        model = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, random_state=args.seed)
        model.fit(X_fit, y_train)

        class WrappedModel:
            def __init__(self):
                self.classes_ = model.classes_

            def predict_proba(self, batch: np.ndarray) -> np.ndarray:
                if args.model == "hybrid_quantum_histgb":
                    return model.predict_proba(projector.hybrid_transform(batch))
                return model.predict_proba(projector.transform(batch))

        fitted = WrappedModel()
    else:
        fitted = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, random_state=args.seed)
        fitted.fit(X_train, y_train)

    output = Path(args.output_dir) / args.model
    output.mkdir(parents=True, exist_ok=True)
    cached_tune = build_split_cache(args.tune_split, base, metrics_df, args)
    cached_test = build_split_cache(args.test_split, base, metrics_df, args)
    cached_tune = attach_probabilities(cached_tune, fitted, args.tune_split)
    cached_test = attach_probabilities(cached_test, fitted, args.test_split)
    trigger = train_hard_trigger(base, metrics_df, args) if args.hard_trigger else None
    cached_tune = attach_trigger_scores(cached_tune, trigger)
    cached_test = attach_trigger_scores(cached_test, trigger)
    remove_threshold, add_threshold, trigger_threshold, tune_table = tune_thresholds(cached_tune, args)
    tune_frames, tune_summary = evaluate_cached_split(args.tune_split, cached_tune, args, remove_threshold, add_threshold, trigger_threshold)
    test_frames, test_summary = evaluate_cached_split(args.test_split, cached_test, args, remove_threshold, add_threshold, trigger_threshold)
    for summary in (tune_summary, test_summary):
        summary["model"] = args.model
        summary["train_split"] = args.train_split
        summary["remove_threshold"] = remove_threshold
        summary["add_threshold"] = add_threshold
        summary["trigger_threshold"] = trigger_threshold
        summary["hard_trigger"] = bool(args.hard_trigger)
    pd.DataFrame([tune_summary, test_summary]).to_csv(output / "summary.csv", index=False)
    tune_table.to_csv(output / "validation_threshold_tuning.csv", index=False)
    pd.concat([tune_frames, test_frames], axis=0).to_csv(output / "per_frame.csv", index=False)
    print(f"[saved] {output / 'summary.csv'}")
    print(pd.DataFrame([tune_summary, test_summary]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
