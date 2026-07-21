"""
sam_prompt_quality_ranker.py
============================
Train prompt rankers on the actual SAM mask quality of candidate prompts.

The previous benchmark trained rankers to predict whether a candidate point hit
the ground-truth mask. That is only an indirect target. This script labels each
candidate point/box prompt by the Dice score of the SAM mask it produces, then
trains classical and projected quantum-kernel prompt rankers to choose prompts
that directly maximize downstream segmentation quality.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
EBTC_EXPERIMENTS = ROOT / "EndoscopicBladderTissue" / "experiments"
if str(EBTC_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EBTC_EXPERIMENTS))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from publication_benchmark_ebtc import ProjectedQuantumKernelSVC
from candidate_ranking_benchmark import build_candidates, fit_low_dim
from sam_quantum_prompt_benchmark import (
    choose_device,
    dice_iou,
    load_sam_predictor,
    load_sequence_map,
    prompt_box,
    row_dataframe,
    set_cached_or_compute_image,
)


def load_or_build_candidates(args):
    data_dir = Path(args.data_dir)
    cache_path = Path(args.candidate_cache) if args.candidate_cache else None
    cache_key = {
        "data_dir": str(data_dir.resolve()),
        "top_n": args.top_n,
        "grid_stride": args.grid_stride,
        "nms_dist": args.nms_dist,
        "patch_radius": args.patch_radius,
        "use_image_features": True,
    }
    if cache_path is not None and cache_path.exists():
        payload = pickle.loads(cache_path.read_bytes())
        if payload.get("cache_key") == cache_key:
            print(f"[cache] loaded candidates from {cache_path}")
            return payload["X"], payload["y"], payload["rows"]

    X, y, rows = build_candidates(
        data_dir=data_dir,
        top_n=args.top_n,
        grid_stride=args.grid_stride,
        nms_dist=args.nms_dist,
        radius=args.patch_radius,
        use_image_features=True,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps({"cache_key": cache_key, "X": X, "y": y, "rows": rows}))
        print(f"[cache] saved candidates to {cache_path}")
    return X, y, rows


def sam_prompt_masks_all(predictor, point_yxs: list[tuple[int, int]], image_shape: tuple[int, int], radius: int):
    coords = np.asarray([[[x, y]] for y, x in point_yxs], dtype=np.float32)
    coords = predictor.transform.apply_coords(coords, image_shape)
    point_coords = torch.as_tensor(coords, dtype=torch.float32, device=predictor.device)
    point_labels = torch.ones((len(point_yxs), 1), dtype=torch.int, device=predictor.device)
    boxes = np.asarray([prompt_box(point, image_shape, radius) for point in point_yxs], dtype=np.float32)
    boxes = predictor.transform.apply_boxes(boxes, image_shape)
    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=predictor.device)
    masks, scores, _ = predictor.predict_torch(
        point_coords=point_coords,
        point_labels=point_labels,
        boxes=boxes,
        multimask_output=True,
        return_logits=False,
    )
    best_idx = torch.argmax(scores, dim=1)
    out_masks = []
    out_scores = []
    for i, mask_idx in enumerate(best_idx.tolist()):
        out_masks.append(masks[i, mask_idx].detach().cpu().numpy().astype(bool))
        out_scores.append(float(scores[i, mask_idx].item()))
    return out_masks, out_scores


def build_prompt_quality_cache(args):
    data_dir = Path(args.data_dir)
    quality_csv = Path(args.prompt_quality_csv)
    quality_features = Path(args.prompt_quality_features)
    if args.use_prompt_quality_cache and quality_csv.exists() and quality_features.exists():
        print(f"[cache] using prompt quality cache {quality_csv}")
        return pd.read_csv(quality_csv), np.load(quality_features)

    X, _, rows = load_or_build_candidates(args)
    sequence_map = load_sequence_map(Path(args.endoscopy_repo))
    df = row_dataframe(rows, data_dir / "export_summary.csv", sequence_map)
    predictor = load_sam_predictor(Path(args.checkpoint), args.model_type, choose_device(args.device))
    embedding_cache = Path(args.sam_embedding_cache_dir) if args.sam_embedding_cache_dir else None

    out_rows = []
    out_features = []
    sample_ids = []
    if args.max_samples_per_split > 0:
        for split in ["train", "val", "test"]:
            split_ids = sorted(df.loc[df["split"].eq(split), "sample_id"].unique())
            sample_ids.extend(split_ids[: args.max_samples_per_split])
    else:
        sample_ids = sorted(df["sample_id"].unique())

    for sample_count, sid in enumerate(sample_ids, start=1):
        sample_all = df[df["sample_id"].eq(sid)].copy()
        if args.candidate_pool == "all":
            sample = sample_all
        elif args.candidate_pool == "heatmap":
            sample = sample_all.sort_values("heatmap_score", ascending=False).head(args.candidates_per_sample)
        elif args.candidate_pool == "mixed":
            heatmap = sample_all.sort_values("heatmap_score", ascending=False).head(args.candidates_per_sample)
            diverse = sample_all.sort_values("source_index", ascending=True).head(args.candidates_per_sample)
            grid = sample_all[sample_all["source_index"] >= args.top_n].head(args.candidates_per_sample)
            sample = pd.concat([heatmap, diverse, grid], axis=0).drop_duplicates()
        else:
            raise ValueError(f"Unknown candidate_pool={args.candidate_pool}")
        image = np.load(data_dir / f"image_{sid}.npy")
        gt = np.load(data_dir / f"gt_mask_{sid}.npy")
        set_cached_or_compute_image(predictor, image, sid, embedding_cache)
        points = [(int(row.y), int(row.x)) for row in sample.itertuples(index=False)]
        for radius in args.radii:
            masks, sam_scores = sam_prompt_masks_all(predictor, points, image.shape[:2], radius)
            for local_idx, (row, mask, sam_score) in enumerate(zip(sample.itertuples(index=True), masks, sam_scores)):
                dice, iou = dice_iou(mask, gt)
                out_rows.append({
                    "sample_id": sid,
                    "candidate_index": int(row.Index),
                    "source_file": row.source_file,
                    "sequence_id": int(row.sequence_id),
                    "split": row.split,
                    "y": int(row.y),
                    "x": int(row.x),
                    "radius": int(radius),
                    "heatmap_score": float(row.heatmap_score),
                    "point_hit": int(row.label > 0),
                    "center_dist": float(row.center_dist),
                    "sam_score": float(sam_score),
                    "sam_dice": float(dice),
                    "sam_iou": float(iou),
                })
                radius_feature = np.asarray([radius / 128.0, sam_score], dtype=float)
                out_features.append(np.concatenate([X[int(row.Index)], radius_feature]))
        if sample_count % 25 == 0 or sample_count == len(sample_ids):
            print(f"[quality] {sample_count}/{len(sample_ids)} samples", flush=True)

    qdf = pd.DataFrame(out_rows)
    Q = np.vstack(out_features).astype(np.float32)
    quality_csv.parent.mkdir(parents=True, exist_ok=True)
    qdf.to_csv(quality_csv, index=False)
    np.save(quality_features, Q)
    print(f"[saved] {quality_csv}")
    print(f"[saved] {quality_features}")
    return qdf, Q


def sample_level_eval(qdf: pd.DataFrame, scores: np.ndarray, name: str) -> dict:
    rows = qdf.copy()
    rows["_score"] = scores
    chosen = rows.sort_values("_score", ascending=False).groupby("sample_id", as_index=False).head(1)
    return {
        "strategy": name,
        "samples": int(chosen["sample_id"].nunique()),
        "prompt_hit": float(chosen["point_hit"].mean()),
        "dice_mean": float(chosen["sam_dice"].mean()),
        "iou_mean": float(chosen["sam_iou"].mean()),
        "sam_score_mean": float(chosen["sam_score"].mean()),
    }


def per_sample_normalize(values: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for sid in np.unique(sample_ids):
        mask = sample_ids == sid
        sample_values = values[mask].astype(float)
        lo = float(np.nanmin(sample_values))
        hi = float(np.nanmax(sample_values))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out[mask] = (sample_values - lo) / (hi - lo)
    return out


def blended_sample_scores(
    qdf: pd.DataFrame,
    model_scores: np.ndarray,
    model_weight: float,
    sam_weight: float,
    heatmap_weight: float,
) -> np.ndarray:
    sample_ids = qdf["sample_id"].to_numpy()
    model_norm = per_sample_normalize(np.asarray(model_scores, dtype=float), sample_ids)
    sam_norm = per_sample_normalize(qdf["sam_score"].to_numpy(dtype=float), sample_ids)
    heatmap_norm = per_sample_normalize(qdf["heatmap_score"].to_numpy(dtype=float), sample_ids)
    return model_weight * model_norm + sam_weight * sam_norm + heatmap_weight * heatmap_norm


def projected_quantum_features(X_angle: np.ndarray, reps: int) -> np.ndarray:
    features = [np.cos(X_angle), np.sin(X_angle)]
    if reps >= 2:
        features.extend([np.cos(2 * X_angle), np.sin(2 * X_angle)])
    if reps >= 3:
        features.extend([np.cos(3 * X_angle), np.sin(3 * X_angle)])
    return np.concatenate(features, axis=1)


def evaluate_rankers(args):
    qdf, Q = build_prompt_quality_cache(args)
    if args.eval_split == "val":
        train_mask = qdf["sequence_id"].le(23).to_numpy()
        eval_mask = qdf["split"].eq("val").to_numpy()
    else:
        train_mask = qdf["sequence_id"].le(26).to_numpy()
        eval_mask = qdf["split"].eq("test").to_numpy()

    train = qdf[train_mask].copy()
    eval_df = qdf[eval_mask].copy()
    X_train = Q[train_mask]
    X_eval = Q[eval_mask]
    y_train_reg = train["sam_dice"].to_numpy(dtype=float)
    good_threshold = args.good_threshold
    if good_threshold <= 0:
        good_threshold = float(np.quantile(y_train_reg, 0.70))
    y_train_good = (y_train_reg >= good_threshold).astype(int)

    results = []
    results.append(sample_level_eval(eval_df, eval_df["heatmap_score"].to_numpy(dtype=float), "heatmap_score"))
    results.append(sample_level_eval(eval_df, eval_df["sam_score"].to_numpy(dtype=float), "sam_score"))
    results.append(sample_level_eval(eval_df, eval_df["sam_dice"].to_numpy(dtype=float), "oracle_prompt_quality"))

    regressors = {
        "Classical_HistGBReg": HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=args.seed),
        "Classical_RidgeReg": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }
    if args.model_set == "pqk_only":
        regressors = {}
    elif args.model_set == "full":
        regressors.update({
            "Classical_ExtraTreesReg": ExtraTreesRegressor(n_estimators=400, random_state=args.seed),
            "Classical_RandomForestReg": RandomForestRegressor(n_estimators=250, random_state=args.seed),
        })
    for name, model in regressors.items():
        model.fit(X_train, y_train_reg)
        pred = model.predict(X_eval)
        row = sample_level_eval(eval_df, pred, name)
        row["train_rmse"] = float(mean_squared_error(y_train_reg, model.predict(X_train)) ** 0.5)
        results.append(row)
        for model_w, sam_w, heatmap_w in args.blend_weights:
            blend = blended_sample_scores(eval_df, pred, model_w, sam_w, heatmap_w)
            results.append(sample_level_eval(
                eval_df,
                blend,
                f"{name}_blend_m{model_w:g}_s{sam_w:g}_h{heatmap_w:g}",
            ))

    if len(np.unique(y_train_good)) == 2:
        for n_components in args.pqk_components:
            X_train_q, X_eval_q, pca_info = fit_low_dim(X_train, X_eval, n_components, args.seed)
            Z_train = projected_quantum_features(X_train_q, args.pqk_reps)
            Z_eval = projected_quantum_features(X_eval_q, args.pqk_reps)
            if args.model_set == "pqk_only":
                qregressors = {}
            else:
                qregressors = {
                    f"QML_PQF_RidgeReg_{n_components}pc": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                    f"QML_PQF_HistGBReg_{n_components}pc": HistGradientBoostingRegressor(
                        max_iter=180, learning_rate=0.05, random_state=args.seed
                    ),
                }
                if args.model_set == "full":
                    qregressors[f"QML_PQF_ExtraTreesReg_{n_components}pc"] = ExtraTreesRegressor(
                        n_estimators=300, random_state=args.seed
                    )
            for qname, qmodel in qregressors.items():
                qmodel.fit(Z_train, y_train_reg)
                qscores = qmodel.predict(Z_eval)
                row = sample_level_eval(eval_df, qscores, qname)
                row["pca_variance"] = pca_info["pca_variance_retained"]
                results.append(row)
                for model_w, sam_w, heatmap_w in args.blend_weights:
                    blend = blended_sample_scores(eval_df, qscores, model_w, sam_w, heatmap_w)
                    blend_row = sample_level_eval(
                        eval_df,
                        blend,
                        f"{qname}_blend_m{model_w:g}_s{sam_w:g}_h{heatmap_w:g}",
                    )
                    blend_row["pca_variance"] = pca_info["pca_variance_retained"]
                    results.append(blend_row)

            pqk = ProjectedQuantumKernelSVC(gamma="scale", reps=args.pqk_reps, C=args.pqk_c, class_weight="balanced")
            pqk.fit(X_train_q, y_train_good)
            scores = pqk.predict_proba(X_eval_q)[:, 1]
            row = sample_level_eval(eval_df, scores, f"QML_PQK_quality_{n_components}pc")
            row["pca_variance"] = pca_info["pca_variance_retained"]
            row["good_threshold"] = good_threshold
            results.append(row)
            for model_w, sam_w, heatmap_w in args.blend_weights:
                blend = blended_sample_scores(eval_df, scores, model_w, sam_w, heatmap_w)
                blend_row = sample_level_eval(
                    eval_df,
                    blend,
                    f"QML_PQK_quality_{n_components}pc_blend_m{model_w:g}_s{sam_w:g}_h{heatmap_w:g}",
                )
                blend_row["pca_variance"] = pca_info["pca_variance_retained"]
                blend_row["good_threshold"] = good_threshold
                results.append(blend_row)

    summary = pd.DataFrame(results).sort_values("dice_mean", ascending=False)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"[saved] {output_csv}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    parser = argparse.ArgumentParser(description="Train rankers on actual SAM prompt Dice")
    parser.add_argument("--data_dir", default="endoscopy_guidance/exports/cvc_all_rgb")
    parser.add_argument("--endoscopy_repo", default="/Users/ilaydadilek/Documents/FAMS LAB/endoscopy-multitask-guidance")
    parser.add_argument("--checkpoint", default="models/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--candidate_cache", default="endoscopy_guidance/results/sam_prompt_candidate_cache.pkl")
    parser.add_argument("--sam_embedding_cache_dir", default="endoscopy_guidance/results/sam_embedding_cache_vit_b")
    parser.add_argument("--prompt_quality_csv", default="endoscopy_guidance/results/sam_prompt_quality_dataset.csv")
    parser.add_argument("--prompt_quality_features", default="endoscopy_guidance/results/sam_prompt_quality_features.npy")
    parser.add_argument("--output_csv", default="endoscopy_guidance/results/sam_prompt_quality_ranker_summary.csv")
    parser.add_argument("--grid_stride", type=int, default=32)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--nms_dist", type=int, default=24)
    parser.add_argument("--patch_radius", type=int, default=24)
    parser.add_argument("--candidates_per_sample", type=int, default=20)
    parser.add_argument("--candidate_pool", choices=["heatmap", "mixed", "all"], default="mixed")
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--radii", type=int, nargs="+", default=[32, 48, 64, 96])
    parser.add_argument("--eval_split", choices=["val", "test"], default="val")
    parser.add_argument("--pqk_components", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--pqk_reps", type=int, default=2)
    parser.add_argument("--pqk_c", type=float, default=1.0)
    parser.add_argument("--good_threshold", type=float, default=0.5)
    parser.add_argument(
        "--model_set",
        choices=["fast", "full", "pqk_only"],
        default="full",
        help="Use fast to skip the slowest tree-heavy regressors on full prompt-quality caches.",
    )
    parser.add_argument(
        "--blend_weights",
        type=float,
        nargs=3,
        action="append",
        metavar=("MODEL", "SAM", "HEATMAP"),
        default=[(0.7, 0.2, 0.1), (0.6, 0.2, 0.2), (0.5, 0.3, 0.2)],
        help="Per-sample normalized model/SAM/heatmap score blends to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use_prompt_quality_cache", action="store_true")
    args = parser.parse_args()
    evaluate_rankers(args)


if __name__ == "__main__":
    main()
