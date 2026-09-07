from __future__ import annotations

import math

import numpy as np
from skimage.morphology import dilation, disk
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def _peak_index(heatmap: np.ndarray) -> tuple[int, int]:
    peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(peak[0]), int(peak[1])


def _largest_component_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    labeled = label(mask.astype(np.uint8))
    props = regionprops(labeled)
    if not props:
        return None
    largest = max(props, key=lambda prop: prop.area)
    cy, cx = largest.centroid
    return float(cy), float(cx)


def confusion_counts(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[int, int, int, int]:
    pred = _as_bool_mask(pred_mask)
    gt = _as_bool_mask(gt_mask)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return tp, fp, fn, tn


def component_count(mask: np.ndarray) -> int:
    return int(label(_as_bool_mask(mask).astype(np.uint8)).max())


def mask_area_ratio(mask: np.ndarray) -> float:
    mask_bool = _as_bool_mask(mask)
    return float(mask_bool.sum() / mask_bool.size) if mask_bool.size else 0.0


def heatmap_peak_value(heatmap: np.ndarray) -> float:
    heatmap_array = np.asarray(heatmap, dtype=np.float32)
    if heatmap_array.size == 0:
        return 0.0
    return float(heatmap_array.max())


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = _as_bool_mask(pred_mask)
    gt = _as_bool_mask(gt_mask)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0

    intersection = (pred & gt).sum()
    return float(2.0 * intersection / (pred.sum() + gt.sum()))


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = _as_bool_mask(pred_mask)
    gt = _as_bool_mask(gt_mask)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0

    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection / union) if union > 0 else 0.0


def precision_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    tp, fp, _, _ = confusion_counts(pred_mask, gt_mask)
    if tp == 0 and fp == 0:
        return 1.0 if not _as_bool_mask(gt_mask).any() else 0.0
    return float(tp / (tp + fp))


def recall_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    tp, _, fn, _ = confusion_counts(pred_mask, gt_mask)
    if tp == 0 and fn == 0:
        return 1.0
    return float(tp / (tp + fn))


def f_beta_score(pred_mask: np.ndarray, gt_mask: np.ndarray, beta: float = 2.0) -> float:
    precision = precision_score(pred_mask, gt_mask)
    recall = recall_score(pred_mask, gt_mask)
    beta_sq = beta**2
    denominator = beta_sq * precision + recall
    if denominator == 0.0:
        return 0.0
    return float((1.0 + beta_sq) * precision * recall / denominator)


def mae_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = _as_bool_mask(pred_mask).astype(np.float32)
    gt = _as_bool_mask(gt_mask).astype(np.float32)
    if pred.size == 0:
        return 0.0
    return float(np.abs(pred - gt).mean())


def boundary_f1_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance: int = 2,
) -> float:
    pred = _as_bool_mask(pred_mask)
    gt = _as_bool_mask(gt_mask)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    pred_boundary = find_boundaries(pred, mode="inner")
    gt_boundary = find_boundaries(gt, mode="inner")
    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    footprint = disk(max(1, tolerance))
    pred_dilated = dilation(pred_boundary, footprint)
    gt_dilated = dilation(gt_boundary, footprint)

    boundary_precision = float((pred_boundary & gt_dilated).sum() / pred_boundary.sum())
    boundary_recall = float((gt_boundary & pred_dilated).sum() / gt_boundary.sum())
    if boundary_precision + boundary_recall == 0.0:
        return 0.0
    return float(
        2.0 * boundary_precision * boundary_recall
        / (boundary_precision + boundary_recall)
    )


def pointing_game(heatmap: np.ndarray, gt_mask: np.ndarray) -> int:
    peak_y, peak_x = _peak_index(np.asarray(heatmap))
    gt = _as_bool_mask(gt_mask)
    return int(gt[peak_y, peak_x])


def peak_to_center_distance(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    centroid = _largest_component_centroid(_as_bool_mask(gt_mask))
    if centroid is None:
        return float("inf")

    peak_y, peak_x = _peak_index(np.asarray(heatmap))
    cy, cx = centroid
    return float(math.sqrt((peak_y - cy) ** 2 + (peak_x - cx) ** 2))


def mask_energy_ratio(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    gt = _as_bool_mask(gt_mask)
    total_energy = float(heatmap.sum())
    if total_energy <= 0.0:
        return 0.0
    inside_energy = float(heatmap[gt].sum())
    return inside_energy / total_energy
