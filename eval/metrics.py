from __future__ import annotations

import math

import numpy as np
from skimage.measure import label, regionprops


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
