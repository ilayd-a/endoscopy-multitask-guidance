import numpy as np
from skimage.measure import label, regionprops


# segmentation metrics

def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute the Dice coefficient between two binary masks

    Parameters:
    pred_mask : np.ndarray
        predicted binary mask (H, W), values in {0, 1}
    gt_mask : np.ndarray
        ground-truth binary mask (H, W), values in {0, 1}

    Return:
    float, dice coefficient in [0, 1]
    return 1.0 when both masks are empty
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0

    intersection = (pred & gt).sum()
    return float(2.0 * intersection / (pred.sum() + gt.sum()))


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute the IoU between two binary masks

    Parameters:
    pred_mask : np.ndarray
        predicted binary mask (H, W), values in {0, 1}
    gt_mask : np.ndarray
        ground-truth binary mask (H, W), values in {0, 1}

    Return:
    float, IoU in [0, 1]
    Returns 1.0 when both masks are empty
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0

    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection / union)


# heatmap metrics

def pointing_game(heatmap: np.ndarray, gt_mask: np.ndarray) -> int:
    """Pointing Game accuracy for a single sample
    Parameters:
    heatmap : np.ndarray
        predicted heatmap (H, W), float values.
    gt_mask : np.ndarray
        ground-truth binary mask (H, W), values in {0, 1}.

    Return:
    int, 1 if the heatmap peak is inside the mask, 0 otherwise.
    """
    peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(gt_mask[peak[0], peak[1]] > 0)


def peak_to_center_distance(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    """Euclidean distance between the heatmap peak and the ground-truth mask centroid

    Parameters:
    heatmap : np.ndarray
        predicted heatmap (H, W), float values.
    gt_mask : np.ndarray
        ground-truth binary mask (H, W), values in {0, 1}.

    Return:
    float, Euclidean distance in pixels
    return float('inf') if the mask is empty
    """
    peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    labeled = label(gt_mask.astype(np.uint8))
    props = regionprops(labeled)
    if len(props) == 0:
        return float("inf")

    largest = max(props, key=lambda r: r.area)
    centroid = largest.centroid

    dist = np.sqrt((peak[0] - centroid[0]) ** 2 + (peak[1] - centroid[1]) ** 2)
    return float(dist)
