from __future__ import annotations

import numpy as np

from .constants import MARKER_SUBSETS


def nearest_marker_distances(markers: np.ndarray, object_points: np.ndarray) -> np.ndarray:
    """Return each marker's nearest object-point distance with shape ``[T, M]``."""

    marker_arr = np.asarray(markers, dtype=np.float64)
    object_arr = np.asarray(object_points, dtype=np.float64)
    if marker_arr.ndim != 3 or marker_arr.shape[-1] != 3:
        raise ValueError(f"markers must have shape [T, M, 3], got {marker_arr.shape}")
    if object_arr.ndim != 3 or object_arr.shape[-1] != 3 or object_arr.shape[0] != marker_arr.shape[0]:
        raise ValueError(f"object_points must have shape [T, P, 3], got {object_arr.shape}")

    marker_norm = np.sum(marker_arr * marker_arr, axis=-1)[:, :, None]
    object_norm = np.sum(object_arr * object_arr, axis=-1)[:, None, :]
    squared = marker_norm + object_norm - 2.0 * np.einsum(
        "tmd,tpd->tmp",
        marker_arr,
        object_arr,
        optimize=True,
    )
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared.min(axis=2))


def nearest_subset_distance(markers: np.ndarray, object_points: np.ndarray, marker_ids: tuple[int, ...]) -> np.ndarray:
    marker_arr = np.asarray(markers, dtype=np.float64)
    if marker_arr.ndim != 3 or marker_arr.shape[-1] != 3:
        raise ValueError(f"markers must have shape [T, M, 3], got {marker_arr.shape}")
    subset_distances = nearest_marker_distances(marker_arr[:, list(marker_ids), :], object_points)
    return subset_distances.min(axis=1)


def marker_object_contact(
    markers: np.ndarray,
    object_points: np.ndarray,
    marker_ids: tuple[int, ...],
    *,
    threshold_m: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    dist = nearest_subset_distance(markers, object_points, marker_ids)
    return dist < float(threshold_m), dist


def contact_metrics(
    pred_markers: np.ndarray,
    gt_markers: np.ndarray,
    object_points: np.ndarray,
    *,
    frame_mask: np.ndarray | None = None,
    threshold_m: float = 0.1,
    subsets: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, float]:
    """Paired marker-to-object contact metrics.

    ``object_points`` is the world-space object point cloud used for both
    generated and GT marker contact checks.
    """

    pred = np.asarray(pred_markers, dtype=np.float64)
    gt = np.asarray(gt_markers, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"marker shapes differ: {pred.shape} vs {gt.shape}")
    if frame_mask is None:
        valid = np.ones((pred.shape[0],), dtype=bool)
    else:
        valid = np.asarray(frame_mask, dtype=bool)
        if valid.shape != (pred.shape[0],):
            raise ValueError(f"frame_mask must have shape [{pred.shape[0]}], got {valid.shape}")

    selected = subsets if subsets is not None else MARKER_SUBSETS
    out: dict[str, float] = {}
    for name, ids in selected.items():
        pred_contact, pred_dist = marker_object_contact(pred, object_points, ids, threshold_m=threshold_m)
        gt_contact, gt_dist = marker_object_contact(gt, object_points, ids, threshold_m=threshold_m)
        p = pred_contact[valid]
        g = gt_contact[valid]
        tp = int(np.logical_and(p, g).sum())
        fp = int(np.logical_and(p, ~g).sum())
        tn = int(np.logical_and(~p, ~g).sum())
        fn = int(np.logical_and(~p, g).sum())
        accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
        if fp == 0 and fn == 0:
            # Perfect classification, including the all-true-negative case where this
            # subset never contacts the object in the clip: correctly predicting "no
            # contact" is a success, so precision/recall/F1 are 1.0 rather than 0/0->0.
            precision = recall = f1 = 1.0
        else:
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            f1 = _safe_div(2.0 * precision * recall, precision + recall)

        gt_contact_valid = gt_contact & valid
        gt_count = int(gt_contact_valid.sum())
        out[f"contact_precision.{name}"] = precision
        out[f"contact_recall.{name}"] = recall
        out[f"contact_accuracy.{name}"] = accuracy
        out[f"contact_f1.{name}"] = f1
        out[f"contact_gt_frame_count.{name}"] = float(gt_count)
        out[f"contact_distance_cm.{name}"] = float(pred_dist[gt_contact_valid].mean() * 100.0) if gt_count else 0.0
        out[f"gt_contact_distance_cm.{name}"] = float(gt_dist[gt_contact_valid].mean() * 100.0) if gt_count else 0.0
    return out


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom > 0 else 0.0
