from __future__ import annotations

import numpy as np


def nearest_marker_object_distance(markers: np.ndarray, object_points: np.ndarray) -> np.ndarray:
    marker_arr = np.asarray(markers, dtype=np.float64)
    object_arr = np.asarray(object_points, dtype=np.float64)
    if marker_arr.ndim != 3 or marker_arr.shape[-1] != 3:
        raise ValueError(f"markers must have shape [T, M, 3], got {marker_arr.shape}")
    if object_arr.ndim != 3 or object_arr.shape[-1] != 3 or object_arr.shape[0] != marker_arr.shape[0]:
        raise ValueError(f"object_points must have shape [T, P, 3], got {object_arr.shape}")
    diff = marker_arr[:, :, None, :] - object_arr[:, None, :, :]
    return np.linalg.norm(diff, axis=-1).min(axis=-1)


def marker_object_distance_error_cm(
    pred_markers: np.ndarray,
    gt_markers: np.ndarray,
    object_points: np.ndarray,
    frame_mask: np.ndarray | None = None,
) -> float:
    pred_dist = nearest_marker_object_distance(pred_markers, object_points)
    gt_dist = nearest_marker_object_distance(gt_markers, object_points)
    per_frame = np.abs(pred_dist - gt_dist).mean(axis=-1)
    if frame_mask is None:
        return float(per_frame.mean() * 100.0) if per_frame.size else 0.0
    mask = np.asarray(frame_mask, dtype=bool)
    if mask.shape != (per_frame.shape[0],):
        raise ValueError(f"frame_mask must have shape [{per_frame.shape[0]}], got {mask.shape}")
    if not mask.any():
        return 0.0
    return float(per_frame[mask].mean() * 100.0)
