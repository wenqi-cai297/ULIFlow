from __future__ import annotations

import numpy as np


def marker_mpjpe_cm(pred_markers: np.ndarray, gt_markers: np.ndarray, frame_mask: np.ndarray | None = None) -> float:
    pred, gt = _validate_markers(pred_markers, gt_markers)
    error_m = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)
    return _masked_mean(error_m, frame_mask) * 100.0


def marker_velocity_error_cm_per_frame(
    pred_markers: np.ndarray,
    gt_markers: np.ndarray,
    frame_mask: np.ndarray | None = None,
) -> float:
    pred, gt = _validate_markers(pred_markers, gt_markers)
    if pred.shape[0] < 2:
        return 0.0
    pred_vel = pred[1:] - pred[:-1]
    gt_vel = gt[1:] - gt[:-1]
    valid = None
    if frame_mask is not None:
        mask = np.asarray(frame_mask, dtype=bool)
        if mask.shape != (pred.shape[0],):
            raise ValueError(f"frame_mask must have shape [{pred.shape[0]}], got {mask.shape}")
        valid = mask[1:] & mask[:-1]
    error_m = np.linalg.norm(pred_vel - gt_vel, axis=-1).mean(axis=-1)
    return _masked_mean(error_m, valid) * 100.0


def _validate_markers(pred_markers: np.ndarray, gt_markers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred_markers, dtype=np.float64)
    gt = np.asarray(gt_markers, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"marker shapes differ: {pred.shape} vs {gt.shape}")
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"markers must have shape [T, M, 3], got {pred.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("markers contain non-finite values")
    return pred, gt


def _masked_mean(values: np.ndarray, mask: np.ndarray | None) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if mask is None:
        return float(arr.mean()) if arr.size else 0.0
    valid = np.asarray(mask, dtype=bool)
    if valid.shape != arr.shape:
        raise ValueError(f"mask shape {valid.shape} does not match values shape {arr.shape}")
    if not valid.any():
        return 0.0
    return float(arr[valid].mean())
