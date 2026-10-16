from __future__ import annotations

import numpy as np


def object_translation_error_cm(pred_trans: np.ndarray, gt_trans: np.ndarray, frame_mask: np.ndarray | None = None) -> float:
    pred = np.asarray(pred_trans, dtype=np.float64)
    gt = np.asarray(gt_trans, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[-1] != 3:
        raise ValueError(f"object translations must both have shape [T, 3], got {pred.shape} and {gt.shape}")
    err = np.linalg.norm(pred - gt, axis=-1)
    return _masked_mean(err, frame_mask) * 100.0


def object_rotation_geodesic_deg(
    pred_rotmat: np.ndarray,
    gt_rotmat: np.ndarray,
    frame_mask: np.ndarray | None = None,
) -> float:
    pred = np.asarray(pred_rotmat, dtype=np.float64)
    gt = np.asarray(gt_rotmat, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-2:] != (3, 3):
        raise ValueError(f"rotations must both have shape [T, 3, 3], got {pred.shape} and {gt.shape}")
    relative = np.einsum("tij,tkj->tik", pred, gt)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    deg = np.degrees(np.arccos(cos))
    return _masked_mean(deg, frame_mask)


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
