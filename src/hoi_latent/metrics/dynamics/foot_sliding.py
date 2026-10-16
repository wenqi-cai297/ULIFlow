from __future__ import annotations

import numpy as np

from hoi_latent.metrics.interaction.constants import FOOT_MARKERS


def foot_sliding_interact_mm(
    markers: np.ndarray,
    *,
    frame_mask: np.ndarray | None = None,
    foot_marker_ids: tuple[int, ...] = FOOT_MARKERS,
    height_threshold_m: float = 0.04,
    up_axis: int = 1,
    floor_height_m: float | None = None,
    fps: float = 30.0,
) -> dict[str, float]:
    """Generated-only InterAct object2human-style foot sliding score.

    Mirrors ``external/InterAct/object2human/evaluation_metrics.py``:
    floor-normalize by the minimum human height, gate low foot markers, weight
    displacement by ``2 - 2 ** (height / threshold)``, and divide by
    ``T * num_foot_markers``.
    """

    arr = _as_markers(markers)
    valid = _frame_mask(arr.shape[0], frame_mask)
    floor = _floor_height(arr, valid, foot_marker_ids, up_axis) if floor_height_m is None else float(floor_height_m)
    foot = arr[:, list(foot_marker_ids), :].copy()
    height = foot[:, :, up_axis] - floor
    foot[:, :, up_axis] = height
    stance = (height[:-1] < float(height_threshold_m)) & valid[:-1, None] & valid[1:, None]
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    disp = np.linalg.norm(foot[1:, :, horizontal_axes] - foot[:-1, :, horizontal_axes], axis=-1)
    weight = 2.0 - np.power(2.0, foot[:-1, :, up_axis] / float(height_threshold_m))
    stats = np.abs(disp * weight) * stance.astype(np.float64)
    denom = max(arr.shape[0] * len(foot_marker_ids), 1)
    value = float(stats.sum() / denom * 1000.0)
    return {
        "foot_sliding_interact_mm": value,
        "foot_sliding_contact_count": float(stance.sum()),
        "foot_sliding_fps": float(fps),
    }


def foot_skate_stance_cm_per_frame(
    pred_markers: np.ndarray,
    gt_markers: np.ndarray,
    *,
    frame_mask: np.ndarray | None = None,
    foot_marker_ids: tuple[int, ...] = FOOT_MARKERS,
    gt_height_threshold_m: float = 0.05,
    gt_speed_threshold_m_per_frame: float = 0.005,
    up_axis: int = 1,
    floor_height_m: float | None = None,
) -> dict[str, float]:
    """Paired stance-gated foot skating diagnostic in centimeters/frame."""

    pred = _as_markers(pred_markers)
    gt = _as_markers(gt_markers)
    if pred.shape != gt.shape:
        raise ValueError(f"marker shapes differ: {pred.shape} vs {gt.shape}")
    valid = _frame_mask(gt.shape[0], frame_mask)
    if gt.shape[0] < 2:
        return {"foot_skate_stance_cm_per_frame": 0.0, "foot_skate_stance_count": 0.0}
    floor = _floor_height(gt, valid, foot_marker_ids, up_axis) if floor_height_m is None else float(floor_height_m)
    pred_foot = pred[:, list(foot_marker_ids), :]
    gt_foot = gt[:, list(foot_marker_ids), :]
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    gt_height = gt_foot[:-1, :, up_axis] - floor
    gt_speed = np.linalg.norm(gt_foot[1:, :, horizontal_axes] - gt_foot[:-1, :, horizontal_axes], axis=-1)
    pred_speed = np.linalg.norm(pred_foot[1:, :, horizontal_axes] - pred_foot[:-1, :, horizontal_axes], axis=-1)
    transition_valid = valid[1:] & valid[:-1]
    stance = (gt_height < float(gt_height_threshold_m)) & (gt_speed < float(gt_speed_threshold_m_per_frame)) & transition_valid[:, None]
    if not stance.any():
        value = 0.0
    else:
        value = float(pred_speed[stance].mean() * 100.0)
    return {"foot_skate_stance_cm_per_frame": value, "foot_skate_stance_count": float(stance.sum())}


def _as_markers(markers: np.ndarray) -> np.ndarray:
    arr = np.asarray(markers, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"markers must have shape [T, M, 3], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("markers contain non-finite values")
    return arr


def _frame_mask(length: int, frame_mask: np.ndarray | None) -> np.ndarray:
    if frame_mask is None:
        return np.ones((length,), dtype=bool)
    mask = np.asarray(frame_mask, dtype=bool)
    if mask.shape != (length,):
        raise ValueError(f"frame_mask must have shape [{length}], got {mask.shape}")
    return mask


def _floor_height(markers: np.ndarray, frame_mask: np.ndarray, marker_ids: tuple[int, ...], up_axis: int) -> float:
    valid_feet = markers[frame_mask][:, list(marker_ids), up_axis]
    return float(valid_feet.min()) if valid_feet.size else 0.0
