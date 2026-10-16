from __future__ import annotations

import numpy as np


def mean_acceleration_cm_per_frame2(markers: np.ndarray, frame_mask: np.ndarray | None = None) -> float:
    arr = _as_markers(markers)
    if arr.shape[0] < 3:
        return 0.0
    acc = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
    valid = _transition_mask(arr.shape[0], frame_mask, order=2)
    per_frame = np.linalg.norm(acc, axis=-1).mean(axis=-1)
    return _masked_mean(per_frame, valid) * 100.0


def mean_jerk_cm_per_frame3(markers: np.ndarray, frame_mask: np.ndarray | None = None) -> float:
    arr = _as_markers(markers)
    if arr.shape[0] < 4:
        return 0.0
    jerk = arr[3:] - 3.0 * arr[2:-1] + 3.0 * arr[1:-2] - arr[:-3]
    valid = _transition_mask(arr.shape[0], frame_mask, order=3)
    per_frame = np.linalg.norm(jerk, axis=-1).mean(axis=-1)
    return _masked_mean(per_frame, valid) * 100.0


def jerk_profile_cm(markers: np.ndarray, frame_mask: np.ndarray | None = None) -> dict[str, float]:
    """CMDM-style long-horizon smoothness from the per-frame jerk magnitude series.

    Returns mean jerk, Peak-Jerk (PJ, the series max) and Area-Under-Jerk (AUJ, the
    series integral over valid transition frames), all in cm/frame^3 units.
    """

    arr = _as_markers(markers)
    if arr.shape[0] < 4:
        return {"mean_jerk_cm": 0.0, "peak_jerk_cm": 0.0, "area_under_jerk_cm": 0.0}
    jerk = arr[3:] - 3.0 * arr[2:-1] + 3.0 * arr[1:-2] - arr[:-3]
    valid = _transition_mask(arr.shape[0], frame_mask, order=3)
    per_frame = np.linalg.norm(jerk, axis=-1).mean(axis=-1) * 100.0
    if valid is not None:
        per_frame = per_frame[valid]
    if per_frame.size == 0:
        return {"mean_jerk_cm": 0.0, "peak_jerk_cm": 0.0, "area_under_jerk_cm": 0.0}
    return {
        "mean_jerk_cm": float(per_frame.mean()),
        "peak_jerk_cm": float(per_frame.max()),
        "area_under_jerk_cm": float(per_frame.sum()),
    }


def boundary_jump_cm(markers: np.ndarray, *, block_size_frames: int, frame_mask: np.ndarray | None = None) -> dict[str, float]:
    arr = _as_markers(markers)
    if block_size_frames <= 0:
        raise ValueError(f"block_size_frames must be positive, got {block_size_frames}")
    valid = np.ones((arr.shape[0],), dtype=bool) if frame_mask is None else np.asarray(frame_mask, dtype=bool)
    if valid.shape != (arr.shape[0],):
        raise ValueError(f"frame_mask must have shape [{arr.shape[0]}], got {valid.shape}")
    values: list[float] = []
    for boundary in range(block_size_frames, arr.shape[0], block_size_frames):
        if valid[boundary - 1] and valid[boundary]:
            values.append(float(np.linalg.norm(arr[boundary] - arr[boundary - 1], axis=-1).mean() * 100.0))
    if not values:
        return {"boundary_jump_cm": 0.0, "boundary_jump_count": 0.0, "boundary_jump_max_cm": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "boundary_jump_cm": float(array.mean()),
        "boundary_jump_count": float(array.size),
        "boundary_jump_max_cm": float(array.max()),
    }


def _as_markers(markers: np.ndarray) -> np.ndarray:
    arr = np.asarray(markers, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"markers must have shape [T, M, 3], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("markers contain non-finite values")
    return arr


def _transition_mask(length: int, frame_mask: np.ndarray | None, *, order: int) -> np.ndarray | None:
    if frame_mask is None:
        return None
    mask = np.asarray(frame_mask, dtype=bool)
    if mask.shape != (length,):
        raise ValueError(f"frame_mask must have shape [{length}], got {mask.shape}")
    valid = mask[: length - order].copy()
    for offset in range(1, order + 1):
        valid &= mask[offset : length - order + offset]
    return valid


def _masked_mean(values: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is None:
        return float(values.mean()) if values.size else 0.0
    if not mask.any():
        return 0.0
    return float(values[mask].mean())
