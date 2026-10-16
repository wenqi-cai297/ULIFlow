from __future__ import annotations

import numpy as np

from .constants import FULL_MARKERS, HAND_MARKERS
from .contact import nearest_marker_distances


def interaction_realism_statistics(
    human_markers: np.ndarray,
    object_points: np.ndarray,
    *,
    frame_mask: np.ndarray | None = None,
    object_trans: np.ndarray | None = None,
    contact_threshold_m: float = 0.10,
    far_threshold_m: float = 0.20,
) -> dict[str, float]:
    """Compute GT-independent temporal human-object interaction statistics.

    These statistics describe one generated or real clip without assuming that a
    generated sample should match a paired GT trajectory frame by frame.
    """

    markers = np.asarray(human_markers, dtype=np.float64)
    points = np.asarray(object_points, dtype=np.float64)
    if markers.ndim != 3 or markers.shape[1:] != (77, 3):
        raise ValueError(f"human_markers must have shape [T,77,3], got {markers.shape}")
    if points.ndim != 3 or points.shape[0] != markers.shape[0] or points.shape[-1] != 3:
        raise ValueError(f"object_points must have shape [T,P,3], got {points.shape}")
    if contact_threshold_m <= 0.0:
        raise ValueError("contact_threshold_m must be positive")
    if far_threshold_m <= contact_threshold_m:
        raise ValueError("far_threshold_m must be greater than contact_threshold_m")

    valid_frames = _valid_prefix_length(frame_mask, markers.shape[0])
    markers = markers[:valid_frames]
    points = points[:valid_frames]
    if valid_frames == 0:
        raise ValueError("at least one valid frame is required")

    marker_distances = nearest_marker_distances(markers, points)
    full_distance = marker_distances[:, list(FULL_MARKERS)].min(axis=1)
    hand_distance = marker_distances[:, list(HAND_MARKERS)].min(axis=1)
    full_contact = full_distance < float(contact_threshold_m)
    hand_contact = hand_distance < float(contact_threshold_m)

    if object_trans is None:
        object_center = points.mean(axis=1)
    else:
        trans = np.asarray(object_trans, dtype=np.float64)
        if trans.shape != (human_markers.shape[0], 3):
            raise ValueError(f"object_trans must have shape [{human_markers.shape[0]},3], got {trans.shape}")
        object_center = trans[:valid_frames]

    object_step = np.linalg.norm(np.diff(object_center, axis=0), axis=-1)
    total_object_path = float(object_step.sum())
    near_step = np.logical_or(full_contact[:-1], full_contact[1:])
    far_step = np.logical_and(full_distance[:-1] > far_threshold_m, full_distance[1:] > far_threshold_m)

    return {
        "valid_frames": float(valid_frames),
        "full_contact_rate": float(full_contact.mean()),
        "hand_contact_rate": float(hand_contact.mean()),
        "full_distance_mean_cm": float(full_distance.mean() * 100.0),
        "full_distance_p10_cm": float(np.percentile(full_distance, 10.0) * 100.0),
        "hand_distance_mean_cm": float(hand_distance.mean() * 100.0),
        "hand_distance_p10_cm": float(np.percentile(hand_distance, 10.0) * 100.0),
        "contact_transitions_per_100_frames": _transition_rate_per_100(full_contact),
        "object_speed_mean_cm_per_frame": float(object_step.mean() * 100.0) if object_step.size else 0.0,
        "object_path_near_contact_fraction": _path_fraction(object_step, near_step, total_object_path),
        "object_path_far_fraction": _path_fraction(object_step, far_step, total_object_path),
        "distance_velocity_mean_cm_per_frame": _mean_abs_difference_cm(full_distance, order=1),
        "distance_acceleration_mean_cm_per_frame2": _mean_abs_difference_cm(full_distance, order=2),
        "distance_jerk_mean_cm_per_frame3": _mean_abs_difference_cm(full_distance, order=3),
    }


def _valid_prefix_length(frame_mask: np.ndarray | None, frames: int) -> int:
    if frame_mask is None:
        return int(frames)
    mask = np.asarray(frame_mask, dtype=bool)
    if mask.shape != (frames,):
        raise ValueError(f"frame_mask must have shape [{frames}], got {mask.shape}")
    valid = int(mask.sum())
    expected = np.arange(frames) < valid
    if not np.array_equal(mask, expected):
        raise ValueError("frame_mask must be prefix-contiguous")
    return valid


def _transition_rate_per_100(contact: np.ndarray) -> float:
    if contact.size < 2:
        return 0.0
    return float(np.not_equal(contact[1:], contact[:-1]).mean() * 100.0)


def _path_fraction(step: np.ndarray, selected: np.ndarray, total: float) -> float:
    if total <= 1e-12 or step.size == 0:
        return 0.0
    return float(step[selected].sum() / total)


def _mean_abs_difference_cm(values_m: np.ndarray, *, order: int) -> float:
    if values_m.size <= order:
        return 0.0
    return float(np.abs(np.diff(values_m, n=order)).mean() * 100.0)
