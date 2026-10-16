from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .decoded_records import Stage2WorldMotionArrays


@dataclass
class InterActFeaturePreprocessOutput:
    raw_features: torch.Tensor
    normalized_features: torch.Tensor
    lengths: list[int]


def preprocess_interact_marker_bps_samples(
    samples: Sequence[Stage2WorldMotionArrays],
    *,
    mean: np.ndarray | torch.Tensor,
    std: np.ndarray | torch.Tensor,
    bps_basis: np.ndarray | torch.Tensor,
    device: torch.device | str | None = None,
) -> InterActFeaturePreprocessOutput:
    """Build InterAct markers+BPS evaluator features from world-space samples.

    The reference evaluator uses marker positions flattened to 231 dimensions
    and BPS deltas from translated, non-rotated basis points to the nearest
    object surface point. Inputs are world-space; object rotations are not used
    here because object points are already transformed by the decode/render path.
    """

    if not samples:
        raise ValueError("samples must contain at least one Stage2WorldMotionArrays item.")
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    basis = _as_tensor(bps_basis, name="bps_basis", device=target_device)
    if basis.ndim != 2 or basis.shape[-1] != 3:
        raise ValueError(f"bps_basis must have shape [K,3], got {tuple(basis.shape)}.")

    lengths: list[int] = []
    frame_counts: list[int] = []
    for sample in samples:
        length = _prefix_mask_length(sample.frame_mask)
        if length <= 0:
            raise ValueError("each sample must contain at least one valid frame.")
        markers = np.asarray(sample.human_markers)
        if markers.ndim != 3 or markers.shape[1:] != (77, 3):
            raise ValueError(f"human_markers must have shape [T,77,3], got {markers.shape}.")
        lengths.append(length)
        frame_counts.append(int(markers.shape[0]))

    max_frames = max(frame_counts)
    feature_dim = 77 * 3 + int(basis.shape[0]) * 3
    raw_batch = torch.zeros(len(samples), max_frames, feature_dim, dtype=torch.float32, device=target_device)
    for index, sample in enumerate(samples):
        raw = marker_bps_features_for_sample(sample, bps_basis=basis, device=target_device)
        if raw.shape[0] != frame_counts[index]:
            raise ValueError(f"sample {index} produced {raw.shape[0]} frames, expected {frame_counts[index]}.")
        if raw.shape[-1] != feature_dim:
            raise ValueError(f"all samples must share feature dim {feature_dim}, got {raw.shape[-1]}.")
        raw_batch[index, : raw.shape[0]] = raw
        del raw

    normalized = normalize_interact_features(raw_batch, mean=mean, std=std)
    return InterActFeaturePreprocessOutput(raw_features=raw_batch, normalized_features=normalized, lengths=lengths)


def marker_bps_features_for_sample(
    sample: Stage2WorldMotionArrays,
    *,
    bps_basis: np.ndarray | torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    basis = _as_tensor(bps_basis, name="bps_basis", device=device)
    markers = _as_tensor(sample.human_markers, name="human_markers", device=device)
    object_points = _as_tensor(sample.object_points, name="object_points", device=device)
    if sample.object_trans is None:
        raise ValueError("InterAct markers+BPS preprocessing requires sample.object_trans.")
    object_trans = _as_tensor(sample.object_trans, name="object_trans", device=device)

    if markers.ndim != 3 or markers.shape[1:] != (77, 3):
        raise ValueError(f"human_markers must have shape [T,77,3], got {tuple(markers.shape)}.")
    if object_points.ndim != 3 or object_points.shape[0] != markers.shape[0] or object_points.shape[-1] != 3:
        raise ValueError(f"object_points must have shape [T,P,3], got {tuple(object_points.shape)}.")
    if object_trans.shape != (markers.shape[0], 3):
        raise ValueError(f"object_trans must have shape [T,3], got {tuple(object_trans.shape)}.")
    if basis.ndim != 2 or basis.shape[-1] != 3:
        raise ValueError(f"bps_basis must have shape [K,3], got {tuple(basis.shape)}.")

    marker_flat = markers.reshape(markers.shape[0], 77 * 3)
    basis_world = basis[None, :, :] + object_trans[:, None, :]
    bps_deltas = nearest_point_deltas(object_points, basis_world)
    return torch.cat([marker_flat, bps_deltas.reshape(markers.shape[0], -1)], dim=-1)


def nearest_point_deltas(object_points_world: torch.Tensor, basis_world: torch.Tensor) -> torch.Tensor:
    points = _as_tensor(object_points_world, name="object_points_world", device=object_points_world.device)
    basis = _as_tensor(basis_world, name="basis_world", device=points.device)
    if points.ndim != 3 or basis.ndim != 3:
        raise ValueError(f"points and basis must be [T,N,3], got {tuple(points.shape)} and {tuple(basis.shape)}.")
    if points.shape[0] != basis.shape[0] or points.shape[-1] != 3 or basis.shape[-1] != 3:
        raise ValueError("points and basis must share frame count and 3D coordinates.")
    distances = torch.cdist(basis.float(), points.float(), p=2)
    closest_ids = distances.argmin(dim=-1)
    gather_ids = closest_ids.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    closest = points.gather(1, gather_ids)
    return closest - basis


def normalize_interact_features(
    raw_features: torch.Tensor,
    *,
    mean: np.ndarray | torch.Tensor,
    std: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    features = _as_tensor(raw_features, name="raw_features", device=raw_features.device)
    mean_t = _as_tensor(mean, name="mean", device=features.device).flatten()
    std_t = _as_tensor(std, name="std", device=features.device).flatten()
    if features.shape[-1] != mean_t.numel() or features.shape[-1] != std_t.numel():
        raise ValueError(
            f"feature dim {features.shape[-1]} must match mean/std dims {mean_t.numel()} and {std_t.numel()}."
        )
    if torch.any(std_t <= 0):
        raise ValueError("std must be strictly positive for evaluator normalization.")
    return (features.float() - mean_t[None, None, :]) / std_t[None, None, :]


def _as_tensor(
    value: np.ndarray | torch.Tensor,
    *,
    name: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach() if not value.requires_grad else value
        tensor = tensor.to(device=device) if device is not None else tensor
        return tensor.float()
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    except Exception as exc:  # pragma: no cover - torch gives type-specific messages.
        raise TypeError(f"{name} must be convertible to a float tensor.") from exc
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    return tensor.float()


def _prefix_mask_length(frame_mask: np.ndarray) -> int:
    mask = np.asarray(frame_mask, dtype=bool).reshape(-1)
    length = int(mask.sum())
    if length > 0 and not mask[:length].all():
        raise ValueError("frame_mask must be a prefix mask for InterAct feature evaluation.")
    if length < mask.shape[0] and mask[length:].any():
        raise ValueError("frame_mask must not contain valid frames after padding starts.")
    return length
