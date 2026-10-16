from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from hoi_latent.data.transforms.structured_normalization import Stage1StructuredNormalizer
from hoi_latent.models.stage1.decoder.readout_heads import Stage1CoarsePrediction


@dataclass
class ObjectWorldPointsLossOutput:
    loss: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class ObjectFiniteDifferenceLossOutput:
    velocity_loss: torch.Tensor
    acceleration_loss: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def rot6d_to_matrix_torch(rot6d: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Convert 6D rotation representation to matrices using the Stage1 convention."""
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d last dimension must be 6, got {tuple(rot6d.shape)}.")
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def object_world_points_loss(
    prediction: Stage1CoarsePrediction,
    batch: dict,
    *,
    structured_normalizer: Stage1StructuredNormalizer,
    error_scale_m: float = 0.05,
    beta: float = 1.0,
) -> ObjectWorldPointsLossOutput:
    """Surface-point loss in world coordinates for the decoded object pose."""
    if error_scale_m <= 0.0:
        raise ValueError("error_scale_m must be positive.")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")

    pred_trans, pred_rotmat = _prediction_object_pose_world(prediction, structured_normalizer)
    raw_object = _require_mapping(_require_mapping(batch, "raw"), "object")
    gt_trans = _require_tensor(raw_object, "trans", "batch.raw.object").to(
        device=pred_trans.device, dtype=pred_trans.dtype
    )
    gt_rotmat = _require_tensor(raw_object, "rotmat", "batch.raw.object").to(
        device=pred_trans.device, dtype=pred_trans.dtype
    )
    if tuple(gt_trans.shape) != tuple(pred_trans.shape):
        raise ValueError(f"raw object trans shape {tuple(gt_trans.shape)} does not match {tuple(pred_trans.shape)}.")
    if tuple(gt_rotmat.shape) != tuple(pred_rotmat.shape):
        raise ValueError(f"raw object rotmat shape {tuple(gt_rotmat.shape)} does not match {tuple(pred_rotmat.shape)}.")

    geometry = _require_mapping(batch, "geometry")
    points_local = _require_tensor(geometry, "object_points_local", "batch.geometry").to(
        device=pred_trans.device, dtype=pred_trans.dtype
    )
    if points_local.ndim != 3 or points_local.shape[-1] != 3:
        raise ValueError(f"object_points_local must be [B,P,3], got {tuple(points_local.shape)}.")
    if int(points_local.shape[0]) != int(pred_trans.shape[0]):
        raise ValueError(
            f"object_points_local batch {points_local.shape[0]} does not match prediction batch {pred_trans.shape[0]}."
        )

    point_mask = _object_point_mask(batch, points_local).to(device=pred_trans.device)
    frame_mask = _object_frame_mask(batch, pred_trans).to(device=pred_trans.device)
    mask = frame_mask[:, :, None] & point_mask[:, None, :]

    pred_points = _object_points_world_torch(points_local, pred_rotmat, pred_trans)
    gt_points = _object_points_world_torch(points_local, gt_rotmat, gt_trans)
    error = pred_points - gt_points
    loss = _masked_smooth_l1_scaled(error, mask, scale=error_scale_m, beta=beta)
    diagnostics = {
        "object_world_points_mean_l2_cm": _masked_mean_l2_cm(error, mask).detach(),
        "object_world_point_error_scale_m": pred_trans.new_tensor(float(error_scale_m)).detach(),
    }
    return ObjectWorldPointsLossOutput(loss=loss, diagnostics=diagnostics)


def object_translation_finite_difference_loss(
    prediction: Stage1CoarsePrediction,
    batch: dict,
    *,
    structured_normalizer: Stage1StructuredNormalizer,
    velocity_error_scale_m: float = 0.05,
    acceleration_error_scale_m: float = 0.02,
    beta: float = 1.0,
) -> ObjectFiniteDifferenceLossOutput:
    """Finite-difference velocity/acceleration consistency on decoded object translation."""
    if velocity_error_scale_m <= 0.0:
        raise ValueError("velocity_error_scale_m must be positive.")
    if acceleration_error_scale_m <= 0.0:
        raise ValueError("acceleration_error_scale_m must be positive.")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")

    pred_trans = _denormalize_object_field(prediction, structured_normalizer, "trans")
    raw_object = _require_mapping(_require_mapping(batch, "raw"), "object")
    gt_trans = _require_tensor(raw_object, "trans", "batch.raw.object").to(
        device=pred_trans.device, dtype=pred_trans.dtype
    )
    if tuple(gt_trans.shape) != tuple(pred_trans.shape):
        raise ValueError(f"raw object trans shape {tuple(gt_trans.shape)} does not match {tuple(pred_trans.shape)}.")

    frame_mask = _object_frame_mask(batch, pred_trans).to(device=pred_trans.device)
    pred_vel = pred_trans[:, 1:] - pred_trans[:, :-1]
    gt_vel = gt_trans[:, 1:] - gt_trans[:, :-1]
    vel_mask = frame_mask[:, 1:] & frame_mask[:, :-1] if frame_mask.shape[1] > 1 else frame_mask[:, :0]
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
    acc_mask = vel_mask[:, 1:] & vel_mask[:, :-1] if vel_mask.shape[1] > 1 else vel_mask[:, :0]

    vel_error = pred_vel - gt_vel
    acc_error = pred_acc - gt_acc
    velocity_loss = _masked_smooth_l1_scaled(vel_error, vel_mask, scale=velocity_error_scale_m, beta=beta)
    acceleration_loss = _masked_smooth_l1_scaled(
        acc_error,
        acc_mask,
        scale=acceleration_error_scale_m,
        beta=beta,
    )
    diagnostics = {
        "object_fd_velocity_l2_cm": _masked_mean_l2_cm(vel_error, vel_mask).detach(),
        "object_fd_acceleration_l2_cm": _masked_mean_l2_cm(acc_error, acc_mask).detach(),
        "object_trans_jitter_cm": _masked_mean_l2_cm(pred_vel, vel_mask).detach(),
        "object_trans_accel_jitter_cm": _masked_mean_l2_cm(pred_acc, acc_mask).detach(),
        "object_fd_velocity_error_scale_m": pred_trans.new_tensor(float(velocity_error_scale_m)).detach(),
        "object_fd_acceleration_error_scale_m": pred_trans.new_tensor(float(acceleration_error_scale_m)).detach(),
    }
    return ObjectFiniteDifferenceLossOutput(
        velocity_loss=velocity_loss,
        acceleration_loss=acceleration_loss,
        diagnostics=diagnostics,
    )


def _prediction_object_pose_world(
    prediction: Stage1CoarsePrediction,
    structured_normalizer: Stage1StructuredNormalizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_trans = _denormalize_object_field(prediction, structured_normalizer, "trans")
    pred_rot6d = _denormalize_object_field(prediction, structured_normalizer, "rot6d")
    return pred_trans, rot6d_to_matrix_torch(pred_rot6d.float()).to(dtype=pred_trans.dtype)


def _denormalize_object_field(
    prediction: Stage1CoarsePrediction,
    structured_normalizer: Stage1StructuredNormalizer,
    field_name: str,
) -> torch.Tensor:
    value = _require_tensor(prediction.object, field_name, "prediction.object").float()
    return structured_normalizer.denormalize_group_torch("object", field_name, value, require_stats=True)


def _object_points_world_torch(points_local: torch.Tensor, rotmat: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    if rotmat.ndim != 4 or rotmat.shape[-2:] != (3, 3):
        raise ValueError(f"rotmat must be [B,T,3,3], got {tuple(rotmat.shape)}.")
    if trans.ndim != 3 or trans.shape[-1] != 3:
        raise ValueError(f"trans must be [B,T,3], got {tuple(trans.shape)}.")
    if int(points_local.shape[0]) != int(rotmat.shape[0]):
        raise ValueError("points_local and rotmat batch sizes differ.")
    return torch.einsum("bpc,btjc->btpj", points_local, rotmat) + trans[:, :, None, :]


def _object_frame_mask(batch: dict, reference: torch.Tensor) -> torch.Tensor:
    masks = batch.get("masks")
    if not isinstance(masks, dict):
        raise KeyError("Object jitter losses require batch['masks'].")
    for key in ("frame", "valid", "object"):
        value = masks.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            if tuple(value.shape) != tuple(reference.shape[:2]):
                raise ValueError(f"batch.masks.{key} shape {tuple(value.shape)} does not match {tuple(reference.shape[:2])}.")
            return value.to(device=reference.device, dtype=torch.bool)
    raise KeyError("Object jitter losses require batch['masks']['object'], ['frame'], or ['valid'].")


def _object_point_mask(batch: dict, points_local: torch.Tensor) -> torch.Tensor:
    geometry = _require_mapping(batch, "geometry")
    value = geometry.get("object_points_mask")
    if isinstance(value, torch.Tensor):
        if value.ndim != 2 or tuple(value.shape) != tuple(points_local.shape[:2]):
            raise ValueError(
                f"object_points_mask shape {tuple(value.shape)} does not match {tuple(points_local.shape[:2])}."
            )
        return value.to(device=points_local.device, dtype=torch.bool)
    return torch.ones(points_local.shape[:2], device=points_local.device, dtype=torch.bool)


def _masked_smooth_l1_scaled(error_m: torch.Tensor, mask: torch.Tensor, *, scale: float, beta: float) -> torch.Tensor:
    loss = F.smooth_l1_loss(error_m / float(scale), torch.zeros_like(error_m), beta=beta, reduction="none")
    mask = mask.to(device=loss.device, dtype=torch.bool)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(loss).to(dtype=loss.dtype)
    return (loss * expanded).sum() / expanded.sum().clamp_min(1.0)


def _masked_mean_l2_cm(error_m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dist = error_m.norm(dim=-1)
    mask = mask.to(device=dist.device, dtype=torch.bool)
    count = mask.to(dtype=dist.dtype).sum()
    return (dist * mask.to(dtype=dist.dtype)).sum() / count.clamp_min(1.0) * 100.0


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Expected mapping at key {key!r}.")
    return value


def _require_tensor(mapping: dict[str, Any], key: str, context: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"{context} missing tensor field {key!r}.")
    return value
