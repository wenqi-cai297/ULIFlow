from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hoi_latent.data.transforms.structured_normalization import Stage1StructuredNormalizer
from hoi_latent.losses.stage1.object_jitter import _masked_mean_l2_cm, _masked_smooth_l1_scaled
from hoi_latent.models.stage1.decoder.readout_heads import Stage1CoarsePrediction


@dataclass
class HumanFiniteDifferenceLossOutput:
    velocity_loss: torch.Tensor
    acceleration_loss: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def human_world_marker_finite_difference_loss(
    prediction: Stage1CoarsePrediction,
    batch: dict,
    *,
    structured_normalizer: Stage1StructuredNormalizer,
    velocity_error_scale_m: float = 0.05,
    acceleration_error_scale_m: float = 0.02,
    beta: float = 1.0,
) -> HumanFiniteDifferenceLossOutput:
    """Finite-difference velocity/acceleration consistency on decoded human world markers."""
    if velocity_error_scale_m <= 0.0:
        raise ValueError("velocity_error_scale_m must be positive.")
    if acceleration_error_scale_m <= 0.0:
        raise ValueError("acceleration_error_scale_m must be positive.")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")

    pred_world = _prediction_human_world_markers(prediction, structured_normalizer)
    gt_world, target_source = _human_world_marker_target(batch, pred_world, structured_normalizer)
    if tuple(gt_world.shape) != tuple(pred_world.shape):
        raise ValueError(f"human world marker target shape {tuple(gt_world.shape)} does not match {tuple(pred_world.shape)}.")

    marker_mask = _human_marker_mask(batch, pred_world).to(device=pred_world.device)
    pred_vel = pred_world[:, 1:] - pred_world[:, :-1]
    gt_vel = gt_world[:, 1:] - gt_world[:, :-1]
    vel_mask = marker_mask[:, 1:] & marker_mask[:, :-1] if marker_mask.shape[1] > 1 else marker_mask[:, :0]
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
        "human_fd_velocity_l2_cm": _masked_mean_l2_cm(vel_error, vel_mask).detach(),
        "human_fd_acceleration_l2_cm": _masked_mean_l2_cm(acc_error, acc_mask).detach(),
        "human_marker_jitter_cm": _masked_mean_l2_cm(pred_vel, vel_mask).detach(),
        "human_marker_accel_jitter_cm": _masked_mean_l2_cm(pred_acc, acc_mask).detach(),
        "human_fd_velocity_error_scale_m": pred_world.new_tensor(float(velocity_error_scale_m)).detach(),
        "human_fd_acceleration_error_scale_m": pred_world.new_tensor(float(acceleration_error_scale_m)).detach(),
        "human_fd_target_source": pred_world.new_tensor(int(target_source)).detach(),
    }
    return HumanFiniteDifferenceLossOutput(
        velocity_loss=velocity_loss,
        acceleration_loss=acceleration_loss,
        diagnostics=diagnostics,
    )


def _prediction_human_world_markers(
    prediction: Stage1CoarsePrediction,
    structured_normalizer: Stage1StructuredNormalizer,
) -> torch.Tensor:
    pred_pos = _denormalize_human_field(prediction, structured_normalizer, "pos_rootrel")
    pred_root = _denormalize_human_field(prediction, structured_normalizer, "root")
    return pred_pos + pred_root[:, :, None, :]


def _denormalize_human_field(
    prediction: Stage1CoarsePrediction,
    structured_normalizer: Stage1StructuredNormalizer,
    field_name: str,
) -> torch.Tensor:
    value = _require_tensor(prediction.human, field_name, "prediction.human").float()
    return structured_normalizer.denormalize_group_torch("human", field_name, value, require_stats=True)


def _human_world_marker_target(
    batch: dict,
    reference: torch.Tensor,
    structured_normalizer: Stage1StructuredNormalizer,
) -> tuple[torch.Tensor, int]:
    raw = batch.get("raw")
    if isinstance(raw, dict):
        raw_human = raw.get("human")
        if isinstance(raw_human, dict) and isinstance(raw_human.get("pos"), torch.Tensor):
            return raw_human["pos"].to(device=reference.device, dtype=reference.dtype), 1
    model_human = _require_mapping(_require_mapping(batch, "model"), "human")
    target_pos = structured_normalizer.denormalize_group_torch(
        "human",
        "pos_rootrel",
        _require_tensor(model_human, "pos_rootrel", "batch.model.human").to(
            device=reference.device, dtype=reference.dtype
        ),
        require_stats=True,
    )
    target_root = structured_normalizer.denormalize_group_torch(
        "human",
        "root",
        _require_tensor(model_human, "root", "batch.model.human").to(device=reference.device, dtype=reference.dtype),
        require_stats=True,
    )
    return target_pos + target_root[:, :, None, :], 0


def _human_marker_mask(batch: dict, reference: torch.Tensor) -> torch.Tensor:
    masks = batch.get("masks")
    if not isinstance(masks, dict):
        raise KeyError("Human jitter losses require batch['masks'].")
    human = masks.get("human")
    if isinstance(human, torch.Tensor) and human.ndim == 3:
        if tuple(human.shape) != tuple(reference.shape[:3]):
            raise ValueError(f"batch.masks.human shape {tuple(human.shape)} does not match {tuple(reference.shape[:3])}.")
        return human.to(device=reference.device, dtype=torch.bool)
    frame_mask = None
    for key in ("frame", "valid", "object"):
        value = masks.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            if tuple(value.shape) != tuple(reference.shape[:2]):
                raise ValueError(f"batch.masks.{key} shape {tuple(value.shape)} does not match {tuple(reference.shape[:2])}.")
            frame_mask = value.to(device=reference.device, dtype=torch.bool)
            break
    if frame_mask is None:
        raise KeyError("Human jitter losses require batch['masks']['human'], ['frame'], ['valid'], or ['object'].")
    return frame_mask[:, :, None].expand(reference.shape[:3])


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
