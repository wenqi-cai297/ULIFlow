from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hoi_latent.models.stage2.stem_types import (
    NUM_STAGE2_LATENT_SUBSTREAMS,
    Stage2LatentSubstreamId,
)


@dataclass
class Stage2RFLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


class Stage2RectifiedFlowLoss(nn.Module):
    """Masked rectified-flow velocity MSE in the transformed Stage1 latent space."""

    def forward(
        self,
        velocity_pred: torch.Tensor,
        velocity_target: torch.Tensor,
        latent_mask: torch.Tensor,
        *,
        latent_substream_ids: torch.Tensor | None = None,
    ) -> Stage2RFLossOutput:
        loss = masked_velocity_mse(velocity_pred, velocity_target, latent_mask)
        diagnostics = _velocity_diagnostics(
            velocity_pred,
            velocity_target,
            latent_mask,
            velocity_mse=loss,
        )
        if latent_substream_ids is not None:
            diagnostics.update(
                _substream_mse_diagnostics(
                    velocity_pred,
                    velocity_target,
                    latent_mask,
                    latent_substream_ids,
                )
            )
        return Stage2RFLossOutput(
            total=loss,
            components={"velocity_mse": loss},
            diagnostics=diagnostics,
        )


def masked_velocity_mse(
    velocity_pred: torch.Tensor,
    velocity_target: torch.Tensor,
    latent_mask: torch.Tensor,
) -> torch.Tensor:
    if velocity_pred.shape != velocity_target.shape:
        raise ValueError(
            "velocity_pred and velocity_target must have the same shape, "
            f"got {tuple(velocity_pred.shape)} and {tuple(velocity_target.shape)}."
        )
    if velocity_pred.ndim != 3:
        raise ValueError(f"velocity_pred must have shape [B, N_z, d_z], got {tuple(velocity_pred.shape)}.")
    if latent_mask.shape != velocity_pred.shape[:2]:
        raise ValueError(
            "latent_mask must have shape [B, N_z], "
            f"got {tuple(latent_mask.shape)} for velocity shape {tuple(velocity_pred.shape)}."
        )

    valid = latent_mask.to(dtype=torch.bool).unsqueeze(-1)
    diff2 = (velocity_pred.float() - velocity_target.float()).pow(2).masked_fill(~valid, 0.0)
    denom = valid.sum().clamp_min(1) * velocity_pred.shape[-1]
    return diff2.sum() / denom.to(dtype=diff2.dtype)


def _velocity_diagnostics(
    velocity_pred: torch.Tensor,
    velocity_target: torch.Tensor,
    latent_mask: torch.Tensor,
    *,
    velocity_mse: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pred_rms = _valid_rms(velocity_pred, latent_mask)
    target_rms = _valid_rms(velocity_target, latent_mask)
    ratio = pred_rms.float() / target_rms.float().clamp_min(1.0e-8)
    normalized_mse = velocity_mse.float() / target_rms.float().square().clamp_min(1.0e-8)
    valid_count = latent_mask.to(dtype=torch.bool).sum().to(device=velocity_pred.device, dtype=torch.float32)
    return {
        "normalized_rf_mse": normalized_mse,
        "velocity_pred_rms": pred_rms,
        "velocity_target_rms": target_rms,
        "velocity_rms_ratio": ratio,
        "valid_latent_token_count": valid_count,
    }


def _substream_mse_diagnostics(
    velocity_pred: torch.Tensor,
    velocity_target: torch.Tensor,
    latent_mask: torch.Tensor,
    latent_substream_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if latent_substream_ids.shape != latent_mask.shape:
        raise ValueError(
            "latent_substream_ids must have shape [B, N_z], "
            f"got {tuple(latent_substream_ids.shape)} for mask shape {tuple(latent_mask.shape)}."
        )
    ids = latent_substream_ids.to(device=velocity_pred.device, dtype=torch.long)
    valid = latent_mask.to(device=velocity_pred.device, dtype=torch.bool)
    valid_ids = ids[valid]
    if valid_ids.numel():
        bad = (valid_ids < 0) | (valid_ids >= NUM_STAGE2_LATENT_SUBSTREAMS)
        if bad.any():
            bad_values = torch.unique(valid_ids[bad]).detach().cpu().tolist()
            raise ValueError(f"latent_substream_ids contains out-of-range valid ids: {bad_values}.")

    diagnostics: dict[str, torch.Tensor] = {}
    for substream in Stage2LatentSubstreamId:
        mask = valid & (ids == int(substream))
        key = f"velocity_mse_by_latent_substream.{_substream_name(substream)}"
        diagnostics[key] = masked_velocity_mse(velocity_pred, velocity_target, mask)
    return diagnostics


def _valid_rms(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != value.shape[:2]:
        raise ValueError(f"mask must have shape {tuple(value.shape[:2])}, got {tuple(mask.shape)}.")
    valid = mask.to(device=value.device, dtype=torch.bool).unsqueeze(-1).expand_as(value)
    selected = value.float()[valid]
    if selected.numel() == 0:
        return value.new_zeros(())
    return selected.pow(2).mean().sqrt().to(dtype=value.dtype)


def _substream_name(substream: Stage2LatentSubstreamId) -> str:
    if substream == Stage2LatentSubstreamId.HUMAN_DYNAMIC:
        return "human"
    if substream == Stage2LatentSubstreamId.OBJECT_POSE:
        return "object_pose"
    if substream == Stage2LatentSubstreamId.OBJECT_GEOMETRY_LATENT:
        return "object_geometry_latent"
    if substream == Stage2LatentSubstreamId.RELATION_DYNAMIC:
        return "relation"
    if substream == Stage2LatentSubstreamId.FLAT_DYNAMIC:
        return "flat_dynamic"
    raise ValueError(f"Unknown Stage2 latent substream id: {substream}.")
