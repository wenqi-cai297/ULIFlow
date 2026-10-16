from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from hoi_latent.inference.stage2.decode_render import (
    Stage2WorldRenderArrays,
    prediction_to_world_markers_and_object_points,
)
from hoi_latent.models.stage1.autoencoder.latent_bottleneck import Stage1LatentFieldOutput
from hoi_latent.models.stage1.decoder.readout_heads import Stage1CoarsePrediction
from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.training.stage1.source_autoencoder import (
    decode_stage1_source_prediction,
    encode_stage1_source_latent,
)


@dataclass
class Stage1CeilingOutput:
    prediction: Stage1CoarsePrediction
    world: Stage2WorldRenderArrays
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


@torch.no_grad()
def compute_stage1_reconstruction_ceiling_prediction(
    *,
    stage1_model: torch.nn.Module,
    whitening: Any,
    batch: Mapping[str, Any],
    target_frame_mask: torch.Tensor | None = None,
    validate: bool = False,
) -> tuple[Stage1CoarsePrediction, dict[str, torch.Tensor]]:
    """Decode the Stage1 reconstruction ceiling through the Stage2 target space.

    The caller must provide a frozen Stage1 model built with the same anchor
    policy used by the Stage2 target/whitening artifact. For the current
    absolute-grid Stage2 flow-forcing runs, that means the Stage1 config must
    already have the absolute-grid anchor override applied before this helper is
    called. Otherwise the ceiling is encoded in a different latent layout and is
    not a valid upper-bound reference for the generated z_white path.
    """
    latent = encode_stage1_source_latent(
        stage1_model,
        dict(batch),
        return_diagnostics=False,
        validate=validate,
    )
    mask = latent.mask.bool()
    z_white = whitening.whiten(latent.z_rec.float(), mask).float()
    z_white = zero_invalid_tokens(z_white, mask)
    z_rec_roundtrip = whitening.unwhiten(z_white, mask).float()
    z_rec_roundtrip = zero_invalid_tokens(z_rec_roundtrip, mask)
    ceiling_latent = Stage1LatentFieldOutput(
        z_rec=z_rec_roundtrip,
        z_raw=z_rec_roundtrip,
        mask=latent.mask,
        modality_ids=latent.modality_ids,
        token_type_ids=latent.token_type_ids,
        time_ids=latent.time_ids,
        entity_ids=latent.entity_ids,
        time_mask=latent.time_mask,
    )
    if target_frame_mask is None:
        target_frame_mask = _target_frame_mask_from_batch(batch, device=z_rec_roundtrip.device)
    else:
        target_frame_mask = target_frame_mask.to(device=z_rec_roundtrip.device, dtype=torch.bool)
    target_num_frames = int(target_frame_mask.shape[1])
    prediction = decode_stage1_source_prediction(
        stage1_model,
        ceiling_latent,
        dict(batch),
        z_override=z_rec_roundtrip,
        target_num_frames=target_num_frames,
        target_frame_mask=target_frame_mask,
        return_diagnostics=False,
        validate=validate,
    )
    valid = mask.unsqueeze(-1)
    diff = (z_rec_roundtrip - latent.z_rec.float()).masked_fill(~valid, 0.0)
    denom = valid.sum().clamp_min(1) * z_rec_roundtrip.shape[-1]
    diagnostics = {
        "stage1_ceiling_whiten_roundtrip_z_rec_rms": (diff.pow(2).sum() / denom).sqrt(),
        "stage1_ceiling_valid_latent_tokens": mask.sum(),
    }
    return prediction, diagnostics


@torch.no_grad()
def compute_stage1_reconstruction_ceiling(
    *,
    stage1_model: torch.nn.Module,
    whitening: Any,
    batch: Mapping[str, Any],
    structured_normalizer: Any,
    object_points_local: torch.Tensor | None = None,
    target_frame_mask: torch.Tensor | None = None,
    validate: bool = False,
) -> Stage1CeilingOutput:
    prediction, diagnostics = compute_stage1_reconstruction_ceiling_prediction(
        stage1_model=stage1_model,
        whitening=whitening,
        batch=batch,
        target_frame_mask=target_frame_mask,
        validate=validate,
    )
    points = object_points_local if object_points_local is not None else _object_points_local_from_batch(batch)
    world = prediction_to_world_markers_and_object_points(
        prediction,
        structured_normalizer=structured_normalizer,
        object_points_local=points,
    )
    return Stage1CeilingOutput(prediction=prediction, world=world, diagnostics=diagnostics)


def _target_frame_mask_from_batch(batch: Mapping[str, Any], *, device: torch.device) -> torch.Tensor:
    masks = batch.get("masks")
    if not isinstance(masks, Mapping):
        raise KeyError(
            "Stage1 ceiling requires batch['masks']['frame'] or batch['masks']['valid'] "
            "when target_frame_mask is not provided."
        )
    frame = masks.get("frame")
    mask_name = "frame"
    if frame is None:
        frame = masks.get("valid")
        mask_name = "valid"
    if not isinstance(frame, torch.Tensor):
        raise TypeError(f"batch['masks']['{mask_name}'] must be a tensor.")
    return frame.to(device=device, dtype=torch.bool)


def _object_points_local_from_batch(batch: Mapping[str, Any]) -> torch.Tensor:
    geometry = batch.get("geometry")
    if not isinstance(geometry, Mapping) or "object_points_local" not in geometry:
        raise KeyError("Stage1 ceiling world conversion requires batch['geometry']['object_points_local'].")
    value = geometry["object_points_local"]
    if not isinstance(value, torch.Tensor):
        raise TypeError("batch['geometry']['object_points_local'] must be a tensor.")
    return value
