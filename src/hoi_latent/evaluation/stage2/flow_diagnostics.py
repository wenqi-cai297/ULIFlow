from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from hoi_latent.evaluation.stage1 import prediction_reconstruction_error_per_frame
from hoi_latent.inference.stage2 import build_gt_stage1_coarse_prediction_from_batch
from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.models.stage2.flow_time import (
    Stage2FlowAnchorLayout,
    Stage2FlowTimeField,
    build_stage2_flow_anchor_layout,
    build_stage2_flow_time_field_from_blocks,
)
from hoi_latent.training.stage1.source_autoencoder import (
    decode_stage1_source_prediction,
    encode_stage1_source_latent,
)
from hoi_latent.training.stage1.phase1_warmup import infer_frame_mask
from hoi_latent.training.stage2.rectified_flow import _stage2_rollout_progress


@dataclass
class Stage2FlowGridDiagnostics:
    flow_time_percent: torch.Tensor
    physical_time_percent: torch.Tensor
    velocity_mse: torch.Tensor
    velocity_observed_mask: torch.Tensor
    output_time_percent: torch.Tensor
    human_marker_error_cm: torch.Tensor
    object_translation_error_cm: torch.Tensor
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_time_percent": self.flow_time_percent.tolist(),
            "physical_time_percent": self.physical_time_percent.tolist(),
            "velocity_mse": self.velocity_mse.tolist(),
            "velocity_observed_mask": self.velocity_observed_mask.tolist(),
            "output_time_percent": self.output_time_percent.tolist(),
            "human_marker_error_cm": self.human_marker_error_cm.tolist(),
            "object_translation_error_cm": self.object_translation_error_cm.tolist(),
            "sample_count": int(self.sample_count),
        }


def build_uniform_stage2_flow_time(
    layout: Stage2FlowAnchorLayout,
    *,
    flow_time: float,
    rollout_progress: torch.Tensor,
) -> Stage2FlowTimeField:
    """Assign one RF time to every generated latent coordinate."""

    value = float(flow_time)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"flow_time must be in [0,1], got {value}.")
    batch_size = int(layout.anchor_mask.shape[0])
    block_counts = []
    for sample_index in range(batch_size):
        used = layout.anchor_block_index[sample_index][layout.anchor_mask[sample_index]]
        if used.numel() == 0:
            raise ValueError("Each sample must contain at least one valid anchor block.")
        block_counts.append(int(used.max().item()) + 1)
    max_blocks = max(block_counts)
    t_block = torch.full(
        (batch_size, max_blocks),
        value,
        device=layout.anchor_mask.device,
        dtype=torch.float32,
    )
    block_mask = torch.zeros_like(t_block, dtype=torch.bool)
    for sample_index, count in enumerate(block_counts):
        block_mask[sample_index, :count] = True
    return build_stage2_flow_time_field_from_blocks(
        layout=layout,
        t_block=t_block,
        block_mask=block_mask,
        t_static=torch.full(
            (batch_size,),
            value,
            device=layout.anchor_mask.device,
            dtype=torch.float32,
        ),
        rollout_progress=rollout_progress,
    )


def velocity_mse_by_physical_time_bins(
    velocity_prediction: torch.Tensor,
    velocity_target: torch.Tensor,
    *,
    time_ids: torch.Tensor,
    dynamic_mask: torch.Tensor,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MSE sums and token counts over normalized physical-time bins."""

    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if velocity_prediction.shape != velocity_target.shape or velocity_prediction.ndim != 3:
        raise ValueError("velocity_prediction and velocity_target must have matching [B,N,D] shapes.")
    expected = velocity_prediction.shape[:2]
    if time_ids.shape != expected or dynamic_mask.shape != expected:
        raise ValueError(f"time_ids and dynamic_mask must have shape {tuple(expected)}.")

    token_mse = (velocity_prediction.float() - velocity_target.float()).square().mean(dim=-1)
    sums = token_mse.new_zeros(num_bins, dtype=torch.float64)
    counts = token_mse.new_zeros(num_bins, dtype=torch.float64)
    for sample_index in range(int(token_mse.shape[0])):
        valid = dynamic_mask[sample_index].bool()
        if not valid.any():
            continue
        times = time_ids[sample_index][valid].float()
        minimum = times.min()
        span = (times.max() - minimum).clamp_min(1.0)
        normalized = (times - minimum) / span
        indices = torch.round(normalized * float(num_bins - 1)).long().clamp(0, num_bins - 1)
        values = token_mse[sample_index][valid].to(dtype=torch.float64)
        sums.scatter_add_(0, indices, values)
        counts.scatter_add_(0, indices, torch.ones_like(values))
    return sums, counts


def interpolate_missing_time_bins(
    values: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    """Linearly fill unobserved physical-time bins along the last axis."""

    if values.ndim != 2:
        raise ValueError(f"values must have shape [R,T], got {tuple(values.shape)}.")
    if observed_mask.ndim == 1:
        observed_mask = observed_mask[None, :].expand_as(values)
    if observed_mask.shape != values.shape:
        raise ValueError(
            "observed_mask must have shape [T] or match values; "
            f"got {tuple(observed_mask.shape)} for {tuple(values.shape)}."
        )

    result = values.clone()
    all_indices = torch.arange(values.shape[1], device=values.device)
    for row_index in range(values.shape[0]):
        observed_indices = all_indices[observed_mask[row_index].bool()]
        if observed_indices.numel() == 0:
            raise ValueError("Each row must contain at least one observed time bin.")
        if observed_indices.numel() == 1:
            result[row_index].fill_(values[row_index, observed_indices[0]])
            continue
        observed_values = values[row_index, observed_indices]
        right_positions = torch.searchsorted(observed_indices, all_indices)
        left_positions = (right_positions - 1).clamp(0, observed_indices.numel() - 1)
        right_positions = right_positions.clamp(0, observed_indices.numel() - 1)
        left_indices = observed_indices[left_positions]
        right_indices = observed_indices[right_positions]
        denominator = (right_indices - left_indices).clamp_min(1).to(values.dtype)
        weight = (all_indices - left_indices).to(values.dtype) / denominator
        interpolated = observed_values[left_positions] + weight * (
            observed_values[right_positions] - observed_values[left_positions]
        )
        result[row_index] = torch.where(
            observed_mask[row_index].bool(),
            values[row_index],
            interpolated,
        )
    return result


@torch.no_grad()
def evaluate_stage2_flow_grid(
    *,
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
    whitening: torch.nn.Module,
    batches: Iterable[dict[str, Any]],
    structured_normalizer: Any,
    flow_times: Sequence[float],
    physical_time_bins: int,
    block_size_anchors: int,
    seed: int,
) -> Stage2FlowGridDiagnostics:
    """Evaluate full-coordinate RF and decoder errors on a fixed teacher-forced grid."""

    if not flow_times:
        raise ValueError("flow_times must not be empty.")
    if physical_time_bins <= 0 or block_size_anchors <= 0:
        raise ValueError("physical_time_bins and block_size_anchors must be positive.")
    flow_values = [float(value) for value in flow_times]
    if any(not 0.0 <= value <= 1.0 for value in flow_values):
        raise ValueError("Every flow time must be in [0,1].")

    velocity_sums = torch.zeros(len(flow_values), physical_time_bins, dtype=torch.float64)
    velocity_counts = torch.zeros_like(velocity_sums)
    human_sums: torch.Tensor | None = None
    human_counts: torch.Tensor | None = None
    object_sums: torch.Tensor | None = None
    object_counts: torch.Tensor | None = None
    sample_count = 0

    for batch_index, batch in enumerate(batches):
        latent = encode_stage1_source_latent(
            stage1_model,
            batch,
            return_diagnostics=False,
            validate=False,
        )
        mask = latent.mask.bool()
        z1 = zero_invalid_tokens(whitening.whiten(latent.z_rec.float(), mask).float(), mask)
        generator = torch.Generator(device=z1.device).manual_seed(int(seed) + batch_index)
        z0 = zero_invalid_tokens(
            torch.randn(z1.shape, device=z1.device, dtype=torch.float32, generator=generator),
            mask,
        )
        velocity_target = zero_invalid_tokens(z1 - z0, mask)
        layout = build_stage2_flow_anchor_layout(
            latent_mask=mask,
            token_type_ids=latent.token_type_ids,
            time_ids=latent.time_ids,
            time_mask=latent.time_mask,
            block_size_anchors=int(block_size_anchors),
        )
        rollout_progress = _stage2_rollout_progress(batch, device=z1.device)
        target_prediction = build_gt_stage1_coarse_prediction_from_batch(
            _batch_with_explicit_frame_mask(batch)
        )

        with _autocast_context(z1.device):
            text = stage2_model.text_encoder(batch, return_diagnostics=False)
            geometry = stage2_model.geometry_encoder(batch, return_diagnostics=False)

        for flow_index, value in enumerate(flow_values):
            flow_time = build_uniform_stage2_flow_time(
                layout,
                flow_time=value,
                rollout_progress=rollout_progress,
            )
            t_token = flow_time.t_latent_token[..., None].float()
            z_t = zero_invalid_tokens((1.0 - t_token) * z0 + t_token * z1, mask)
            with _autocast_context(z1.device):
                latent_tokens = stage2_model.latent_encoder(
                    z_white_t=z_t,
                    latent_mask=mask,
                    modality_ids=latent.modality_ids,
                    token_type_ids=latent.token_type_ids,
                    time_ids=latent.time_ids,
                    entity_ids=latent.entity_ids,
                    time_mask=latent.time_mask,
                    return_diagnostics=False,
                )
                packed = stage2_model.packer(latent_tokens, text, geometry)
                velocity_prediction = stage2_model.backbone(
                    packed,
                    flow_time,
                    return_diagnostics=False,
                ).velocity.float()

            sums, counts = velocity_mse_by_physical_time_bins(
                velocity_prediction,
                velocity_target,
                time_ids=latent.time_ids,
                dynamic_mask=layout.dynamic_token_mask,
                num_bins=physical_time_bins,
            )
            velocity_sums[flow_index] += sums.cpu()
            velocity_counts[flow_index] += counts.cpu()

            z1_prediction = zero_invalid_tokens(
                z_t + (1.0 - t_token) * velocity_prediction,
                mask,
            )
            z_rec_prediction = whitening.unwhiten(z1_prediction, mask).float()
            with _autocast_context(z1.device):
                decoded = decode_stage1_source_prediction(
                    stage1_model,
                    latent,
                    batch,
                    z_override=z_rec_prediction,
                    return_diagnostics=False,
                    validate=False,
                )
            errors = prediction_reconstruction_error_per_frame(
                decoded,
                target_prediction,
                structured_normalizer,
            )
            human_sums, human_counts = _accumulate_frame_errors(
                human_sums,
                human_counts,
                flow_index,
                errors["human_cm"],
                decoded.masks["human"] & target_prediction.masks["human"],
                flow_count=len(flow_values),
            )
            object_sums, object_counts = _accumulate_frame_errors(
                object_sums,
                object_counts,
                flow_index,
                errors["object_cm"],
                decoded.masks["object"] & target_prediction.masks["object"],
                flow_count=len(flow_values),
            )
        sample_count += int(z1.shape[0])

    if sample_count == 0 or human_sums is None or object_sums is None:
        raise ValueError("Stage2 flow-grid evaluation received no batches.")
    velocity_observed_mask = velocity_counts > 0.0
    velocity_mse = interpolate_missing_time_bins(
        velocity_sums / velocity_counts.clamp_min(1.0),
        velocity_observed_mask,
    )
    human_error = human_sums / human_counts.clamp_min(1.0)
    object_error = object_sums / object_counts.clamp_min(1.0)
    frame_count = int(human_error.shape[1])
    return Stage2FlowGridDiagnostics(
        flow_time_percent=torch.tensor(flow_values, dtype=torch.float64) * 100.0,
        physical_time_percent=(torch.arange(physical_time_bins, dtype=torch.float64) + 0.5)
        * (100.0 / float(physical_time_bins)),
        velocity_mse=velocity_mse,
        velocity_observed_mask=velocity_observed_mask,
        output_time_percent=torch.linspace(0.0, 100.0, frame_count, dtype=torch.float64),
        human_marker_error_cm=human_error,
        object_translation_error_cm=object_error,
        sample_count=sample_count,
    )


def _accumulate_frame_errors(
    sums: torch.Tensor | None,
    counts: torch.Tensor | None,
    flow_index: int,
    errors: torch.Tensor,
    mask: torch.Tensor,
    *,
    flow_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if errors.ndim != 2:
        raise ValueError(f"Per-frame errors must have shape [B,T], got {tuple(errors.shape)}.")
    valid = mask.bool()
    while valid.ndim > 2:
        valid = valid.any(dim=-1)
    if valid.shape != errors.shape:
        raise ValueError(f"Frame mask shape {tuple(valid.shape)} does not match errors {tuple(errors.shape)}.")
    if sums is None or counts is None:
        sums = torch.zeros(flow_count, errors.shape[1], dtype=torch.float64)
        counts = torch.zeros_like(sums)
    sums[flow_index] += (errors.double() * valid.to(dtype=torch.float64)).sum(dim=0).cpu()
    counts[flow_index] += valid.double().sum(dim=0).cpu()
    return sums, counts


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _batch_with_explicit_frame_mask(batch: dict[str, Any]) -> dict[str, Any]:
    masks = batch.get("masks")
    if not isinstance(masks, dict):
        raise KeyError("Stage2 flow diagnostics require batch['masks'].")
    if "frame" in masks:
        return batch
    augmented = dict(batch)
    augmented["masks"] = dict(masks)
    augmented["masks"]["frame"] = infer_frame_mask(batch)
    return augmented
