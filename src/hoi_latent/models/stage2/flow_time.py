from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from hoi_latent.models.stage1.tokenizer_types import TokenTypeId


@dataclass
class Stage2FlowAnchorLayout:
    anchor_time_ids: torch.Tensor
    anchor_mask: torch.Tensor
    anchor_block_index: torch.Tensor
    token_anchor_index: torch.Tensor
    token_block_index: torch.Tensor
    dynamic_token_mask: torch.Tensor
    static_geometry_token_mask: torch.Tensor
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class Stage2FlowTimeField:
    layout: Stage2FlowAnchorLayout
    t_block: torch.Tensor
    block_mask: torch.Tensor
    t_anchor: torch.Tensor
    # Static generated object-geometry latent noise time; this is not the
    # condition-stream rollout clock.
    t_static: torch.Tensor
    rollout_progress: torch.Tensor
    t_latent_token: torch.Tensor
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


def temporal_dynamic_token_mask(
    time_mask: torch.Tensor,
    latent_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
) -> torch.Tensor:
    if time_mask.shape != latent_mask.shape or token_type_ids.shape != latent_mask.shape:
        raise ValueError(
            "time_mask, latent_mask, and token_type_ids must have matching shapes, "
            f"got {tuple(time_mask.shape)}, {tuple(latent_mask.shape)}, {tuple(token_type_ids.shape)}."
        )
    return (
        time_mask.to(dtype=torch.bool)
        & latent_mask.to(dtype=torch.bool)
        & (token_type_ids.to(dtype=torch.long) != int(TokenTypeId.OBJECT_GEOMETRY))
    )


def build_stage2_flow_anchor_layout(
    *,
    latent_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    time_ids: torch.Tensor,
    time_mask: torch.Tensor,
    block_size_anchors: int = 8,
) -> Stage2FlowAnchorLayout:
    if block_size_anchors <= 0:
        raise ValueError(f"block_size_anchors must be positive, got {block_size_anchors}.")
    _validate_same_shape(
        latent_mask=latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=time_mask,
    )
    device = latent_mask.device
    latent_mask_bool = latent_mask.to(dtype=torch.bool)
    token_type_long = token_type_ids.to(device=device, dtype=torch.long)
    dynamic = temporal_dynamic_token_mask(time_mask, latent_mask_bool, token_type_long)
    static_geometry = latent_mask_bool & (token_type_long == int(TokenTypeId.OBJECT_GEOMETRY))

    batch, num_tokens = latent_mask_bool.shape
    unique_times_by_sample: list[torch.Tensor] = []
    max_anchors = 0
    for b in range(batch):
        valid_times = time_ids[b].to(device=device, dtype=torch.long)[dynamic[b]]
        if valid_times.numel() == 0:
            raise ValueError("Each sample must contain at least one dynamic latent anchor.")
        unique_times = torch.unique(valid_times, sorted=True)
        unique_times_by_sample.append(unique_times)
        max_anchors = max(max_anchors, int(unique_times.numel()))

    anchor_time_ids = torch.zeros(batch, max_anchors, device=device, dtype=torch.long)
    anchor_mask = torch.zeros(batch, max_anchors, device=device, dtype=torch.bool)
    anchor_block_index = torch.zeros(batch, max_anchors, device=device, dtype=torch.long)
    token_anchor_index = torch.zeros(batch, num_tokens, device=device, dtype=torch.long)
    token_block_index = torch.zeros(batch, num_tokens, device=device, dtype=torch.long)

    time_long = time_ids.to(device=device, dtype=torch.long)
    for b, unique_times in enumerate(unique_times_by_sample):
        count = int(unique_times.numel())
        anchor_time_ids[b, :count] = unique_times
        anchor_mask[b, :count] = True
        anchor_block_index[b, :count] = torch.arange(count, device=device, dtype=torch.long) // int(block_size_anchors)
        token_indices = torch.searchsorted(unique_times, time_long[b][dynamic[b]])
        token_anchor_index[b, dynamic[b]] = token_indices
        token_block_index[b, dynamic[b]] = token_indices // int(block_size_anchors)
        if not torch.equal(unique_times[token_indices], time_long[b][dynamic[b]]):
            raise ValueError("t_anchor does not cover every valid dynamic time_id.")

    num_anchors = anchor_mask.sum(dim=1)
    num_blocks = torch.div(num_anchors + int(block_size_anchors) - 1, int(block_size_anchors), rounding_mode="floor")
    return Stage2FlowAnchorLayout(
        anchor_time_ids=anchor_time_ids,
        anchor_mask=anchor_mask,
        anchor_block_index=anchor_block_index,
        token_anchor_index=token_anchor_index,
        token_block_index=token_block_index,
        dynamic_token_mask=dynamic,
        static_geometry_token_mask=static_geometry,
        diagnostics={
            "num_flow_anchors": num_anchors.detach(),
            "num_flow_blocks": num_blocks.detach(),
        },
    )


def build_stage2_flow_time_field_from_blocks(
    *,
    layout: Stage2FlowAnchorLayout,
    t_block: torch.Tensor,
    block_mask: torch.Tensor,
    t_static: torch.Tensor | None = None,
    rollout_progress: torch.Tensor | None = None,
) -> Stage2FlowTimeField:
    _validate_layout(layout)
    if t_block.shape != block_mask.shape:
        raise ValueError(f"t_block and block_mask must have matching shapes, got {tuple(t_block.shape)} and {tuple(block_mask.shape)}.")
    if t_block.ndim != 2 or t_block.shape[0] != layout.anchor_mask.shape[0]:
        raise ValueError(f"t_block must have shape [B,K], got {tuple(t_block.shape)}.")
    device = layout.anchor_mask.device
    dtype = t_block.dtype
    t_block = t_block.to(device=device)
    block_mask = block_mask.to(device=device, dtype=torch.bool)
    _validate_times_in_unit_interval(t_block[block_mask], name="t_block")

    max_used_block = layout.anchor_block_index[layout.anchor_mask].max()
    if t_block.shape[1] <= int(max_used_block.item()):
        raise ValueError("t_block does not cover every valid anchor block.")
    block_index = layout.anchor_block_index.clamp_min(0).clamp_max(t_block.shape[1] - 1)
    used_block_valid = torch.gather(block_mask, dim=1, index=block_index)
    if (layout.anchor_mask & ~used_block_valid).any():
        raise ValueError("block_mask marks a block invalid even though it is used by a valid anchor.")

    gathered_anchor = torch.gather(t_block, dim=1, index=block_index)
    t_anchor = torch.where(layout.anchor_mask, gathered_anchor, torch.zeros_like(gathered_anchor))
    if t_static is None:
        t_static = _masked_mean(t_anchor, layout.anchor_mask)
    else:
        if t_static.shape != (layout.anchor_mask.shape[0],):
            raise ValueError(f"t_static must have shape {(layout.anchor_mask.shape[0],)}, got {tuple(t_static.shape)}.")
        t_static = t_static.to(device=device, dtype=dtype)
        _validate_times_in_unit_interval(t_static, name="t_static")
    if rollout_progress is None:
        raise ValueError("rollout_progress is required; do not use t_static as a condition-stream clock.")
    if rollout_progress.shape != (layout.anchor_mask.shape[0],):
        raise ValueError(
            f"rollout_progress must have shape {(layout.anchor_mask.shape[0],)}, got {tuple(rollout_progress.shape)}."
        )
    rollout_progress = rollout_progress.to(device=device, dtype=dtype)
    _validate_times_in_unit_interval(rollout_progress, name="rollout_progress")

    batch, num_tokens = layout.token_anchor_index.shape
    token_anchor_index = layout.token_anchor_index.clamp_min(0).clamp_max(t_anchor.shape[1] - 1)
    gathered_token = torch.gather(t_anchor, dim=1, index=token_anchor_index)
    t_latent_token = torch.zeros(batch, num_tokens, device=device, dtype=dtype)
    t_latent_token = torch.where(layout.dynamic_token_mask, gathered_token, t_latent_token)
    t_static_token = t_static[:, None].expand_as(t_latent_token)
    t_latent_token = torch.where(layout.static_geometry_token_mask, t_static_token, t_latent_token)

    return Stage2FlowTimeField(
        layout=layout,
        t_block=torch.where(block_mask, t_block, torch.zeros_like(t_block)),
        block_mask=block_mask,
        t_anchor=t_anchor,
        t_static=t_static,
        rollout_progress=rollout_progress,
        t_latent_token=t_latent_token,
        diagnostics=_flow_time_diagnostics(
            t_block=t_block,
            block_mask=block_mask,
            t_anchor=t_anchor,
            anchor_mask=layout.anchor_mask,
            t_static=t_static,
            rollout_progress=rollout_progress,
            t_latent_token=t_latent_token,
            layout=layout,
        ),
    )


def build_random_all_blocks_flow_time(
    *,
    latent_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    time_ids: torch.Tensor,
    time_mask: torch.Tensor,
    block_size_anchors: int = 8,
    generator: torch.Generator | None = None,
    t_static: torch.Tensor | None = None,
    rollout_progress: torch.Tensor | None = None,
) -> Stage2FlowTimeField:
    layout = build_stage2_flow_anchor_layout(
        latent_mask=latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=time_mask,
        block_size_anchors=block_size_anchors,
    )
    num_anchors = layout.anchor_mask.sum(dim=1)
    num_blocks = torch.div(num_anchors + int(block_size_anchors) - 1, int(block_size_anchors), rounding_mode="floor")
    max_blocks = int(num_blocks.max().item())
    t_block = torch.zeros(latent_mask.shape[0], max_blocks, device=latent_mask.device, dtype=torch.float32)
    block_mask = torch.zeros_like(t_block, dtype=torch.bool)
    for b in range(latent_mask.shape[0]):
        count = int(num_blocks[b].item())
        block_mask[b, :count] = True
        t_block[b, :count] = torch.rand(count, device=latent_mask.device, dtype=torch.float32, generator=generator)
    return build_stage2_flow_time_field_from_blocks(
        layout=layout,
        t_block=t_block,
        block_mask=block_mask,
        t_static=t_static,
        rollout_progress=rollout_progress,
    )


def build_shared_window_flow_time(
    *,
    latent_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    time_ids: torch.Tensor,
    time_mask: torch.Tensor,
    block_size_anchors: int = 8,
    generator: torch.Generator | None = None,
    t_static: torch.Tensor | None = None,
    rollout_progress: torch.Tensor | None = None,
) -> Stage2FlowTimeField:
    """Sample one flow time per example and share it across all temporal blocks."""
    layout = build_stage2_flow_anchor_layout(
        latent_mask=latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=time_mask,
        block_size_anchors=block_size_anchors,
    )
    num_anchors = layout.anchor_mask.sum(dim=1)
    num_blocks = torch.div(
        num_anchors + int(block_size_anchors) - 1,
        int(block_size_anchors),
        rounding_mode="floor",
    )
    max_blocks = int(num_blocks.max().item())
    shared_time = torch.rand(
        latent_mask.shape[0],
        1,
        device=latent_mask.device,
        dtype=torch.float32,
        generator=generator,
    )
    block_ids = torch.arange(max_blocks, device=latent_mask.device).unsqueeze(0)
    block_mask = block_ids < num_blocks.unsqueeze(1)
    return build_stage2_flow_time_field_from_blocks(
        layout=layout,
        t_block=shared_time.expand(-1, max_blocks),
        block_mask=block_mask,
        t_static=t_static,
        rollout_progress=rollout_progress,
    )


def _validate_same_shape(**values: torch.Tensor) -> None:
    shapes = {name: tuple(value.shape) for name, value in values.items()}
    first = next(iter(shapes.values()))
    if any(shape != first for shape in shapes.values()):
        raise ValueError(f"Expected matching [B,N] shapes, got {shapes}.")
    if len(first) != 2:
        raise ValueError(f"Expected rank-2 [B,N] tensors, got shape {first}.")


def _validate_layout(layout: Stage2FlowAnchorLayout) -> None:
    batch, anchors = layout.anchor_mask.shape
    if layout.anchor_time_ids.shape != (batch, anchors) or layout.anchor_block_index.shape != (batch, anchors):
        raise ValueError("Stage2FlowAnchorLayout anchor tensors must have matching [B,A] shapes.")
    token_shape = layout.token_anchor_index.shape
    for name in ["token_block_index", "dynamic_token_mask", "static_geometry_token_mask"]:
        if getattr(layout, name).shape != token_shape:
            raise ValueError(f"Stage2FlowAnchorLayout.{name} must have shape {token_shape}.")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = value * mask.to(device=value.device, dtype=value.dtype)
    denom = mask.sum(dim=1).clamp_min(1).to(device=value.device, dtype=value.dtype)
    return weighted.sum(dim=1) / denom


def _validate_times_in_unit_interval(value: torch.Tensor, *, name: str) -> None:
    if value.numel() == 0:
        return
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")
    bad = (value < 0.0) | (value > 1.0)
    if bad.any():
        bad_min = float(value[bad].min().detach().cpu().item())
        bad_max = float(value[bad].max().detach().cpu().item())
        raise ValueError(f"{name} values must be in [0,1], got range [{bad_min}, {bad_max}].")


def _flow_time_diagnostics(
    *,
    t_block: torch.Tensor,
    block_mask: torch.Tensor,
    t_anchor: torch.Tensor,
    anchor_mask: torch.Tensor,
    t_static: torch.Tensor,
    rollout_progress: torch.Tensor,
    t_latent_token: torch.Tensor,
    layout: Stage2FlowAnchorLayout,
) -> dict[str, torch.Tensor]:
    valid_block = t_block[block_mask]
    valid_anchor = t_anchor[anchor_mask]
    dynamic = layout.dynamic_token_mask
    static = layout.static_geometry_token_mask
    diagnostics: dict[str, torch.Tensor] = {
        "t_static_mean": t_static.detach().float().mean(),
        "rollout_progress_mean": rollout_progress.detach().float().mean(),
        "rollout_progress_min": rollout_progress.detach().float().min(),
        "rollout_progress_max": rollout_progress.detach().float().max(),
        "num_flow_blocks_mean": block_mask.sum(dim=1).detach().float().mean(),
        "num_flow_blocks_max": block_mask.sum(dim=1).detach().float().max(),
    }
    if valid_block.numel():
        diagnostics["t_block_mean"] = valid_block.detach().float().mean()
        diagnostics["t_block_std"] = valid_block.detach().float().std(unbiased=False)
    if valid_anchor.numel():
        diagnostics["t_anchor_mean"] = valid_anchor.detach().float().mean()
        diagnostics["t_anchor_std"] = valid_anchor.detach().float().std(unbiased=False)
    if dynamic.any():
        diagnostics["t_latent_dynamic_mean"] = t_latent_token[dynamic].detach().float().mean()
    if static.any():
        diagnostics["t_latent_static_mean"] = t_latent_token[static].detach().float().mean()
    return diagnostics
