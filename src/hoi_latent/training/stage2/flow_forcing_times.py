from __future__ import annotations

from typing import Any

import torch

from hoi_latent.models.stage2.flow_time import (
    Stage2FlowTimeField,
    build_random_all_blocks_flow_time,
    build_shared_window_flow_time,
)


def build_random_all_blocks_flow_time_from_config(
    config: dict[str, Any],
    *,
    latent_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    time_ids: torch.Tensor,
    time_mask: torch.Tensor,
    generator: torch.Generator | None = None,
    rollout_progress: torch.Tensor | None = None,
) -> Stage2FlowTimeField:
    cfg = _flow_forcing_cfg(config)
    schedule = str(cfg.get("dynamic_noise_schedule", "random_all_blocks"))
    if schedule != "random_all_blocks":
        raise ValueError(f"Stage2 FF1 requires dynamic_noise_schedule='random_all_blocks', got {schedule!r}.")
    return build_stage2_training_flow_time_from_config(
        config,
        latent_mask=latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=time_mask,
        generator=generator,
        rollout_progress=rollout_progress,
    )


def build_stage2_training_flow_time_from_config(
    config: dict[str, Any],
    *,
    latent_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    time_ids: torch.Tensor,
    time_mask: torch.Tensor,
    generator: torch.Generator | None = None,
    rollout_progress: torch.Tensor | None = None,
) -> Stage2FlowTimeField:
    cfg = _flow_forcing_cfg(config)
    schedule = str(cfg.get("dynamic_noise_schedule", "random_all_blocks"))
    if schedule not in {"random_all_blocks", "shared_window"}:
        raise ValueError(
            "training.stage2.flow_forcing.dynamic_noise_schedule must be "
            f"'random_all_blocks' or 'shared_window', got {schedule!r}."
        )
    static_policy = str(cfg.get("static_time_policy", "independent_static"))
    if static_policy not in {"independent_static", "mean_anchor"}:
        raise ValueError(
            "Stage2 FF1 requires static_time_policy='independent_static' or 'mean_anchor', "
            f"got {static_policy!r}."
        )
    block_size = int(cfg.get("block_size_anchors", 8))
    if block_size <= 0:
        raise ValueError(f"block_size_anchors must be positive, got {block_size}.")
    t_static = None
    if static_policy == "independent_static":
        t_static = torch.rand(
            latent_mask.shape[0],
            device=latent_mask.device,
            dtype=torch.float32,
            generator=generator,
        )
    builder = (
        build_random_all_blocks_flow_time
        if schedule == "random_all_blocks"
        else build_shared_window_flow_time
    )
    return builder(
        latent_mask=latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=time_mask,
        block_size_anchors=block_size,
        generator=generator,
        t_static=t_static,
        rollout_progress=rollout_progress,
    )


def flow_time_log_diagnostics(
    flow_time: Stage2FlowTimeField,
    *,
    block_size_anchors: int | None = None,
    dynamic_noise_schedule: str = "random_all_blocks",
) -> dict[str, torch.Tensor | str]:
    block_size = _infer_block_size(flow_time) if block_size_anchors is None else int(block_size_anchors)
    diagnostics: dict[str, torch.Tensor | str] = {
        "dynamic_noise_schedule": str(dynamic_noise_schedule),
        "block_size_anchors": torch.tensor(
            block_size,
            device=flow_time.t_anchor.device,
            dtype=torch.float32,
        ),
    }
    diagnostics.update(flow_time.diagnostics)
    return diagnostics


def _flow_forcing_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("training", {}).get("stage2", {}).get("flow_forcing", {})


def _infer_block_size(flow_time: Stage2FlowTimeField) -> int:
    used = flow_time.layout.anchor_mask
    if not used.any():
        return 0
    anchor_idx = used.nonzero(as_tuple=False)[0]
    b = int(anchor_idx[0].item())
    blocks = flow_time.layout.anchor_block_index[b][flow_time.layout.anchor_mask[b]]
    if blocks.numel() <= 1:
        return int(blocks.numel())
    first_block = blocks[0]
    return int((blocks == first_block).sum().item())
