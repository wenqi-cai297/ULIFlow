from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Protocol

import torch

from hoi_latent.inference.stage2.latent_template import (
    Stage2InferenceLatentTemplate,
    Stage2InferenceLatentTemplateSpec,
    build_stage2_inference_latent_template,
)
from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.models.stage2.flow_time import (
    Stage2FlowTimeField,
    build_stage2_flow_anchor_layout,
    build_stage2_flow_time_field_from_blocks,
)


@dataclass(frozen=True)
class Stage2GuidanceConfig:
    enabled: bool = False
    mode: str = "none"
    joint_scale: float = 1.0
    text_scale: float = 1.0
    geometry_scale: float = 1.0


@dataclass(frozen=True)
class Stage2FlowForcingRolloutConfig:
    steps_per_block: int = 32
    window_blocks: int = 4
    clean_history_blocks: int = 1
    solver: str = "euler"


@dataclass
class Stage2SamplerStepState:
    z_window: torch.Tensor
    flow_time: Stage2FlowTimeField
    template: Stage2InferenceLatentTemplate
    update_token_mask: torch.Tensor
    dt_token: torch.Tensor
    batch: dict


@dataclass
class Stage2SamplerStepOutput:
    z_next: torch.Tensor
    velocity: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Stage2SamplerStepContext:
    step: int
    window_start: int
    local_to_global_block: torch.Tensor
    should_update_global: torch.Tensor


@dataclass
class Stage2DecodePlan:
    decode_window_starts: torch.Tensor
    preferred_decode_window_start: torch.Tensor
    preferred_local_slot: torch.Tensor


@dataclass
class Stage2FlowForcingRolloutOutput:
    z_blocks: torch.Tensor
    z_static: torch.Tensor
    decode_z_white_windows: torch.Tensor
    decode_window_starts: torch.Tensor
    preferred_decode_window_start: torch.Tensor
    preferred_local_slot: torch.Tensor
    update_counts: torch.Tensor
    static_update_count: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Stage2FlowForcingSolver(Protocol):
    name: str

    def step(
        self,
        model: torch.nn.Module,
        state: Stage2SamplerStepState,
        *,
        guidance: Stage2GuidanceConfig,
    ) -> Stage2SamplerStepOutput:
        ...


class EulerFlowForcingSolver:
    name = "euler"

    def step(
        self,
        model: torch.nn.Module,
        state: Stage2SamplerStepState,
        *,
        guidance: Stage2GuidanceConfig,
    ) -> Stage2SamplerStepOutput:
        velocity, guide_diag = guided_velocity(
            model,
            batch=state.batch,
            z_white_t=state.z_window,
            flow_time=state.flow_time,
            template=state.template,
            guidance=guidance,
        )
        z_update = state.z_window + state.dt_token * velocity
        z_next = torch.where(state.update_token_mask.unsqueeze(-1), z_update, state.z_window)
        return Stage2SamplerStepOutput(
            z_next=z_next,
            velocity=velocity,
            diagnostics={"solver": self.name, "solver_num_velocity_queries": 1, **guide_diag},
        )


class HeunFlowForcingSolver:
    name = "heun"

    def step(
        self,
        model: torch.nn.Module,
        state: Stage2SamplerStepState,
        *,
        guidance: Stage2GuidanceConfig,
    ) -> Stage2SamplerStepOutput:
        v0, guide_diag = guided_velocity(
            model,
            batch=state.batch,
            z_white_t=state.z_window,
            flow_time=state.flow_time,
            template=state.template,
            guidance=guidance,
        )
        z_pred = torch.where(
            state.update_token_mask.unsqueeze(-1),
            state.z_window + state.dt_token * v0,
            state.z_window,
        )
        flow_time_pred = advance_flow_time_for_updated_tokens(
            state.flow_time,
            update_token_mask=state.update_token_mask,
            dt_token=state.dt_token.squeeze(-1),
        )
        v1, _ = guided_velocity(
            model,
            batch=state.batch,
            z_white_t=z_pred,
            flow_time=flow_time_pred,
            template=state.template,
            guidance=guidance,
        )
        velocity = 0.5 * (v0 + v1)
        z_heun = state.z_window + state.dt_token * velocity
        z_next = torch.where(state.update_token_mask.unsqueeze(-1), z_heun, state.z_window)
        return Stage2SamplerStepOutput(
            z_next=z_next,
            velocity=velocity,
            diagnostics={"solver": self.name, "solver_num_velocity_queries": 2, **guide_diag},
        )


@torch.no_grad()
def sample_stage2_flow_forcing_rollout(
    model: torch.nn.Module,
    *,
    batch: dict,
    template_spec: Stage2InferenceLatentTemplateSpec,
    requested_frames: int,
    guidance: Stage2GuidanceConfig | dict[str, Any] | None = None,
    solver: Stage2FlowForcingSolver | str | None = None,
    seed: int = 0,
    config: Stage2FlowForcingRolloutConfig | None = None,
    step_callback: Callable[[Stage2SamplerStepContext, Stage2SamplerStepState], None] | None = None,
) -> Stage2FlowForcingRolloutOutput:
    cfg = config or Stage2FlowForcingRolloutConfig()
    if requested_frames <= 0:
        raise ValueError(f"requested_frames must be positive, got {requested_frames}.")
    if cfg.steps_per_block <= 0:
        raise ValueError(f"steps_per_block must be positive, got {cfg.steps_per_block}.")
    if cfg.clean_history_blocks < 0 or cfg.clean_history_blocks >= cfg.window_blocks:
        raise ValueError("clean_history_blocks must be in [0, window_blocks).")
    solver_obj = _coerce_solver(solver or cfg.solver)
    guidance_cfg = _coerce_guidance(guidance)
    device = _model_device(model)
    batch_size = _batch_size_from_batch(batch)
    window_blocks = int(cfg.window_blocks)
    if window_blocks != _window_blocks(template_spec):
        raise ValueError(f"config.window_blocks={window_blocks} does not match template window blocks {_window_blocks(template_spec)}.")
    num_blocks = _ceil_div(int(requested_frames), _block_size_frames(template_spec))
    active_slots = int(cfg.window_blocks) - int(cfg.clean_history_blocks)
    block_stride_steps = math.ceil(float(cfg.steps_per_block) / float(active_slots))
    start_steps = torch.arange(num_blocks, device=device, dtype=torch.long) * int(block_stride_steps)
    end_steps = start_steps + int(cfg.steps_per_block)
    total_steps = int(end_steps[-1].item()) if num_blocks else 0
    generator = torch.Generator(device=device).manual_seed(int(seed))

    base_template = build_stage2_inference_latent_template(
        template_spec,
        batch_size=batch_size,
        requested_frames=template_spec.window_frames,
        local_to_global_block=torch.arange(window_blocks, device=device),
        device=device,
    )
    dynamic_positions = _dynamic_positions_by_local_block(base_template, window_blocks=window_blocks)
    static_positions = base_template.static_geometry_token_mask[0].nonzero(as_tuple=False).flatten()
    latent_dim = _latent_dim_from_model(model)
    block_token_count = int(dynamic_positions[0].numel())
    z_blocks = torch.randn(batch_size, num_blocks, block_token_count, latent_dim, device=device, generator=generator)
    z_static = torch.randn(batch_size, int(static_positions.numel()), latent_dim, device=device, generator=generator)
    update_counts = torch.zeros(batch_size, num_blocks, device=device, dtype=torch.long)
    static_update_count = torch.zeros(batch_size, device=device, dtype=torch.long)

    committed_until = 0
    last_diag: dict[str, Any] = {}
    for step in range(total_steps):
        window_start, _ = _choose_window_start(
            committed_until,
            num_blocks,
            window_blocks=window_blocks,
            clean_history_blocks=int(cfg.clean_history_blocks),
        )
        local_to_global = _local_to_global(window_start, window_blocks, batch_size=batch_size, device=device)
        template = build_stage2_inference_latent_template(
            template_spec,
            batch_size=batch_size,
            requested_frames=requested_frames,
            local_to_global_block=local_to_global,
            device=device,
        )
        z_window = _gather_local_window(
            z_blocks=z_blocks,
            z_static=z_static,
            template=template,
            dynamic_positions=dynamic_positions,
            static_positions=static_positions,
            latent_dim=latent_dim,
        )
        local_t_block = _local_t_block(
            step,
            local_to_global[0],
            start_steps,
            end_steps,
            steps_per_block=int(cfg.steps_per_block),
            batch_size=batch_size,
        )
        t_static = local_t_block[:, 0].clone() if step < int(cfg.steps_per_block) else torch.ones(batch_size, device=device)
        flow_time = _build_flow_time_for_template(
            template,
            local_t_block=local_t_block,
            t_static=t_static,
            rollout_progress=_rollout_progress(
                window_start,
                requested_frames=requested_frames,
                window_frames=template_spec.window_frames,
                block_size_frames=_block_size_frames(template_spec),
                device=device,
                batch_size=batch_size,
            ),
        )
        should_update_global = (step >= start_steps) & (step < end_steps)
        update_token_mask = _local_update_mask(
            template,
            local_to_global[0],
            should_update_global=should_update_global,
            update_static=step < int(cfg.steps_per_block),
        )
        dt_token = update_token_mask.to(dtype=z_window.dtype).unsqueeze(-1) / float(cfg.steps_per_block)
        state = Stage2SamplerStepState(
            z_window=z_window,
            flow_time=flow_time,
            template=template,
            update_token_mask=update_token_mask,
            dt_token=dt_token,
            batch=batch,
        )
        if step_callback is not None:
            step_callback(
                Stage2SamplerStepContext(
                    step=int(step),
                    window_start=int(window_start),
                    local_to_global_block=local_to_global.detach().clone(),
                    should_update_global=should_update_global.detach().clone(),
                ),
                state,
            )
        out = solver_obj.step(
            model,
            state,
            guidance=guidance_cfg,
        )
        _scatter_local_window(
            z_next=out.z_next,
            z_blocks=z_blocks,
            z_static=z_static,
            local_to_global=local_to_global[0],
            should_update_global=should_update_global,
            update_static=step < int(cfg.steps_per_block),
            dynamic_positions=dynamic_positions,
            static_positions=static_positions,
        )
        for g in local_to_global[0].tolist():
            if 0 <= int(g) < num_blocks and bool(should_update_global[int(g)].item()):
                update_counts[:, int(g)] += 1
        if step < int(cfg.steps_per_block):
            static_update_count += 1
        while committed_until < num_blocks and (step + 1) >= int(end_steps[committed_until].item()):
            committed_until += 1
        last_diag = out.diagnostics

    decode_windows, decode_starts = _assemble_decode_windows(
        z_blocks=z_blocks,
        z_static=z_static,
        template_spec=template_spec,
        requested_frames=requested_frames,
        dynamic_positions=dynamic_positions,
        static_positions=static_positions,
        device=device,
    )
    decode_plan = build_preferred_decode_plan(
        num_blocks=num_blocks,
        window_blocks=window_blocks,
        clean_history_blocks=int(cfg.clean_history_blocks),
        device=device,
    )
    return Stage2FlowForcingRolloutOutput(
        z_blocks=z_blocks,
        z_static=z_static,
        decode_z_white_windows=decode_windows,
        decode_window_starts=decode_starts,
        preferred_decode_window_start=decode_plan.preferred_decode_window_start,
        preferred_local_slot=decode_plan.preferred_local_slot,
        update_counts=update_counts,
        static_update_count=static_update_count,
        diagnostics={
            "num_requested_frames": torch.tensor(int(requested_frames), device=device),
            "num_blocks": torch.tensor(int(num_blocks), device=device),
            "steps_per_block": torch.tensor(int(cfg.steps_per_block), device=device),
            "block_stride_steps": torch.tensor(int(block_stride_steps), device=device),
            "num_model_forwards": torch.tensor(
                int(total_steps) * _solver_velocity_queries(solver_obj) * _guidance_forward_count(guidance_cfg),
                device=device,
            ),
            "committed_blocks": torch.tensor(int(committed_until), device=device),
            "static_geometry_policy": "bootstrap_then_freeze",
            "solver": solver_obj.name,
            **last_diag,
        },
    )


def guided_velocity(
    model: torch.nn.Module,
    *,
    batch: dict,
    z_white_t: torch.Tensor,
    flow_time: Stage2FlowTimeField,
    template: Stage2InferenceLatentTemplate,
    guidance: Stage2GuidanceConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    guidance = _coerce_guidance(guidance)
    batch_size = z_white_t.shape[0]
    device = z_white_t.device
    none = torch.zeros(batch_size, device=device, dtype=torch.bool)
    all_ = torch.ones(batch_size, device=device, dtype=torch.bool)

    def call(text_null: torch.Tensor, geometry_null: torch.Tensor) -> torch.Tensor:
        return model(
            batch=batch,
            z_white_t=z_white_t,
            flow_time=flow_time,
            latent_mask=template.latent_mask,
            modality_ids=template.modality_ids,
            token_type_ids=template.token_type_ids,
            time_ids=template.time_ids,
            entity_ids=template.entity_ids,
            time_mask=template.time_mask,
            force_null_text_mask=text_null,
            force_null_geometry_mask=geometry_null,
            return_diagnostics=False,
        ).velocity.float()

    mode = guidance.mode if guidance.enabled else "none"
    if mode == "none":
        velocity = call(none, none)
        return velocity, {
            "guidance_num_forwards_per_step": torch.tensor(1, device=device),
            "guidance_delta_joint_rms": velocity.new_tensor(0.0),
            "guidance_delta_text_rms": velocity.new_tensor(0.0),
            "guidance_delta_geometry_rms": velocity.new_tensor(0.0),
        }
    if mode == "joint":
        v_cond = call(none, none)
        v_null = call(all_, all_)
        delta = v_cond - v_null
        velocity = v_null + float(guidance.joint_scale) * delta
        return velocity, {
            "guidance_num_forwards_per_step": torch.tensor(2, device=device),
            "guidance_delta_joint_rms": _rms(delta),
            "guidance_delta_text_rms": velocity.new_tensor(0.0),
            "guidance_delta_geometry_rms": velocity.new_tensor(0.0),
        }
    if mode == "separated":
        v_both = call(none, none)
        v_text_only = call(none, all_)
        v_geom_only = call(all_, none)
        v_null = call(all_, all_)
        text_delta = 0.5 * ((v_text_only - v_null) + (v_both - v_geom_only))
        geometry_delta = 0.5 * ((v_geom_only - v_null) + (v_both - v_text_only))
        velocity = v_null + float(guidance.text_scale) * text_delta + float(guidance.geometry_scale) * geometry_delta
        return velocity, {
            "guidance_num_forwards_per_step": torch.tensor(4, device=device),
            "guidance_delta_joint_rms": _rms(v_both - v_null),
            "guidance_delta_text_rms": _rms(text_delta),
            "guidance_delta_geometry_rms": _rms(geometry_delta),
        }
    raise ValueError(f"Unsupported Stage2 guidance mode: {mode!r}.")


def advance_flow_time_for_updated_tokens(
    flow_time: Stage2FlowTimeField,
    *,
    update_token_mask: torch.Tensor,
    dt_token: torch.Tensor,
) -> Stage2FlowTimeField:
    layout = flow_time.layout
    if update_token_mask.shape != layout.dynamic_token_mask.shape:
        raise ValueError(f"update_token_mask must have shape {tuple(layout.dynamic_token_mask.shape)}.")
    if dt_token.shape != update_token_mask.shape:
        raise ValueError(f"dt_token must have shape {tuple(update_token_mask.shape)}, got {tuple(dt_token.shape)}.")
    update = update_token_mask.to(device=flow_time.t_block.device, dtype=torch.bool)
    dt = dt_token.to(device=flow_time.t_block.device, dtype=flow_time.t_block.dtype)
    t_block = flow_time.t_block.clone()
    dynamic_update = update & layout.dynamic_token_mask.to(device=update.device)
    for b in range(update.shape[0]):
        for block in range(t_block.shape[1]):
            idx = dynamic_update[b] & (layout.token_block_index[b].to(device=update.device) == block)
            if idx.any():
                t_block[b, block] = (t_block[b, block] + dt[b][idx].max()).clamp(0.0, 1.0)
    t_static = flow_time.t_static.clone()
    static_update = update & layout.static_geometry_token_mask.to(device=update.device)
    for b in range(update.shape[0]):
        if static_update[b].any():
            t_static[b] = (t_static[b] + dt[b][static_update[b]].max()).clamp(0.0, 1.0)
    return build_stage2_flow_time_field_from_blocks(
        layout=layout,
        t_block=t_block,
        block_mask=flow_time.block_mask,
        t_static=t_static,
        rollout_progress=flow_time.rollout_progress,
    )


def build_stage2_flow_forcing_solver(config: dict[str, Any] | str | None) -> Stage2FlowForcingSolver:
    if config is None:
        name = "euler"
    elif isinstance(config, str):
        name = config
    else:
        sampler = config.get("inference", {}).get("stage2", {}).get("sampler", {})
        solver_cfg = sampler.get("solver", "euler")
        name = solver_cfg.get("name", "euler") if isinstance(solver_cfg, dict) else str(solver_cfg)
    if name == "euler":
        return EulerFlowForcingSolver()
    if name == "heun":
        return HeunFlowForcingSolver()
    raise ValueError(f"Unsupported Stage2 FF sampler solver: {name!r}")


def build_preferred_decode_plan(
    *,
    num_blocks: int,
    window_blocks: int = 4,
    clean_history_blocks: int = 1,
    device: torch.device | str = "cpu",
) -> Stage2DecodePlan:
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
    device = torch.device(device)
    tail_start = max(0, int(num_blocks) - int(window_blocks))
    decode_window_starts = torch.arange(tail_start + 1, device=device, dtype=torch.long)
    preferred_start = torch.zeros(num_blocks, device=device, dtype=torch.long)
    preferred_slot = torch.zeros(num_blocks, device=device, dtype=torch.long)
    for g in range(num_blocks):
        if num_blocks <= window_blocks or g == 0:
            start = 0
        else:
            start = min(g - int(clean_history_blocks), tail_start)
        preferred_start[g] = start
        preferred_slot[g] = g - start
    return Stage2DecodePlan(
        decode_window_starts=decode_window_starts,
        preferred_decode_window_start=preferred_start,
        preferred_local_slot=preferred_slot,
    )


def _build_flow_time_for_template(
    template: Stage2InferenceLatentTemplate,
    *,
    local_t_block: torch.Tensor,
    t_static: torch.Tensor,
    rollout_progress: torch.Tensor,
) -> Stage2FlowTimeField:
    layout = build_stage2_flow_anchor_layout(
        latent_mask=template.latent_mask,
        token_type_ids=template.token_type_ids,
        time_ids=template.time_ids,
        time_mask=template.time_mask,
        block_size_anchors=int(template.block_size_anchors),
    )
    return build_stage2_flow_time_field_from_blocks(
        layout=layout,
        t_block=local_t_block,
        block_mask=template.local_block_mask,
        t_static=t_static,
        rollout_progress=rollout_progress,
    )


def _dynamic_positions_by_local_block(template: Stage2InferenceLatentTemplate, *, window_blocks: int) -> list[torch.Tensor]:
    positions = []
    reference_types = None
    reference_entities = None
    for block in range(window_blocks):
        idx = (template.dynamic_token_mask[0] | (template.time_mask[0].to(dtype=torch.bool))) & (
            template.token_block_index[0] == block
        )
        pos = idx.nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            raise ValueError(f"Local block {block} has no dynamic template tokens.")
        types = template.token_type_ids[0, pos]
        entities = template.entity_ids[0, pos]
        if reference_types is None:
            reference_types = types
            reference_entities = entities
        elif not torch.equal(types, reference_types):
            raise ValueError("Dynamic token type pattern must match across local blocks.")
        elif not torch.equal(entities, reference_entities):
            flat_unknown_types = bool((types < 0).all().item() and (reference_types < 0).all().item())
            if not flat_unknown_types:
                raise ValueError("Dynamic token entity pattern must match across local blocks.")
        positions.append(pos)
    return positions


def _gather_local_window(
    *,
    z_blocks: torch.Tensor,
    z_static: torch.Tensor,
    template: Stage2InferenceLatentTemplate,
    dynamic_positions: list[torch.Tensor],
    static_positions: torch.Tensor,
    latent_dim: int,
) -> torch.Tensor:
    z_window = torch.zeros(template.latent_mask.shape[0], template.latent_mask.shape[1], latent_dim, device=z_blocks.device)
    local_to_global = template.local_to_global_block[0]
    for slot, positions in enumerate(dynamic_positions):
        if bool(template.local_block_mask[0, slot].item()):
            z_window[:, positions] = z_blocks[:, int(local_to_global[slot].item())]
    if static_positions.numel():
        z_window[:, static_positions] = z_static
    return zero_invalid_tokens(z_window, template.latent_mask)


def _scatter_local_window(
    *,
    z_next: torch.Tensor,
    z_blocks: torch.Tensor,
    z_static: torch.Tensor,
    local_to_global: torch.Tensor,
    should_update_global: torch.Tensor,
    update_static: bool,
    dynamic_positions: list[torch.Tensor],
    static_positions: torch.Tensor,
) -> None:
    for slot, positions in enumerate(dynamic_positions):
        g = int(local_to_global[slot].item())
        if 0 <= g < z_blocks.shape[1] and bool(should_update_global[g].item()):
            z_blocks[:, g] = z_next[:, positions]
    if update_static and static_positions.numel():
        z_static.copy_(z_next[:, static_positions])


def _assemble_decode_windows(
    *,
    z_blocks: torch.Tensor,
    z_static: torch.Tensor,
    template_spec: Stage2InferenceLatentTemplateSpec,
    requested_frames: int,
    dynamic_positions: list[torch.Tensor],
    static_positions: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = z_blocks.shape[1]
    window_blocks = _window_blocks(template_spec)
    tail_start = max(0, num_blocks - window_blocks)
    starts = torch.arange(tail_start + 1, device=device, dtype=torch.long)
    windows = []
    for start in starts.tolist():
        template = build_stage2_inference_latent_template(
            template_spec,
            batch_size=z_blocks.shape[0],
            requested_frames=requested_frames,
            local_to_global_block=torch.arange(start, start + window_blocks, device=device),
            device=device,
        )
        windows.append(
            _gather_local_window(
                z_blocks=z_blocks,
                z_static=z_static,
                template=template,
                dynamic_positions=dynamic_positions,
                static_positions=static_positions,
                latent_dim=z_blocks.shape[-1],
            )
        )
    return torch.stack(windows, dim=1), starts


def _local_update_mask(
    template: Stage2InferenceLatentTemplate,
    local_to_global: torch.Tensor,
    *,
    should_update_global: torch.Tensor,
    update_static: bool,
) -> torch.Tensor:
    update = torch.zeros_like(template.latent_mask, dtype=torch.bool)
    for slot, g_value in enumerate(local_to_global.tolist()):
        if 0 <= int(g_value) < should_update_global.numel() and bool(should_update_global[int(g_value)].item()):
            update |= template.dynamic_token_mask & (template.token_block_index == slot)
    if update_static:
        update |= template.static_geometry_token_mask
    return update & template.latent_mask


def _local_t_block(
    step: int,
    local_to_global: torch.Tensor,
    start_steps: torch.Tensor,
    end_steps: torch.Tensor,
    *,
    steps_per_block: int,
    batch_size: int,
) -> torch.Tensor:
    values = torch.zeros(batch_size, local_to_global.numel(), device=start_steps.device, dtype=torch.float32)
    for slot, g_value in enumerate(local_to_global.tolist()):
        if 0 <= int(g_value) < start_steps.numel():
            t = (float(step) - float(start_steps[int(g_value)].item())) / float(steps_per_block)
            values[:, slot] = min(1.0, max(0.0, t))
    return values


def _choose_window_start(
    committed_until: int,
    num_blocks: int,
    *,
    window_blocks: int,
    clean_history_blocks: int,
) -> tuple[int, bool]:
    if num_blocks <= window_blocks:
        return 0, True
    candidate = max(0, int(committed_until) - int(clean_history_blocks))
    tail_start = max(0, int(num_blocks) - int(window_blocks))
    if candidate >= tail_start:
        return tail_start, True
    return candidate, False


def _local_to_global(window_start: int, window_blocks: int, *, batch_size: int, device: torch.device) -> torch.Tensor:
    return (torch.arange(window_blocks, device=device, dtype=torch.long) + int(window_start)).unsqueeze(0).expand(batch_size, -1).clone()


def _rollout_progress(
    window_start: int,
    *,
    requested_frames: int,
    window_frames: int,
    block_size_frames: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    start_frame = int(window_start) * int(block_size_frames)
    end_frame = min(int(start_frame) + int(window_frames), int(requested_frames))
    center = 0.5 * (float(start_frame) + float(max(start_frame, end_frame - 1)))
    denom = float(max(int(requested_frames) - 1, 1))
    return torch.full((batch_size,), center / denom, device=device, dtype=torch.float32)


def _coerce_guidance(value: Stage2GuidanceConfig | dict[str, Any] | None) -> Stage2GuidanceConfig:
    if value is None:
        return Stage2GuidanceConfig()
    if isinstance(value, Stage2GuidanceConfig):
        return value
    return Stage2GuidanceConfig(**value)


def _coerce_solver(value: Stage2FlowForcingSolver | str) -> Stage2FlowForcingSolver:
    if isinstance(value, str):
        return build_stage2_flow_forcing_solver(value)
    return value


def _guidance_forward_count(guidance: Stage2GuidanceConfig) -> int:
    if not guidance.enabled or guidance.mode == "none":
        return 1
    if guidance.mode == "joint":
        return 2
    if guidance.mode == "separated":
        return 4
    raise ValueError(f"Unsupported Stage2 guidance mode: {guidance.mode!r}.")


def _solver_velocity_queries(solver: Stage2FlowForcingSolver) -> int:
    return 2 if solver.name == "heun" else 1


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _latent_dim_from_model(model: torch.nn.Module) -> int:
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "latent_dim"):
        return int(backbone.latent_dim)
    return 16


def _batch_size_from_batch(batch: dict) -> int:
    meta = batch.get("meta", {})
    text = meta.get("text") if isinstance(meta, dict) else None
    if isinstance(text, list):
        return len(text)
    geometry = batch.get("geometry", {})
    if isinstance(geometry, dict):
        for value in geometry.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return int(value.shape[0])
    raise ValueError("Unable to infer batch size from batch meta.text or geometry tensors.")


def _window_blocks(spec: Stage2InferenceLatentTemplateSpec) -> int:
    return _ceil_div(int(spec.anchors_per_window), int(spec.block_size_anchors))


def _block_size_frames(spec: Stage2InferenceLatentTemplateSpec) -> int:
    return int(spec.anchor_stride_frames) * int(spec.block_size_anchors)


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().pow(2).mean().sqrt()
