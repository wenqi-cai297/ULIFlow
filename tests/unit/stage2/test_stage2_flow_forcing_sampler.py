from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hoi_latent.inference.stage2 import (
    EulerFlowForcingSolver,
    HeunFlowForcingSolver,
    Stage2SamplerStepContext,
    Stage2FlowForcingRolloutConfig,
    Stage2GuidanceConfig,
    advance_flow_time_for_updated_tokens,
    build_preferred_decode_plan,
    build_stage2_flow_forcing_solver,
    build_stage2_inference_latent_template,
    extract_stage2_inference_latent_template_spec,
    guided_velocity,
    sample_stage2_flow_forcing_rollout,
    validate_stage2_template_content_independence,
)
from hoi_latent.models.stage1.autoencoder.latent_bottleneck import Stage1LatentFieldOutput
from hoi_latent.models.stage1.tokenizer_types import ModalityId, TokenTypeId
from hoi_latent.models.stage2.flow_time import build_stage2_flow_anchor_layout, build_stage2_flow_time_field_from_blocks


def test_template_spec_records_provenance_hash_and_no_latent_values() -> None:
    latent_a = _latent()
    latent_b = _latent(mask_last=False, z_offset=10.0)

    diagnostics = validate_stage2_template_content_independence([latent_a, latent_b])
    spec = extract_stage2_inference_latent_template_spec(
        latent_a,
        window_frames=16,
        anchor_stride_frames=4,
        block_size_anchors=2,
        source_stage1_checkpoint_sha256="abc123",
        stage1_anchor_override={"mode": "absolute_grid", "stride": 4, "grid_origin": 0},
    )
    changed = extract_stage2_inference_latent_template_spec(
        latent_a,
        window_frames=16,
        anchor_stride_frames=4,
        block_size_anchors=2,
        source_stage1_checkpoint_sha256="abc123",
        stage1_anchor_override={"mode": "absolute_grid", "stride": 8, "grid_origin": 0},
    )

    assert diagnostics["template_content_examples"].item() == 2
    assert diagnostics["template_latent_mask_all_equal"].item() is False
    assert spec.source_stage1_checkpoint_sha256 == "abc123"
    assert spec.stage1_anchor_override["mode"] == "absolute_grid"
    assert "z_rec" not in spec.metadata_1d
    assert set(spec.metadata_1d) == {"latent_mask", "modality_ids", "token_type_ids", "time_ids", "entity_ids", "time_mask"}
    assert spec.spec_hash != changed.spec_hash


def test_template_content_independence_rejects_content_dependent_metadata() -> None:
    latent_a = _latent()
    latent_b = _latent()
    latent_b.token_type_ids[0, 0] = int(TokenTypeId.RELATION_BODY_SLOT)

    with pytest.raises(ValueError, match="token_type_ids"):
        validate_stage2_template_content_independence([latent_a, latent_b])


def test_template_builder_masks_partial_frames_and_keeps_static_geometry() -> None:
    spec = _spec()
    template = build_stage2_inference_latent_template(
        spec,
        batch_size=1,
        requested_frames=10,
        local_to_global_block=torch.tensor([0, 1]),
        device="cpu",
    )

    assert template.latent_mask.tolist() == [[True, True, True, True, False]]
    assert template.time_mask.tolist() == [[True, True, False, True, False]]
    assert template.static_geometry_token_mask.tolist() == [[False, False, True, False, False]]
    assert template.local_block_mask.tolist() == [[True, True]]
    assert template.local_anchor_time_ids.tolist() == [[0, 4, 8, 12]]


def test_template_builder_rebases_dynamic_time_ids_to_local_window() -> None:
    latent = _latent()
    latent.time_ids = latent.time_ids + 80
    spec = extract_stage2_inference_latent_template_spec(
        latent,
        window_frames=16,
        anchor_stride_frames=4,
        block_size_anchors=2,
        source_stage1_checkpoint_sha256="abc123",
        stage1_anchor_override={"mode": "absolute_grid", "stride": 4, "grid_origin": 0},
    )
    template = build_stage2_inference_latent_template(
        spec,
        batch_size=1,
        requested_frames=16,
        local_to_global_block=torch.tensor([0, 1]),
        device="cpu",
    )

    assert template.time_ids.tolist() == [[0, 4, 80, 8, 12]]


def test_rollout_updates_each_block_exactly_and_freezes_static_after_bootstrap() -> None:
    model = ConstantVelocityModel(value=0.0)
    out = sample_stage2_flow_forcing_rollout(
        model,
        batch=_batch(batch_size=2),
        template_spec=_spec(),
        requested_frames=24,
        guidance=Stage2GuidanceConfig(),
        solver=EulerFlowForcingSolver(),
        seed=3,
        config=Stage2FlowForcingRolloutConfig(steps_per_block=2, window_blocks=2, clean_history_blocks=1),
    )

    assert out.z_blocks.shape[:3] == (2, 3, 2)
    assert out.update_counts.tolist() == [[2, 2, 2], [2, 2, 2]]
    assert out.static_update_count.tolist() == [2, 2]
    assert out.decode_z_white_windows.shape[:3] == (2, 2, 5)
    assert out.preferred_decode_window_start.tolist() == [0, 0, 1]
    assert out.preferred_local_slot.tolist() == [0, 1, 1]
    assert out.diagnostics["committed_blocks"].item() == 3


def test_rollout_accepts_flat_tokens_with_absolute_entity_ids() -> None:
    model = ConstantVelocityModel(value=0.0)
    spec = extract_stage2_inference_latent_template_spec(
        _flat_latent(),
        window_frames=16,
        anchor_stride_frames=4,
        block_size_anchors=2,
        source_stage1_checkpoint_sha256="abc123",
        stage1_anchor_override={"mode": "source"},
    )

    out = sample_stage2_flow_forcing_rollout(
        model,
        batch=_batch(),
        template_spec=spec,
        requested_frames=16,
        guidance=Stage2GuidanceConfig(),
        solver=EulerFlowForcingSolver(),
        seed=3,
        config=Stage2FlowForcingRolloutConfig(steps_per_block=1, window_blocks=2, clean_history_blocks=1),
    )

    assert out.z_blocks.shape == (1, 2, 2, 16)
    assert out.update_counts.tolist() == [[1, 1]]


def test_rollout_step_callback_receives_pre_solver_state() -> None:
    model = ConstantVelocityModel(value=0.0)
    seen: list[tuple[Stage2SamplerStepContext, torch.Tensor, torch.Tensor]] = []

    def callback(context, state) -> None:
        seen.append(
            (
                context,
                state.update_token_mask.detach().clone(),
                state.flow_time.t_block.detach().clone(),
            )
        )

    sample_stage2_flow_forcing_rollout(
        model,
        batch=_batch(batch_size=1),
        template_spec=_spec(),
        requested_frames=24,
        guidance=Stage2GuidanceConfig(),
        solver=EulerFlowForcingSolver(),
        seed=3,
        config=Stage2FlowForcingRolloutConfig(steps_per_block=2, window_blocks=2, clean_history_blocks=1),
        step_callback=callback,
    )

    assert len(seen) == 6
    first_context, first_update, first_t_block = seen[0]
    assert first_context.step == 0
    assert first_context.window_start == 0
    assert first_context.local_to_global_block.tolist() == [[0, 1]]
    assert first_context.should_update_global.tolist() == [True, False, False]
    assert first_update.any()
    assert first_t_block.tolist() == [[0.0, 0.0]]


def test_rollout_tail_pinned_default_window_updates_all_blocks_exactly() -> None:
    model = ConstantVelocityModel(value=0.0)
    out = sample_stage2_flow_forcing_rollout(
        model,
        batch=_batch(batch_size=1),
        template_spec=_spec(block_size_anchors=1),
        requested_frames=24,
        guidance=Stage2GuidanceConfig(),
        solver=EulerFlowForcingSolver(),
        seed=7,
        config=Stage2FlowForcingRolloutConfig(steps_per_block=5, window_blocks=4, clean_history_blocks=1),
    )

    assert out.update_counts.tolist() == [[5, 5, 5, 5, 5, 5]]
    assert out.static_update_count.tolist() == [5]
    assert out.decode_window_starts.tolist() == [0, 1, 2]
    assert out.preferred_decode_window_start.tolist() == [0, 0, 1, 2, 2, 2]
    assert out.preferred_local_slot.tolist() == [0, 1, 1, 1, 2, 3]


def test_rollout_feeds_wavefront_staircase_and_static_bootstrap_clock() -> None:
    model = CapturingVelocityModel()
    sample_stage2_flow_forcing_rollout(
        model,
        batch=_batch(batch_size=1),
        template_spec=_spec(block_size_anchors=1),
        requested_frames=16,
        guidance=Stage2GuidanceConfig(),
        solver=EulerFlowForcingSolver(),
        seed=1,
        config=Stage2FlowForcingRolloutConfig(steps_per_block=3, window_blocks=4, clean_history_blocks=1),
    )

    # block_stride_steps = ceil(3/3) = 1; start_steps = [0,1,2,3]; one fixed window [0,1,2,3].
    assert len(model.t_blocks) == 6
    third = pytest.approx(1.0 / 3.0)
    two_third = pytest.approx(2.0 / 3.0)
    assert model.t_blocks[2] == [two_third, third, 0.0, 0.0]
    assert model.t_blocks[3] == [pytest.approx(1.0), two_third, third, 0.0]
    assert model.t_blocks[4] == [pytest.approx(1.0), pytest.approx(1.0), two_third, third]
    # t_static tracks block-0 time while block 0 is still denoising, then freezes at 1.0.
    assert model.t_statics[0] == pytest.approx(0.0)
    assert model.t_statics[1] == third
    assert model.t_statics[2] == two_third
    assert model.t_statics[3:] == [pytest.approx(1.0)] * 3


def test_euler_and_heun_solver_updates_and_query_counts() -> None:
    template = _template()
    flow_time = _flow_time(template, t0=0.25)
    update = template.dynamic_token_mask.clone()
    dt = update.to(torch.float32).unsqueeze(-1) * 0.5
    z = torch.zeros(1, 5, 16)

    euler_model = ConstantVelocityModel(value=2.0)
    euler = EulerFlowForcingSolver().step(
        euler_model,
        _state(z, flow_time, template, update, dt),
        guidance=Stage2GuidanceConfig(),
    )
    assert torch.allclose(euler.z_next[update], torch.full_like(euler.z_next[update], 1.0))
    assert euler_model.calls == 1
    assert euler.diagnostics["solver_num_velocity_queries"] == 1

    heun_model = SequenceVelocityModel([2.0, 4.0])
    heun = HeunFlowForcingSolver().step(
        heun_model,
        _state(z, flow_time, template, update, dt),
        guidance=Stage2GuidanceConfig(),
    )
    assert torch.allclose(heun.z_next[update], torch.full_like(heun.z_next[update], 1.5))
    assert torch.allclose(heun.velocity[update], torch.full_like(heun.velocity[update], 3.0))
    assert heun_model.calls == 2
    assert heun.diagnostics["solver_num_velocity_queries"] == 2


def test_advance_flow_time_advances_only_updated_tokens_and_respects_static_freeze() -> None:
    template = _template()
    field = _flow_time(template, t0=0.25, t_static=0.4)
    update = torch.zeros_like(template.latent_mask)
    update[:, 0] = True
    update[:, 2] = True
    dt = torch.zeros_like(field.t_latent_token)
    dt[update] = 0.25

    advanced = advance_flow_time_for_updated_tokens(field, update_token_mask=update, dt_token=dt)
    assert advanced.t_block[0, 0].item() == pytest.approx(0.5)
    assert advanced.t_block[0, 1].item() == pytest.approx(0.25)
    assert advanced.t_static[0].item() == pytest.approx(0.65)

    frozen_update = update.clone()
    frozen_update[:, 2] = False
    frozen = advance_flow_time_for_updated_tokens(field, update_token_mask=frozen_update, dt_token=dt)
    assert frozen.t_static[0].item() == pytest.approx(0.4)


def test_cfg_modes_forward_counts_and_identity_points() -> None:
    template = _template()
    flow_time = _flow_time(template)
    z = torch.zeros(1, 5, 16)

    model = GuidanceVelocityModel()
    none, none_diag = guided_velocity(
        model,
        batch=_batch(),
        z_white_t=z,
        flow_time=flow_time,
        template=template,
        guidance=Stage2GuidanceConfig(enabled=False, mode="none"),
    )
    assert torch.allclose(none, torch.full_like(none, 10.0))
    assert none_diag["guidance_num_forwards_per_step"].item() == 1

    model = GuidanceVelocityModel()
    joint, joint_diag = guided_velocity(
        model,
        batch=_batch(),
        z_white_t=z,
        flow_time=flow_time,
        template=template,
        guidance=Stage2GuidanceConfig(enabled=True, mode="joint", joint_scale=1.0),
    )
    assert torch.allclose(joint, torch.full_like(joint, 10.0))
    assert model.calls == 2
    assert joint_diag["guidance_num_forwards_per_step"].item() == 2

    model = GuidanceVelocityModel()
    separated, sep_diag = guided_velocity(
        model,
        batch=_batch(),
        z_white_t=z,
        flow_time=flow_time,
        template=template,
        guidance=Stage2GuidanceConfig(enabled=True, mode="separated", text_scale=1.0, geometry_scale=1.0),
    )
    assert torch.allclose(separated, torch.full_like(separated, 10.0))
    assert model.calls == 4
    assert sep_diag["guidance_num_forwards_per_step"].item() == 4


def test_solver_builder_and_preferred_decode_plan() -> None:
    assert isinstance(build_stage2_flow_forcing_solver({"inference": {"stage2": {"sampler": {"solver": {"name": "euler"}}}}}), EulerFlowForcingSolver)
    assert isinstance(build_stage2_flow_forcing_solver("heun"), HeunFlowForcingSolver)
    with pytest.raises(ValueError, match="Unsupported"):
        build_stage2_flow_forcing_solver("bad")

    plan = build_preferred_decode_plan(num_blocks=6, window_blocks=4, clean_history_blocks=1)
    assert plan.decode_window_starts.tolist() == [0, 1, 2]
    assert plan.preferred_decode_window_start.tolist() == [0, 0, 1, 2, 2, 2]
    assert plan.preferred_local_slot.tolist() == [0, 1, 1, 1, 2, 3]


def _spec(*, block_size_anchors: int = 2):
    return extract_stage2_inference_latent_template_spec(
        _latent(),
        window_frames=16,
        anchor_stride_frames=4,
        block_size_anchors=block_size_anchors,
        source_stage1_checkpoint_sha256="abc123",
        stage1_anchor_override={"mode": "absolute_grid", "stride": 4, "grid_origin": 0},
    )


def _template():
    return build_stage2_inference_latent_template(
        _spec(),
        batch_size=1,
        requested_frames=16,
        local_to_global_block=torch.tensor([0, 1]),
        device="cpu",
    )


def _flow_time(template, *, t0: float = 0.25, t_static: float = 0.5):
    layout = build_stage2_flow_anchor_layout(
        latent_mask=template.latent_mask,
        token_type_ids=template.token_type_ids,
        time_ids=template.time_ids,
        time_mask=template.time_mask,
        block_size_anchors=2,
    )
    return build_stage2_flow_time_field_from_blocks(
        layout=layout,
        t_block=torch.tensor([[t0, t0]]),
        block_mask=torch.tensor([[True, True]]),
        t_static=torch.tensor([t_static]),
        rollout_progress=torch.tensor([0.5]),
    )


def _state(z, flow_time, template, update, dt):
    return SimpleNamespace(
        z_window=z,
        flow_time=flow_time,
        template=template,
        update_token_mask=update,
        dt_token=dt,
        batch=_batch(),
    )


def _latent(*, mask_last: bool = True, z_offset: float = 0.0) -> Stage1LatentFieldOutput:
    z = torch.arange(5 * 16, dtype=torch.float32).reshape(1, 5, 16) + float(z_offset)
    mask = torch.tensor([[True, True, True, True, mask_last]])
    modality_ids = torch.tensor(
        [[int(ModalityId.HUMAN), int(ModalityId.HUMAN), int(ModalityId.OBJECT), int(ModalityId.HUMAN), int(ModalityId.HUMAN)]]
    )
    token_type_ids = torch.tensor(
        [
            [
                int(TokenTypeId.HUMAN_ROOT),
                int(TokenTypeId.HUMAN_ROOT),
                int(TokenTypeId.OBJECT_GEOMETRY),
                int(TokenTypeId.HUMAN_ROOT),
                int(TokenTypeId.HUMAN_ROOT),
            ]
        ]
    )
    time_ids = torch.tensor([[0, 4, 0, 8, 12]])
    entity_ids = torch.tensor([[0, 0, 0, 0, 0]])
    time_mask = torch.tensor([[True, True, False, True, True]])
    return Stage1LatentFieldOutput(
        z_rec=z,
        z_raw=z.clone(),
        mask=mask,
        modality_ids=modality_ids,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        entity_ids=entity_ids,
        time_mask=time_mask,
    )


def _flat_latent() -> Stage1LatentFieldOutput:
    return Stage1LatentFieldOutput(
        z_rec=torch.zeros(1, 4, 16),
        z_raw=torch.zeros(1, 4, 16),
        mask=torch.ones(1, 4, dtype=torch.bool),
        modality_ids=torch.full((1, 4), -1, dtype=torch.long),
        token_type_ids=torch.full((1, 4), -1, dtype=torch.long),
        time_ids=torch.tensor([[0, 4, 8, 12]]),
        entity_ids=torch.tensor([[0, 1, 2, 3]]),
        time_mask=torch.ones(1, 4, dtype=torch.bool),
    )


def _batch(batch_size: int = 1) -> dict:
    return {
        "meta": {"text": ["open drawer"] * batch_size},
        "geometry": {"object_bps": torch.zeros(batch_size, 4, 3), "has_object_bps": torch.ones(batch_size, dtype=torch.bool)},
    }


class ConstantVelocityModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.backbone = SimpleNamespace(latent_dim=16)
        self.calls = 0
        self.value = float(value)

    def forward(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(velocity=torch.full_like(kwargs["z_white_t"], self.value))


class SequenceVelocityModel(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.backbone = SimpleNamespace(latent_dim=16)
        self.values = list(values)
        self.calls = 0

    def forward(self, **kwargs):
        value = self.values[self.calls]
        self.calls += 1
        return SimpleNamespace(velocity=torch.full_like(kwargs["z_white_t"], float(value)))


class CapturingVelocityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SimpleNamespace(latent_dim=16)
        self.calls = 0
        self.t_blocks: list[list[float]] = []
        self.t_statics: list[float] = []

    def forward(self, **kwargs):
        self.calls += 1
        flow_time = kwargs["flow_time"]
        self.t_blocks.append([float(v) for v in flow_time.t_block[0].tolist()])
        self.t_statics.append(float(flow_time.t_static[0].item()))
        return SimpleNamespace(velocity=torch.zeros_like(kwargs["z_white_t"]))


class GuidanceVelocityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SimpleNamespace(latent_dim=16)
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        text_null = bool(kwargs["force_null_text_mask"][0].item())
        geometry_null = bool(kwargs["force_null_geometry_mask"][0].item())
        if not text_null and not geometry_null:
            value = 10.0
        elif not text_null and geometry_null:
            value = 7.0
        elif text_null and not geometry_null:
            value = 5.0
        else:
            value = 2.0
        return SimpleNamespace(velocity=torch.full_like(kwargs["z_white_t"], value))
