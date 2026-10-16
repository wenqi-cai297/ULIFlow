from __future__ import annotations

import numpy as np
import pytest
import torch

from hoi_latent.data.transforms.structured_normalization import Stage1StructuredNormalizer
from hoi_latent.losses.stage1 import Stage1LossOutput, Stage1Phase1WarmupLoss
from hoi_latent.models.stage1 import (
    Stage1CoarsePrediction,
    Stage1LatentFieldOutput,
    build_stage1_autoencoder_from_config,
)
from test_stage1_autoencoder import _config as _autoencoder_config
from test_stage1_tokenizers import make_synthetic_stage1_structured_batch


def test_stage1_phase1_warmup_loss_forward_is_finite_and_reports_components():
    frame_mask = torch.ones(2, 4, dtype=torch.bool)
    batch = _make_zero_batch(frame_mask)
    coarse = _make_zero_coarse(frame_mask)
    latent = _make_latent(batch_size=2)
    loss_fn = Stage1Phase1WarmupLoss(normalizer=_identity_normalizer())

    out = loss_fn(coarse, latent, batch, global_step=40000)

    assert isinstance(out, Stage1LossOutput)
    assert torch.isfinite(out.total)
    for key in [
        "total",
        "human",
        "object",
        "relation",
        "world_marker",
        "world_marker_scaled_loss",
        "latent_cov",
        "human.pos_rootrel",
        "object.rot6d",
        "relation.delta_m2o",
        "relation.dist",
    ]:
        assert key in out.components
        assert torch.isfinite(out.components[key])
    for key in [
        "valid_frame_count",
        "valid_velocity_frame_count",
        "valid_acceleration_frame_count",
        "lambda_cov_floor",
        "latent_z_effective_rank",
        "world_marker_mpjpe_m",
    ]:
        assert key in out.diagnostics
    assert 0.0 <= out.diagnostics["latent_z_effective_rank"].item() <= latent.z_rec.shape[-1]


def test_stage1_phase1_warmup_loss_all_invalid_is_zero_and_finite():
    frame_mask = torch.zeros(2, 4, dtype=torch.bool)
    batch = _make_zero_batch(frame_mask)
    coarse = _make_zero_coarse(frame_mask, fill=10.0)
    latent = _make_latent(batch_size=2, all_invalid=True)
    loss_fn = Stage1Phase1WarmupLoss(normalizer=_identity_normalizer())

    out = loss_fn(coarse, latent, batch, global_step=40000)

    assert torch.isfinite(out.total)
    assert out.total.item() == 0.0
    assert out.diagnostics["valid_frame_count"].item() == 0


def test_stage1_phase1_warmup_loss_uses_derivative_masks_for_boundaries():
    frame_mask = torch.ones(1, 4, dtype=torch.bool)
    batch = _make_zero_batch(frame_mask)
    coarse = _make_zero_coarse(frame_mask)
    latent = _make_latent(batch_size=1)
    coarse.human["vel_rootrel"][:, -1] = 100.0
    coarse.human["acc_rootrel"][:, -2:] = 100.0
    loss_fn = Stage1Phase1WarmupLoss(
        lambda_object=0.0,
        lambda_relation=0.0,
        lambda_world_marker=0.0,
        lambda_latent_cov_floor=0.0,
        human_field_weights={
            "pos_rootrel": 0.0,
            "root": 0.0,
            "vel_rootrel": 1.0,
            "root_vel": 0.0,
            "acc_rootrel": 1.0,
            "root_acc": 0.0,
            "left_heights": 0.0,
            "right_heights": 0.0,
        },
    )

    out = loss_fn(coarse, latent, batch, global_step=0)

    assert out.components["human.vel_rootrel"].item() == 0.0
    assert out.components["human.acc_rootrel"].item() == 0.0
    assert out.total.item() == 0.0


def test_stage1_phase1_world_marker_loss_backpropagates_to_pos_and_root():
    frame_mask = torch.ones(1, 2, dtype=torch.bool)
    batch = _make_zero_batch(frame_mask)
    coarse = _make_zero_coarse(frame_mask, requires_grad=True)
    latent = _make_latent(batch_size=1)
    with torch.no_grad():
        coarse.human["pos_rootrel"][..., 0] = 0.05
        coarse.human["root"][..., 0] = 0.05
    loss_fn = _world_only_loss()

    out = loss_fn(coarse, latent, batch, global_step=0)
    out.total.backward()

    assert coarse.human["pos_rootrel"].grad is not None
    assert coarse.human["root"].grad is not None
    assert coarse.human["pos_rootrel"].grad.abs().sum().item() > 0.0
    assert coarse.human["root"].grad.abs().sum().item() > 0.0


def test_stage1_phase1_world_marker_mask_ignores_invalid_markers():
    frame_mask = torch.ones(1, 2, dtype=torch.bool)
    batch = _make_zero_batch(frame_mask)
    batch["masks"]["human"][:, :, 0] = False
    coarse = _make_zero_coarse(frame_mask)
    coarse.human["pos_rootrel"][:, :, 0, 0] = 10.0
    latent = _make_latent(batch_size=1)
    loss_fn = _world_only_loss()

    out = loss_fn(coarse, latent, batch, global_step=0)

    assert out.components["world_marker_scaled_loss"].item() == 0.0
    assert out.total.item() == 0.0


def test_stage1_phase1_world_marker_uses_scaled_units_not_meter_tiny_huber():
    frame_mask = torch.ones(1, 2, dtype=torch.bool)
    batch = _make_zero_batch(frame_mask)
    coarse = _make_zero_coarse(frame_mask)
    coarse.human["pos_rootrel"][..., 0] = 0.05
    latent = _make_latent(batch_size=1)
    loss_fn = _world_only_loss()

    out = loss_fn(coarse, latent, batch, global_step=0)

    assert out.components["world_marker_scaled_loss"].item() > 0.1
    assert out.diagnostics["world_marker_mpjpe_m"].item() == pytest.approx(0.05, rel=1.0e-5)


def test_stage1_phase1_world_marker_requires_normalizer_stats():
    frame_mask = torch.ones(1, 2, dtype=torch.bool)
    batch = _make_zero_batch(frame_mask)
    coarse = _make_zero_coarse(frame_mask)
    latent = _make_latent(batch_size=1)
    loss_fn = Stage1Phase1WarmupLoss()

    with pytest.raises(ValueError, match="requires structured normalizer stats"):
        loss_fn(coarse, latent, batch, global_step=0)


def test_stage1_phase1_latent_cov_floor_schedule():
    loss_fn = Stage1Phase1WarmupLoss(lambda_world_marker=0.0)

    assert loss_fn.latent_cov_floor_weight(0) == 0.0
    assert loss_fn.latent_cov_floor_weight(9999) == 0.0
    assert loss_fn.latent_cov_floor_weight(20000) == pytest.approx(5.0e-5)
    assert loss_fn.latent_cov_floor_weight(30000) == pytest.approx(1.0e-4)
    assert loss_fn.latent_cov_floor_weight(40000) == pytest.approx(1.0e-4)


def test_stage1_phase1_warmup_path_trains_decoder_without_refiner():
    batch = make_synthetic_stage1_structured_batch(batch_size=1, seq_len=5)
    batch["raw"]["human"] = {
        "pos": batch["model"]["human"]["pos_rootrel"] + batch["model"]["human"]["root"][:, :, None, :]
    }
    module = build_stage1_autoencoder_from_config(_autoencoder_config(), batch=batch)
    loss_fn = Stage1Phase1WarmupLoss(normalizer=_identity_normalizer())

    tokenizer_outputs = module.tokenizer(batch, validate=True)
    latent = module.encoder(tokenizer_outputs, batch=batch, validate=True)
    coarse = module.decoder(latent, target_frame_mask=batch["masks"]["valid"], validate=True)
    out = loss_fn(coarse, latent, batch, global_step=0)
    out.total.backward()

    assert any(_has_grad(param) for param in module.decoder.parameters())
    assert not hasattr(module, "refiner")


def test_stage1_phase1_derivative_mask_matches_data_convention():
    # The loss reimplements the data's numpy derivative_mask in torch (batched).
    # A convention drift (forward vs backward diff, or interior-gap handling) would
    # silently mis-mask vel/acc supervision relative to how GT vel/acc are built
    # (forward diff, interact_stage1_structured.diff_pad_last_valid). The existing
    # boundary test only checks the all-valid case; lock the full convention here.
    from hoi_latent.data.representations.interact_stage1_structured import derivative_mask
    from hoi_latent.losses.stage1.phase1_warmup import _derivative_mask

    patterns = [
        [True, True, True, True],
        [True, True, True, True, True, True],
        [False, True, True, True],       # leading invalid
        [True, True, True, False],       # trailing invalid
        [True, False, True, True],       # interior gap
        [True, False, False, True],
        [True, True],                    # short
        [True],                          # T <= order
    ]
    for order in (1, 2):
        for pat in patterns:
            expected = torch.from_numpy(derivative_mask(np.array(pat, dtype=bool), order))
            got = _derivative_mask(torch.tensor(pat, dtype=torch.bool)[None, :], order)[0]
            assert torch.equal(got, expected), (order, pat, got.tolist(), expected.tolist())


def test_stage1_phase1_world_marker_fallback_target_without_raw():
    # No batch["raw"] -> the world-marker target must fall back to denorm(model)
    # recovery (source_id=0). This is the likely production path and is otherwise
    # never exercised (the helper batch always provides raw.human.pos).
    frame_mask = torch.ones(1, 2, dtype=torch.bool)
    batch = _make_zero_batch(frame_mask)
    del batch["raw"]
    coarse = _make_zero_coarse(frame_mask)
    coarse.human["pos_rootrel"][..., 0] = 0.05
    latent = _make_latent(batch_size=1)
    loss_fn = _world_only_loss()

    out = loss_fn(coarse, latent, batch, global_step=0)

    assert out.diagnostics["world_marker_target_source"].item() == 0
    assert out.diagnostics["world_marker_mpjpe_m"].item() == pytest.approx(0.05, rel=1.0e-5)
    assert out.components["world_marker_scaled_loss"].item() > 0.1


def _world_only_loss() -> Stage1Phase1WarmupLoss:
    return Stage1Phase1WarmupLoss(
        normalizer=_identity_normalizer(),
        lambda_human=0.0,
        lambda_object=0.0,
        lambda_relation=0.0,
        lambda_world_marker=1.0,
        lambda_latent_cov_floor=0.0,
    )


def _identity_normalizer() -> Stage1StructuredNormalizer:
    return Stage1StructuredNormalizer(
        {
            "human": {
                "pos_rootrel": {
                    "mean": torch.zeros(77, 3).numpy(),
                    "std": torch.ones(77, 3).numpy(),
                    "count": 1,
                },
                "root": {"mean": torch.zeros(3).numpy(), "std": torch.ones(3).numpy(), "count": 1},
            }
        }
    )


def _make_zero_batch(frame_mask: torch.Tensor) -> dict:
    batch_size, num_frames = frame_mask.shape
    human_mask = frame_mask[:, :, None].expand(batch_size, num_frames, 77).clone()
    relation_mask = frame_mask[:, :, None].expand(batch_size, num_frames, 78).clone()
    human = {
        "pos_rootrel": torch.zeros(batch_size, num_frames, 77, 3),
        "vel_rootrel": torch.zeros(batch_size, num_frames, 77, 3),
        "acc_rootrel": torch.zeros(batch_size, num_frames, 77, 3),
        "root": torch.zeros(batch_size, num_frames, 3),
        "root_vel": torch.zeros(batch_size, num_frames, 3),
        "root_acc": torch.zeros(batch_size, num_frames, 3),
        "left_heights": torch.zeros(batch_size, num_frames, 7),
        "right_heights": torch.zeros(batch_size, num_frames, 7),
    }
    obj = {
        "rot6d": torch.zeros(batch_size, num_frames, 6),
        "trans": torch.zeros(batch_size, num_frames, 3),
        "trans_vel": torch.zeros(batch_size, num_frames, 3),
        "trans_acc": torch.zeros(batch_size, num_frames, 3),
    }
    relation = {
        "delta_m2o": torch.zeros(batch_size, num_frames, 78, 3),
        "dist": torch.zeros(batch_size, num_frames, 78, 1),
        "delta_vel": torch.zeros(batch_size, num_frames, 78, 3),
        "delta_acc": torch.zeros(batch_size, num_frames, 78, 3),
        "point_canon": torch.zeros(batch_size, num_frames, 78, 3),
        "root_object_delta": torch.zeros(batch_size, num_frames, 3),
        "root_object_dist": torch.zeros(batch_size, num_frames, 1),
        "root_object_delta_vel": torch.zeros(batch_size, num_frames, 3),
        "root_object_delta_acc": torch.zeros(batch_size, num_frames, 3),
    }
    return {
        "model": {"human": human, "object": obj, "relation": relation},
        "raw": {"human": {"pos": torch.zeros(batch_size, num_frames, 77, 3)}},
        "masks": {
            "valid": frame_mask.clone(),
            "human": human_mask,
            "object": frame_mask.clone(),
            "relation": relation_mask,
        },
    }


def _make_zero_coarse(
    frame_mask: torch.Tensor,
    *,
    fill: float = 0.0,
    requires_grad: bool = False,
) -> Stage1CoarsePrediction:
    batch_size, num_frames = frame_mask.shape
    human_mask = frame_mask[:, :, None].expand(batch_size, num_frames, 77).clone()
    relation_mask = frame_mask[:, :, None].expand(batch_size, num_frames, 78).clone()
    human = {
        "pos_rootrel": _tensor((batch_size, num_frames, 77, 3), fill, requires_grad),
        "vel_rootrel": _tensor((batch_size, num_frames, 77, 3), fill, requires_grad),
        "acc_rootrel": _tensor((batch_size, num_frames, 77, 3), fill, requires_grad),
        "root": _tensor((batch_size, num_frames, 3), fill, requires_grad),
        "root_vel": _tensor((batch_size, num_frames, 3), fill, requires_grad),
        "root_acc": _tensor((batch_size, num_frames, 3), fill, requires_grad),
        "left_heights": _tensor((batch_size, num_frames, 7), fill, requires_grad),
        "right_heights": _tensor((batch_size, num_frames, 7), fill, requires_grad),
    }
    obj = {
        "rot6d": _tensor((batch_size, num_frames, 6), fill, requires_grad),
        "trans": _tensor((batch_size, num_frames, 3), fill, requires_grad),
        "trans_vel": _tensor((batch_size, num_frames, 3), fill, requires_grad),
        "trans_acc": _tensor((batch_size, num_frames, 3), fill, requires_grad),
    }
    relation = {
        "delta_m2o": _tensor((batch_size, num_frames, 78, 3), fill, requires_grad),
        "dist": _tensor((batch_size, num_frames, 78, 1), fill, requires_grad),
        "delta_vel": _tensor((batch_size, num_frames, 78, 3), fill, requires_grad),
        "delta_acc": _tensor((batch_size, num_frames, 78, 3), fill, requires_grad),
        "point_canon": _tensor((batch_size, num_frames, 78, 3), fill, requires_grad),
        "root_object_delta": _tensor((batch_size, num_frames, 3), fill, requires_grad),
        "root_object_dist": _tensor((batch_size, num_frames, 1), fill, requires_grad),
        "root_object_delta_vel": _tensor((batch_size, num_frames, 3), fill, requires_grad),
        "root_object_delta_acc": _tensor((batch_size, num_frames, 3), fill, requires_grad),
    }
    return Stage1CoarsePrediction(
        human=human,
        object=obj,
        relation=relation,
        masks={
            "frame": frame_mask,
            "human": human_mask,
            "object": frame_mask,
            "relation": relation_mask,
            "velocity": frame_mask,
            "acceleration": frame_mask,
        },
        aux={},
    )


def _make_latent(batch_size: int, *, all_invalid: bool = False) -> Stage1LatentFieldOutput:
    z = torch.randn(batch_size, 6, 4, requires_grad=True)
    mask = torch.zeros(batch_size, 6, dtype=torch.bool) if all_invalid else torch.ones(batch_size, 6, dtype=torch.bool)
    z = z * mask.to(dtype=z.dtype).unsqueeze(-1)
    ids = torch.zeros(batch_size, 6, dtype=torch.long)
    return Stage1LatentFieldOutput(
        z_rec=z,
        z_raw=z.detach().clone(),
        mask=mask,
        modality_ids=ids,
        token_type_ids=ids,
        time_ids=ids,
        entity_ids=ids,
        time_mask=mask.clone(),
        aux={},
    )


def _tensor(shape: tuple[int, ...], fill: float, requires_grad: bool) -> torch.Tensor:
    value = torch.full(shape, fill, dtype=torch.float32)
    if requires_grad:
        value.requires_grad_()
    return value


def _has_grad(param: torch.nn.Parameter) -> bool:
    return param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum().item() > 0.0
