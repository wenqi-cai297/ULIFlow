from __future__ import annotations

import hashlib

import torch
import pytest

from hoi_latent.config.stage1_anchor_overrides import apply_stage1_anchor_override
from hoi_latent.losses.stage2 import Stage2RectifiedFlowLoss, masked_velocity_mse
from hoi_latent.models.stage1.tokenizer_types import TokenTypeId
from hoi_latent.models.stage2 import (
    build_stage2_flow_anchor_layout,
    build_stage2_flow_time_field_from_blocks,
)
from hoi_latent.models.stage2.stem_types import Stage2LatentSubstreamId
from hoi_latent.training.stage2 import TrainableParameterEMA, sample_categorical_condition_dropout
from hoi_latent.training.stage2.rectified_flow import (
    Stage2RectifiedFlowTrainer,
    Stage2SourceArtifacts,
    _load_stage2_state_without_frozen_clip,
    _stage2_rollout_progress,
    _stage2_state_dict_without_frozen_clip,
    _validate_source_whitening_provenance,
    _validate_resume_and_init_stage2_weights_are_exclusive,
    _validate_whitening_anchor_override,
    _velocity_cosine,
    apply_stage2_rectified_flow_cli_overrides,
    build_stage2_optimizer_from_config,
    build_stage2_scheduler_from_config,
    init_stage2_weights_from_checkpoint,
    load_stage2_rectified_flow_training_config,
    validate_stage2_first_batch,
)
from hoi_latent.training.stage2.flow_forcing_times import (
    build_random_all_blocks_flow_time_from_config,
    build_stage2_training_flow_time_from_config,
)


def test_masked_velocity_mse_ignores_invalid_tokens() -> None:
    pred = torch.zeros(2, 3, 4)
    target = torch.ones(2, 3, 4)
    target[0, 2] = 1000.0
    mask = torch.tensor([[True, True, False], [False, False, False]])

    loss = masked_velocity_mse(pred, target, mask)

    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(1.0))


def test_rectified_flow_loss_all_invalid_is_finite_zero() -> None:
    pred = torch.randn(2, 3, 4)
    target = torch.randn(2, 3, 4)
    mask = torch.zeros(2, 3, dtype=torch.bool)

    out = Stage2RectifiedFlowLoss()(pred, target, mask)

    assert torch.isfinite(out.total)
    assert out.total.item() == 0.0
    assert out.diagnostics["normalized_rf_mse"].item() == 0.0
    assert out.diagnostics["valid_latent_token_count"].item() == 0.0


def test_rectified_flow_normalized_mse_is_invariant_to_target_scale() -> None:
    mask = torch.tensor([[True, True, False]])
    pred = torch.zeros(1, 3, 2)
    target = torch.ones(1, 3, 2)
    target[:, 2] = 1000.0

    unit = Stage2RectifiedFlowLoss()(pred, target, mask)
    scaled = Stage2RectifiedFlowLoss()(pred, target * 7.0, mask)

    assert unit.total.item() == pytest.approx(1.0)
    assert scaled.total.item() == pytest.approx(49.0)
    assert unit.diagnostics["normalized_rf_mse"].item() == pytest.approx(1.0)
    assert scaled.diagnostics["normalized_rf_mse"].item() == pytest.approx(1.0)


def test_rectified_flow_loss_reports_substream_metrics() -> None:
    pred = torch.zeros(1, 4, 2)
    target = torch.ones(1, 4, 2)
    mask = torch.ones(1, 4, dtype=torch.bool)
    substreams = torch.tensor(
        [
            [
                int(Stage2LatentSubstreamId.HUMAN_DYNAMIC),
                int(Stage2LatentSubstreamId.OBJECT_POSE),
                int(Stage2LatentSubstreamId.OBJECT_GEOMETRY_LATENT),
                int(Stage2LatentSubstreamId.RELATION_DYNAMIC),
            ]
        ]
    )

    out = Stage2RectifiedFlowLoss()(pred, target, mask, latent_substream_ids=substreams)

    assert out.diagnostics["velocity_mse_by_latent_substream.human"].item() == 1.0
    assert out.diagnostics["velocity_mse_by_latent_substream.object_pose"].item() == 1.0
    assert out.diagnostics["velocity_mse_by_latent_substream.object_geometry_latent"].item() == 1.0
    assert out.diagnostics["velocity_mse_by_latent_substream.relation"].item() == 1.0


def test_trainable_ema_uses_fp32_shadow_and_restores_num_updates() -> None:
    model = torch.nn.Linear(3, 2)
    ema = TrainableParameterEMA(model, decay=0.9, warmup=False)
    with torch.no_grad():
        model.weight.add_(1.0)
        model.bias.add_(1.0)
    ema.update(model)
    state = ema.state_dict()

    restored = TrainableParameterEMA(model, decay=0.1, warmup=True)
    restored.load_state_dict(state)

    assert restored.num_updates == 1
    assert restored.decay == 0.9
    assert restored.warmup is False
    assert all(value.dtype == torch.float32 for value in restored.shadow.values())


def test_categorical_condition_dropout_reports_forced_and_effective_null() -> None:
    batch = {
        "meta": {"text": ["pick up object", None]},
        "geometry": {"has_object_bps": torch.tensor([True, False])},
    }
    cfg = {"both": 0.0, "text_only": 0.0, "geometry_only": 0.0, "none": 1.0}

    force_text, force_geometry, diagnostics = sample_categorical_condition_dropout(
        batch,
        cfg,
        device=torch.device("cpu"),
    )

    assert force_text.tolist() == [True, True]
    assert force_geometry.tolist() == [True, True]
    assert diagnostics["condition_dropout.forced_text_fraction"].item() == 1.0
    assert diagnostics["condition_dropout.effective_geometry_null_fraction"].item() == 1.0


def test_categorical_condition_dropout_rejects_negative_probabilities() -> None:
    batch = {"meta": {"text": ["pick up"]}, "geometry": {"has_object_bps": torch.tensor([True])}}

    with pytest.raises(ValueError, match="non-negative"):
        sample_categorical_condition_dropout(
            batch,
            {"both": 1.0, "text_only": -0.1, "geometry_only": 0.0, "none": 0.0},
            device=torch.device("cpu"),
        )


def test_stage1_anchor_cli_overrides_map_into_resolved_config() -> None:
    cfg = apply_stage2_rectified_flow_cli_overrides(
        {},
        {
            "stage1_anchor_mode": "absolute_grid",
            "stage1_anchor_stride": 4,
            "stage1_anchor_grid_origin": 0,
        },
    )

    train_cfg = cfg["training"]["stage2"]
    assert train_cfg["stage1_anchor_mode"] == "absolute_grid"
    assert train_cfg["stage1_anchor_stride"] == 4
    assert train_cfg["stage1_anchor_grid_origin"] == 0

    override = apply_stage1_anchor_override(
        cfg,
        anchor_mode=train_cfg["stage1_anchor_mode"],
        anchor_stride=train_cfg["stage1_anchor_stride"],
        anchor_grid_origin=train_cfg["stage1_anchor_grid_origin"],
    )

    temporal = cfg["model"]["stage1"]["token_compressor"]["temporal"]
    assert temporal == {"anchor_mode": "absolute_grid", "anchor_stride": 4, "anchor_grid_origin": 0}
    assert override["after"] == temporal


def test_stage2_init_weights_cli_mapping_and_resume_mutex() -> None:
    cfg = apply_stage2_rectified_flow_cli_overrides({}, {"init_stage2_weights": "old.pt"})

    assert cfg["training"]["stage2"]["init_stage2_weights"] == "old.pt"
    _validate_resume_and_init_stage2_weights_are_exclusive({"resume": None, "init_stage2_weights": "old.pt"})
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_resume_and_init_stage2_weights_are_exclusive(
            {"resume": "latest.pt", "init_stage2_weights": "old.pt"}
        )


def test_whitening_anchor_override_validation_reads_raw_payload_metadata(monkeypatch) -> None:
    expected = {
        "before": {"anchor_mode": None, "anchor_stride": None, "anchor_grid_origin": None},
        "after": {"anchor_mode": "absolute_grid", "anchor_stride": 4, "anchor_grid_origin": 0},
    }
    loaded: list[tuple[object, object, object]] = []

    def fake_load(path, *, map_location=None, weights_only=None):
        loaded.append((path, map_location, weights_only))
        if str(path) == "missing_meta.pt":
            return {"metadata": {}}
        return {"metadata": {"anchor_override": expected}}

    monkeypatch.setattr(torch, "load", fake_load)

    _validate_whitening_anchor_override("whitening.pt", expected)
    assert loaded[-1] == ("whitening.pt", "cpu", False)

    wrong = {
        "before": expected["before"],
        "after": {"anchor_mode": "absolute_grid", "anchor_stride": 8, "anchor_grid_origin": 0},
    }
    with pytest.raises(ValueError, match="Stage1-anchor override"):
        _validate_whitening_anchor_override("whitening.pt", wrong)

    with pytest.raises(ValueError, match="Stage1-anchor override"):
        _validate_whitening_anchor_override("missing_meta.pt", expected)


def test_source_whitening_provenance_still_rejects_mismatch_without_anchor_override() -> None:
    with pytest.raises(ValueError, match="does not match source checkpoint provenance"):
        _validate_source_whitening_provenance({"whitening_artifact": "recorded.pt"}, "requested.pt")


@pytest.mark.parametrize("same_path", [False, True])
def test_source_whitening_hash_accepts_relocation_and_rejects_changed_bytes(tmp_path, same_path) -> None:
    original = tmp_path / "original.pt"
    original.write_bytes(b"verified whitening")
    requested = original if same_path else tmp_path / "relocated.pt"
    requested.write_bytes(original.read_bytes())
    payload = {
        "whitening_artifact": str(original),
        "whitening_artifact_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    }
    if not same_path:
        original.unlink()
    _validate_source_whitening_provenance(payload, requested)
    requested.write_bytes(b"changed whitening")
    with pytest.raises(ValueError, match="hash does not match"):
        _validate_source_whitening_provenance(payload, requested)


def test_legacy_source_whitening_provenance_uses_readable_original_for_relocation(tmp_path) -> None:
    original = tmp_path / "original.pt"
    relocated = tmp_path / "relocated.pt"
    original.write_bytes(b"legacy whitening")
    relocated.write_bytes(original.read_bytes())
    payload = {"whitening_artifact": str(original)}
    _validate_source_whitening_provenance(payload, original)
    _validate_source_whitening_provenance(payload, relocated)
    relocated.write_bytes(b"different whitening")
    with pytest.raises(ValueError, match="does not match source checkpoint provenance"):
        _validate_source_whitening_provenance(payload, relocated)
    original.unlink()
    with pytest.raises(ValueError, match="does not match source checkpoint provenance"):
        _validate_source_whitening_provenance(payload, relocated)
    _validate_source_whitening_provenance({}, relocated)


def test_resume_source_artifacts_compare_stage1_anchor_override() -> None:
    current = {
        "before": {"anchor_mode": None, "anchor_stride": None, "anchor_grid_origin": None},
        "after": {"anchor_mode": "absolute_grid", "anchor_stride": 4, "anchor_grid_origin": 0},
    }
    trainer = object.__new__(Stage2RectifiedFlowTrainer)
    trainer.source_artifacts = Stage2SourceArtifacts(
        stage1_checkpoint="source.pt",
        stage1_checkpoint_sha256="stage1-hash",
        whitening_artifact="whitening.pt",
        whitening_artifact_sha256="white-hash",
        whitening_payload={},
        stage1_anchor_override=current,
        normalizers={},
        normalizer_hashes={},
    )

    trainer._validate_resume_source_artifacts(
        {
            "source_artifacts": {
                "stage1_checkpoint_sha256": "stage1-hash",
                "whitening_artifact_sha256": "white-hash",
                "stage1_anchor_override": current,
                "normalizer_hashes": {},
            }
        }
    )

    with pytest.raises(ValueError, match="stage1_anchor_override mismatch"):
        trainer._validate_resume_source_artifacts(
            {
                "source_artifacts": {
                    "stage1_checkpoint_sha256": "stage1-hash",
                    "whitening_artifact_sha256": "white-hash",
                    "stage1_anchor_override": None,
                    "normalizer_hashes": {},
                }
            }
        )


class _TinyStage2Like(torch.nn.Module):
    def __init__(self, *, freeze_clip: bool) -> None:
        super().__init__()
        self.text_encoder = torch.nn.Module()
        self.text_encoder.clip_model = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 2)
        if freeze_clip:
            for param in self.text_encoder.clip_model.parameters():
                param.requires_grad_(False)


class _TinyStage2LikeTextModel(torch.nn.Module):
    def __init__(self, *, freeze_text_encoder: bool) -> None:
        super().__init__()
        self.text_encoder = torch.nn.Module()
        self.text_encoder.text_model = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 2)
        if freeze_text_encoder:
            for param in self.text_encoder.text_model.parameters():
                param.requires_grad_(False)


def test_stage2_checkpoint_filter_drops_only_frozen_clip_weights() -> None:
    frozen = _TinyStage2Like(freeze_clip=True)
    trainable = _TinyStage2Like(freeze_clip=False)

    frozen_state = _stage2_state_dict_without_frozen_clip(frozen)
    trainable_state = _stage2_state_dict_without_frozen_clip(trainable)

    assert not any("clip_model" in key for key in frozen_state)
    assert any("clip_model" in key for key in trainable_state)
    with pytest.raises(RuntimeError, match="missing"):
        _load_stage2_state_without_frozen_clip(trainable, frozen_state)


def test_stage2_checkpoint_filter_drops_only_frozen_text_model_weights() -> None:
    frozen = _TinyStage2LikeTextModel(freeze_text_encoder=True)
    trainable = _TinyStage2LikeTextModel(freeze_text_encoder=False)

    frozen_state = _stage2_state_dict_without_frozen_clip(frozen)
    trainable_state = _stage2_state_dict_without_frozen_clip(trainable)

    assert not any("text_model" in key for key in frozen_state)
    assert any("text_model" in key for key in trainable_state)
    _load_stage2_state_without_frozen_clip(frozen, frozen_state)
    with pytest.raises(RuntimeError, match="missing"):
        _load_stage2_state_without_frozen_clip(trainable, frozen_state)


def test_init_stage2_weights_loads_shape_matched_trainable_weights(tmp_path) -> None:
    model = _TinyStage2Like(freeze_clip=True)
    donor = _TinyStage2Like(freeze_clip=True)
    with torch.no_grad():
        donor.head.weight.fill_(3.0)
        donor.head.bias.fill_(4.0)
    state = _stage2_state_dict_without_frozen_clip(donor)
    state["head.bias"] = torch.ones(3)  # shape mismatch: should be skipped, not fatal
    path = tmp_path / "donor.pt"
    torch.save({"stage2": state}, path)

    report = init_stage2_weights_from_checkpoint(model, path)

    assert report["loaded_count"] == 1
    assert report["skipped_count"] == 1
    assert report["missing_count"] == 0
    assert torch.allclose(model.head.weight, torch.full_like(model.head.weight, 3.0))
    assert not torch.allclose(model.head.bias, torch.full_like(model.head.bias, 4.0))

    empty = tmp_path / "empty.pt"
    torch.save({"stage2": {"bad.weight": torch.ones(1)}}, empty)
    with pytest.raises(RuntimeError, match="loaded_count=0"):
        init_stage2_weights_from_checkpoint(model, empty)


def test_formal_stage2_config_uses_t5_77_token_contract() -> None:
    _, _, resolved = load_stage2_rectified_flow_training_config(
        "configs/stage1/data_interact_motion962_all_v0.yaml",
        "configs/stage2/train_rectified_flow_full_mot_post_jitter_v0.yaml",
    )

    text_cfg = resolved["model"]["stage2"]["text"]
    filter_cfg = resolved["data"]["text_filter"]
    assert text_cfg["backend"] == "t5_encoder"
    assert text_cfg["backbone"] == "google/flan-t5-xl"
    assert text_cfg["freeze_text_encoder"] is True
    assert text_cfg["encoder_dtype"] == "bf16"
    assert text_cfg["max_seq_len"] == 77
    assert filter_cfg["tokenizer"] == "google/flan-t5-xl"
    assert filter_cfg["max_tokens"] == 77
    assert filter_cfg["enabled"] is True


def test_validate_stage2_first_batch_rejects_missing_object_geometry() -> None:
    batch = {
        "raw": {"x": torch.zeros(1)},
        "model": {"x": torch.zeros(1)},
        "geometry": {
            "object_bps": torch.zeros(1, 4, 3),
            "object_bps_mask": torch.ones(1, 4, dtype=torch.bool),
            "has_object_bps": torch.tensor([False]),
        },
        "ids": {
            "marker_ids": torch.zeros(1, dtype=torch.long),
            "relation_type_ids": torch.zeros(1, dtype=torch.long),
            "dataset_id": torch.zeros(1, dtype=torch.long),
            "object_id": torch.zeros(1, dtype=torch.long),
        },
        "masks": {
            "frame": torch.ones(1, 2, dtype=torch.bool),
            "human": torch.ones(1, 2, 77, dtype=torch.bool),
            "relation": torch.ones(1, 2, 78, dtype=torch.bool),
        },
        "meta": {
            "text": ["pick up"],
            "start": torch.tensor([0]),
            "end": torch.tensor([2]),
            "sequence_length": torch.tensor([10]),
        },
    }

    with pytest.raises(ValueError, match="requires object geometry"):
        validate_stage2_first_batch(batch, window_size=2)


def test_validate_stage2_first_batch_requires_window_start_metadata() -> None:
    batch = {
        "raw": {"x": torch.zeros(1)},
        "model": {"x": torch.zeros(1)},
        "geometry": {
            "object_bps": torch.zeros(1, 4, 3),
            "object_bps_mask": torch.ones(1, 4, dtype=torch.bool),
            "has_object_bps": torch.tensor([True]),
        },
        "ids": {
            "marker_ids": torch.zeros(1, dtype=torch.long),
            "relation_type_ids": torch.zeros(1, dtype=torch.long),
            "dataset_id": torch.zeros(1, dtype=torch.long),
            "object_id": torch.zeros(1, dtype=torch.long),
        },
        "masks": {
            "frame": torch.ones(1, 2, dtype=torch.bool),
            "human": torch.ones(1, 2, 77, dtype=torch.bool),
            "relation": torch.ones(1, 2, 78, dtype=torch.bool),
        },
        "meta": {"text": ["pick up"], "sequence_length": torch.tensor([10])},
    }

    with pytest.raises(KeyError, match="meta.*start"):
        validate_stage2_first_batch(batch, window_size=2)

    batch["meta"]["start"] = torch.tensor([5])
    batch["meta"]["end"] = torch.tensor([7])
    validate_stage2_first_batch(batch, window_size=2)


def test_validate_stage2_first_batch_requires_sequence_length_metadata() -> None:
    batch = {
        "raw": {"x": torch.zeros(1)},
        "model": {"x": torch.zeros(1)},
        "geometry": {
            "object_bps": torch.zeros(1, 4, 3),
            "object_bps_mask": torch.ones(1, 4, dtype=torch.bool),
            "has_object_bps": torch.tensor([True]),
        },
        "ids": {
            "marker_ids": torch.zeros(1, dtype=torch.long),
            "relation_type_ids": torch.zeros(1, dtype=torch.long),
            "dataset_id": torch.zeros(1, dtype=torch.long),
            "object_id": torch.zeros(1, dtype=torch.long),
        },
        "masks": {
            "frame": torch.ones(1, 2, dtype=torch.bool),
            "human": torch.ones(1, 2, 77, dtype=torch.bool),
            "relation": torch.ones(1, 2, 78, dtype=torch.bool),
        },
        "meta": {"text": ["pick up"], "start": torch.tensor([0]), "end": torch.tensor([2])},
    }

    with pytest.raises(KeyError, match="sequence_length"):
        validate_stage2_first_batch(batch, window_size=2)

    batch["meta"]["sequence_length"] = torch.tensor([10])
    validate_stage2_first_batch(batch, window_size=2)


def test_stage2_rollout_progress_uses_window_center_and_sequence_length() -> None:
    batch = {
        "meta": {
            "start": torch.tensor([0, 32, 72]),
            "end": torch.tensor([128, 160, 200]),
            "sequence_length": torch.tensor([128, 256, 200]),
        }
    }

    progress = _stage2_rollout_progress(batch, device=torch.device("cpu"))

    assert progress[0].item() == pytest.approx(63.5 / 127.0)
    assert progress[1].item() == pytest.approx(95.5 / 255.0)
    assert progress[2].item() == pytest.approx(135.5 / 199.0)


def test_velocity_cosine_uses_valid_tokens_only() -> None:
    pred = torch.tensor([[[1.0, 0.0], [1000.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0], [-1000.0, 0.0]]])
    mask = torch.tensor([[True, False]])

    cosine = _velocity_cosine(pred, target, mask)

    assert torch.allclose(cosine, torch.tensor(1.0))


def test_flow_time_layout_and_block_expansion_handles_partial_blocks() -> None:
    latent_mask = torch.tensor([[True, True, True, True, True, False], [True, True, True, True, True, True]])
    token_type_ids = torch.tensor(
        [
            [
                int(TokenTypeId.HUMAN_ROOT),
                int(TokenTypeId.RELATION_BODY_SLOT),
                int(TokenTypeId.OBJECT_GEOMETRY),
                int(TokenTypeId.OBJECT_POSE),
                int(TokenTypeId.HUMAN_BODY_SLOT),
                int(TokenTypeId.HUMAN_BODY_SLOT),
            ],
            [
                int(TokenTypeId.HUMAN_ROOT),
                int(TokenTypeId.RELATION_BODY_SLOT),
                int(TokenTypeId.OBJECT_GEOMETRY),
                int(TokenTypeId.OBJECT_POSE),
                int(TokenTypeId.HUMAN_BODY_SLOT),
                int(TokenTypeId.HUMAN_BODY_SLOT),
            ],
        ]
    )
    time_ids = torch.tensor([[0, 0, 0, 4, 8, 0], [0, 4, 0, 8, 12, 16]])
    time_mask = torch.tensor([[True, True, False, True, True, False], [True, True, False, True, True, True]])

    layout = build_stage2_flow_anchor_layout(
        latent_mask=latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=time_mask,
        block_size_anchors=2,
    )
    assert layout.anchor_mask.tolist() == [[True, True, True, False, False], [True, True, True, True, True]]
    assert layout.anchor_block_index[0].tolist()[:3] == [0, 0, 1]
    assert layout.anchor_block_index[1].tolist()[:5] == [0, 0, 1, 1, 2]

    t_block = torch.tensor([[0.1, 0.8, 0.0], [0.2, 0.4, 0.6]])
    block_mask = torch.tensor([[True, True, False], [True, True, True]])
    rollout_progress = torch.tensor([0.25, 0.75])
    field = build_stage2_flow_time_field_from_blocks(
        layout=layout,
        t_block=t_block,
        block_mask=block_mask,
        rollout_progress=rollout_progress,
    )

    assert torch.allclose(field.t_anchor[0, :3], torch.tensor([0.1, 0.1, 0.8]))
    assert torch.allclose(field.t_anchor[1, :5], torch.tensor([0.2, 0.2, 0.4, 0.4, 0.6]))
    assert field.t_anchor[0, 3:].abs().sum().item() == 0.0
    assert field.t_block[0, 2].item() == 0.0
    assert field.t_static[0].item() == pytest.approx((0.1 + 0.1 + 0.8) / 3)
    assert field.t_static[1].item() == pytest.approx((0.2 + 0.2 + 0.4 + 0.4 + 0.6) / 5)
    assert torch.allclose(field.rollout_progress, rollout_progress)
    assert field.t_latent_token[0, 0].item() == pytest.approx(0.1)
    assert field.t_latent_token[0, 1].item() == pytest.approx(0.1)
    assert field.t_latent_token[0, 2].item() == pytest.approx(field.t_static[0].item())
    assert field.t_latent_token[0, 4].item() == pytest.approx(0.8)
    assert field.t_latent_token[0, 5].item() == 0.0


def test_flow_time_requires_explicit_rollout_progress() -> None:
    latent_mask = torch.ones(1, 3, dtype=torch.bool)
    token_type_ids = torch.tensor(
        [[int(TokenTypeId.HUMAN_ROOT), int(TokenTypeId.OBJECT_GEOMETRY), int(TokenTypeId.RELATION_BODY_SLOT)]]
    )
    time_ids = torch.tensor([[0, 0, 4]])
    time_mask = torch.tensor([[True, False, True]])
    layout = build_stage2_flow_anchor_layout(
        latent_mask=latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=time_mask,
        block_size_anchors=2,
    )

    with pytest.raises(ValueError, match="rollout_progress is required"):
        build_stage2_flow_time_field_from_blocks(
            layout=layout,
            t_block=torch.tensor([[0.25]]),
            block_mask=torch.tensor([[True]]),
        )


def test_random_all_blocks_flow_time_from_config_is_deterministic_with_seed() -> None:
    latent_mask = torch.ones(1, 5, dtype=torch.bool)
    token_type_ids = torch.tensor(
        [[
            int(TokenTypeId.HUMAN_ROOT),
            int(TokenTypeId.OBJECT_POSE),
            int(TokenTypeId.OBJECT_GEOMETRY),
            int(TokenTypeId.RELATION_BODY_SLOT),
            int(TokenTypeId.HUMAN_BODY_SLOT),
        ]]
    )
    time_ids = torch.tensor([[0, 4, 0, 8, 12]])
    time_mask = torch.tensor([[True, True, False, True, True]])
    cfg = {"training": {"stage2": {"flow_forcing": {"block_size_anchors": 2}}}}

    field_a = build_random_all_blocks_flow_time_from_config(
        cfg,
        latent_mask=latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=time_mask,
        generator=torch.Generator().manual_seed(7),
        rollout_progress=torch.tensor([0.5]),
    )
    field_b = build_random_all_blocks_flow_time_from_config(
        cfg,
        latent_mask=latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=time_mask,
        generator=torch.Generator().manual_seed(7),
        rollout_progress=torch.tensor([0.5]),
    )

    assert torch.allclose(field_a.t_block, field_b.t_block)
    assert torch.allclose(field_a.t_anchor, field_b.t_anchor)
    assert torch.allclose(field_a.t_static, field_b.t_static)
    mean_anchor = field_a.t_anchor[field_a.layout.anchor_mask].mean()
    assert not torch.allclose(field_a.t_static[0], mean_anchor)
    assert field_a.block_mask.tolist() == [[True, True]]
    assert field_a.t_anchor[0, 0].item() == pytest.approx(field_a.t_anchor[0, 1].item())
    assert field_a.t_anchor[0, 2].item() == pytest.approx(field_a.t_anchor[0, 3].item())


def test_flow_forcing_mean_anchor_static_policy_remains_available() -> None:
    latent_mask = torch.ones(1, 5, dtype=torch.bool)
    token_type_ids = torch.tensor(
        [[
            int(TokenTypeId.HUMAN_ROOT),
            int(TokenTypeId.OBJECT_POSE),
            int(TokenTypeId.OBJECT_GEOMETRY),
            int(TokenTypeId.RELATION_BODY_SLOT),
            int(TokenTypeId.HUMAN_BODY_SLOT),
        ]]
    )
    time_ids = torch.tensor([[0, 4, 0, 8, 12]])
    time_mask = torch.tensor([[True, True, False, True, True]])

    field = build_random_all_blocks_flow_time_from_config(
        {"training": {"stage2": {"flow_forcing": {"block_size_anchors": 2, "static_time_policy": "mean_anchor"}}}},
        latent_mask=latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=time_mask,
        generator=torch.Generator().manual_seed(7),
        rollout_progress=torch.tensor([0.5]),
    )

    mean_anchor = field.t_anchor[field.layout.anchor_mask].mean()
    assert torch.allclose(field.t_static[0], mean_anchor)


def test_shared_window_training_flow_time_uses_one_time_for_every_block() -> None:
    latent_mask = torch.ones(2, 5, dtype=torch.bool)
    token_type_ids = torch.tensor(
        [[
            int(TokenTypeId.HUMAN_ROOT),
            int(TokenTypeId.OBJECT_POSE),
            int(TokenTypeId.OBJECT_GEOMETRY),
            int(TokenTypeId.RELATION_BODY_SLOT),
            int(TokenTypeId.HUMAN_BODY_SLOT),
        ]]
    ).expand(2, -1)
    time_ids = torch.tensor([[0, 4, 0, 8, 12]]).expand(2, -1)
    time_mask = torch.tensor([[True, True, False, True, True]]).expand(2, -1)
    config = {
        "training": {
            "stage2": {
                "flow_forcing": {
                    "dynamic_noise_schedule": "shared_window",
                    "static_time_policy": "mean_anchor",
                    "block_size_anchors": 2,
                }
            }
        }
    }

    field = build_stage2_training_flow_time_from_config(
        config,
        latent_mask=latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=time_mask,
        generator=torch.Generator().manual_seed(11),
        rollout_progress=torch.tensor([0.25, 0.75]),
    )

    for batch_index in range(2):
        valid_block_times = field.t_block[batch_index][field.block_mask[batch_index]]
        assert torch.allclose(valid_block_times, valid_block_times[:1].expand_as(valid_block_times))
        assert torch.allclose(field.t_static[batch_index], valid_block_times[0])
    assert not torch.allclose(field.t_block[0, 0], field.t_block[1, 0])


def test_training_flow_time_rejects_unknown_schedule() -> None:
    latent_mask = torch.ones(1, 3, dtype=torch.bool)
    token_type_ids = torch.tensor(
        [[int(TokenTypeId.HUMAN_ROOT), int(TokenTypeId.OBJECT_GEOMETRY), int(TokenTypeId.RELATION_BODY_SLOT)]]
    )
    time_ids = torch.tensor([[0, 0, 4]])
    time_mask = torch.tensor([[True, False, True]])

    with pytest.raises(ValueError, match="dynamic_noise_schedule"):
        build_stage2_training_flow_time_from_config(
            {"training": {"stage2": {"flow_forcing": {"dynamic_noise_schedule": "unknown"}}}},
            latent_mask=latent_mask,
            token_type_ids=token_type_ids,
            time_ids=time_ids,
            time_mask=time_mask,
            rollout_progress=torch.tensor([0.5]),
        )


def test_flow_forcing_config_rejects_non_ff1_modes() -> None:
    latent_mask = torch.ones(1, 3, dtype=torch.bool)
    token_type_ids = torch.tensor(
        [[int(TokenTypeId.HUMAN_ROOT), int(TokenTypeId.OBJECT_GEOMETRY), int(TokenTypeId.RELATION_BODY_SLOT)]]
    )
    time_ids = torch.tensor([[0, 0, 4]])
    time_mask = torch.tensor([[True, False, True]])

    with pytest.raises(ValueError, match="dynamic_noise_schedule"):
        build_random_all_blocks_flow_time_from_config(
            {"training": {"stage2": {"flow_forcing": {"dynamic_noise_schedule": "same_t"}}}},
            latent_mask=latent_mask,
            token_type_ids=token_type_ids,
            time_ids=time_ids,
            time_mask=time_mask,
            rollout_progress=torch.tensor([0.5]),
        )
    with pytest.raises(ValueError, match="static_time_policy"):
        build_random_all_blocks_flow_time_from_config(
            {"training": {"stage2": {"flow_forcing": {"static_time_policy": "random_static"}}}},
            latent_mask=latent_mask,
            token_type_ids=token_type_ids,
            time_ids=time_ids,
            time_mask=time_mask,
            rollout_progress=torch.tensor([0.5]),
        )


def test_stage2_optimizer_default_weight_decay_matches_stage2_config_default() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = build_stage2_optimizer_from_config({"training": {"stage2": {"optimizer": {"type": "adamw"}}}}, model)
    weight_decays = {float(group["weight_decay"]) for group in optimizer.param_groups}

    assert 1.0e-2 in weight_decays


def test_stage2_scheduler_warmup_constant_does_not_decay_to_horizon() -> None:
    model = torch.nn.Linear(2, 2)
    cfg = {
        "training": {
            "stage2": {
                "max_steps": 100000,
                "optimizer": {"type": "adamw", "lr": 1.0e-4},
                "scheduler": {"type": "warmup_constant", "warmup_steps": 2},
            }
        }
    }
    optimizer = build_stage2_optimizer_from_config(cfg, model)
    scheduler = build_stage2_scheduler_from_config(cfg, optimizer)
    assert scheduler is not None
    assert getattr(scheduler, "phase1_metadata")["max_steps"] == 2

    observed = []
    for _ in range(5):
        observed.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert observed[0] == pytest.approx(5.0e-5)
    assert observed[1] == pytest.approx(1.0e-4)
    assert observed[-1] == pytest.approx(1.0e-4)

    bad = {
        "training": {
            "stage2": {
                "optimizer": {"type": "adamw", "lr": 1.0e-4},
                "scheduler": {"type": "warmup_cosine", "warmup_steps": 2},
            }
        }
    }
    with pytest.raises(ValueError, match="warmup_constant_cooldown"):
        build_stage2_scheduler_from_config(bad, optimizer)
