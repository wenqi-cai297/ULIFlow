from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from hoi_latent.models.stage1.tokenizer_types import ModalityId, TokenTypeId
from hoi_latent.models.stage2 import (
    Stage2StreamPacker,
    Stage2TriStreamMoTBackbone,
    Stage2TriStreamMoTModel,
    TextConditionEncoder,
    ObjectGeometryConditionEncoder,
    LatentNoisyStreamEncoder,
    build_random_all_blocks_flow_time,
    build_stage2_tri_stream_mot_model_from_config,
)


def test_stage2_model_wrapper_forward_shapes_intermediates_and_diagnostics():
    model = _make_model()
    batch = _batch()
    latent_inputs = _latent_inputs()

    out = model(
        batch,
        flow_time=_flow_time(latent_inputs),
        return_intermediates=True,
        return_diagnostics=True,
        force_null_text_mask=torch.tensor([False, True]),
        force_null_geometry_mask=torch.tensor([False, True]),
        **latent_inputs,
    )

    assert out.velocity.shape == (2, 5, 16)
    assert out.backbone.velocity.shape == out.velocity.shape
    assert out.text is not None
    assert out.geometry is not None
    assert out.latent is not None
    assert out.packed is not None
    assert out.diagnostics["text"]["forced_null_text_count"].item() == 1
    assert out.diagnostics["geometry"]["forced_null_geometry_count"].item() == 1
    assert out.diagnostics["pack"]["total_tokens"] == 5 + 8 + 16
    assert "backbone" in out.diagnostics

    no_intermediates = model(batch, flow_time=_flow_time(latent_inputs), **latent_inputs)
    assert no_intermediates.velocity.shape == (2, 5, 16)
    assert no_intermediates.text is None
    assert no_intermediates.geometry is None
    assert no_intermediates.latent is None
    assert no_intermediates.packed is None


def test_stage2_model_builder_wires_shared_dimensions_with_fake_clip():
    config = {
        "model": {
            "stage2": {
                "d_model": 24,
                "latent_dim": 16,
                "num_heads": 4,
                "dropout": 0.0,
                "text": {"contextualizer_depth": 1, "clip_dim": 32, "max_seq_len": 8},
                "geometry": {"max_bps_points": 8, "num_geometry_tokens": 16},
                "backbone": {"depth": 1, "head_dim": 6, "mlp_ratio": 2},
            }
        }
    }
    model = build_stage2_tri_stream_mot_model_from_config(
        config,
        clip_model=FakeCLIPTextModel(hidden_size=32),
        tokenizer=FakeCLIPTokenizer(max_length=8),
    )

    assert isinstance(model, Stage2TriStreamMoTModel)
    assert model.backbone.d_model == 24
    latent_inputs = _latent_inputs()
    out = model(_batch(), flow_time=_flow_time(latent_inputs), **latent_inputs)
    assert out.velocity.shape == (2, 5, 16)


def test_stage2_model_builder_wires_fake_t5_backend():
    config = {
        "model": {
            "stage2": {
                "d_model": 24,
                "latent_dim": 16,
                "num_heads": 4,
                "dropout": 0.0,
                "text": {
                    "backend": "t5_encoder",
                    "contextualizer_depth": 1,
                    "max_seq_len": 8,
                    "freeze_text_encoder": True,
                },
                "geometry": {"max_bps_points": 8, "num_geometry_tokens": 16},
                "backbone": {"depth": 1, "head_dim": 6, "mlp_ratio": 2},
            }
        }
    }
    model = build_stage2_tri_stream_mot_model_from_config(
        config,
        text_model=FakeT5EncoderModel(hidden_size=40),
        tokenizer=FakeCLIPTokenizer(max_length=8),
    )

    assert model.text_encoder.backend == "t5_encoder"
    assert model.text_encoder.encoder_dim == 40
    latent_inputs = _latent_inputs()
    out = model(_batch(), flow_time=_flow_time(latent_inputs), **latent_inputs)
    assert out.velocity.shape == (2, 5, 16)


def _make_model() -> Stage2TriStreamMoTModel:
    return Stage2TriStreamMoTModel(
        text_encoder=TextConditionEncoder(
            d_model=24,
            num_heads=4,
            clip_model=FakeCLIPTextModel(hidden_size=32),
            tokenizer=FakeCLIPTokenizer(max_length=8),
            contextualizer_depth=1,
            freeze_clip=True,
        ),
        geometry_encoder=ObjectGeometryConditionEncoder(d_model=24, num_heads=4, max_bps_points=8, num_geometry_tokens=16),
        latent_encoder=LatentNoisyStreamEncoder(d_model=24, latent_dim=16, num_entity_embeddings=32),
        packer=Stage2StreamPacker(),
        backbone=Stage2TriStreamMoTBackbone(
            d_model=24,
            latent_dim=16,
            depth=1,
            num_heads=4,
            head_dim=6,
            mlp_ratio=2,
        ),
    )


def _batch() -> dict:
    bps = torch.randn(2, 5, 3)
    return {
        "meta": {"text": ["pick up the cup", None]},
        "geometry": {
            "object_bps": bps,
            "object_bps_mask": torch.ones(2, 5, dtype=torch.bool),
            "has_object_bps": torch.ones(2, dtype=torch.bool),
        },
    }


def _latent_inputs() -> dict:
    z = torch.randn(2, 5, 16)
    mask = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    modality_ids = torch.tensor(
        [
            [int(ModalityId.HUMAN), int(ModalityId.OBJECT), int(ModalityId.OBJECT), int(ModalityId.RELATION), 999],
            [int(ModalityId.HUMAN), int(ModalityId.OBJECT), int(ModalityId.OBJECT), int(ModalityId.RELATION), int(ModalityId.HUMAN)],
        ]
    )
    token_type_ids = torch.tensor(
        [
            [
                int(TokenTypeId.HUMAN_ROOT),
                int(TokenTypeId.OBJECT_POSE),
                int(TokenTypeId.OBJECT_GEOMETRY),
                int(TokenTypeId.RELATION_BODY_SLOT),
                999,
            ],
            [
                int(TokenTypeId.HUMAN_ROOT),
                int(TokenTypeId.OBJECT_POSE),
                int(TokenTypeId.OBJECT_GEOMETRY),
                int(TokenTypeId.RELATION_BODY_SLOT),
                int(TokenTypeId.HUMAN_BODY_SLOT),
            ],
        ]
    )
    return {
        "z_white_t": z,
        "latent_mask": mask,
        "modality_ids": modality_ids,
        "token_type_ids": token_type_ids,
        "time_ids": torch.tensor([[0, 1, 0, 1, 0], [0, 1, 0, 1, 2]]),
        "entity_ids": torch.tensor([[0, 0, 1, 2, 999], [0, 0, 1, 2, 3]]),
        "time_mask": torch.tensor([[True, True, False, True, True], [True, True, False, True, True]]),
    }


def _flow_time(latent_inputs: dict):
    generator = torch.Generator().manual_seed(123)
    return build_random_all_blocks_flow_time(
        latent_mask=latent_inputs["latent_mask"],
        token_type_ids=latent_inputs["token_type_ids"],
        time_ids=latent_inputs["time_ids"],
        time_mask=latent_inputs["time_mask"],
        block_size_anchors=2,
        generator=generator,
        rollout_progress=torch.tensor([0.25, 0.75]),
    )


class FakeCLIPTokenizer:
    def __init__(self, max_length: int = 8) -> None:
        self.model_max_length = max_length

    def __call__(self, texts, padding: str, truncation: bool, max_length: int, return_tensors: str):
        del padding, truncation, return_tensors
        input_ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_length, dtype=torch.long)
        for row, text in enumerate(texts):
            length = min(max_length, len(str(text).split()) + 2)
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeCLIPTextModel(nn.Module):
    def __init__(self, hidden_size: int = 32, vocab_size: int = 128) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        del output_hidden_states
        hidden = self.proj(self.embedding(input_ids))
        lengths = attention_mask.sum(dim=1).clamp_min(1)
        batch = torch.arange(input_ids.shape[0], device=input_ids.device)
        pooled = hidden[batch, lengths - 1]
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=pooled)


class FakeT5EncoderModel(nn.Module):
    def __init__(self, hidden_size: int = 40, vocab_size: int = 128) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        del attention_mask, output_hidden_states
        hidden = self.proj(self.embedding(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)
