from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hoi_latent.models.stage1.tokenizer_types import ModalityId, TokenTypeId
from hoi_latent.models.stage2 import (
    LatentNoisyStreamEncoder,
    ObjectGeometryConditionEncoder,
    Stage2FlowRouteId,
    Stage2LatentSubstreamId,
    Stage2StreamPacker,
    TextConditionEncoder,
    derive_stage2_latent_substream_ids,
)


def test_text_condition_encoder_fake_clip_shapes_masks_and_gradients():
    clip = FakeCLIPTextModel(hidden_size=32)
    tokenizer = FakeCLIPTokenizer(max_length=8)
    stem = TextConditionEncoder(
        d_model=24,
        num_heads=4,
        clip_model=clip,
        tokenizer=tokenizer,
        contextualizer_depth=2,
        residual_gate_init=0.0,
        freeze_clip=True,
    )

    out = stem({"meta": {"text": ["pick up the cup", None]}}, return_diagnostics=True)

    assert out.tokens.shape == (2, tokenizer.model_max_length, 24)
    assert out.mask.shape == (2, tokenizer.model_max_length)
    assert out.pooled.shape == (2, 24)
    assert out.input_ids.shape == (2, tokenizer.model_max_length)
    assert torch.isfinite(out.tokens).all()
    assert torch.isfinite(out.pooled).all()
    assert out.flow_route_ids.unique().tolist() == [int(Stage2FlowRouteId.TEXT_CONDITION)]
    assert out.diagnostics["missing_text_count"].item() == 1
    assert out.tokens[~out.mask].abs().max().item() == 0.0
    assert all(not param.requires_grad for param in stem.clip_model.parameters())

    loss = out.tokens.sum() + out.pooled.sum()
    loss.backward()
    assert all(param.grad is None for param in stem.clip_model.parameters())
    assert stem.text_token_proj.weight.grad is not None
    assert any(block.gate_attn.grad is not None for block in stem.contextualizer)


def test_text_condition_encoder_fake_t5_shapes_masks_and_gradients():
    text_model = FakeT5EncoderModel(hidden_size=40)
    tokenizer = FakeCLIPTokenizer(max_length=7)
    stem = TextConditionEncoder(
        d_model=24,
        num_heads=4,
        backend="t5_encoder",
        text_model=text_model,
        tokenizer=tokenizer,
        contextualizer_depth=2,
        residual_gate_init=0.0,
        freeze_text_encoder=True,
    )

    out = stem({"meta": {"text": ["pick up the cup", None]}}, return_diagnostics=True)

    assert stem.backend == "t5_encoder"
    assert stem.encoder_dim == 40
    assert stem.encoder_pool_proj is None
    assert out.tokens.shape == (2, tokenizer.model_max_length, 24)
    assert out.mask.shape == (2, tokenizer.model_max_length)
    assert out.pooled.shape == (2, 24)
    assert torch.isfinite(out.tokens).all()
    assert torch.isfinite(out.pooled).all()
    assert out.diagnostics["missing_text_count"].item() == 1
    assert out.tokens[~out.mask].abs().max().item() == 0.0
    assert all(not param.requires_grad for param in stem.text_model.parameters())

    loss = out.tokens.sum() + out.pooled.sum()
    loss.backward()
    assert all(param.grad is None for param in stem.text_model.parameters())
    assert stem.text_token_proj.weight.grad is not None
    assert stem.token_pool_proj.weight.grad is not None


def test_text_condition_encoder_real_clip_integration():
    """Real CLIP ViT-B/32 path: FakeCLIP cannot validate the real transformers API
    (pooler_output presence, hidden_size=512, model_max_length=77, max_length padding)."""
    pytest.importorskip("transformers")
    try:
        stem = TextConditionEncoder(d_model=384, num_heads=6, contextualizer_depth=2, freeze_clip=True)
    except Exception as exc:  # offline / no cached weights
        pytest.skip(f"Real CLIP unavailable: {exc!r}")

    assert stem.max_seq_len == 77
    assert stem.clip_dim == 512

    out = stem({"meta": {"text": ["a person picks up the box", None, "sit on the chair"]}}, return_diagnostics=True)
    assert out.tokens.shape == (3, 77, 384)
    assert out.pooled.shape == (3, 384)
    assert out.mask.shape == (3, 77)
    assert torch.isfinite(out.tokens).all()
    assert torch.isfinite(out.pooled).all()
    assert out.tokens[~out.mask].abs().max().item() == 0.0
    assert all(not param.requires_grad for param in stem.clip_model.parameters())
    assert out.diagnostics["missing_text_count"].item() == 1

    (out.tokens.sum() + out.pooled.sum()).backward()
    assert all(param.grad is None for param in stem.clip_model.parameters())
    assert stem.text_token_proj.weight.grad is not None


def test_text_condition_encoder_force_null_uses_valid_padding_mask():
    stem = TextConditionEncoder(
        d_model=24,
        num_heads=4,
        clip_model=FakeCLIPTextModel(hidden_size=32),
        tokenizer=FakeCLIPTokenizer(max_length=8),
        contextualizer_depth=1,
        freeze_clip=True,
    )
    out = stem(
        {"meta": {"text": ["long prompt with several words", "another prompt"]}},
        force_null_text_mask=torch.tensor([False, True]),
        return_diagnostics=True,
    )
    assert out.diagnostics["forced_null_text_count"].item() == 1
    assert out.mask[1].sum().item() == 2
    assert out.tokens[~out.mask].abs().max().item() == 0.0


def test_geometry_condition_encoder_handles_missing_without_nan_gradients():
    stem = ObjectGeometryConditionEncoder(d_model=24, num_heads=4, max_bps_points=8, num_geometry_tokens=16)
    batch = _geometry_batch(batch_size=2, num_bps=5, rank4=False)
    batch["geometry"]["object_bps_mask"][1] = False
    batch["geometry"]["has_object_bps"] = torch.tensor([True, False])

    out = stem(batch, return_diagnostics=True)

    assert out.tokens.shape == (2, 16, 24)
    assert out.mask.shape == (2, 16)
    assert out.pooled.shape == (2, 24)
    assert out.missing.tolist() == [False, True]
    assert torch.isfinite(out.tokens).all()
    assert torch.isfinite(out.pooled).all()
    assert out.flow_route_ids.unique().tolist() == [int(Stage2FlowRouteId.OBJECT_GEOMETRY_CONDITION)]
    assert out.diagnostics["missing_geometry_count"].item() == 1

    loss = out.tokens.square().sum() + out.pooled.square().sum()
    loss.backward()
    grads = [param.grad for param in stem.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_geometry_condition_encoder_rank4_bps_and_force_null():
    stem = ObjectGeometryConditionEncoder(d_model=24, num_heads=4, max_bps_points=8, num_geometry_tokens=16)
    batch = _geometry_batch(batch_size=2, num_bps=5, rank4=True)
    out = stem(batch, force_null_geometry_mask=torch.tensor([False, True]), return_diagnostics=True)

    assert out.tokens.shape == (2, 16, 24)
    assert out.source_mask.shape == (2, 5)
    assert out.missing.tolist() == [False, True]
    assert out.diagnostics["forced_null_geometry_count"].item() == 1
    assert torch.isfinite(out.tokens).all()


def test_geometry_condition_encoder_rejects_too_many_bps_points():
    stem = ObjectGeometryConditionEncoder(d_model=24, num_heads=4, max_bps_points=4, num_geometry_tokens=16)
    with pytest.raises(ValueError, match="max_bps_points"):
        stem(_geometry_batch(batch_size=1, num_bps=5, rank4=False))


def test_latent_noisy_stream_encoder_metadata_masks_and_optional_substream():
    stem = LatentNoisyStreamEncoder(d_model=24, latent_dim=16, num_entity_embeddings=32)
    inputs = _latent_inputs()

    out = stem(**inputs, return_diagnostics=True)

    assert out.tokens.shape == (2, 5, 24)
    assert out.flow_route_ids.unique().tolist() == [int(Stage2FlowRouteId.LATENT_GENERATED)]
    assert stem.latent_substream_embedding is None
    assert torch.isfinite(out.tokens).all()
    assert out.tokens[~out.mask].abs().max().item() == 0.0
    assert out.apply_temporal_rope[0, 0].item()
    assert not out.apply_temporal_rope[0, 2].item()
    assert not out.apply_temporal_rope[0, 4].item()
    assert out.token_type_ids[0, 4].item() == 0
    assert out.entity_ids[0, 4].item() == 0

    substream_stem = LatentNoisyStreamEncoder(
        d_model=24,
        latent_dim=16,
        num_entity_embeddings=32,
        use_latent_substream_embedding=True,
    )
    substream_out = substream_stem(**inputs)
    assert substream_stem.latent_substream_embedding is not None
    assert torch.isfinite(substream_out.tokens).all()


def test_latent_noisy_stream_encoder_accepts_flat_compressed_metadata_when_enabled():
    inputs = _flat_latent_inputs()

    with pytest.raises(ValueError, match="out-of-range valid ids"):
        LatentNoisyStreamEncoder(d_model=24, latent_dim=315, num_entity_embeddings=64)(**inputs)

    stem = LatentNoisyStreamEncoder(
        d_model=24,
        latent_dim=315,
        num_entity_embeddings=64,
        use_latent_substream_embedding=True,
        allow_unknown_token_type_ids=True,
    )
    out = stem(**inputs, return_diagnostics=True)
    substreams = derive_stage2_latent_substream_ids(
        inputs["modality_ids"],
        inputs["token_type_ids"],
        mask=inputs["latent_mask"],
    )

    assert out.tokens.shape == (2, 5, 24)
    assert torch.isfinite(out.tokens).all()
    assert out.tokens[~out.mask].abs().max().item() == 0.0
    assert out.token_type_ids[0, 0].item() == -1
    assert out.token_type_ids[0, 3].item() == int(TokenTypeId.OBJECT_GEOMETRY)
    assert out.token_type_ids[0, 4].item() == 0
    assert out.apply_temporal_rope[0, 0].item()
    assert out.apply_temporal_rope[0, 1].item()
    assert not out.apply_temporal_rope[0, 3].item()
    assert substreams[0, 0].item() == int(Stage2LatentSubstreamId.FLAT_DYNAMIC)
    assert substreams[0, 3].item() == int(Stage2LatentSubstreamId.OBJECT_GEOMETRY_LATENT)
    assert out.diagnostics["allow_unknown_token_type_ids"]


def test_stage2_stream_packer_aligns_slices_and_zeroes_invalid_tokens():
    text_stem = TextConditionEncoder(
        d_model=24,
        num_heads=4,
        clip_model=FakeCLIPTextModel(hidden_size=32),
        tokenizer=FakeCLIPTokenizer(max_length=8),
        contextualizer_depth=1,
        freeze_clip=True,
    )
    geometry_stem = ObjectGeometryConditionEncoder(d_model=24, num_heads=4, max_bps_points=8, num_geometry_tokens=16)
    latent_stem = LatentNoisyStreamEncoder(d_model=24, latent_dim=16, num_entity_embeddings=32)

    text = text_stem({"meta": {"text": ["pick cup", None]}})
    geometry = geometry_stem(_geometry_batch(batch_size=2, num_bps=5, rank4=False))
    latent = latent_stem(**_latent_inputs())
    packed = Stage2StreamPacker()(latent, text, geometry)

    assert packed.tokens.shape == (2, 5 + 8 + 16, 24)
    assert packed.latent_slice == slice(0, 5)
    assert packed.text_slice == slice(5, 13)
    assert packed.geometry_slice == slice(13, 29)
    assert torch.equal(packed.tokens[:, packed.latent_slice], latent.tokens)
    assert torch.equal(packed.attention_mask[:, packed.text_slice], text.mask)
    assert not packed.apply_temporal_rope[:, packed.text_slice].any()
    assert not packed.apply_temporal_rope[:, packed.geometry_slice].any()
    assert packed.tokens[~packed.attention_mask].abs().max().item() == 0.0
    assert packed.pooled_text.shape == (2, 24)
    assert packed.pooled_geometry.shape == (2, 24)


class FakeCLIPTokenizer:
    def __init__(self, max_length: int = 8) -> None:
        self.model_max_length = max_length

    def __call__(
        self,
        texts,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ):
        del padding, truncation, return_tensors
        input_ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_length, dtype=torch.long)
        for row, text in enumerate(texts):
            words = str(text).split()
            length = min(max_length, len(words) + 2)
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


def _geometry_batch(batch_size: int, num_bps: int, rank4: bool) -> dict:
    bps = torch.randn(batch_size, num_bps, 3)
    mask = torch.ones(batch_size, num_bps, dtype=torch.bool)
    if rank4:
        bps = bps[:, None]
        mask = mask[:, None]
    return {
        "geometry": {
            "object_bps": bps,
            "object_bps_mask": mask,
            "has_object_bps": torch.ones(batch_size, dtype=torch.bool),
        }
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
    entity_ids = torch.tensor([[0, 0, 1, 2, 999], [0, 0, 1, 2, 3]])
    time_ids = torch.tensor([[0, 0, 0, 0, 0], [0, 1, 0, 1, 2]])
    time_mask = torch.tensor([[True, True, False, True, True], [True, True, False, True, True]])
    return {
        "z_white_t": z,
        "latent_mask": mask,
        "modality_ids": modality_ids,
        "token_type_ids": token_type_ids,
        "time_ids": time_ids,
        "entity_ids": entity_ids,
        "time_mask": time_mask,
    }


def _flat_latent_inputs() -> dict:
    z = torch.randn(2, 5, 315)
    mask = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    modality_ids = torch.tensor(
        [
            [-1, -1, -1, int(ModalityId.OBJECT), 999],
            [-1, -1, -1, int(ModalityId.OBJECT), -1],
        ]
    )
    token_type_ids = torch.tensor(
        [
            [-1, -1, -1, int(TokenTypeId.OBJECT_GEOMETRY), 999],
            [-1, -1, -1, int(TokenTypeId.OBJECT_GEOMETRY), -1],
        ]
    )
    entity_ids = torch.tensor([[0, 1, 2, 3, 999], [0, 1, 2, 3, 4]])
    time_ids = torch.tensor([[0, 4, 8, 0, 0], [0, 4, 8, 0, 12]])
    time_mask = torch.tensor([[True, True, True, False, True], [True, True, True, False, True]])
    return {
        "z_white_t": z,
        "latent_mask": mask,
        "modality_ids": modality_ids,
        "token_type_ids": token_type_ids,
        "time_ids": time_ids,
        "entity_ids": entity_ids,
        "time_mask": time_mask,
    }
