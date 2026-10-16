from __future__ import annotations

import pytest
import torch

from hoi_latent.models.stage1 import (
    HumanTokenizer,
    ModalityId,
    ObjectTokenizer,
    RelationTokenizer,
    Stage1HOITokenizer,
    Stage1TokenizerModule,
    TokenTypeId,
    concat_tokenizer_outputs,
    validate_stage1_tokenizer_config,
    validate_tokenizer_output,
)


def test_tokenizer_shapes_time_masks_and_concat():
    batch = make_synthetic_stage1_structured_batch(batch_size=2, seq_len=5)
    d_model = 64
    p_h, g, p_r = 8, 4, 8
    h_tok = HumanTokenizer(d_model, 5, 4, p_h, 100)
    o_tok = ObjectTokenizer(d_model, 5, 4, g, geometry_source="bps")
    r_tok = RelationTokenizer(d_model, 5, 4, p_r, 100, 2)

    h_out = h_tok(batch)
    o_out = o_tok(batch)
    r_out = r_tok(batch, return_aux=True)
    joint = concat_tokenizer_outputs([h_out, o_out, r_out])

    assert h_out.tokens.shape == (2, 5 * (1 + p_h + 2), d_model)
    assert o_out.tokens.shape == (2, 5 + g, d_model)
    assert r_out.tokens.shape == (2, 5 * (p_r + 2), d_model)
    assert joint.tokens.shape == (2, 114, d_model)

    for name, out in [("human", h_out), ("object", o_out), ("relation", r_out), ("joint", joint)]:
        validate_tokenizer_output(out, name)
        assert (out.time_ids >= 0).all()
        _assert_invalid_tokens_exactly_zero(out)

    assert h_out.modality_ids.unique().tolist() == [int(ModalityId.HUMAN)]
    assert o_out.modality_ids.unique().tolist() == [int(ModalityId.OBJECT)]
    assert r_out.modality_ids.unique().tolist() == [int(ModalityId.RELATION)]
    assert torch.equal(h_out.time_mask, h_out.mask)
    assert torch.equal(r_out.time_mask, r_out.mask)
    assert torch.equal(o_out.time_mask[:, :5], batch["masks"]["object"])
    assert not o_out.time_mask[:, 5:].any()
    assert (o_out.time_ids[:, 5:] == 0).all()
    assert (o_out.entity_ids[:, :5] == 0).all()
    assert torch.equal(o_out.entity_ids[0, 5:], torch.arange(1, g + 1))

    assert r_out.aux["ground_relation_index"] == 77
    per_frame_types = r_out.token_type_ids[0].reshape(5, p_r + 2)
    assert (per_frame_types[:, p_r] == int(TokenTypeId.RELATION_GROUND)).all()
    assert (per_frame_types[:, p_r + 1] == int(TokenTypeId.RELATION_ROOT_OBJECT)).all()


def test_slot_identity_and_part_bias_initialization():
    h_tok = HumanTokenizer(64, 5, 4, 8, 100)
    o_tok = ObjectTokenizer(64, 5, 4, 4, geometry_source="bps")
    r_tok = RelationTokenizer(64, 5, 4, 8, 100, 2)

    assert hasattr(h_tok, "body_slot_output_embedding")
    assert hasattr(o_tok, "geometry_slot_output_embedding")
    assert hasattr(r_tok, "relation_slot_output_embedding")
    assert hasattr(o_tok, "bps_index_embedding")
    assert torch.allclose(h_tok.body_part_bias.weight, torch.zeros_like(h_tok.body_part_bias.weight))
    assert torch.allclose(r_tok.relation_part_bias.weight, torch.zeros_like(r_tok.relation_part_bias.weight))


def test_object_geometry_sources_and_bps_rank4_shape():
    for source in ["points", "bps", "both"]:
        batch = make_synthetic_stage1_structured_batch(batch_size=2, seq_len=5, bps_rank4=True)
        tok = ObjectTokenizer(64, 5, 4, 4, geometry_source=source)
        out = tok(batch, return_aux=True)

        validate_tokenizer_output(out, f"object_{source}")
        assert out.tokens.shape == (2, 9, 64)
        assert torch.isfinite(out.tokens).all()
        _assert_invalid_tokens_exactly_zero(out)
        assert not out.time_mask[:, 5:].any()
        if source in {"bps", "both"}:
            assert hasattr(tok, "bps_index_embedding")
        assert out.aux["geometry_source"] == source.replace("both", "bps+points")


def test_object_geometry_required_and_bps_max_points():
    batch = make_synthetic_stage1_structured_batch()
    tok = ObjectTokenizer(64, 5, 4, 4, geometry_source="bps")
    bad = dict(batch)
    bad["geometry"] = {}
    with pytest.raises(KeyError, match="geometry.object_bps"):
        tok(bad)

    too_many = make_synthetic_stage1_structured_batch(num_bps=6)
    small_tok = ObjectTokenizer(64, 5, 4, 4, geometry_source="bps", max_bps_points=5)
    with pytest.raises(ValueError, match="max_bps_points"):
        small_tok(too_many)


def test_all_invalid_pooling_does_not_nan():
    batch = make_synthetic_stage1_structured_batch()
    batch["masks"]["human"][1, 2] = False
    batch["masks"]["relation"][1, 3, :77] = False

    human = HumanTokenizer(64, 5, 4, 8, 100)(batch)
    relation = RelationTokenizer(64, 5, 4, 8, 100, 2)(batch)

    assert torch.isfinite(human.tokens).all()
    assert torch.isfinite(relation.tokens).all()


def test_relation_raw_distance_bias_requires_raw_distance():
    batch = make_synthetic_stage1_structured_batch()
    tok = RelationTokenizer(64, 5, 4, 8, 100, 2, use_raw_distance_for_bias=True)
    missing_raw = dict(batch)
    missing_raw.pop("raw")

    with pytest.raises(KeyError, match=r"batch\['raw'\]\['relation'\]\['dist'\]"):
        tok(missing_raw)

    fallback_tok = RelationTokenizer(64, 5, 4, 8, 100, 2, use_raw_distance_for_bias=False)
    out = fallback_tok(missing_raw, return_aux=True)
    assert out.aux["used_proximity_bias"] is True
    validate_tokenizer_output(out, "relation_model_distance_bias")


def test_max_seq_len_validation():
    batch = make_synthetic_stage1_structured_batch(seq_len=5)
    with pytest.raises(ValueError, match="max_seq_len"):
        HumanTokenizer(64, 4, 4, 8, 100)(batch)
    with pytest.raises(ValueError, match="data.window_size"):
        validate_stage1_tokenizer_config(data_window_size=5, max_seq_len=4)


def test_stage1_tokenizer_module_contract_and_gradients_flow():
    batch = make_synthetic_stage1_structured_batch()
    tok = Stage1TokenizerModule(
        human=HumanTokenizer(64, 5, 4, 8, 100),
        object=ObjectTokenizer(64, 5, 4, 4, geometry_source="bps"),
        relation=RelationTokenizer(64, 5, 4, 8, 100, 2),
    )
    outputs = tok(batch, return_concat=True, validate=True)
    assert set(outputs) == {"human", "object", "relation", "concat"}
    assert tok.checkpoint_key == "tokenizer"
    assert tok.output_keys == ("human", "object", "relation")
    assert outputs["concat"].tokens.shape[1] == sum(outputs[key].tokens.shape[1] for key in tok.output_keys)
    assert any(key.startswith("human.") for key in tok.state_dict())
    assert any(key.startswith("object.") for key in tok.state_dict())
    assert any(key.startswith("relation.") for key in tok.state_dict())

    ckpt = {tok.checkpoint_key: tok.state_dict()}
    assert ckpt["tokenizer"]

    legacy_tok = Stage1HOITokenizer(
        human=HumanTokenizer(64, 5, 4, 8, 100),
        object=ObjectTokenizer(64, 5, 4, 4, geometry_source="bps"),
        relation=RelationTokenizer(64, 5, 4, 8, 100, 2),
    )
    assert isinstance(legacy_tok, Stage1TokenizerModule)

    loss = sum(outputs[key].tokens.sum() for key in tok.output_keys)
    loss.backward()

    grads = [
        param.grad
        for param in tok.parameters()
        if param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all()
    ]
    assert grads
    assert any(float(grad.abs().sum().item()) > 0.0 for grad in grads)


def make_synthetic_stage1_structured_batch(
    batch_size: int = 2,
    seq_len: int = 5,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    bps_rank4: bool = False,
    num_bps: int = 1024,
) -> dict:
    torch.manual_seed(7)
    valid = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    if batch_size > 1 and seq_len >= 2:
        valid[1, -2:] = False
    human_mask = valid[:, :, None].expand(batch_size, seq_len, 77).clone()
    object_mask = valid.clone()
    relation_mask = valid[:, :, None].expand(batch_size, seq_len, 78).clone()
    relation_type_ids = torch.zeros(78, dtype=torch.long, device=device)
    relation_type_ids[-1] = 1
    relation_part_ids = torch.zeros(78, dtype=torch.long, device=device)
    relation_part_ids[-1] = 99

    human = {
        "pos_rootrel": torch.randn(batch_size, seq_len, 77, 3, dtype=dtype, device=device),
        "vel_rootrel": torch.randn(batch_size, seq_len, 77, 3, dtype=dtype, device=device),
        "acc_rootrel": torch.randn(batch_size, seq_len, 77, 3, dtype=dtype, device=device),
        "root": torch.randn(batch_size, seq_len, 3, dtype=dtype, device=device),
        "root_vel": torch.randn(batch_size, seq_len, 3, dtype=dtype, device=device),
        "root_acc": torch.randn(batch_size, seq_len, 3, dtype=dtype, device=device),
        "left_heights": torch.randn(batch_size, seq_len, 7, dtype=dtype, device=device),
        "right_heights": torch.randn(batch_size, seq_len, 7, dtype=dtype, device=device),
        "vel_official": torch.randn(batch_size, seq_len, 77, 3, dtype=dtype, device=device),
    }
    obj = {
        "rot6d": torch.randn(batch_size, seq_len, 6, dtype=dtype, device=device),
        "trans": torch.randn(batch_size, seq_len, 3, dtype=dtype, device=device),
        "trans_vel": torch.randn(batch_size, seq_len, 3, dtype=dtype, device=device),
        "trans_acc": torch.randn(batch_size, seq_len, 3, dtype=dtype, device=device),
        "vel9_official": torch.randn(batch_size, seq_len, 9, dtype=dtype, device=device),
    }
    dist = torch.rand(batch_size, seq_len, 78, 1, dtype=dtype, device=device)
    relation = {
        "delta_m2o": torch.randn(batch_size, seq_len, 78, 3, dtype=dtype, device=device),
        "dist": dist,
        "delta_vel": torch.randn(batch_size, seq_len, 78, 3, dtype=dtype, device=device),
        "delta_acc": torch.randn(batch_size, seq_len, 78, 3, dtype=dtype, device=device),
        "point_canon": torch.randn(batch_size, seq_len, 78, 3, dtype=dtype, device=device),
        "root_object_delta": torch.randn(batch_size, seq_len, 3, dtype=dtype, device=device),
        "root_object_dist": torch.rand(batch_size, seq_len, 1, dtype=dtype, device=device),
        "root_object_delta_vel": torch.randn(batch_size, seq_len, 3, dtype=dtype, device=device),
        "root_object_delta_acc": torch.randn(batch_size, seq_len, 3, dtype=dtype, device=device),
    }
    bps = torch.randn(batch_size, num_bps, 3, dtype=dtype, device=device)
    bps_mask = torch.ones(batch_size, num_bps, dtype=torch.bool, device=device)
    if bps_rank4:
        bps = bps[:, None]
        bps_mask = bps_mask[:, None]
    return {
        "model": {
            "human": human,
            "object": obj,
            "relation": relation,
        },
        "raw": {
            "relation": {
                "dist": dist.clone(),
            },
        },
        "geometry": {
            "object_points_local": torch.randn(batch_size, 340, 3, dtype=dtype, device=device),
            "object_points_mask": torch.ones(batch_size, 340, dtype=torch.bool, device=device),
            "object_bps": bps,
            "object_bps_mask": bps_mask,
        },
        "ids": {
            "marker_ids": torch.arange(77, dtype=torch.long, device=device),
            "marker_part_ids": torch.zeros(77, dtype=torch.long, device=device),
            "relation_ids": torch.arange(78, dtype=torch.long, device=device),
            "relation_marker_ids": torch.cat(
                [torch.arange(77, dtype=torch.long, device=device), torch.tensor([-1], dtype=torch.long, device=device)]
            ),
            "relation_part_ids": relation_part_ids,
            "relation_type_ids": relation_type_ids,
            "dataset_id": torch.zeros(batch_size, dtype=torch.long, device=device),
            "object_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        },
        "masks": {
            "valid": valid,
            "human": human_mask,
            "object": object_mask,
            "relation": relation_mask,
            "vel": valid.clone(),
            "acc": valid.clone(),
        },
    }


def _assert_invalid_tokens_exactly_zero(out) -> None:
    invalid = ~out.mask
    if invalid.any():
        assert out.tokens[invalid].abs().max().item() == 0.0
