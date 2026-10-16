from __future__ import annotations

from pathlib import Path

import numpy as np
from types import SimpleNamespace

from hoi_latent.data.representations.interact_marker_metadata import (
    PART_GROUND,
    PART_UNKNOWN,
    build_stage1_ids,
    build_marker_part_diagnostics,
)
from hoi_latent.data.representations.interact_motion_962 import InterActMotion962
from hoi_latent.data.representations.interact_stage1_structured import (
    build_stage1_raw_groups_from_motion_window,
    derivative_mask,
)
from hoi_latent.data.transforms.structured_normalization import (
    Stage1StructuredNormalizer,
    compute_streaming_structured_normalizer,
    zero_stage1_model_groups_by_masks,
)


def test_parser_backward_compatibility_roundtrip():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 5, InterActMotion962.DIM)).astype(np.float32)
    recon = InterActMotion962.compose(InterActMotion962.parse(x))
    assert np.max(np.abs(x - recon)) == 0.0


def test_structured_relation_sign_and_shapes():
    motion = np.zeros((4, InterActMotion962.DIM), dtype=np.float32)
    groups = InterActMotion962.parse(motion)
    groups.human_pos[:] = np.arange(4, dtype=np.float32)[:, None, None]
    groups.object_trans[:] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    groups.relation[..., 0:3] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    groups.relation[..., 3:6] = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    motion = InterActMotion962.compose(groups)

    valid = np.asarray([True, True, True, False])
    raw = build_stage1_raw_groups_from_motion_window(motion, valid)

    assert raw["human"]["pos"].shape == (4, 77, 3)
    assert raw["object"]["rotmat"].shape == (4, 3, 3)
    assert raw["relation"]["delta_m2o"].shape == (4, 78, 3)
    np.testing.assert_allclose(raw["relation"]["delta_o2m"], -raw["relation"]["delta_m2o"])
    np.testing.assert_allclose(
        raw["relation"]["nearest_point_world"][:3, :77],
        raw["human"]["pos"][:3] + raw["relation"]["delta_m2o"][:3, :77],
    )
    np.testing.assert_allclose(
        raw["relation"]["nearest_point_world"][:3, 77],
        raw["relation"]["ground_marker"][:3] + raw["relation"]["delta_m2o"][:3, 77],
    )
    assert np.all(raw["human"]["vel"][2:] == 0)
    assert np.all(raw["human"]["acc"][1:] == 0)


def test_structured_normalizer_keeps_rotmat_geometric_and_zeros_padding():
    motion = np.zeros((4, InterActMotion962.DIM), dtype=np.float32)
    groups = InterActMotion962.parse(motion)
    groups.human_pos[:] = 3.0
    groups.object_rot6d[:] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    motion = InterActMotion962.compose(groups)
    valid = np.asarray([True, True, False, False])
    raw = build_stage1_raw_groups_from_motion_window(motion, valid)
    stats = {
        "human": {
            "pos": {
                "mean": np.ones((77, 3), dtype=np.float32),
                "std": np.full((77, 3), 2.0, dtype=np.float32),
                "count": 2,
                "min_std_before_clamp": 2.0,
                "max_std_before_clamp": 2.0,
                "num_std_below_eps": 0,
            },
        },
        "object": {
            "rotmat": {
                "mean": np.ones((3, 3), dtype=np.float32) * 100.0,
                "std": np.ones((3, 3), dtype=np.float32),
                "count": 2,
                "min_std_before_clamp": 1.0,
                "max_std_before_clamp": 1.0,
                "num_std_below_eps": 0,
            },
        },
    }
    normalizer = Stage1StructuredNormalizer(stats)
    model = normalizer.normalize_sample_raw_groups(raw)

    np.testing.assert_allclose(model["human"]["pos"][:2], 1.0)
    np.testing.assert_allclose(model["object"]["rotmat"], raw["object"]["rotmat"])

    zeroed = zero_stage1_model_groups_by_masks(
        model,
        valid,
        derivative_mask(valid, order=1),
        derivative_mask(valid, order=2),
    )
    assert np.all(zeroed["human"]["pos"][2:] == 0)
    assert np.all(zeroed["human"]["vel"][1:] == 0)
    assert np.all(zeroed["human"]["acc"] == 0)
    assert np.all(zeroed["object"]["rotmat"][2:] == 0)


def test_structured_normalizer_stats_are_mask_aware(tmp_path):
    motion = np.zeros((4, InterActMotion962.DIM), dtype=np.float32)
    motion_path = tmp_path / "motion.npy"
    np.save(motion_path, motion)
    record = SimpleNamespace(split="train", motion_path=str(motion_path))

    normalizer, metadata = compute_streaming_structured_normalizer([record], split="train")

    assert normalizer.stats["human"]["pos"]["count"] == 4
    assert normalizer.stats["human"]["vel"]["count"] == 3
    assert normalizer.stats["human"]["acc"]["count"] == 2
    assert normalizer.stats["human"]["vel"]["mask"] == "vel"
    assert normalizer.stats["human"]["acc"]["mask"] == "acc"
    assert "rotmat" not in normalizer.stats["object"]
    assert metadata["fields"]["human"]["vel"]["mask"] == "vel"


def test_marker_part_diagnostics_exposes_unknown_only_map():
    marker_part_ids = np.full((77,), PART_UNKNOWN, dtype=np.int64)
    diagnostics = build_marker_part_diagnostics(None, marker_part_ids)

    assert diagnostics["marker_part_map_exists"] is False
    assert diagnostics["all_marker_part_ids_unknown"] is True
    assert diagnostics["part_pooling_available"] is False
    assert diagnostics["recommended_tokenization"] == "marker_tokens_only"


def test_marker_part_map_v1_covers_all_markers_and_enables_part_pooling():
    ids = build_stage1_ids("assets/interact/marker_part_map_v1.json")
    marker_part_ids = ids["marker_part_ids"]
    relation_part_ids = ids["relation_part_ids"]
    diagnostics = build_marker_part_diagnostics("assets/interact/marker_part_map_v1.json", marker_part_ids)

    assert marker_part_ids.shape == (77,)
    assert relation_part_ids.shape == (78,)
    assert np.all(marker_part_ids != PART_UNKNOWN)
    np.testing.assert_array_equal(relation_part_ids[:77], marker_part_ids)
    assert relation_part_ids[-1] == PART_GROUND
    assert diagnostics["all_marker_part_ids_unknown"] is False
    assert diagnostics["part_pooling_available"] is True
    assert diagnostics["recommended_tokenization"] == "marker_and_part_tokens"


def test_stage1_configs_use_marker_part_map_v1():
    for path in ["configs/stage1/data_interact_motion962_all_v0.yaml"]:
        text = Path(path).read_text(encoding="utf-8")
        assert "marker_part_map_v1.json" in text
        assert "marker_part_map_v0.json" not in text
