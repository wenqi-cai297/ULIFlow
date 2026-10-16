from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hoi_latent.evaluation.stage2.decoded_records import Stage2WorldMotionArrays
from hoi_latent.evaluation.stage2.feature_metrics import compute_stage2_feature_metrics
from hoi_latent.evaluation.stage2.feature_preprocess import (
    marker_bps_features_for_sample,
    preprocess_interact_marker_bps_samples,
)
from hoi_latent.evaluation.stage2.interact_feature_evaluator import (
    InterActDistilbertActorAgnosticTextEncoder,
    InterActFeatureEvaluator,
    InterActFeatureEvaluatorUnavailable,
    InterActMotionActorAgnosticEncoder,
    InterActFeatureArtifactPaths,
    _samples_from_fixture_inputs,
    lengths_to_mask,
    load_interact_feature_artifacts,
    run_interact_feature_parity_check,
    split_interact_markersbps_checkpoint_state,
)
from hoi_latent.inference.stage2.decode_render import object_points_world


def test_marker_bps_preprocessing_matches_reference_delta_convention() -> None:
    sample = _sample()
    basis = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    features = marker_bps_features_for_sample(sample, bps_basis=basis)
    assert tuple(features.shape) == (2, 231 + 6)
    np.testing.assert_allclose(features[:, :231].numpy(), sample.human_markers.reshape(2, 231))
    # Basis translates with the object; nearest deltas are closest_point - translated_basis.
    expected_frame0 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32).reshape(-1)
    expected_frame1 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32).reshape(-1)
    np.testing.assert_allclose(features[0, 231:].numpy(), expected_frame0)
    np.testing.assert_allclose(features[1, 231:].numpy(), expected_frame1)


def test_preprocess_normalizes_and_reports_lengths() -> None:
    sample = _sample()
    feature_dim = 231 + 6
    out = preprocess_interact_marker_bps_samples(
        [sample],
        mean=np.ones(feature_dim, dtype=np.float32),
        std=np.full(feature_dim, 2.0, dtype=np.float32),
        bps_basis=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert out.lengths == [2]
    assert out.normalized_features.shape == out.raw_features.shape
    np.testing.assert_allclose(out.normalized_features.numpy(), (out.raw_features.numpy() - 1.0) / 2.0)


def test_preprocess_rejects_non_prefix_frame_mask() -> None:
    sample = _sample()
    sample.frame_mask = np.array([True, False, True])
    sample.human_markers = np.concatenate([sample.human_markers, sample.human_markers[:1]], axis=0)
    sample.object_points = np.concatenate([sample.object_points, sample.object_points[:1]], axis=0)
    sample.object_trans = np.concatenate([sample.object_trans, sample.object_trans[:1]], axis=0)
    with pytest.raises(ValueError, match="prefix mask"):
        preprocess_interact_marker_bps_samples(
            [sample],
            mean=np.zeros(237, dtype=np.float32),
            std=np.ones(237, dtype=np.float32),
            bps_basis=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        )


def test_artifact_loader_fails_loudly_when_missing(tmp_path) -> None:
    paths = InterActFeatureArtifactPaths(
        checkpoint=tmp_path / "markersbps.ckpt",
        mean=tmp_path / "mean.npy",
        std=tmp_path / "std.npy",
        bps_basis=tmp_path / "bps.npy",
    )
    with pytest.raises(InterActFeatureEvaluatorUnavailable, match="missing artifact"):
        load_interact_feature_artifacts(paths)


def test_artifact_loader_rejects_wrong_official_shapes(tmp_path) -> None:
    paths = InterActFeatureArtifactPaths(
        checkpoint=tmp_path / "markersbps.ckpt",
        mean=tmp_path / "mean.npy",
        std=tmp_path / "std.npy",
        bps_basis=tmp_path / "bps.npy",
    )
    torch.save({"state_dict": {"textencoder.x": torch.zeros(1), "motionencoder.y": torch.zeros(1)}}, paths.checkpoint)
    np.save(paths.mean, np.zeros(237, dtype=np.float32))
    np.save(paths.std, np.ones(237, dtype=np.float32))
    np.save(paths.bps_basis, np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="615 dims"):
        load_interact_feature_artifacts(paths)


def test_artifact_loader_validates_manifest_hashes(tmp_path) -> None:
    paths = InterActFeatureArtifactPaths(
        checkpoint=tmp_path / "markersbps.ckpt",
        mean=tmp_path / "mean.npy",
        std=tmp_path / "std.npy",
        bps_basis=tmp_path / "bps.npy",
        manifest=tmp_path / "manifest.json",
    )
    torch.save({"state_dict": {"textencoder.x": torch.zeros(1), "motionencoder.y": torch.zeros(1)}}, paths.checkpoint)
    np.save(paths.mean, np.zeros(615, dtype=np.float32))
    np.save(paths.std, np.ones(615, dtype=np.float32))
    np.save(paths.bps_basis, np.zeros((128, 3), dtype=np.float32))
    paths.manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": {
                    "mean": {"sha256": "bad-sha"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InterActFeatureEvaluatorUnavailable, match="manifest sha256 mismatch"):
        load_interact_feature_artifacts(paths)


def test_checkpoint_state_split_requires_both_encoder_prefixes() -> None:
    state = {
        "textencoder.projection.1.weight": torch.zeros(2, 2),
        "motionencoder.skel_embedding.weight": torch.zeros(2, 2),
    }
    text, motion = split_interact_markersbps_checkpoint_state(state)
    assert "projection.1.weight" in text
    assert "skel_embedding.weight" in motion
    with pytest.raises(ValueError, match="textencoder.* and motionencoder"):
        split_interact_markersbps_checkpoint_state({"textencoder.x": torch.zeros(1)})


def test_motion_and_fake_text_encoder_forward_return_loc() -> None:
    motion_encoder = InterActMotionActorAgnosticEncoder(
        nfeats=5,
        latent_dim=8,
        ff_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    ).eval()
    motion_dist = motion_encoder(torch.randn(2, 3, 5), [3, 2])
    assert tuple(motion_dist.loc.shape) == (2, 8)

    text_encoder = InterActDistilbertActorAgnosticTextEncoder(
        tokenizer=_FakeTokenizer(),
        text_model=_FakeTextModel(hidden_size=6),
        latent_dim=8,
        ff_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    ).eval()
    text_dist = text_encoder(["pick up", "sit"])
    assert tuple(text_dist.loc.shape) == (2, 8)
    assert not any(param.requires_grad for param in text_encoder.text_model.parameters())


def test_parity_fixture_passes_and_detects_stale_golden(tmp_path) -> None:
    torch.manual_seed(123)
    evaluator = InterActFeatureEvaluator(
        mean=np.zeros(237, dtype=np.float32),
        std=np.ones(237, dtype=np.float32),
        bps_basis=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        motion_encoder=InterActMotionActorAgnosticEncoder(
            nfeats=237,
            latent_dim=8,
            ff_size=16,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        ),
        text_encoder=InterActDistilbertActorAgnosticTextEncoder(
            tokenizer=_FakeTokenizer(),
            text_model=_FakeTextModel(hidden_size=6),
            latent_dim=8,
            ff_size=16,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        ),
    ).eval()
    sample = _sample()
    preprocess = evaluator.preprocess([sample])
    motion_loc = evaluator.encode_motion(preprocess.normalized_features, preprocess.lengths).cpu().numpy()
    text_loc = evaluator.encode_text(["pick up"]).cpu().numpy()

    np.savez(
        tmp_path / "fixture_inputs.npz",
        human_markers_world=sample.human_markers[None],
        object_points_world=sample.object_points[None],
        object_trans=sample.object_trans[None],
        frame_mask=sample.frame_mask[None],
    )
    (tmp_path / "fixture_texts.json").write_text(json.dumps(["pick up"]), encoding="utf-8")
    np.savez(
        tmp_path / "golden_features.npz",
        raw_feature_615=preprocess.raw_features.cpu().numpy(),
        normalized_feature_615=preprocess.normalized_features.cpu().numpy(),
        motion_loc=motion_loc,
        text_loc=text_loc,
    )
    report = run_interact_feature_parity_check(evaluator, parity_dir=tmp_path)
    assert report.passed
    assert report.golden_source == "unknown"
    assert report.fixture_sha256["golden_features"]

    (tmp_path / "golden_metadata.json").write_text(
        json.dumps({"golden_source": "external", "external_interact_git": "abc123"}),
        encoding="utf-8",
    )
    external_report = run_interact_feature_parity_check(evaluator, parity_dir=tmp_path)
    assert external_report.passed
    assert external_report.golden_source == "external"
    assert external_report.golden_metadata["external_interact_git"] == "abc123"

    np.savez(
        tmp_path / "golden_features.npz",
        raw_feature_615=preprocess.raw_features.cpu().numpy() + 1.0,
        normalized_feature_615=preprocess.normalized_features.cpu().numpy(),
        motion_loc=motion_loc,
        text_loc=text_loc,
    )
    assert not run_interact_feature_parity_check(evaluator, parity_dir=tmp_path).passed


def test_fixture_canonical_fallback_uses_decode_render_object_transform() -> None:
    markers = np.zeros((1, 2, 77, 3), dtype=np.float32)
    canonical = np.array([[[1.0, 2.0, 0.0], [-3.0, 0.5, 1.0]]], dtype=np.float32)
    rotmat = np.array(
        [
            [
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        ],
        dtype=np.float32,
    )
    trans = np.array([[[0.5, 0.0, 0.0], [0.0, -0.5, 0.25]]], dtype=np.float32)
    samples = _samples_from_fixture_inputs(
        {
            "human_markers_world": markers,
            "object_points_canonical": canonical,
            "object_rotmat": rotmat,
            "object_trans": trans,
        }
    )
    expected = object_points_world(canonical[0], rotmat[0], trans[0])
    np.testing.assert_allclose(samples[0].object_points, expected)
    wrong_untransposed = np.einsum(
        "tpj,tji->tpi",
        np.broadcast_to(canonical[0][None], expected.shape),
        rotmat[0],
    ) + trans[0, :, None, :]
    assert not np.allclose(samples[0].object_points, wrong_untransposed)


def test_feature_metrics_combine_formula_layer() -> None:
    real = np.eye(4, dtype=np.float32)
    generated = real.copy()
    result = compute_stage2_feature_metrics(
        real_motion_features=real,
        generated_motion_features=generated,
        text_features=real,
        r_precision_pool_size=2,
    )
    assert result.values["feature/fid"] == pytest.approx(0.0)
    assert result.values["feature/r_precision_top1"] == pytest.approx(1.0)
    assert result.values["feature/mm_dist"] == pytest.approx(0.0)


def test_lengths_to_mask_matches_interact_semantics() -> None:
    mask = lengths_to_mask([2, 3], torch.device("cpu"), max_len=4)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, False]]


def _sample() -> Stage2WorldMotionArrays:
    human = np.zeros((2, 77, 3), dtype=np.float32)
    human[0, 0] = [1.0, 2.0, 3.0]
    human[1, 0] = [4.0, 5.0, 6.0]
    obj = np.array(
        [
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    trans = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    return Stage2WorldMotionArrays(
        human_markers=human,
        object_points=obj,
        frame_mask=np.array([True, True]),
        object_trans=trans,
    )


class _FakeTokenizer:
    def __call__(self, texts, *, return_tensors, padding):
        max_len = max(len(text.split()) for text in texts) + 1
        ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        mask = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, text in enumerate(texts):
            length = len(text.split()) + 1
            ids[i, :length] = torch.arange(length)
            mask[i, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _FakeTextModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = torch.nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.proj(input_ids))
