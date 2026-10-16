from __future__ import annotations

import pytest
import torch

from hoi_latent.models.stage1 import (
    Stage1LatentWhiteningTransform,
    fit_stage1_latent_whitening_from_values,
    load_stage1_latent_whitening_transform,
    make_identity_stage1_latent_whitening_stats,
    save_stage1_latent_whitening_stats,
)


def test_stage1_latent_whitening_fit_makes_mean_zero_and_covariance_identity():
    values = _make_correlated_values(num_values=4096, z_dim=8)
    stats = fit_stage1_latent_whitening_from_values(values)
    transform = Stage1LatentWhiteningTransform(stats)
    mask = torch.ones(1, values.shape[0], dtype=torch.bool)

    z_white = transform.whiten(values[None], mask)[0]
    mean = z_white.mean(dim=0)
    covariance = _covariance(z_white)

    assert stats.mode == "zca_full"
    assert stats.z_dim == 8
    assert stats.valid_count == values.shape[0]
    assert torch.allclose(mean, torch.zeros_like(mean), atol=2e-5)
    assert torch.allclose(covariance, torch.eye(8), atol=2e-4)
    diagnostics = transform.diagnostics()
    assert diagnostics["post_condition_number"].item() < 1.01
    assert diagnostics["post_effective_rank"].item() > 7.95
    assert not bool(diagnostics["rank_deficient"])
    assert int(diagnostics["active_dimension_count"]) == 8
    assert int(diagnostics["floored_dimension_count"]) == 0


def test_stage1_latent_whitening_reports_rank_deficient_active_subspace():
    generator = torch.Generator().manual_seed(42)
    factors = torch.randn(4096, 4, generator=generator)
    mixing = torch.randn(4, 8, generator=generator)
    values = factors @ mixing
    transform = Stage1LatentWhiteningTransform(fit_stage1_latent_whitening_from_values(values))

    diagnostics = transform.diagnostics()

    active_count = int(diagnostics["active_dimension_count"])
    floored_count = int(diagnostics["floored_dimension_count"])
    assert bool(diagnostics["rank_deficient"])
    assert active_count + floored_count == 8
    assert active_count == 4
    assert floored_count == 4
    assert diagnostics["raw_effective_rank"].item() < 4.1
    assert diagnostics["active_post_condition_number"].item() < 1.01


def test_stage1_latent_whitening_round_trip_reconstructs_valid_tokens_and_zeros_invalid():
    values = _make_correlated_values(num_values=256, z_dim=6)
    stats = fit_stage1_latent_whitening_from_values(values)
    transform = Stage1LatentWhiteningTransform(stats)
    z = values[:24].reshape(3, 8, 6).clone()
    mask = torch.tensor(
        [
            [True, True, False, True, True, True, False, True],
            [True, False, False, True, True, False, True, True],
            [False, False, False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )

    z_white = transform.whiten(z, mask)
    z_roundtrip = transform.unwhiten(z_white, mask)

    assert z_white[~mask].abs().max().item() == 0.0
    assert z_roundtrip[~mask].abs().max().item() == 0.0
    assert torch.allclose(z_roundtrip[mask], z[mask], atol=2e-5)
    assert torch.isfinite(z_white).all()
    assert torch.isfinite(z_roundtrip).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_stage1_latent_whitening_preserves_input_dtype_for_half_precision(dtype):
    values = _make_correlated_values(num_values=512, z_dim=4)
    transform = Stage1LatentWhiteningTransform(fit_stage1_latent_whitening_from_values(values))
    z = values[:12].reshape(2, 6, 4).to(dtype=dtype)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 3:] = False

    z_white = transform.whiten(z, mask)
    z_roundtrip = transform.unwhiten(z_white, mask)

    assert z_white.dtype == dtype
    assert z_roundtrip.dtype == dtype
    assert z_white[~mask].abs().max().item() == 0.0
    assert z_roundtrip[~mask].abs().max().item() == 0.0


def test_stage1_latent_whitening_save_and_load_round_trip(tmp_path):
    values = _make_correlated_values(num_values=512, z_dim=5)
    stats = fit_stage1_latent_whitening_from_values(values)
    path = tmp_path / "whitening.pt"

    save_stage1_latent_whitening_stats(stats, path, metadata={"source": "unit-test"})
    loaded = load_stage1_latent_whitening_transform(path)

    z = values[:20].reshape(4, 5, 5)
    mask = torch.ones(4, 5, dtype=torch.bool)
    original = Stage1LatentWhiteningTransform(stats)

    assert torch.allclose(original.whiten(z, mask), loaded.whiten(z, mask))
    assert torch.allclose(original.unwhiten(original.whiten(z, mask), mask), loaded.unwhiten(loaded.whiten(z, mask), mask))


def test_identity_stage1_latent_whitening_is_no_op_and_zeros_invalid():
    stats = make_identity_stage1_latent_whitening_stats(4, valid_count=123)
    transform = Stage1LatentWhiteningTransform(stats)
    z = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, False, True], [False, True, True]])

    z_white = transform.whiten(z, mask)
    z_rec = transform.unwhiten(z_white, mask)

    assert stats.valid_count == 123
    assert torch.allclose(z_white[mask], z[mask])
    assert torch.allclose(z_rec[mask], z[mask])
    assert z_white[~mask].abs().max().item() == 0.0
    assert z_rec[~mask].abs().max().item() == 0.0


def test_stage1_latent_whitening_rejects_unsupported_modes_and_bad_values():
    values = _make_correlated_values(num_values=128, z_dim=4)

    with pytest.raises(ValueError, match="mode='zca_full'"):
        fit_stage1_latent_whitening_from_values(values, mode="pca_rank")
    with pytest.raises(ValueError, match="at least two"):
        fit_stage1_latent_whitening_from_values(values[:1])
    bad = values.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        fit_stage1_latent_whitening_from_values(bad)


def test_stage1_latent_whitening_rejects_shape_and_mask_mismatch():
    values = _make_correlated_values(num_values=128, z_dim=4)
    transform = Stage1LatentWhiteningTransform(fit_stage1_latent_whitening_from_values(values))

    with pytest.raises(ValueError, match=r"\[B,N,D\]"):
        transform.whiten(values, torch.ones(values.shape[0], dtype=torch.bool))
    with pytest.raises(ValueError, match="last dim"):
        transform.whiten(torch.randn(2, 3, 5), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="mask must have shape"):
        transform.whiten(torch.randn(2, 3, 4), torch.ones(2, 4, dtype=torch.bool))
    bad = torch.randn(2, 3, 4)
    bad[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        transform.whiten(bad, torch.ones(2, 3, dtype=torch.bool))


def test_stage1_latent_whitening_all_invalid_transform_has_no_nan_and_stays_zero():
    values = _make_correlated_values(num_values=256, z_dim=4)
    transform = Stage1LatentWhiteningTransform(fit_stage1_latent_whitening_from_values(values))
    z = torch.randn(2, 5, 4)
    mask = torch.zeros(2, 5, dtype=torch.bool)

    z_white = transform.whiten(z, mask)
    z_rec = transform.unwhiten(z_white, mask)

    assert torch.isfinite(z_white).all()
    assert torch.isfinite(z_rec).all()
    assert z_white.abs().max().item() == 0.0
    assert z_rec.abs().max().item() == 0.0


def test_stage1_latent_whitening_full_rank_keeps_original_dimension():
    values = _make_correlated_values(num_values=512, z_dim=7)
    stats = fit_stage1_latent_whitening_from_values(values)
    transform = Stage1LatentWhiteningTransform(stats)
    z = values[:14].reshape(2, 7, 7)
    mask = torch.ones(2, 7, dtype=torch.bool)

    z_white = transform.whiten(z, mask)

    assert z_white.shape == z.shape
    assert transform.z_dim == 7
    assert transform.whiten_matrix.shape == (7, 7)
    assert transform.unwhiten_matrix.shape == (7, 7)


def _make_correlated_values(num_values: int, z_dim: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234 + z_dim)
    base = torch.randn(num_values, z_dim, generator=generator)
    mix = torch.randn(z_dim, z_dim, generator=generator)
    q, _ = torch.linalg.qr(mix)
    scales = torch.linspace(0.4, 2.0, z_dim)
    values = base @ torch.diag(scales) @ q.T
    values = values + torch.linspace(-0.5, 0.5, z_dim)
    return values.float()


def _covariance(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(values.shape[0] - 1)
