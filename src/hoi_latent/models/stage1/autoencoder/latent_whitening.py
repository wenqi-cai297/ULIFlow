from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens


@dataclass
class Stage1LatentWhiteningStats:
    mean: torch.Tensor
    covariance: torch.Tensor
    eigvals: torch.Tensor
    eigvecs: torch.Tensor
    whiten: torch.Tensor
    unwhiten: torch.Tensor
    eps: float
    mode: str
    z_dim: int
    valid_count: int

    def to_payload(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mean": self.mean.detach().cpu(),
            "covariance": self.covariance.detach().cpu(),
            "eigvals": self.eigvals.detach().cpu(),
            "eigvecs": self.eigvecs.detach().cpu(),
            "whiten": self.whiten.detach().cpu(),
            "unwhiten": self.unwhiten.detach().cpu(),
            "eps": float(self.eps),
            "mode": self.mode,
            "z_dim": int(self.z_dim),
            "valid_count": int(self.valid_count),
        }
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Stage1LatentWhiteningStats":
        required = [
            "mean",
            "covariance",
            "eigvals",
            "eigvecs",
            "whiten",
            "unwhiten",
            "eps",
            "mode",
            "z_dim",
            "valid_count",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"Stage1LatentWhiteningStats payload missing keys: {missing}.")
        stats = cls(
            mean=_as_tensor(payload["mean"], "mean"),
            covariance=_as_tensor(payload["covariance"], "covariance"),
            eigvals=_as_tensor(payload["eigvals"], "eigvals"),
            eigvecs=_as_tensor(payload["eigvecs"], "eigvecs"),
            whiten=_as_tensor(payload["whiten"], "whiten"),
            unwhiten=_as_tensor(payload["unwhiten"], "unwhiten"),
            eps=float(payload["eps"]),
            mode=str(payload["mode"]),
            z_dim=int(payload["z_dim"]),
            valid_count=int(payload["valid_count"]),
        )
        validate_stage1_latent_whitening_stats(stats)
        return stats


class Stage1LatentWhiteningTransform(nn.Module):
    """Fixed epsilon-regularized ZCA boundary transform for Stage1 z_rec latents."""

    def __init__(self, stats: Stage1LatentWhiteningStats) -> None:
        super().__init__()
        validate_stage1_latent_whitening_stats(stats)
        self.z_dim = int(stats.z_dim)
        self.mode = str(stats.mode)
        self.eps = float(stats.eps)
        self.valid_count = int(stats.valid_count)
        self.register_buffer("mean", stats.mean.detach().float().clone())
        self.register_buffer("covariance", stats.covariance.detach().float().clone())
        self.register_buffer("eigvals", stats.eigvals.detach().float().clone())
        self.register_buffer("eigvecs", stats.eigvecs.detach().float().clone())
        self.register_buffer("whiten_matrix", stats.whiten.detach().float().clone())
        self.register_buffer("unwhiten_matrix", stats.unwhiten.detach().float().clone())

    def whiten(self, z_rec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _apply_whitening_transform(
            z_rec,
            mask,
            matrix=self.whiten_matrix,
            mean=self.mean,
            subtract_mean=True,
            z_dim=self.z_dim,
            name="z_rec",
        )

    def unwhiten(self, z_white: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _apply_whitening_transform(
            z_white,
            mask,
            matrix=self.unwhiten_matrix,
            mean=self.mean,
            subtract_mean=False,
            z_dim=self.z_dim,
            name="z_white",
        )

    def diagnostics(self) -> dict[str, torch.Tensor]:
        post_cov = self.whiten_matrix @ self.covariance @ self.whiten_matrix.transpose(0, 1)
        post_eigvals = torch.linalg.eigvalsh(post_cov.float()).clamp_min(0.0)
        pre_eigvals = self.eigvals.float().clamp_min(0.0)
        raw_cov_in_fit_basis = self.eigvecs.transpose(0, 1) @ self.covariance @ self.eigvecs
        raw_eigvals = torch.diagonal(raw_cov_in_fit_basis).float().clamp_min(0.0)
        active_mask = raw_eigvals > self.eps
        active_count = active_mask.sum()
        floored_count = self.z_dim - active_count
        active_basis = self.eigvecs[:, active_mask]
        if active_basis.shape[1] > 0:
            active_post_cov = active_basis.transpose(0, 1) @ post_cov @ active_basis
            active_post_cov = 0.5 * (active_post_cov + active_post_cov.transpose(0, 1))
            active_post_eigvals = torch.linalg.eigvalsh(active_post_cov.float()).clamp_min(0.0)
            active_post_condition = _condition_number_from_eigvals(
                active_post_eigvals,
                torch.finfo(active_post_eigvals.dtype).eps,
            )
        else:
            active_post_eigvals = post_eigvals.new_zeros((0,))
            active_post_condition = post_eigvals.new_zeros(())
        return {
            "mode": self.mean.new_tensor(0) if self.mode == "zca_full" else self.mean.new_tensor(-1),
            "valid_count": self.mean.new_tensor(self.valid_count, dtype=torch.long),
            "pre_condition_number": _condition_number_from_eigvals(pre_eigvals, self.eps),
            "post_condition_number": _condition_number_from_eigvals(post_eigvals, self.eps),
            "pre_effective_rank": _effective_rank_from_eigvals(pre_eigvals, self.eps),
            "post_effective_rank": _effective_rank_from_eigvals(post_eigvals, self.eps),
            "raw_effective_rank": _effective_rank_from_eigvals(raw_eigvals, self.eps),
            "active_post_condition_number": active_post_condition,
            "active_dimension_count": active_count,
            "floored_dimension_count": floored_count,
            "active_dimension_fraction": active_count.float() / float(self.z_dim),
            "rank_deficient": floored_count > 0,
            "pre_mean_abs": self.mean.abs().mean(),
            "post_covariance": post_cov,
            "post_eigvals": post_eigvals,
            "raw_eigvals": raw_eigvals,
            "active_dimension_mask": active_mask,
            "active_post_eigvals": active_post_eigvals,
        }

    def to_stats(self) -> Stage1LatentWhiteningStats:
        return Stage1LatentWhiteningStats(
            mean=self.mean.detach().cpu(),
            covariance=self.covariance.detach().cpu(),
            eigvals=self.eigvals.detach().cpu(),
            eigvecs=self.eigvecs.detach().cpu(),
            whiten=self.whiten_matrix.detach().cpu(),
            unwhiten=self.unwhiten_matrix.detach().cpu(),
            eps=self.eps,
            mode=self.mode,
            z_dim=self.z_dim,
            valid_count=self.valid_count,
        )


def fit_stage1_latent_whitening_from_values(
    values: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    mode: str = "zca_full",
) -> Stage1LatentWhiteningStats:
    if mode != "zca_full":
        raise ValueError("fit_stage1_latent_whitening_from_values only supports mode='zca_full'.")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    if values.ndim != 2:
        raise ValueError(f"values must have shape [M,D], got {tuple(values.shape)}.")
    if values.shape[0] < 2:
        raise ValueError("values must contain at least two valid latent rows.")
    if values.shape[1] <= 0:
        raise ValueError("values must have a positive latent dimension.")
    if not torch.isfinite(values).all():
        raise ValueError("values contains non-finite entries.")

    work = values.detach().to(dtype=torch.float64, device="cpu")
    valid_count, z_dim = work.shape
    mean = work.mean(dim=0)
    centered = work - mean
    covariance = centered.transpose(0, 1).matmul(centered) / float(max(valid_count - 1, 1))
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    eigvals = eigvals.clamp_min(float(eps))
    whiten = eigvecs.matmul(torch.diag(eigvals.rsqrt())).matmul(eigvecs.transpose(0, 1))
    unwhiten = eigvecs.matmul(torch.diag(eigvals.sqrt())).matmul(eigvecs.transpose(0, 1))
    stats = Stage1LatentWhiteningStats(
        mean=mean.float(),
        covariance=covariance.float(),
        eigvals=eigvals.float(),
        eigvecs=eigvecs.float(),
        whiten=whiten.float(),
        unwhiten=unwhiten.float(),
        eps=float(eps),
        mode=mode,
        z_dim=int(z_dim),
        valid_count=int(valid_count),
    )
    validate_stage1_latent_whitening_stats(stats)
    return stats


def make_identity_stage1_latent_whitening_stats(
    z_dim: int,
    *,
    eps: float = 1.0e-6,
    valid_count: int = 0,
) -> Stage1LatentWhiteningStats:
    """Build a no-op whitening artifact that preserves the Stage2 source protocol."""

    if z_dim <= 0:
        raise ValueError(f"z_dim must be positive, got {z_dim}.")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    eye = torch.eye(int(z_dim), dtype=torch.float32)
    stats = Stage1LatentWhiteningStats(
        mean=torch.zeros(int(z_dim), dtype=torch.float32),
        covariance=eye.clone(),
        eigvals=torch.ones(int(z_dim), dtype=torch.float32),
        eigvecs=eye.clone(),
        whiten=eye.clone(),
        unwhiten=eye.clone(),
        eps=float(eps),
        mode="zca_full",
        z_dim=int(z_dim),
        valid_count=int(valid_count),
    )
    validate_stage1_latent_whitening_stats(stats)
    return stats


def save_stage1_latent_whitening_stats(
    stats: Stage1LatentWhiteningStats,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    validate_stage1_latent_whitening_stats(stats)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats.to_payload(metadata=metadata), target)


def load_stage1_latent_whitening_transform(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Stage1LatentWhiteningTransform:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected whitening payload dict, got {type(payload)!r}.")
    return Stage1LatentWhiteningTransform(Stage1LatentWhiteningStats.from_payload(payload))


def validate_stage1_latent_whitening_stats(stats: Stage1LatentWhiteningStats) -> None:
    if stats.mode != "zca_full":
        raise ValueError(f"Only mode='zca_full' is supported, got {stats.mode!r}.")
    if stats.eps <= 0.0:
        raise ValueError(f"eps must be positive, got {stats.eps}.")
    if stats.z_dim <= 0:
        raise ValueError(f"z_dim must be positive, got {stats.z_dim}.")
    if stats.valid_count < 0:
        raise ValueError(f"valid_count must be non-negative, got {stats.valid_count}.")
    d = int(stats.z_dim)
    _check_vector(stats.mean, d, "mean")
    _check_vector(stats.eigvals, d, "eigvals")
    for name, value in [
        ("covariance", stats.covariance),
        ("eigvecs", stats.eigvecs),
        ("whiten", stats.whiten),
        ("unwhiten", stats.unwhiten),
    ]:
        _check_square(value, d, name)
    if (stats.eigvals < 0).any():
        raise ValueError("eigvals must be non-negative after eps clamping.")


def _apply_whitening_transform(
    z: torch.Tensor,
    mask: torch.Tensor,
    *,
    matrix: torch.Tensor,
    mean: torch.Tensor,
    subtract_mean: bool,
    z_dim: int,
    name: str,
) -> torch.Tensor:
    _validate_latent_and_mask(z, mask, z_dim, name)
    original_dtype = z.dtype
    work = z.float()
    mat = matrix.to(device=z.device, dtype=torch.float32)
    mu = mean.to(device=z.device, dtype=torch.float32)
    if subtract_mean:
        work = work - mu
    out = work.matmul(mat.transpose(0, 1))
    if not subtract_mean:
        out = out + mu
    return zero_invalid_tokens(out.to(dtype=original_dtype), mask.to(torch.bool))


def _validate_latent_and_mask(z: torch.Tensor, mask: torch.Tensor, z_dim: int, name: str) -> None:
    if z.ndim != 3:
        raise ValueError(f"{name} must have shape [B,N,D], got {tuple(z.shape)}.")
    if z.shape[-1] != z_dim:
        raise ValueError(f"{name} last dim must be {z_dim}, got {z.shape[-1]}.")
    if tuple(mask.shape) != tuple(z.shape[:2]):
        raise ValueError(f"mask must have shape {tuple(z.shape[:2])}, got {tuple(mask.shape)}.")
    if not torch.isfinite(z).all():
        raise ValueError(f"{name} contains non-finite values.")


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")
    return value.detach().clone()


def _check_vector(value: torch.Tensor, z_dim: int, name: str) -> None:
    if tuple(value.shape) != (z_dim,):
        raise ValueError(f"{name} must have shape ({z_dim},), got {tuple(value.shape)}.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")


def _check_square(value: torch.Tensor, z_dim: int, name: str) -> None:
    if tuple(value.shape) != (z_dim, z_dim):
        raise ValueError(f"{name} must have shape ({z_dim}, {z_dim}), got {tuple(value.shape)}.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")


def _condition_number_from_eigvals(eigvals: torch.Tensor, eps: float) -> torch.Tensor:
    positive = eigvals[eigvals > eps]
    if positive.numel() == 0:
        return eigvals.new_zeros(())
    return positive.max() / positive.min().clamp_min(eps)


def _effective_rank_from_eigvals(eigvals: torch.Tensor, eps: float) -> torch.Tensor:
    total = eigvals.sum()
    if total <= eps:
        return eigvals.new_zeros(())
    probs = eigvals / total.clamp_min(eps)
    entropy = -(probs * (probs + eps).log()).sum()
    return entropy.exp()
