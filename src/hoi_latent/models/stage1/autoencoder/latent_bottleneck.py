from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.models.stage1.tokenizer_types import (
    ModalityId,
    TokenizerOutput,
    TokenTypeId,
    validate_tokenizer_output,
)


@dataclass
class Stage1LatentFieldOutput:
    z_rec: torch.Tensor
    z_raw: torch.Tensor
    mask: torch.Tensor
    modality_ids: torch.Tensor
    token_type_ids: torch.Tensor
    time_ids: torch.Tensor
    entity_ids: torch.Tensor
    time_mask: torch.Tensor
    aux: dict[str, Any] = field(default_factory=dict)


class EMALatentNormalizer(nn.Module):
    """Masked EMA mean/std normalizer for the Stage1 reconstruction latent."""

    def __init__(
        self,
        z_dim: int,
        momentum: float = 0.99,
        eps: float = 1.0e-5,
        update_with_stopgrad: bool = True,
    ) -> None:
        super().__init__()
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if not update_with_stopgrad:
            raise ValueError("EMALatentNormalizer v1 requires update_with_stopgrad=True.")
        self.z_dim = z_dim
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.update_with_stopgrad = bool(update_with_stopgrad)
        self.register_buffer("running_mean", torch.zeros(z_dim))
        self.register_buffer("running_var", torch.ones(z_dim))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def forward(self, z_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_latent_tensor_and_mask(z_raw, mask, "z_raw")
        if z_raw.shape[-1] != self.z_dim:
            raise ValueError(f"z_raw last dim must be {self.z_dim}, got {z_raw.shape[-1]}.")
        if self.training:
            self.update(z_raw, mask)
        mean = self.running_mean.to(device=z_raw.device, dtype=z_raw.dtype)
        var = self.running_var.to(device=z_raw.device, dtype=z_raw.dtype)
        return (z_raw - mean.view(1, 1, -1)) * torch.rsqrt(var.view(1, 1, -1) + self.eps)

    @torch.no_grad()
    def update(self, z_raw: torch.Tensor, mask: torch.Tensor) -> None:
        valid = mask.to(torch.bool)
        if not valid.any():
            return
        values = z_raw.detach().float()[valid]
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False).clamp_min(self.eps)
        if int(self.num_updates.item()) == 0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var)
        else:
            self.running_mean.mul_(self.momentum).add_(batch_mean, alpha=1.0 - self.momentum)
            self.running_var.mul_(self.momentum).add_(batch_var, alpha=1.0 - self.momentum)
        self.num_updates.add_(1)


class IdentityLatentNormalizer(nn.Module):
    """No-op latent normalizer for latent-regularization ablations."""

    def __init__(self, z_dim: int) -> None:
        super().__init__()
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}.")
        self.z_dim = int(z_dim)
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def forward(self, z_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_latent_tensor_and_mask(z_raw, mask, "z_raw")
        if z_raw.shape[-1] != self.z_dim:
            raise ValueError(f"z_raw last dim must be {self.z_dim}, got {z_raw.shape[-1]}.")
        return zero_invalid_tokens(z_raw, mask.to(torch.bool))


def _build_input_norm(input_norm: str, d_model: int) -> nn.Module:
    if input_norm == "identity":
        return nn.Identity()
    if input_norm == "layernorm":
        return nn.LayerNorm(d_model)
    raise ValueError("input_norm must be 'identity' or 'layernorm'.")


def _build_activation(activation: str) -> nn.Module:
    if activation == "silu":
        return nn.SiLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError("activation must be 'silu' or 'gelu'.")


class SharedNonlinearLatentDownNeck(nn.Module):
    """Shared per-token nonlinear channel taper for Stage1 z_rec."""

    def __init__(
        self,
        d_model: int,
        z_dim: int,
        taper_dim: int = 64,
        activation: str = "silu",
        input_norm: str = "identity",
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}.")
        if taper_dim <= 0:
            raise ValueError(f"taper_dim must be positive, got {taper_dim}.")
        self.d_model = int(d_model)
        self.z_dim = int(z_dim)
        self.taper_dim = int(taper_dim)
        self.input_norm_type = input_norm
        self.activation_name = activation
        self.input_norm = _build_input_norm(input_norm, d_model)
        self.taper_in = nn.Linear(d_model, taper_dim)
        self.activation = _build_activation(activation)
        self.taper_out = nn.Linear(taper_dim, z_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B,N,D], got {tuple(x.shape)}.")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"x last dim must be {self.d_model}, got {x.shape[-1]}.")
        if tuple(mask.shape) != tuple(x.shape[:2]):
            raise ValueError(f"mask must have shape {tuple(x.shape[:2])}, got {tuple(mask.shape)}.")
        x = zero_invalid_tokens(x, mask)
        x = self.input_norm(x)
        h = self.taper_in(x)
        h = self.activation(h)
        z = self.taper_out(h)
        return zero_invalid_tokens(z, mask)


class SharedLinearLatentProjector(nn.Module):
    """Shared linear latent projector kept for ablation."""

    def __init__(
        self,
        d_model: int,
        z_dim: int,
        input_norm: str = "identity",
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}.")
        self.d_model = int(d_model)
        self.z_dim = int(z_dim)
        self.input_norm_type = input_norm
        self.input_norm = _build_input_norm(input_norm, d_model)
        self.projection = nn.Linear(d_model, z_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B,N,D], got {tuple(x.shape)}.")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"x last dim must be {self.d_model}, got {x.shape[-1]}.")
        if tuple(mask.shape) != tuple(x.shape[:2]):
            raise ValueError(f"mask must have shape {tuple(x.shape[:2])}, got {tuple(mask.shape)}.")
        x = zero_invalid_tokens(x, mask)
        x = self.input_norm(x)
        z = self.projection(x)
        return zero_invalid_tokens(z, mask)


class Stage1ZRecLatentFieldBottleneck(nn.Module):
    checkpoint_key = "latent_bottleneck"
    output_key = "z_rec"

    def __init__(
        self,
        d_model: int = 256,
        z_dim: int = 16,
        projection_mode: str = "shared_nonlinear_taper",
        input_norm: str = "identity",
        taper_dim: int = 64,
        activation: str = "silu",
        normalization_type: str = "ema_mean_std",
        ema_momentum: float = 0.99,
        ema_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}.")
        self.d_model = int(d_model)
        self.z_dim = int(z_dim)
        if projection_mode not in {"shared_nonlinear_taper", "shared_linear"}:
            raise ValueError("projection_mode must be 'shared_nonlinear_taper' or 'shared_linear'.")
        self.projection_mode = projection_mode
        self.input_norm_type = input_norm
        self.activation_name = activation if projection_mode == "shared_nonlinear_taper" else None
        self.taper_dim = int(taper_dim) if projection_mode == "shared_nonlinear_taper" else None
        if projection_mode == "shared_nonlinear_taper":
            self.latent_projector: nn.Module = SharedNonlinearLatentDownNeck(
                d_model=d_model,
                z_dim=z_dim,
                taper_dim=taper_dim,
                activation=activation,
                input_norm=input_norm,
            )
        else:
            self.latent_projector = SharedLinearLatentProjector(
                d_model=d_model,
                z_dim=z_dim,
                input_norm=input_norm,
            )
        if normalization_type == "none":
            normalization_type = "identity"
        if normalization_type not in {"ema_mean_std", "identity"}:
            raise ValueError("normalization_type must be 'ema_mean_std', 'identity', or 'none'.")
        self.normalization_type = normalization_type
        if normalization_type == "ema_mean_std":
            self.latent_normalizer: nn.Module = EMALatentNormalizer(
                z_dim=z_dim,
                momentum=ema_momentum,
                eps=ema_eps,
                update_with_stopgrad=True,
            )
        else:
            self.latent_normalizer = IdentityLatentNormalizer(z_dim=z_dim)

    def forward(
        self,
        joint: TokenizerOutput,
        return_diagnostics: bool = False,
        validate: bool = False,
    ) -> Stage1LatentFieldOutput:
        if validate:
            validate_tokenizer_output(joint, "joint")
        if joint.tokens.shape[-1] != self.d_model:
            raise ValueError(f"joint.tokens last dim must be {self.d_model}, got {joint.tokens.shape[-1]}.")
        mask = joint.mask.to(torch.bool)
        time_mask = joint.time_mask.to(torch.bool) if joint.time_mask is not None else mask.clone()

        x = zero_invalid_tokens(joint.tokens, mask)
        z_raw = self.latent_projector(x, mask)
        z_raw = zero_invalid_tokens(z_raw, mask)
        z_rec = self.latent_normalizer(z_raw, mask)
        z_rec = zero_invalid_tokens(z_rec, mask)

        latent_aux: dict[str, Any] = {
            "d_model": self.d_model,
            "z_dim": self.z_dim,
            "projection_mode": self.projection_mode,
            "input_norm": self.input_norm_type,
            "activation": self.activation_name,
            "taper_dim": self.taper_dim,
            "normalization": self.normalization_type,
            "diagnostics_enabled": bool(return_diagnostics),
            "ema_num_updates": self.latent_normalizer.num_updates.detach().clone(),
        }
        if joint.time_mask is None:
            latent_aux["time_mask_fallback"] = "mask"
        if return_diagnostics:
            latent_aux["diagnostics"] = {
                "z_raw": compute_aggregate_latent_stats(
                    z_raw.detach(),
                    mask,
                    modality_ids=joint.modality_ids,
                    token_type_ids=joint.token_type_ids,
                    time_ids=joint.time_ids,
                ),
                "z_rec": compute_aggregate_latent_stats(
                    z_rec.detach(),
                    mask,
                    modality_ids=joint.modality_ids,
                    token_type_ids=joint.token_type_ids,
                    time_ids=joint.time_ids,
                ),
            }

        out = Stage1LatentFieldOutput(
            z_rec=z_rec,
            z_raw=z_raw,
            mask=mask,
            modality_ids=joint.modality_ids,
            token_type_ids=joint.token_type_ids,
            time_ids=joint.time_ids,
            entity_ids=joint.entity_ids,
            time_mask=time_mask,
            aux={**joint.aux, "latent_bottleneck": latent_aux},
        )
        if validate:
            validate_latent_field_output(out)
        return out


def validate_latent_field_output(out: Stage1LatentFieldOutput, name: str = "Stage1LatentFieldOutput") -> None:
    if out.z_rec.ndim != 3:
        raise ValueError(f"{name}.z_rec must be [B,N,d_z], got {tuple(out.z_rec.shape)}.")
    if tuple(out.z_raw.shape) != tuple(out.z_rec.shape):
        raise ValueError(f"{name}.z_raw must match z_rec shape, got {tuple(out.z_raw.shape)}.")
    batch_size, num_tokens, _ = out.z_rec.shape
    expected = (batch_size, num_tokens)
    for field_name in ["mask", "modality_ids", "token_type_ids", "time_ids", "entity_ids", "time_mask"]:
        value = getattr(out, field_name)
        if tuple(value.shape) != expected:
            raise ValueError(f"{name}.{field_name} must have shape {expected}, got {tuple(value.shape)}.")
    if (out.time_ids < 0).any():
        raise ValueError(f"{name}.time_ids contains negative ids.")
    if not torch.isfinite(out.z_raw).all():
        raise ValueError(f"{name}.z_raw contains non-finite values.")
    if not torch.isfinite(out.z_rec).all():
        raise ValueError(f"{name}.z_rec contains non-finite values.")
    invalid = ~out.mask.to(torch.bool)
    if invalid.any() and out.z_raw[invalid].abs().max().item() != 0.0:
        raise ValueError(f"{name}.z_raw has non-zero invalid tokens.")
    if invalid.any() and out.z_rec[invalid].abs().max().item() != 0.0:
        raise ValueError(f"{name}.z_rec has non-zero invalid tokens.")


def compute_aggregate_latent_stats(
    z_rec: torch.Tensor,
    mask: torch.Tensor,
    *,
    modality_ids: torch.Tensor | None = None,
    token_type_ids: torch.Tensor | None = None,
    time_ids: torch.Tensor | None = None,
    outlier_abs_threshold: float = 4.0,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    _validate_latent_tensor_and_mask(z_rec, mask, "z_rec")
    valid = mask.to(torch.bool)
    values = z_rec[valid]
    z_dim = z_rec.shape[-1]
    if values.numel() == 0:
        zero = z_rec.new_zeros(())
        return {
            "valid_count": z_rec.new_tensor(0, dtype=torch.long),
            "mean": z_rec.new_zeros(z_dim),
            "std": z_rec.new_zeros(z_dim),
            "covariance": z_rec.new_zeros(z_dim, z_dim),
            "covariance_condition_number": zero,
            "effective_rank": zero,
            "outlier_ratio": zero,
            "offdiag_covariance_energy": zero,
            "object_geometry_retained_variance": zero,
            "same_anchor_human_relation_latent_correlation": zero,
            "same_anchor_object_pose_relation_consistency": zero,
        }

    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    covariance = _covariance(values)
    diag = torch.diagonal(covariance)
    offdiag = covariance - torch.diag_embed(diag)
    condition_number, effective_rank = _covariance_shape_stats(covariance, eps=eps)
    outlier_ratio = (values.abs() > outlier_abs_threshold).any(dim=-1).to(z_rec.dtype).mean()

    geometry_variance = z_rec.new_zeros(())
    human_relation = z_rec.new_zeros(())
    object_relation = z_rec.new_zeros(())
    if modality_ids is not None and token_type_ids is not None:
        _validate_metadata_shape(modality_ids, mask, "modality_ids")
        _validate_metadata_shape(token_type_ids, mask, "token_type_ids")
        geometry = valid & (token_type_ids == int(TokenTypeId.OBJECT_GEOMETRY))
        if geometry.any():
            geometry_variance = z_rec[geometry].var(dim=0, unbiased=False).mean()
        if time_ids is not None:
            _validate_metadata_shape(time_ids, mask, "time_ids")
            human_relation = _same_anchor_cosine(
                z_rec,
                valid,
                time_ids,
                modality_ids == int(ModalityId.HUMAN),
                modality_ids == int(ModalityId.RELATION),
            )
            object_relation = _same_anchor_cosine(
                z_rec,
                valid,
                time_ids,
                token_type_ids == int(TokenTypeId.OBJECT_POSE),
                modality_ids == int(ModalityId.RELATION),
            )

    return {
        "valid_count": z_rec.new_tensor(values.shape[0], dtype=torch.long),
        "mean": mean,
        "std": std,
        "covariance": covariance,
        "covariance_condition_number": condition_number,
        "effective_rank": effective_rank,
        "outlier_ratio": outlier_ratio,
        "offdiag_covariance_energy": _offdiag_square_mean(offdiag),
        "object_geometry_retained_variance": geometry_variance,
        "same_anchor_human_relation_latent_correlation": human_relation,
        "same_anchor_object_pose_relation_consistency": object_relation,
    }


def aggregate_mean_covariance_regularization(
    z_rec: torch.Tensor,
    mask: torch.Tensor,
    *,
    mean_weight: float = 1.0,
    offdiag_weight: float = 1.0,
    diag_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    _validate_latent_tensor_and_mask(z_rec, mask, "z_rec")
    values = z_rec[mask.to(torch.bool)]
    if values.numel() == 0:
        zero = z_rec.sum() * 0.0
        return {"total": zero, "mean": zero, "cov_offdiag": zero, "cov_diag": zero}
    mean = values.mean(dim=0)
    covariance = _covariance(values)
    diag = torch.diagonal(covariance)
    offdiag = covariance - torch.diag_embed(diag)
    mean_loss = mean.square().mean()
    offdiag_loss = _offdiag_square_mean(offdiag)
    diag_loss = (diag - 1.0).square().mean()
    total = mean_weight * mean_loss + offdiag_weight * offdiag_loss + diag_weight * diag_loss
    return {
        "total": total,
        "mean": mean_loss,
        "cov_offdiag": offdiag_loss,
        "cov_diag": diag_loss,
    }


def add_decoder_latent_jitter(
    z_rec: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
    training: bool,
) -> torch.Tensor:
    _validate_latent_tensor_and_mask(z_rec, mask, "z_rec")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}.")
    if not training or sigma == 0.0:
        return zero_invalid_tokens(z_rec, mask)
    jittered = z_rec + torch.randn_like(z_rec) * float(sigma)
    return zero_invalid_tokens(jittered, mask)


def build_stage1_latent_bottleneck_from_config(config: dict) -> Stage1ZRecLatentFieldBottleneck:
    stage1_cfg = config.get("model", {}).get("stage1", {})
    latent_cfg = stage1_cfg.get("latent_bottleneck", {})
    d_model = int(latent_cfg.get("d_model", stage1_cfg.get("d_model", 256)))
    z_dim = int(latent_cfg.get("z_dim", 16))
    if latent_cfg.get("keep_token_count", True) is not True:
        raise ValueError("latent_bottleneck.keep_token_count must be true for v1.")

    projection_cfg = latent_cfg.get("projection", {})
    projection_mode = projection_cfg.get("mode", "shared_nonlinear_taper")
    if projection_mode not in {"shared_nonlinear_taper", "shared_linear"}:
        raise ValueError(
            "latent_bottleneck.projection.mode v1 supports "
            "'shared_nonlinear_taper' or 'shared_linear'."
        )
    input_norm = projection_cfg.get("input_norm", "identity")
    if input_norm not in {"identity", "layernorm"}:
        raise ValueError("latent_bottleneck.projection.input_norm v1 supports 'identity' or 'layernorm'.")
    activation = projection_cfg.get("activation", "silu")
    if activation not in {"silu", "gelu"}:
        raise ValueError("latent_bottleneck.projection.activation v1 supports 'silu' or 'gelu'.")
    taper_dim = int(projection_cfg.get("taper_dim", 64))
    if taper_dim <= 0:
        raise ValueError("latent_bottleneck.projection.taper_dim must be positive.")
    mixer_depth = int(projection_cfg.get("post_projection_mixer_depth", 0))
    if mixer_depth != 0:
        raise ValueError("latent_bottleneck.projection.post_projection_mixer_depth must be 0 in v1.")

    normalization_cfg = latent_cfg.get("normalization", {})
    normalization_type = normalization_cfg.get("type", "ema_mean_std")
    if normalization_type not in {"ema_mean_std", "identity", "none"}:
        raise ValueError(
            "latent_bottleneck.normalization.type v1 supports 'ema_mean_std', 'identity', or 'none'."
        )
    if normalization_type == "ema_mean_std" and normalization_cfg.get("update_with_stopgrad", True) is not True:
        raise ValueError("latent_bottleneck.normalization.update_with_stopgrad must be true.")

    return Stage1ZRecLatentFieldBottleneck(
        d_model=d_model,
        z_dim=z_dim,
        projection_mode=projection_mode,
        input_norm=input_norm,
        taper_dim=taper_dim,
        activation=activation,
        normalization_type=normalization_type,
        ema_momentum=float(normalization_cfg.get("momentum", 0.99)),
        ema_eps=float(normalization_cfg.get("eps", 1.0e-5)),
    )


def _validate_latent_tensor_and_mask(z: torch.Tensor, mask: torch.Tensor, name: str) -> None:
    if z.ndim != 3:
        raise ValueError(f"{name} must have shape [B,N,D], got {tuple(z.shape)}.")
    if tuple(mask.shape) != tuple(z.shape[:2]):
        raise ValueError(f"mask must have shape {tuple(z.shape[:2])}, got {tuple(mask.shape)}.")
    if not torch.isfinite(z).all():
        raise ValueError(f"{name} contains non-finite values.")


def _validate_metadata_shape(value: torch.Tensor, mask: torch.Tensor, name: str) -> None:
    if tuple(value.shape) != tuple(mask.shape):
        raise ValueError(f"{name} must have shape {tuple(mask.shape)}, got {tuple(value.shape)}.")


def _covariance(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    denom = max(values.shape[0] - 1, 1)
    return centered.transpose(0, 1).matmul(centered) / denom


def _offdiag_square_mean(offdiag: torch.Tensor) -> torch.Tensor:
    if offdiag.ndim != 2 or offdiag.shape[0] != offdiag.shape[1]:
        raise ValueError(f"offdiag must be square [D,D], got {tuple(offdiag.shape)}.")
    dim = offdiag.shape[0]
    if dim <= 1:
        return offdiag.sum() * 0.0
    return offdiag.square().sum() / float(dim * (dim - 1))


def _covariance_shape_stats(covariance: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if covariance.numel() == 0:
        zero = covariance.new_zeros(())
        return zero, zero
    eigvals = torch.linalg.eigvalsh(covariance.float()).to(device=covariance.device, dtype=covariance.dtype)
    eigvals = eigvals.clamp_min(0.0)
    total = eigvals.sum()
    if total <= eps:
        zero = covariance.new_zeros(())
        return zero, zero
    positive = eigvals[eigvals > eps]
    if positive.numel() == 0:
        condition = covariance.new_zeros(())
    else:
        condition = positive.max() / positive.min().clamp_min(eps)
    probs = eigvals / total.clamp_min(eps)
    entropy = -(probs * (probs + eps).log()).sum()
    effective_rank = entropy.exp()
    return condition, effective_rank


def _same_anchor_cosine(
    z: torch.Tensor,
    mask: torch.Tensor,
    time_ids: torch.Tensor,
    left_selector: torch.Tensor,
    right_selector: torch.Tensor,
) -> torch.Tensor:
    scores = []
    batch_size = z.shape[0]
    for batch_idx in range(batch_size):
        left = mask[batch_idx] & left_selector[batch_idx]
        right = mask[batch_idx] & right_selector[batch_idx]
        times = torch.unique(time_ids[batch_idx][left | right])
        for time in times:
            left_at_time = left & (time_ids[batch_idx] == time)
            right_at_time = right & (time_ids[batch_idx] == time)
            if not left_at_time.any() or not right_at_time.any():
                continue
            left_value = z[batch_idx, left_at_time].mean(dim=0)
            right_value = z[batch_idx, right_at_time].mean(dim=0)
            scores.append(F.cosine_similarity(left_value, right_value, dim=0))
    if not scores:
        return z.new_zeros(())
    return torch.stack(scores).mean()
