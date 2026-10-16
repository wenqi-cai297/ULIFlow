from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hoi_latent.models.stage1.autoencoder import (
    Stage1LatentFieldOutput,
    validate_latent_field_output,
)
from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.models.stage1.tokenizer_types import TokenizerOutput, validate_tokenizer_output


def _build_input_norm(input_norm: str, z_dim: int) -> nn.Module:
    if input_norm == "identity":
        return nn.Identity()
    if input_norm == "layernorm":
        return nn.LayerNorm(z_dim)
    raise ValueError("input_norm must be 'identity' or 'layernorm'.")


def _build_activation(activation: str) -> nn.Module:
    if activation == "silu":
        return nn.SiLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError("activation must be 'silu' or 'gelu'.")


class SharedNonlinearLatentUpNeck(nn.Module):
    """Shared per-token nonlinear expansion from z_rec channels to decoder width."""

    def __init__(
        self,
        z_dim: int = 16,
        d_model: int = 256,
        taper_dim: int = 64,
        activation: str = "silu",
        input_norm: str = "identity",
    ) -> None:
        super().__init__()
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}.")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if taper_dim <= 0:
            raise ValueError(f"taper_dim must be positive, got {taper_dim}.")
        self.z_dim = int(z_dim)
        self.d_model = int(d_model)
        self.taper_dim = int(taper_dim)
        self.activation_name = activation
        self.input_norm_type = input_norm
        self.input_norm = _build_input_norm(input_norm, z_dim)
        self.up_in = nn.Linear(z_dim, taper_dim)
        self.activation = _build_activation(activation)
        self.up_out = nn.Linear(taper_dim, d_model)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_latent_values(z, mask, self.z_dim, "z")
        z = zero_invalid_tokens(z, mask)
        z = self.input_norm(z)
        h = self.up_in(z)
        h = self.activation(h)
        x = self.up_out(h)
        return zero_invalid_tokens(x, mask)


class SharedLinearLatentUpNeck(nn.Module):
    """Shared linear up-neck kept as an ablation."""

    def __init__(
        self,
        z_dim: int = 16,
        d_model: int = 256,
        input_norm: str = "identity",
    ) -> None:
        super().__init__()
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}.")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        self.z_dim = int(z_dim)
        self.d_model = int(d_model)
        self.input_norm_type = input_norm
        self.input_norm = _build_input_norm(input_norm, z_dim)
        self.projection = nn.Linear(z_dim, d_model)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_latent_values(z, mask, self.z_dim, "z")
        z = zero_invalid_tokens(z, mask)
        z = self.input_norm(z)
        x = self.projection(z)
        return zero_invalid_tokens(x, mask)


class Stage1DecoderUpNeckModule(nn.Module):
    output_key = "decoder_input"

    def __init__(
        self,
        z_dim: int = 16,
        d_model: int = 256,
        mode: str = "shared_nonlinear_taper",
        taper_dim: int = 64,
        activation: str = "silu",
        input_norm: str = "identity",
    ) -> None:
        super().__init__()
        if mode not in {"shared_nonlinear_taper", "shared_linear"}:
            raise ValueError("mode must be 'shared_nonlinear_taper' or 'shared_linear'.")
        self.z_dim = int(z_dim)
        self.d_model = int(d_model)
        self.mode = mode
        self.input_norm_type = input_norm
        self.activation_name = activation if mode == "shared_nonlinear_taper" else None
        self.taper_dim = int(taper_dim) if mode == "shared_nonlinear_taper" else None
        if mode == "shared_nonlinear_taper":
            self.up_neck: nn.Module = SharedNonlinearLatentUpNeck(
                z_dim=z_dim,
                d_model=d_model,
                taper_dim=taper_dim,
                activation=activation,
                input_norm=input_norm,
            )
        else:
            self.up_neck = SharedLinearLatentUpNeck(
                z_dim=z_dim,
                d_model=d_model,
                input_norm=input_norm,
            )

    def forward(
        self,
        latent: Stage1LatentFieldOutput,
        z_override: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        validate: bool = False,
    ) -> TokenizerOutput:
        if validate:
            validate_latent_field_output(latent)
        z = latent.z_rec if z_override is None else z_override
        if tuple(z.shape) != tuple(latent.z_rec.shape):
            raise ValueError(
                f"z_override must match latent.z_rec shape {tuple(latent.z_rec.shape)}, got {tuple(z.shape)}."
            )
        mask = latent.mask.to(torch.bool)
        _validate_latent_values(z, mask, self.z_dim, "latent.z_rec" if z_override is None else "z_override")

        tokens = self.up_neck(z, mask)
        tokens = zero_invalid_tokens(tokens, mask)
        aux: dict[str, Any] = {
            **latent.aux,
            "decoder_up_neck": {
                "mode": self.mode,
                "z_dim": self.z_dim,
                "d_model": self.d_model,
                "taper_dim": self.taper_dim,
                "activation": self.activation_name,
                "input_norm": self.input_norm_type,
                "used_z_override": z_override is not None,
                "normalization": "none",
                "consumes_normalized_z_rec": True,
                "diagnostics_enabled": bool(return_diagnostics),
            },
        }
        out = TokenizerOutput(
            tokens=tokens,
            mask=mask,
            modality_ids=latent.modality_ids,
            token_type_ids=latent.token_type_ids,
            time_ids=latent.time_ids,
            entity_ids=latent.entity_ids,
            time_mask=latent.time_mask,
            aux=aux,
        )
        if validate:
            validate_tokenizer_output(out, "decoder_up_neck")
        return out


def build_stage1_decoder_up_neck_from_config(config: dict) -> Stage1DecoderUpNeckModule:
    stage1_cfg = config.get("model", {}).get("stage1", {})
    decoder_cfg = stage1_cfg.get("decoder", {})
    up_cfg = decoder_cfg.get("up_neck", {})
    latent_cfg = stage1_cfg.get("latent_bottleneck", {})
    z_dim = int(up_cfg.get("z_dim", latent_cfg.get("z_dim", 16)))
    d_model = int(up_cfg.get("d_model", stage1_cfg.get("d_model", 256)))
    taper_dim = int(up_cfg.get("taper_dim", 64))
    mode = up_cfg.get("mode", "shared_nonlinear_taper")
    activation = up_cfg.get("activation", "silu")
    input_norm = up_cfg.get("input_norm", "identity")
    if z_dim <= 0:
        raise ValueError("decoder.up_neck.z_dim must be positive.")
    if d_model <= 0:
        raise ValueError("decoder.up_neck.d_model must be positive.")
    if taper_dim <= 0:
        raise ValueError("decoder.up_neck.taper_dim must be positive.")
    if mode not in {"shared_nonlinear_taper", "shared_linear"}:
        raise ValueError("decoder.up_neck.mode must be 'shared_nonlinear_taper' or 'shared_linear'.")
    if activation not in {"silu", "gelu"}:
        raise ValueError("decoder.up_neck.activation must be 'silu' or 'gelu'.")
    if input_norm not in {"identity", "layernorm"}:
        raise ValueError("decoder.up_neck.input_norm must be 'identity' or 'layernorm'.")
    return Stage1DecoderUpNeckModule(
        z_dim=z_dim,
        d_model=d_model,
        mode=mode,
        taper_dim=taper_dim,
        activation=activation,
        input_norm=input_norm,
    )


def _validate_latent_values(z: torch.Tensor, mask: torch.Tensor, z_dim: int, name: str) -> None:
    if z.ndim != 3:
        raise ValueError(f"{name} must have shape [B,N,d_z], got {tuple(z.shape)}.")
    if z.shape[-1] != z_dim:
        raise ValueError(f"{name} last dim must be {z_dim}, got {z.shape[-1]}.")
    if tuple(mask.shape) != tuple(z.shape[:2]):
        raise ValueError(f"mask must have shape {tuple(z.shape[:2])}, got {tuple(mask.shape)}.")
    if not torch.isfinite(z).all():
        raise ValueError(f"{name} contains non-finite values.")
