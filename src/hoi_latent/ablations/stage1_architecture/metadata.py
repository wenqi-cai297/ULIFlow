from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from torch import nn


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if not trainable_only or parameter.requires_grad)


PRODUCTION_NO_REFINER_AUTOENCODER_PARAMS = 60_398_694
PRODUCTION_N_LATENT = 708
PRODUCTION_Z_DIM = 16
PRODUCTION_LATENT_CAPACITY = PRODUCTION_N_LATENT * PRODUCTION_Z_DIM


class ArchitectureAblationMetadataReporter:
    def __init__(
        self,
        *,
        production_no_refiner_params: int = PRODUCTION_NO_REFINER_AUTOENCODER_PARAMS,
        production_n_latent: int = PRODUCTION_N_LATENT,
        production_z_dim: int = PRODUCTION_Z_DIM,
    ) -> None:
        self.production_no_refiner_params = int(production_no_refiner_params)
        self.production_n_latent = int(production_n_latent)
        self.production_z_dim = int(production_z_dim)

    def report(
        self,
        model: nn.Module,
        *,
        variant_name: str,
        n_variant: int,
        z_dim_variant: int,
        input_token_count_before_compression: int | None = None,
        decoder_memory_token_count: int | None = None,
        decoder_output_query_count: int | None = None,
        route_decoupling: str | None = None,
        calibration: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        variant_params = count_parameters(model)
        c_prod = self.production_n_latent * self.production_z_dim
        c_variant = int(n_variant) * int(z_dim_variant)
        metadata: dict[str, Any] = {
            "variant_name": variant_name,
            "production_no_refiner_autoencoder_params": self.production_no_refiner_params,
            "variant_params": variant_params,
            "absolute_param_difference": variant_params - self.production_no_refiner_params,
            "relative_param_difference": abs(variant_params - self.production_no_refiner_params)
            / float(self.production_no_refiner_params),
            "N_prod": self.production_n_latent,
            "z_dim_prod": self.production_z_dim,
            "C_prod": c_prod,
            "N_variant": int(n_variant),
            "z_dim_variant": int(z_dim_variant),
            "C_variant": c_variant,
            "latent_capacity_relative_difference": abs(c_variant - c_prod) / float(c_prod),
            "input_token_count_before_compression": input_token_count_before_compression,
            "decoder_memory_token_count": decoder_memory_token_count,
            "decoder_output_query_count": decoder_output_query_count,
            "route_decoupling": route_decoupling,
        }
        if calibration is not None:
            metadata["calibration"] = asdict(calibration) if is_dataclass(calibration) else calibration
        if extra:
            metadata.update(extra)
        return metadata
