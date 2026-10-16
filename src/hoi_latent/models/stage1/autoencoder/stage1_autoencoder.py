from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import nn

from hoi_latent.models.stage1.autoencoder.latent_bottleneck import (
    Stage1LatentFieldOutput,
    validate_latent_field_output,
)
from hoi_latent.models.stage1.decoder.decoder import Stage1DecoderModule, build_stage1_decoder_from_config
from hoi_latent.models.stage1.decoder.readout_heads import (
    Stage1CoarsePrediction,
    validate_stage1_coarse_prediction,
)
from hoi_latent.models.stage1.encoder.reconstruction_encoder import (
    Stage1EncoderModule,
    build_stage1_encoder_from_config,
)
from hoi_latent.models.stage1.tokenizer_types import TokenizerOutput
from hoi_latent.models.stage1.tokenizers import Stage1TokenizerModule, build_stage1_tokenizer_from_config


@dataclass
class Stage1AutoencoderOutput:
    latent: Stage1LatentFieldOutput
    prediction: Stage1CoarsePrediction
    tokenizer_outputs: dict[str, TokenizerOutput] | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    @property
    def coarse(self) -> Stage1CoarsePrediction:
        return self.prediction

    @property
    def refined(self) -> Stage1CoarsePrediction:
        return self.prediction

    @property
    def human(self) -> dict[str, torch.Tensor]:
        return self.prediction.human

    @property
    def object(self) -> dict[str, torch.Tensor]:
        return self.prediction.object

    @property
    def masks(self) -> dict[str, torch.Tensor]:
        return self.prediction.masks


class Stage1AutoencoderModule(nn.Module):
    """Full Stage 1 reconstruction autoencoder boundary."""

    checkpoint_key = "stage1_autoencoder"
    output_key = "stage1_reconstruction"

    def __init__(
        self,
        tokenizer: Stage1TokenizerModule,
        encoder: Stage1EncoderModule,
        decoder: Stage1DecoderModule,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        batch: dict,
        *,
        z_override: torch.Tensor | None = None,
        target_num_frames: int | None = None,
        target_frame_mask: torch.Tensor | None = None,
        return_tokenizer_outputs: bool = False,
        return_diagnostics: bool = False,
        validate: bool = False,
    ) -> Stage1AutoencoderOutput:
        if target_frame_mask is None:
            target_frame_mask = _infer_frame_mask_from_batch(batch)
        if target_num_frames is None and target_frame_mask is not None:
            target_num_frames = int(target_frame_mask.shape[1])

        tokenizer_outputs = self.tokenizer(
            batch,
            return_aux=return_diagnostics,
            return_concat=False,
            validate=validate,
        )
        latent = self.encoder(
            tokenizer_outputs,
            batch=batch,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
        prediction = self.decoder(
            latent,
            z_override=z_override,
            target_num_frames=target_num_frames,
            target_frame_mask=target_frame_mask,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
        aux = {
            **prediction.aux,
            "stage1_autoencoder": {
                "checkpoint_key": self.checkpoint_key,
                "output_key": self.output_key,
                "diagnostics_enabled": bool(return_diagnostics),
                "returned_tokenizer_outputs": bool(return_tokenizer_outputs),
                "target_num_frames": int(prediction.aux["decoder"]["target_num_frames"]),
                "submodules": {
                    "tokenizer": type(self.tokenizer).__name__,
                    "encoder": type(self.encoder).__name__,
                    "decoder": type(self.decoder).__name__,
                },
                "component_checkpoint_keys": self.component_checkpoint_keys(),
                "refiner_removed": True,
            },
        }
        out = Stage1AutoencoderOutput(
            latent=latent,
            prediction=prediction,
            tokenizer_outputs=tokenizer_outputs if return_tokenizer_outputs else None,
            aux=aux,
        )
        if validate:
            validate_stage1_autoencoder_output(out)
        return out

    def component_checkpoint_keys(self) -> dict[str, str]:
        return {
            "tokenizer": self.tokenizer.checkpoint_key,
            "encoder": self.encoder.checkpoint_key,
            "decoder": self.decoder.checkpoint_key,
        }

    def component_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            self.tokenizer.checkpoint_key: self.tokenizer.state_dict(),
            self.encoder.checkpoint_key: self.encoder.state_dict(),
            self.decoder.checkpoint_key: self.decoder.state_dict(),
        }


def build_stage1_autoencoder_from_config(config: dict, batch: Optional[dict] = None) -> Stage1AutoencoderModule:
    return Stage1AutoencoderModule(
        tokenizer=build_stage1_tokenizer_from_config(config, batch=batch),
        encoder=build_stage1_encoder_from_config(config),
        decoder=build_stage1_decoder_from_config(config),
    )


def validate_stage1_autoencoder_output(output: Stage1AutoencoderOutput) -> None:
    validate_latent_field_output(output.latent)
    validate_stage1_coarse_prediction(output.prediction)
    if output.tokenizer_outputs is not None:
        for key in ("human", "object", "relation"):
            if key not in output.tokenizer_outputs:
                raise KeyError(f"Stage1 autoencoder tokenizer_outputs missing {key!r}.")
    for key in ("stage1_autoencoder", "decoder"):
        if key not in output.aux:
            raise KeyError(f"Stage1 autoencoder aux missing {key!r}.")


def _infer_frame_mask_from_batch(batch: dict) -> torch.Tensor | None:
    masks = batch.get("masks")
    if not isinstance(masks, dict):
        return None
    for key in ("frame", "valid", "object"):
        value = masks.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value.to(dtype=torch.bool)
    return None
