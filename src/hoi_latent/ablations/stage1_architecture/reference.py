from __future__ import annotations

import torch
from torch import nn

from hoi_latent.ablations.stage1_architecture.types import Stage1ArchitectureAblationOutput
from hoi_latent.models.stage1.autoencoder.latent_bottleneck import validate_latent_field_output
from hoi_latent.models.stage1.decoder.decoder import Stage1DecoderModule, build_stage1_decoder_from_config
from hoi_latent.models.stage1.decoder.readout_heads import validate_stage1_coarse_prediction
from hoi_latent.models.stage1.encoder.reconstruction_encoder import Stage1EncoderModule, build_stage1_encoder_from_config
from hoi_latent.models.stage1.tokenizers import Stage1TokenizerModule, build_stage1_tokenizer_from_config


class Stage1ReferenceAutoEncoder(nn.Module):
    """Production MoT autoencoder without the optional relation refiner."""

    checkpoint_key = "stage1_architecture_ablation"
    output_key = "stage1_no_refiner_reconstruction"
    variant_name = "full_mot_reference"
    layout = "production_mot_no_refiner"

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
    ) -> Stage1ArchitectureAblationOutput:
        if target_frame_mask is None:
            masks = batch.get("masks")
            if not isinstance(masks, dict):
                raise KeyError("Expected batch['masks'] for Stage1 autoencoder.")
            for key in ("frame", "valid", "object"):
                value = masks.get(key)
                if isinstance(value, torch.Tensor) and value.ndim == 2:
                    target_frame_mask = value.to(dtype=torch.bool)
                    break
            if target_frame_mask is None:
                raise KeyError("Expected a frame, valid, or object mask in batch['masks'].")
        if target_num_frames is None:
            target_num_frames = int(target_frame_mask.shape[1])
        tokenizer_outputs = self.tokenizer(
            batch, return_aux=return_diagnostics, return_concat=False, validate=validate
        )
        latent = self.encoder(
            tokenizer_outputs, batch=batch, return_diagnostics=return_diagnostics, validate=validate
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
            "stage1_architecture_ablation": {
                "checkpoint_key": self.checkpoint_key,
                "output_key": self.output_key,
                "variant": self.variant_name,
                "layout": self.layout,
                "no_refiner": True,
                "target_num_frames": int(target_num_frames),
                "diagnostics_enabled": bool(return_diagnostics),
                "submodules": {
                    "tokenizer": type(self.tokenizer).__name__,
                    "encoder": type(self.encoder).__name__,
                    "decoder": type(self.decoder).__name__,
                },
                "route_decoupling": "ffn_only",
            },
        }
        output = Stage1ArchitectureAblationOutput(
            latent=latent,
            prediction=prediction,
            aux=aux,
            tokenizer_outputs=tokenizer_outputs if return_tokenizer_outputs else None,
        )
        if validate:
            validate_latent_field_output(output.latent)
            validate_stage1_coarse_prediction(output.prediction)
        return output

    def component_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            self.tokenizer.checkpoint_key: self.tokenizer.state_dict(),
            self.encoder.checkpoint_key: self.encoder.state_dict(),
            self.decoder.checkpoint_key: self.decoder.state_dict(),
        }


def build_stage1_reference_from_config(config: dict, batch: dict | None = None) -> Stage1ReferenceAutoEncoder:
    return Stage1ReferenceAutoEncoder(
        tokenizer=build_stage1_tokenizer_from_config(config, batch=batch),
        encoder=build_stage1_encoder_from_config(config),
        decoder=build_stage1_decoder_from_config(config),
    )
