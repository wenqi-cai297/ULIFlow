from __future__ import annotations

from typing import Any, Optional

from torch import nn

from hoi_latent.models.stage1.autoencoder.latent_bottleneck import (
    Stage1LatentFieldOutput,
    Stage1ZRecLatentFieldBottleneck,
    build_stage1_latent_bottleneck_from_config,
    validate_latent_field_output,
)
from hoi_latent.models.stage1.encoder.joint_mot import (
    Stage1JointMoTEncoderModule,
    build_stage1_joint_mot_encoder_from_config,
)
from hoi_latent.models.stage1.encoder.stem import (
    Stage1StemEncoderModule,
    build_stage1_stem_encoder_from_config,
)
from hoi_latent.models.stage1.encoder.token_compressor import (
    Stage1TokenCompressorAndBudgeter,
    build_stage1_token_compressor_from_config,
)
from hoi_latent.models.stage1.tokenizer_types import TokenizerOutput


class Stage1EncoderModule(nn.Module):
    """Stage1 reconstruction encoder from tokenizer outputs to z_rec."""

    checkpoint_key = "encoder"
    output_key = "z_rec"

    def __init__(
        self,
        stem_encoder: Stage1StemEncoderModule,
        token_compressor: Stage1TokenCompressorAndBudgeter,
        joint_encoder: Stage1JointMoTEncoderModule,
        latent_bottleneck: Stage1ZRecLatentFieldBottleneck,
    ) -> None:
        super().__init__()
        self.stem_encoder = stem_encoder
        self.token_compressor = token_compressor
        self.joint_encoder = joint_encoder
        self.latent_bottleneck = latent_bottleneck

    def forward(
        self,
        tokenizer_outputs: dict[str, TokenizerOutput],
        batch: Optional[dict] = None,
        return_diagnostics: bool = False,
        validate: bool = False,
    ) -> Stage1LatentFieldOutput:
        stem_outputs = self.stem_encoder(tokenizer_outputs, validate=validate)
        compressed_outputs = self.token_compressor(stem_outputs, batch=batch, validate=validate)
        joint_outputs = self.joint_encoder(
            compressed_outputs,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
        latent = self.latent_bottleneck(
            joint_outputs["joint"],
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
        aux: dict[str, Any] = {
            **latent.aux,
            "encoder": {
                "checkpoint_key": self.checkpoint_key,
                "output_key": self.output_key,
                "diagnostics_enabled": bool(return_diagnostics),
                "compressed_token_count": int(joint_outputs["joint"].tokens.shape[1]),
                "z_dim": int(latent.z_rec.shape[-1]),
                "submodules": {
                    "stem_encoder": type(self.stem_encoder).__name__,
                    "token_compressor": type(self.token_compressor).__name__,
                    "joint_encoder": type(self.joint_encoder).__name__,
                    "latent_bottleneck": type(self.latent_bottleneck).__name__,
                },
            },
        }
        out = Stage1LatentFieldOutput(
            z_rec=latent.z_rec,
            z_raw=latent.z_raw,
            mask=latent.mask,
            modality_ids=latent.modality_ids,
            token_type_ids=latent.token_type_ids,
            time_ids=latent.time_ids,
            entity_ids=latent.entity_ids,
            time_mask=latent.time_mask,
            aux=aux,
        )
        if validate:
            validate_latent_field_output(out)
        return out


def build_stage1_encoder_from_config(config: dict) -> Stage1EncoderModule:
    return Stage1EncoderModule(
        stem_encoder=build_stage1_stem_encoder_from_config(config),
        token_compressor=build_stage1_token_compressor_from_config(config),
        joint_encoder=build_stage1_joint_mot_encoder_from_config(config),
        latent_bottleneck=build_stage1_latent_bottleneck_from_config(config),
    )
