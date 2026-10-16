from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hoi_latent.models.stage1.autoencoder.latent_bottleneck import Stage1LatentFieldOutput
from hoi_latent.models.stage1.decoder.joint_stack import (
    Stage1DecoderJointStack,
    build_stage1_decoder_joint_stack_from_config,
)
from hoi_latent.models.stage1.decoder.readout_heads import (
    Stage1CoarsePrediction,
    Stage1CoarseReadoutHeads,
    build_stage1_coarse_readout_heads_from_config,
    validate_stage1_coarse_prediction,
)
from hoi_latent.models.stage1.decoder.temporal_densifier import (
    Stage1TemporalDensifier,
    build_stage1_temporal_densifier_from_config,
)
from hoi_latent.models.stage1.decoder.up_neck import (
    Stage1DecoderUpNeckModule,
    build_stage1_decoder_up_neck_from_config,
)


class Stage1DecoderModule(nn.Module):
    """Stage1 decoder entry point from z_rec to structured reconstruction."""

    checkpoint_key = "decoder"
    output_key = "prediction"

    def __init__(
        self,
        up_neck: Stage1DecoderUpNeckModule,
        joint_stack: Stage1DecoderJointStack,
        temporal_densifier: Stage1TemporalDensifier,
        readout_heads: Stage1CoarseReadoutHeads,
    ) -> None:
        super().__init__()
        self.up_neck = up_neck
        self.joint_stack = joint_stack
        self.temporal_densifier = temporal_densifier
        self.readout_heads = readout_heads

    def forward(
        self,
        latent: Stage1LatentFieldOutput,
        z_override: torch.Tensor | None = None,
        target_num_frames: int | None = None,
        target_frame_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        validate: bool = False,
    ) -> Stage1CoarsePrediction:
        decoder_input = self.up_neck(
            latent,
            z_override=z_override,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
        decoder_joint = self.joint_stack(
            decoder_input,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
        dense_tokens = self.temporal_densifier(
            decoder_joint,
            target_num_frames=target_num_frames,
            target_frame_mask=target_frame_mask,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
        prediction = self.readout_heads(
            dense_tokens,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
        decoder_aux: dict[str, Any] = {
            "checkpoint_key": self.checkpoint_key,
            "output_key": self.output_key,
            "used_z_override": z_override is not None,
            "target_num_frames": int(dense_tokens.aux["temporal_densifier"]["target_num_frames"]),
            "diagnostics_enabled": bool(return_diagnostics),
            "submodules": {
                "up_neck": type(self.up_neck).__name__,
                "joint_stack": type(self.joint_stack).__name__,
                "temporal_densifier": type(self.temporal_densifier).__name__,
                "readout_heads": type(self.readout_heads).__name__,
            },
        }
        aux: dict[str, Any] = {
            **prediction.aux,
            "decoder": decoder_aux,
        }
        out = Stage1CoarsePrediction(
            human=prediction.human,
            object=prediction.object,
            relation=prediction.relation,
            masks=prediction.masks,
            aux=aux,
        )
        if validate:
            validate_stage1_coarse_prediction(out)
        return out


def build_stage1_decoder_from_config(config: dict) -> Stage1DecoderModule:
    return Stage1DecoderModule(
        up_neck=build_stage1_decoder_up_neck_from_config(config),
        joint_stack=build_stage1_decoder_joint_stack_from_config(config),
        temporal_densifier=build_stage1_temporal_densifier_from_config(config),
        readout_heads=build_stage1_coarse_readout_heads_from_config(config),
    )
