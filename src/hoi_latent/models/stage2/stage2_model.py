from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from hoi_latent.models.stage2.backbone import (
    Stage2TriStreamMoTBackbone,
    Stage2TriStreamMoTOutput,
    build_stage2_tri_stream_mot_backbone_from_config,
)
from hoi_latent.models.stage2.flow_time import Stage2FlowTimeField
from hoi_latent.models.stage2.packing import Stage2StreamPacker
from hoi_latent.models.stage2.stem_types import (
    Stage2GeometryStemOutput,
    Stage2LatentStemOutput,
    Stage2PackedStreams,
    Stage2TextStemOutput,
)
from hoi_latent.models.stage2.stems import (
    LatentNoisyStreamEncoder,
    ObjectGeometryConditionEncoder,
    TextConditionEncoder,
)


@dataclass
class Stage2TriStreamMoTModelOutput:
    velocity: torch.Tensor
    backbone: Stage2TriStreamMoTOutput
    text: Stage2TextStemOutput | None = None
    geometry: Stage2GeometryStemOutput | None = None
    latent: Stage2LatentStemOutput | None = None
    packed: Stage2PackedStreams | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Stage2TriStreamMoTModel(nn.Module):
    """Thin wrapper for Stage2 tri-stream MoT velocity prediction.

    This module wires the three source stems, stream packer, and backbone. It
    does not construct RF targets, compute losses, apply CFG guidance, or run a
    sampler.
    """

    checkpoint_key = "stage2_tri_stream_mot"

    def __init__(
        self,
        text_encoder: TextConditionEncoder,
        geometry_encoder: ObjectGeometryConditionEncoder,
        latent_encoder: LatentNoisyStreamEncoder,
        packer: Stage2StreamPacker,
        backbone: Stage2TriStreamMoTBackbone,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.geometry_encoder = geometry_encoder
        self.latent_encoder = latent_encoder
        self.packer = packer
        self.backbone = backbone

    def forward(
        self,
        batch: dict,
        z_white_t: torch.Tensor,
        flow_time: Stage2FlowTimeField,
        latent_mask: torch.Tensor,
        modality_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        time_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        time_mask: torch.Tensor,
        *,
        force_null_text_mask: torch.Tensor | None = None,
        force_null_geometry_mask: torch.Tensor | None = None,
        return_intermediates: bool = False,
        return_diagnostics: bool = False,
    ) -> Stage2TriStreamMoTModelOutput:
        latent = self.latent_encoder(
            z_white_t=z_white_t,
            latent_mask=latent_mask,
            modality_ids=modality_ids,
            token_type_ids=token_type_ids,
            time_ids=time_ids,
            entity_ids=entity_ids,
            time_mask=time_mask,
            return_diagnostics=return_diagnostics,
        )
        text = self.text_encoder(
            batch,
            force_null_text_mask=force_null_text_mask,
            return_diagnostics=return_diagnostics,
        )
        geometry = self.geometry_encoder(
            batch,
            force_null_geometry_mask=force_null_geometry_mask,
            return_diagnostics=return_diagnostics,
        )
        packed = self.packer(latent, text, geometry)
        backbone_out = self.backbone(packed, flow_time, return_diagnostics=return_diagnostics)
        diagnostics = {}
        if return_diagnostics:
            diagnostics = {
                "text": text.diagnostics,
                "geometry": geometry.diagnostics,
                "latent": latent.diagnostics,
                "pack": packed.diagnostics,
                "backbone": backbone_out.diagnostics,
            }
        return Stage2TriStreamMoTModelOutput(
            velocity=backbone_out.velocity,
            backbone=backbone_out,
            text=text if return_intermediates else None,
            geometry=geometry if return_intermediates else None,
            latent=latent if return_intermediates else None,
            packed=packed if return_intermediates else None,
            diagnostics=diagnostics,
        )


def build_stage2_text_condition_encoder_from_config(
    config: Mapping[str, Any],
    *,
    text_model: nn.Module | None = None,
    clip_model: nn.Module | None = None,
    tokenizer: Any | None = None,
) -> TextConditionEncoder:
    stage2_cfg = config.get("model", {}).get("stage2", {})
    text_cfg = stage2_cfg.get("text", {})
    d_model = int(stage2_cfg.get("d_model", 384))
    return TextConditionEncoder(
        d_model=d_model,
        num_heads=int(text_cfg.get("num_heads", stage2_cfg.get("num_heads", 6))),
        backbone=str(text_cfg.get("backbone", "openai/clip-vit-base-patch32")),
        backend=str(text_cfg.get("backend", "clip")),
        freeze_clip=_optional_bool(text_cfg.get("freeze_clip")),
        freeze_text_encoder=_optional_bool(text_cfg.get("freeze_text_encoder")),
        max_seq_len=_optional_int(text_cfg.get("max_seq_len")),
        clip_dim=_optional_int(text_cfg.get("clip_dim")),
        encoder_dim=_optional_int(text_cfg.get("encoder_dim")),
        encoder_dtype=str(text_cfg.get("encoder_dtype", "auto")),
        contextualizer_depth=int(text_cfg.get("contextualizer_depth", 2)),
        contextualizer_dropout=float(text_cfg.get("contextualizer_dropout", stage2_cfg.get("dropout", 0.0))),
        residual_gate_init=float(text_cfg.get("residual_gate_init", 0.0)),
        null_prompt=str(text_cfg.get("null_prompt", "")),
        text_model=text_model,
        clip_model=clip_model,
        tokenizer=tokenizer,
    )


def build_stage2_object_geometry_condition_encoder_from_config(
    config: Mapping[str, Any],
) -> ObjectGeometryConditionEncoder:
    stage2_cfg = config.get("model", {}).get("stage2", {})
    geometry_cfg = stage2_cfg.get("geometry", {})
    return ObjectGeometryConditionEncoder(
        d_model=int(stage2_cfg.get("d_model", 384)),
        num_heads=int(geometry_cfg.get("num_heads", stage2_cfg.get("num_heads", 6))),
        max_bps_points=int(geometry_cfg.get("max_bps_points", 1024)),
        num_geometry_tokens=int(geometry_cfg.get("num_geometry_tokens", 16)),
        dropout=float(geometry_cfg.get("dropout", stage2_cfg.get("dropout", 0.0))),
        allow_missing_bps=bool(geometry_cfg.get("allow_missing_bps", True)),
    )


def build_stage2_latent_noisy_stream_encoder_from_config(
    config: Mapping[str, Any],
) -> LatentNoisyStreamEncoder:
    stage2_cfg = config.get("model", {}).get("stage2", {})
    latent_cfg = stage2_cfg.get("latent", {})
    return LatentNoisyStreamEncoder(
        d_model=int(stage2_cfg.get("d_model", 384)),
        latent_dim=int(stage2_cfg.get("latent_dim", 16)),
        num_entity_embeddings=int(latent_cfg.get("num_entity_embeddings", 1024)),
        use_latent_substream_embedding=bool(latent_cfg.get("use_latent_substream_embedding", False)),
        allow_unknown_token_type_ids=bool(latent_cfg.get("allow_unknown_token_type_ids", False)),
    )


def build_stage2_tri_stream_mot_model_from_config(
    config: Mapping[str, Any],
    *,
    text_model: nn.Module | None = None,
    clip_model: nn.Module | None = None,
    tokenizer: Any | None = None,
) -> Stage2TriStreamMoTModel:
    return Stage2TriStreamMoTModel(
        text_encoder=build_stage2_text_condition_encoder_from_config(
            config,
            text_model=text_model,
            clip_model=clip_model,
            tokenizer=tokenizer,
        ),
        geometry_encoder=build_stage2_object_geometry_condition_encoder_from_config(config),
        latent_encoder=build_stage2_latent_noisy_stream_encoder_from_config(config),
        packer=Stage2StreamPacker(),
        backbone=build_stage2_tri_stream_mot_backbone_from_config(config),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
