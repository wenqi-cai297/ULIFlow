from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch

from hoi_latent.models.stage1.tokenizer_types import ModalityId, TokenTypeId


class Stage2FlowRouteId(IntEnum):
    LATENT_GENERATED = 0
    TEXT_CONDITION = 1
    OBJECT_GEOMETRY_CONDITION = 2


NUM_STAGE2_FLOW_ROUTES = max(int(item) for item in Stage2FlowRouteId) + 1


class Stage2LatentSubstreamId(IntEnum):
    HUMAN_DYNAMIC = 0
    OBJECT_POSE = 1
    OBJECT_GEOMETRY_LATENT = 2
    RELATION_DYNAMIC = 3
    FLAT_DYNAMIC = 4


NUM_STAGE2_LATENT_SUBSTREAMS = max(int(item) for item in Stage2LatentSubstreamId) + 1


@dataclass
class Stage2TextStemOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    pooled: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    flow_route_ids: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage2GeometryStemOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    pooled: torch.Tensor
    source_mask: torch.Tensor
    missing: torch.Tensor
    flow_route_ids: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage2LatentStemOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    time_ids: torch.Tensor
    token_type_ids: torch.Tensor
    entity_ids: torch.Tensor
    flow_route_ids: torch.Tensor
    apply_temporal_rope: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage2PackedStreams:
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    flow_route_ids: torch.Tensor
    time_ids: torch.Tensor
    apply_temporal_rope: torch.Tensor
    latent_slice: slice
    text_slice: slice
    geometry_slice: slice
    pooled_text: torch.Tensor
    pooled_geometry: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


def sanitize_ids_for_embedding(
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    pad_id: int = 0,
    num_embeddings: int | None = None,
    name: str = "ids",
) -> torch.Tensor:
    if ids.shape != mask.shape:
        raise ValueError(f"{name} shape must match mask shape, got {tuple(ids.shape)} and {tuple(mask.shape)}.")
    out = ids.to(dtype=torch.long).clone()
    mask_bool = mask.to(dtype=torch.bool)
    if num_embeddings is not None:
        if pad_id < 0 or pad_id >= num_embeddings:
            raise ValueError(f"pad_id={pad_id} is outside embedding range [0, {num_embeddings}).")
        valid_ids = out[mask_bool]
        if valid_ids.numel():
            bad = (valid_ids < 0) | (valid_ids >= num_embeddings)
            if bad.any():
                bad_values = torch.unique(valid_ids[bad]).detach().cpu().tolist()
                raise ValueError(f"{name} contains out-of-range valid ids: {bad_values}.")
    out[~mask_bool] = int(pad_id)
    return out


def derive_stage2_latent_substream_ids(
    modality_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if modality_ids.shape != token_type_ids.shape:
        raise ValueError(
            "modality_ids and token_type_ids must have the same shape, "
            f"got {tuple(modality_ids.shape)} and {tuple(token_type_ids.shape)}."
        )
    if mask is None:
        valid = torch.ones_like(token_type_ids, dtype=torch.bool)
    else:
        if mask.shape != token_type_ids.shape:
            raise ValueError(f"mask must have shape {tuple(token_type_ids.shape)}, got {tuple(mask.shape)}.")
        valid = mask.to(dtype=torch.bool)

    substream = torch.zeros_like(token_type_ids, dtype=torch.long)
    human = (modality_ids == int(ModalityId.HUMAN)) & valid
    object_pose = (token_type_ids == int(TokenTypeId.OBJECT_POSE)) & valid
    object_geometry = (token_type_ids == int(TokenTypeId.OBJECT_GEOMETRY)) & valid
    relation = (modality_ids == int(ModalityId.RELATION)) & valid
    flat_dynamic = (modality_ids < 0) & (token_type_ids < 0) & valid
    substream[human] = int(Stage2LatentSubstreamId.HUMAN_DYNAMIC)
    substream[object_pose] = int(Stage2LatentSubstreamId.OBJECT_POSE)
    substream[object_geometry] = int(Stage2LatentSubstreamId.OBJECT_GEOMETRY_LATENT)
    substream[relation] = int(Stage2LatentSubstreamId.RELATION_DYNAMIC)
    substream[flat_dynamic] = int(Stage2LatentSubstreamId.FLAT_DYNAMIC)

    assigned = human | object_pose | object_geometry | relation | flat_dynamic
    bad = valid & ~assigned
    if bad.any():
        bad_types = torch.unique(token_type_ids[bad]).detach().cpu().tolist()
        bad_modalities = torch.unique(modality_ids[bad]).detach().cpu().tolist()
        raise ValueError(
            "Unknown Stage2 latent substream for valid metadata: "
            f"token_type_ids={bad_types}, modality_ids={bad_modalities}."
        )
    return substream
