from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import nn

from hoi_latent.models.stage1.tokenizer_types import (
    ModalityId,
    TokenTypeId,
    TokenizerOutput,
    concat_tokenizer_outputs,
)


class HOISubstreamId(IntEnum):
    HUMAN_DYNAMIC = 0
    OBJECT_POSE = 1
    OBJECT_GEOMETRY = 2
    RELATION_DYNAMIC = 3


HOI_SUBSTREAM_NAMES: tuple[str, ...] = (
    "human_dynamic",
    "object_pose",
    "object_geometry",
    "relation_dynamic",
)


@dataclass
class Stage1HOIPack:
    tokens: torch.Tensor
    mask: torch.Tensor
    substream_ids: torch.Tensor
    modality_ids: torch.Tensor
    token_type_ids: torch.Tensor
    time_ids: torch.Tensor
    entity_ids: torch.Tensor
    time_mask: torch.Tensor
    apply_temporal_rope: torch.Tensor
    stream_slices: dict[str, slice]


class Stage1HOISequencePacker(nn.Module):
    """Pack compressed H/O/R TokenizerOutputs into one metadata-rich HOI sequence."""

    input_keys = ("human", "object", "relation")

    def forward(self, compressed_outputs: dict[str, TokenizerOutput]) -> tuple[Stage1HOIPack, TokenizerOutput]:
        missing = [key for key in self.input_keys if key not in compressed_outputs]
        if missing:
            raise KeyError(f"Stage1HOISequencePacker missing required outputs: {missing}.")
        ordered = [compressed_outputs[key] for key in self.input_keys]
        self._validate_object_layout(compressed_outputs["object"])
        concat = concat_tokenizer_outputs(ordered)
        time_mask = concat.time_mask
        if time_mask is None:
            time_mask = concat.mask.clone()
        substream_ids = derive_hoi_substream_ids(concat.modality_ids, concat.token_type_ids)
        apply_temporal_rope = (
            time_mask.to(dtype=torch.bool)
            & concat.mask.to(dtype=torch.bool)
            & (concat.token_type_ids != int(TokenTypeId.OBJECT_GEOMETRY))
        )
        lengths = {key: compressed_outputs[key].tokens.shape[1] for key in self.input_keys}
        start = 0
        stream_slices: dict[str, slice] = {}
        for key in self.input_keys:
            end = start + lengths[key]
            stream_slices[key] = slice(start, end)
            start = end
        pack = Stage1HOIPack(
            tokens=concat.tokens,
            mask=concat.mask,
            substream_ids=substream_ids,
            modality_ids=concat.modality_ids,
            token_type_ids=concat.token_type_ids,
            time_ids=concat.time_ids,
            entity_ids=concat.entity_ids,
            time_mask=time_mask,
            apply_temporal_rope=apply_temporal_rope,
            stream_slices=stream_slices,
        )
        return pack, concat

    @staticmethod
    def _validate_object_layout(out: TokenizerOutput) -> None:
        token_type_ids = out.token_type_ids
        if token_type_ids.ndim != 2:
            raise ValueError(f"object.token_type_ids must be [B,N], got {tuple(token_type_ids.shape)}.")
        first = token_type_ids[0]
        if not torch.equal(token_type_ids, first[None, :].expand_as(token_type_ids)):
            raise ValueError("JointMoT object token_type_ids layout must be identical across batch.")
        pose = first == int(TokenTypeId.OBJECT_POSE)
        geometry = first == int(TokenTypeId.OBJECT_GEOMETRY)
        if not pose.any() or not geometry.any():
            raise ValueError("JointMoT object stream requires both OBJECT_POSE and OBJECT_GEOMETRY tokens.")
        first_geometry = int(torch.nonzero(geometry, as_tuple=False)[0].item())
        if pose[first_geometry:].any():
            raise ValueError("JointMoT object stream must order OBJECT_POSE tokens before OBJECT_GEOMETRY tokens.")


def derive_hoi_substream_ids(modality_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
    if modality_ids.shape != token_type_ids.shape:
        raise ValueError("modality_ids and token_type_ids must have the same shape.")
    substream_ids = torch.full_like(token_type_ids, fill_value=-1)
    human = modality_ids == int(ModalityId.HUMAN)
    object_pose = token_type_ids == int(TokenTypeId.OBJECT_POSE)
    object_geometry = token_type_ids == int(TokenTypeId.OBJECT_GEOMETRY)
    relation = modality_ids == int(ModalityId.RELATION)
    substream_ids[human] = int(HOISubstreamId.HUMAN_DYNAMIC)
    substream_ids[object_pose] = int(HOISubstreamId.OBJECT_POSE)
    substream_ids[object_geometry] = int(HOISubstreamId.OBJECT_GEOMETRY)
    substream_ids[relation] = int(HOISubstreamId.RELATION_DYNAMIC)
    if (substream_ids < 0).any():
        bad = torch.unique(token_type_ids[substream_ids < 0]).tolist()
        raise ValueError(f"Unknown Stage1 JointMoT route for token_type_ids={bad}.")
    return substream_ids


def build_pack_metadata(pack: Stage1HOIPack, include_diagnostics: bool = False) -> dict:
    metadata = {
        "stream_slices": {key: (value.start, value.stop) for key, value in pack.stream_slices.items()},
        "substream_names": HOI_SUBSTREAM_NAMES,
    }
    if include_diagnostics:
        metadata.update(
            {
                "substream_token_counts": {
                    name: (pack.substream_ids == idx).sum(dim=1)
                    for idx, name in enumerate(HOI_SUBSTREAM_NAMES)
                },
                "apply_temporal_rope_count": pack.apply_temporal_rope.sum(dim=1),
            }
        )
    return metadata
