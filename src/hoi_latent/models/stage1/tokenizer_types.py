from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Optional

import torch


class ModalityId(IntEnum):
    HUMAN = 0
    OBJECT = 1
    RELATION = 2


class TokenTypeId(IntEnum):
    HUMAN_ROOT = 0
    HUMAN_BODY_SLOT = 1
    HUMAN_FOOT_LEFT = 2
    HUMAN_FOOT_RIGHT = 3
    OBJECT_POSE = 10
    OBJECT_GEOMETRY = 11
    RELATION_BODY_SLOT = 20
    RELATION_GROUND = 21
    RELATION_ROOT_OBJECT = 22


NUM_TOKEN_TYPES = max(int(item) for item in TokenTypeId) + 1


@dataclass
class TokenizerOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    modality_ids: torch.Tensor
    token_type_ids: torch.Tensor
    time_ids: torch.Tensor
    entity_ids: torch.Tensor
    time_mask: Optional[torch.Tensor] = None
    aux: dict[str, Any] = field(default_factory=dict)


def assert_shape(x: torch.Tensor, expected_rank: int, name: str) -> None:
    if x.ndim != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}, got {tuple(x.shape)}.")


def summarize_tokenizer_output(out: TokenizerOutput) -> dict:
    return {
        "tokens": tuple(out.tokens.shape),
        "mask": tuple(out.mask.shape),
        "valid_ratio": float(out.mask.float().mean().item()) if out.mask.numel() else 0.0,
        "time_mask_valid_ratio": (
            float(out.time_mask.float().mean().item())
            if out.time_mask is not None and out.time_mask.numel()
            else None
        ),
        "modality_ids_unique": sorted(int(v) for v in out.modality_ids.unique().tolist()),
        "token_type_ids_unique": sorted(int(v) for v in out.token_type_ids.unique().tolist()),
    }


def concat_tokenizer_outputs(outputs: Iterable[TokenizerOutput]) -> TokenizerOutput:
    outputs = list(outputs)
    if not outputs:
        raise ValueError("concat_tokenizer_outputs requires at least one output.")
    if all(out.time_mask is not None for out in outputs):
        time_mask = torch.cat([out.time_mask for out in outputs], dim=1)
    elif all(out.time_mask is None for out in outputs):
        time_mask = None
    else:
        pieces = [
            torch.ones_like(out.mask, dtype=torch.bool) if out.time_mask is None else out.time_mask
            for out in outputs
        ]
        time_mask = torch.cat(pieces, dim=1)
    return TokenizerOutput(
        tokens=torch.cat([out.tokens for out in outputs], dim=1),
        mask=torch.cat([out.mask for out in outputs], dim=1),
        modality_ids=torch.cat([out.modality_ids for out in outputs], dim=1),
        token_type_ids=torch.cat([out.token_type_ids for out in outputs], dim=1),
        time_ids=torch.cat([out.time_ids for out in outputs], dim=1),
        entity_ids=torch.cat([out.entity_ids for out in outputs], dim=1),
        time_mask=time_mask,
        aux={str(idx): out.aux for idx, out in enumerate(outputs)},
    )


def validate_tokenizer_output(out: TokenizerOutput, name: str = "TokenizerOutput") -> None:
    if out.tokens.ndim != 3:
        raise ValueError(f"{name}.tokens must be [B,N,D], got {tuple(out.tokens.shape)}.")
    batch_size, num_tokens, _ = out.tokens.shape
    expected = (batch_size, num_tokens)
    for field_name in ["mask", "modality_ids", "token_type_ids", "time_ids", "entity_ids"]:
        value = getattr(out, field_name)
        if tuple(value.shape) != expected:
            raise ValueError(f"{name}.{field_name} must have shape {expected}, got {tuple(value.shape)}.")
    if out.time_mask is not None and tuple(out.time_mask.shape) != expected:
        raise ValueError(f"{name}.time_mask must have shape {expected}, got {tuple(out.time_mask.shape)}.")
    if (out.time_ids < 0).any():
        raise ValueError(f"{name}.time_ids contains negative ids.")
    if not torch.isfinite(out.tokens).all():
        raise ValueError(f"{name}.tokens contains non-finite values.")
    invalid = ~out.mask
    if invalid.any() and out.tokens[invalid].abs().max().item() != 0.0:
        raise ValueError(f"{name}.tokens has non-zero invalid tokens.")
