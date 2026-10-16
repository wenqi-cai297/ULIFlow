from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from hoi_latent.models.stage1.encoder.transformer_blocks import (
    FactorizedTemporalSlotBlock,
    PreNormCrossAttentionBlock,
    PreNormSelfAttentionBlock,
)
from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.models.stage1.tokenizer_types import (
    TokenizerOutput,
    TokenTypeId,
    concat_tokenizer_outputs,
    validate_tokenizer_output,
)


class Stage1StemEncoderModule(nn.Module):
    checkpoint_key = "stem_encoder"
    output_keys = ("human", "object", "relation")
    concat_key = "concat"

    def __init__(self, human: "HumanStemEncoder", object: "ObjectStemEncoder", relation: "RelationStemEncoder"):
        super().__init__()
        self.human = human
        self.object = object
        self.relation = relation

    def forward(
        self,
        tokenizer_outputs: dict[str, TokenizerOutput],
        return_concat: bool = False,
        validate: bool = False,
    ) -> dict[str, TokenizerOutput]:
        missing = [key for key in self.output_keys if key not in tokenizer_outputs]
        if missing:
            raise KeyError(f"StemEncoder input is missing keys: {missing}.")
        outputs = {
            "human": self.human(tokenizer_outputs["human"]),
            "object": self.object(tokenizer_outputs["object"]),
            "relation": self.relation(tokenizer_outputs["relation"]),
        }
        if validate:
            self.validate_outputs(outputs)
        if return_concat:
            outputs[self.concat_key] = concat_tokenizer_outputs(outputs[key] for key in self.output_keys)
            if validate:
                validate_tokenizer_output(outputs[self.concat_key], f"{self.__class__.__name__}.{self.concat_key}")
        return outputs

    def validate_outputs(self, outputs: dict[str, TokenizerOutput]) -> None:
        for key in self.output_keys:
            if key not in outputs:
                raise KeyError(f"StemEncoder output is missing key {key!r}.")
            validate_tokenizer_output(outputs[key], f"{self.__class__.__name__}.{key}")


class HumanStemEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        depth: int = 2,
        slots_per_frame: int = 11,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.slots_per_frame = int(slots_per_frame)
        self.blocks = nn.ModuleList(
            [FactorizedTemporalSlotBlock(d_model, num_heads, dropout, ffn_mult) for _ in range(int(depth))]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, human_out: TokenizerOutput) -> TokenizerOutput:
        x_ts, mask_ts, time_ids_ts = _view_as_time_slots(human_out, self.slots_per_frame, "HumanStemEncoder")
        for block in self.blocks:
            x_ts = block(x_ts, mask_ts, time_ids_ts)
        tokens = self.final_norm(x_ts.reshape(human_out.tokens.shape))
        tokens = zero_invalid_tokens(tokens, human_out.mask)
        return replace_tokens_preserve_metadata(
            human_out,
            tokens,
            aux={
                **human_out.aux,
                "stem_attention": "factorized_temporal_slot",
                "stem_rope": "temporal_1d",
                "candidate_for_compression": True,
            },
        )


class RelationStemEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        depth: int = 2,
        slots_per_frame: int = 10,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.slots_per_frame = int(slots_per_frame)
        self.blocks = nn.ModuleList(
            [FactorizedTemporalSlotBlock(d_model, num_heads, dropout, ffn_mult) for _ in range(int(depth))]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.relation_saliency_head = nn.Linear(d_model, 1)

    def forward(self, relation_out: TokenizerOutput) -> TokenizerOutput:
        x_ts, mask_ts, time_ids_ts = _view_as_time_slots(relation_out, self.slots_per_frame, "RelationStemEncoder")
        for block in self.blocks:
            x_ts = block(x_ts, mask_ts, time_ids_ts)
        x = self.final_norm(x_ts.reshape(relation_out.tokens.shape))
        x_ts = x.reshape_as(x_ts)
        saliency = self.relation_saliency_head(x_ts).squeeze(-1)
        saliency = saliency * mask_ts.to(dtype=saliency.dtype)
        tokens = zero_invalid_tokens(x, relation_out.mask)
        return replace_tokens_preserve_metadata(
            relation_out,
            tokens,
            aux={
                **relation_out.aux,
                "stem_attention": "factorized_temporal_slot",
                "stem_rope": "temporal_1d",
                "temporal_saliency_logits": saliency,
                "candidate_for_compression": True,
            },
        )


class ObjectStemEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pose_depth: int = 2,
        geometry_depth: int = 1,
        pose_geometry_cross_depth: int = 1,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.pose_temporal_encoder = nn.ModuleList(
            [
                PreNormSelfAttentionBlock(d_model, num_heads, dropout, ffn_mult, use_rope=True)
                for _ in range(int(pose_depth))
            ]
        )
        self.geometry_encoder = nn.ModuleList(
            [
                PreNormSelfAttentionBlock(d_model, num_heads, dropout, ffn_mult, use_rope=False)
                for _ in range(int(geometry_depth))
            ]
        )
        self.pose_geometry_cross_attention = nn.ModuleList(
            [
                PreNormCrossAttentionBlock(d_model, num_heads, dropout, ffn_mult)
                for _ in range(int(pose_geometry_cross_depth))
            ]
        )
        self.pose_final_norm = nn.LayerNorm(d_model)
        self.geometry_final_norm = nn.LayerNorm(d_model)

    def forward(self, object_out: TokenizerOutput) -> TokenizerOutput:
        if object_out.time_mask is None:
            raise ValueError("ObjectStemEncoder requires object_out.time_mask.")
        pose_index = _static_type_index(object_out, TokenTypeId.OBJECT_POSE, "ObjectStemEncoder")
        geometry_index = _static_type_index(object_out, TokenTypeId.OBJECT_GEOMETRY, "ObjectStemEncoder")
        if not pose_index.any() or not geometry_index.any():
            raise ValueError("ObjectStemEncoder requires both OBJECT_POSE and OBJECT_GEOMETRY tokens.")

        pose = object_out.tokens[:, pose_index, :]
        geometry = object_out.tokens[:, geometry_index, :]
        pose_valid_mask = object_out.mask[:, pose_index]
        geometry_valid_mask = object_out.mask[:, geometry_index]
        pose_time_ids = object_out.time_ids[:, pose_index]

        for block in self.pose_temporal_encoder:
            pose = block(pose, pose_valid_mask, position_ids=pose_time_ids)
        for block in self.geometry_encoder:
            geometry = block(geometry, geometry_valid_mask)
        for block in self.pose_geometry_cross_attention:
            pose = block(
                query=pose,
                key_value=geometry,
                query_mask=pose_valid_mask,
                key_value_mask=geometry_valid_mask,
            )
        pose = zero_invalid_tokens(self.pose_final_norm(pose), pose_valid_mask)
        geometry = zero_invalid_tokens(self.geometry_final_norm(geometry), geometry_valid_mask)

        tokens = _merge_static_type_tokens_without_scatter(
            pose=pose,
            pose_index=pose_index,
            geometry=geometry,
            geometry_index=geometry_index,
            num_tokens=object_out.tokens.shape[1],
        ).to(dtype=object_out.tokens.dtype)
        tokens = zero_invalid_tokens(tokens, object_out.mask)
        return replace_tokens_preserve_metadata(
            object_out,
            tokens,
            aux={
                **object_out.aux,
                "stem_attention": "pose_temporal_geometry_memory",
                "stem_rope": "pose_temporal_1d",
                "static_geometry_time_mask": False,
                "candidate_for_compression": True,
            },
        )


def _merge_static_type_tokens_without_scatter(
    *,
    pose: torch.Tensor,
    pose_index: torch.Tensor,
    geometry: torch.Tensor,
    geometry_index: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    pose_positions = torch.nonzero(pose_index, as_tuple=False).flatten()
    geometry_positions = torch.nonzero(geometry_index, as_tuple=False).flatten()
    if pose_positions.numel() != pose.shape[1]:
        raise ValueError(
            f"pose token count mismatch: index has {pose_positions.numel()}, tensor has {pose.shape[1]}."
        )
    if geometry_positions.numel() != geometry.shape[1]:
        raise ValueError(
            f"geometry token count mismatch: index has {geometry_positions.numel()}, tensor has {geometry.shape[1]}."
        )
    pose_selector = torch.nn.functional.one_hot(pose_positions, num_classes=int(num_tokens)).to(
        device=pose.device,
        dtype=pose.dtype,
    )
    geometry_selector = torch.nn.functional.one_hot(geometry_positions, num_classes=int(num_tokens)).to(
        device=geometry.device,
        dtype=geometry.dtype,
    )
    return torch.einsum("btd,tn->bnd", pose, pose_selector) + torch.einsum(
        "bgd,gn->bnd",
        geometry,
        geometry_selector,
    )


def build_stage1_stem_encoder_from_config(config: dict) -> Stage1StemEncoderModule:
    stage1_cfg = config.get("model", {}).get("stage1", {})
    stem_cfg = stage1_cfg.get("stem_encoder", {})
    tokenizer_cfg = stage1_cfg.get("tokenizers", {})
    d_model = int(stage1_cfg.get("d_model", 256))
    num_heads = int(stem_cfg.get("num_heads", stage1_cfg.get("num_heads", 4)))
    dropout = float(stem_cfg.get("dropout", stage1_cfg.get("dropout", 0.0)))
    ffn_mult = int(stem_cfg.get("ffn_mult", 4))

    human_cfg = stem_cfg.get("human", {})
    human_tok_cfg = tokenizer_cfg.get("human", {})
    human_slots = int(
        human_cfg.get("slots_per_frame", 1 + int(human_tok_cfg.get("num_body_slots", 8)) + 2)
    )

    object_cfg = stem_cfg.get("object", {})

    relation_cfg = stem_cfg.get("relation", {})
    relation_tok_cfg = tokenizer_cfg.get("relation", {})
    relation_slots = int(
        relation_cfg.get("slots_per_frame", int(relation_tok_cfg.get("num_relation_slots", 8)) + 2)
    )

    return Stage1StemEncoderModule(
        human=HumanStemEncoder(
            d_model=d_model,
            num_heads=num_heads,
            depth=int(human_cfg.get("depth", 2)),
            slots_per_frame=human_slots,
            dropout=float(human_cfg.get("dropout", dropout)),
            ffn_mult=int(human_cfg.get("ffn_mult", ffn_mult)),
        ),
        object=ObjectStemEncoder(
            d_model=d_model,
            num_heads=num_heads,
            pose_depth=int(object_cfg.get("pose_depth", 2)),
            geometry_depth=int(object_cfg.get("geometry_depth", 1)),
            pose_geometry_cross_depth=int(object_cfg.get("pose_geometry_cross_depth", 1)),
            dropout=float(object_cfg.get("dropout", dropout)),
            ffn_mult=int(object_cfg.get("ffn_mult", ffn_mult)),
        ),
        relation=RelationStemEncoder(
            d_model=d_model,
            num_heads=num_heads,
            depth=int(relation_cfg.get("depth", 2)),
            slots_per_frame=relation_slots,
            dropout=float(relation_cfg.get("dropout", dropout)),
            ffn_mult=int(relation_cfg.get("ffn_mult", ffn_mult)),
        ),
    )


def replace_tokens_preserve_metadata(
    out: TokenizerOutput,
    tokens: torch.Tensor,
    aux: Optional[dict] = None,
) -> TokenizerOutput:
    return TokenizerOutput(
        tokens=tokens,
        mask=out.mask,
        modality_ids=out.modality_ids,
        token_type_ids=out.token_type_ids,
        time_ids=out.time_ids,
        entity_ids=out.entity_ids,
        time_mask=out.time_mask,
        aux={} if aux is None else aux,
    )


def _view_as_time_slots(
    out: TokenizerOutput,
    slots_per_frame: int,
    module_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if slots_per_frame <= 0:
        raise ValueError(f"{module_name}.slots_per_frame must be positive, got {slots_per_frame}.")
    b, n, d = out.tokens.shape
    if n % slots_per_frame != 0:
        raise ValueError(
            f"{module_name} expected token count divisible by slots_per_frame={slots_per_frame}, got N={n}."
        )
    t = n // slots_per_frame
    return (
        out.tokens.reshape(b, t, slots_per_frame, d),
        out.mask.reshape(b, t, slots_per_frame),
        out.time_ids.reshape(b, t, slots_per_frame),
    )


def _static_type_index(out: TokenizerOutput, token_type: TokenTypeId, module_name: str) -> torch.Tensor:
    token_type_ids = out.token_type_ids
    if token_type_ids.ndim != 2:
        raise ValueError(f"{module_name}.token_type_ids must be [B,N], got {tuple(token_type_ids.shape)}.")
    first = token_type_ids[0]
    if not torch.equal(token_type_ids, first[None, :].expand_as(token_type_ids)):
        raise ValueError(f"{module_name} requires token_type_ids layout to be identical across batch.")
    return first == int(token_type)
