from __future__ import annotations

import torch
from torch import nn

from hoi_latent.models.stage1.encoder.hoi_packing import (
    HOI_SUBSTREAM_NAMES,
    HOISubstreamId,
    Stage1HOIPack,
    build_pack_metadata,
    derive_hoi_substream_ids,
)
from hoi_latent.models.stage1.encoder.joint_mot import (
    HOIMoTBlock,
    _merge_layer_attention_diagnostics,
    _substream_stat,
)
from hoi_latent.models.stage1.encoder.routed_layers import (
    RoutedAttentionProjection,
    RoutedLayerNorm,
    SubstreamEmbedding,
)
from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.models.stage1.tokenizer_types import TokenTypeId, TokenizerOutput, validate_tokenizer_output


class Stage1DecoderJointStack(nn.Module):
    output_key = "decoder_joint"

    def __init__(
        self,
        d_model: int = 256,
        depth: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
        ffn_mult: int = 4,
        route_decoupling: str = "ffn_only",
        substream_embedding: bool = True,
        full_qkv_init: str = "shared_plus_delta",
        route_delta_gamma_init: float = 0.0,
        rope_base: float = 10000.0,
        apply_temporal_rope_to_dynamic_tokens: bool = True,
        routed_final_norm: bool = True,
        diagnostics_cross_substream_attention: bool = True,
        diagnostics_qk_stats: bool = True,
        diagnostics_invalid_token_check: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        if depth <= 0:
            raise ValueError(f"Decoder joint stack depth must be positive, got {depth}.")
        if ffn_mult <= 0:
            raise ValueError(f"ffn_mult must be positive, got {ffn_mult}.")
        if route_decoupling not in {"embedding_only", "ffn_only", "full"}:
            raise ValueError(f"Unknown route_decoupling={route_decoupling!r}.")
        if route_decoupling == "full" and full_qkv_init not in RoutedAttentionProjection.allowed_init_modes:
            raise ValueError(
                "route_decoupling='full' requires full_qkv_init='shared_plus_delta' or 'tie_at_init'; "
                f"got {full_qkv_init!r}."
            )
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.route_decoupling = route_decoupling
        self.apply_temporal_rope_to_dynamic_tokens = bool(apply_temporal_rope_to_dynamic_tokens)
        self.use_routed_final_norm = route_decoupling != "embedding_only" and bool(routed_final_norm)
        self.diagnostics_cross_substream_attention = bool(diagnostics_cross_substream_attention)
        self.diagnostics_qk_stats = bool(diagnostics_qk_stats)
        self.diagnostics_invalid_token_check = bool(diagnostics_invalid_token_check)
        self.substream_embedding = SubstreamEmbedding(d_model) if substream_embedding else None
        self.blocks = nn.ModuleList(
            [
                HOIMoTBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                    route_decoupling=route_decoupling,
                    full_qkv_init=full_qkv_init,
                    route_delta_gamma_init=route_delta_gamma_init,
                    rope_base=rope_base,
                )
                for _ in range(self.depth)
            ]
        )
        if self.use_routed_final_norm:
            self.final_norm = RoutedLayerNorm(d_model)
        else:
            self.final_norm: nn.Module = nn.LayerNorm(d_model)

    def forward(
        self,
        decoder_input: TokenizerOutput,
        return_diagnostics: bool = False,
        validate: bool = False,
    ) -> TokenizerOutput:
        if validate:
            validate_tokenizer_output(decoder_input, "decoder_joint_stack.input")
        pack = build_decoder_joint_pack(decoder_input)
        if not self.apply_temporal_rope_to_dynamic_tokens:
            pack.apply_temporal_rope = pack.apply_temporal_rope.new_zeros(pack.apply_temporal_rope.shape)
        x = pack.tokens
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Stage1DecoderJointStack d_model={self.d_model} but input token dim is {x.shape[-1]}.")
        if self.substream_embedding is not None:
            x = x + self.substream_embedding(pack.substream_ids)
        x = zero_invalid_tokens(x, pack.mask)

        collect_attention_diagnostics = return_diagnostics and (
            self.diagnostics_cross_substream_attention or self.diagnostics_qk_stats
        )
        layer_diagnostics = []
        for block in self.blocks:
            x, block_diag = block(x, pack, return_diagnostics=collect_attention_diagnostics)
            if collect_attention_diagnostics:
                layer_diagnostics.append(block_diag)

        if self.use_routed_final_norm:
            x = self.final_norm(x, pack.substream_ids)
        else:
            x = self.final_norm(x)
        x = zero_invalid_tokens(x, pack.mask)

        aux = {
            **decoder_input.aux,
            "decoder_pack": build_pack_metadata(pack, include_diagnostics=return_diagnostics),
            "decoder_joint_stack": (
                self._diagnostics(x, pack, layer_diagnostics, collect_attention_diagnostics)
                if return_diagnostics
                else self._minimal_diagnostics()
            ),
        }
        out = TokenizerOutput(
            tokens=x,
            mask=decoder_input.mask,
            modality_ids=decoder_input.modality_ids,
            token_type_ids=decoder_input.token_type_ids,
            time_ids=decoder_input.time_ids,
            entity_ids=decoder_input.entity_ids,
            time_mask=pack.time_mask,
            aux=aux,
        )
        if validate:
            validate_tokenizer_output(out, "decoder_joint_stack")
        return out

    def _minimal_diagnostics(self) -> dict:
        return {
            "route_decoupling": self.route_decoupling,
            "depth": self.depth,
            "diagnostics_enabled": False,
        }

    def _diagnostics(
        self,
        x,
        pack: Stage1HOIPack,
        layer_diagnostics: list[dict],
        include_attention: bool,
    ) -> dict:
        diagnostics = {
            "route_decoupling": self.route_decoupling,
            "depth": self.depth,
            "diagnostics_enabled": True,
            "substream_names": HOI_SUBSTREAM_NAMES,
            "substream_token_counts": {
                name: (pack.substream_ids == idx).sum(dim=1)
                for idx, name in enumerate(HOI_SUBSTREAM_NAMES)
            },
            "post_norm_mean_by_substream": _substream_stat(x, pack.substream_ids, pack.mask, "mean"),
            "post_norm_std_by_substream": _substream_stat(x, pack.substream_ids, pack.mask, "std"),
            "apply_temporal_rope_count": pack.apply_temporal_rope.sum(dim=1),
        }
        if self.diagnostics_invalid_token_check:
            invalid = ~pack.mask
            invalid_max = x[invalid].abs().max() if invalid.any() else x.new_tensor(0.0)
            diagnostics["invalid_token_max_abs"] = invalid_max.detach()
        if include_attention:
            diagnostics.update(
                _merge_layer_attention_diagnostics(
                    layer_diagnostics,
                    include_cross_substream=self.diagnostics_cross_substream_attention,
                    include_qk_stats=self.diagnostics_qk_stats,
                )
            )
        return diagnostics


def build_decoder_joint_pack(decoder_input: TokenizerOutput) -> Stage1HOIPack:
    time_mask = decoder_input.time_mask
    if time_mask is None:
        time_mask = decoder_input.mask.clone()
    mask = decoder_input.mask.to(dtype=torch.bool)
    time_mask = time_mask.to(device=mask.device, dtype=torch.bool)
    substream_ids = derive_hoi_substream_ids(decoder_input.modality_ids, decoder_input.token_type_ids)
    apply_temporal_rope = (
        time_mask
        & mask
        & (decoder_input.token_type_ids != int(TokenTypeId.OBJECT_GEOMETRY))
        & (substream_ids != int(HOISubstreamId.OBJECT_GEOMETRY))
    )
    num_tokens = decoder_input.tokens.shape[1]
    return Stage1HOIPack(
        tokens=decoder_input.tokens,
        mask=mask,
        substream_ids=substream_ids,
        modality_ids=decoder_input.modality_ids,
        token_type_ids=decoder_input.token_type_ids,
        time_ids=decoder_input.time_ids,
        entity_ids=decoder_input.entity_ids,
        time_mask=time_mask,
        apply_temporal_rope=apply_temporal_rope,
        stream_slices={"decoder_input": slice(0, num_tokens)},
    )


def build_stage1_decoder_joint_stack_from_config(config: dict) -> Stage1DecoderJointStack:
    stage1_cfg = config.get("model", {}).get("stage1", {})
    decoder_cfg = stage1_cfg.get("decoder", {})
    stack_cfg = decoder_cfg.get("joint_stack", {})
    attention_cfg = stack_cfg.get("attention", {})
    route_cfg = stack_cfg.get("routes", {})
    full_cfg = route_cfg.get("full", {})
    position_cfg = stack_cfg.get("position_encoding", {})
    final_norm_cfg = stack_cfg.get("final_norm", {})
    diagnostics_cfg = stack_cfg.get("diagnostics", {})
    _validate_decoder_joint_stack_config(stack_cfg, attention_cfg, route_cfg, position_cfg)
    return Stage1DecoderJointStack(
        d_model=int(stack_cfg.get("d_model", stage1_cfg.get("d_model", 256))),
        depth=int(stack_cfg.get("depth", 8)),
        num_heads=int(stack_cfg.get("num_heads", stage1_cfg.get("num_heads", 4))),
        dropout=float(stack_cfg.get("dropout", stage1_cfg.get("dropout", 0.0))),
        ffn_mult=int(stack_cfg.get("ffn_mult", 4)),
        route_decoupling=str(stack_cfg.get("route_decoupling", route_cfg.get("route_decoupling", "ffn_only"))),
        substream_embedding=bool(stack_cfg.get("substream_embedding", route_cfg.get("substream_embedding", True))),
        full_qkv_init=str(full_cfg.get("full_qkv_init", route_cfg.get("full_qkv_init", "shared_plus_delta"))),
        route_delta_gamma_init=float(
            full_cfg.get("route_delta_gamma_init", route_cfg.get("route_delta_gamma_init", 0.0))
        ),
        rope_base=float(position_cfg.get("rope_base", 10000.0)),
        apply_temporal_rope_to_dynamic_tokens=bool(position_cfg.get("apply_to_dynamic_tokens", True)),
        routed_final_norm=bool(stack_cfg.get("routed_final_norm", final_norm_cfg.get("routed_final_norm", True))),
        diagnostics_cross_substream_attention=bool(diagnostics_cfg.get("cross_substream_attention", True)),
        diagnostics_qk_stats=bool(diagnostics_cfg.get("qk_stats", True)),
        diagnostics_invalid_token_check=bool(diagnostics_cfg.get("invalid_token_check", True)),
    )


def _validate_decoder_joint_stack_config(
    stack_cfg: dict,
    attention_cfg: dict,
    route_cfg: dict,
    position_cfg: dict,
) -> None:
    _require_value(stack_cfg, "activation", "gelu")
    _require_value(stack_cfg, "temporal_position_encoding", "temporal_hoi_rope")
    _require_value(attention_cfg, "mode", "full_bidirectional")
    _require_value(attention_cfg, "implementation", "safe_sdpa")
    _require_value(attention_cfg, "all_invalid_policy", "zero_output")
    _require_value(route_cfg, "learned_router", False)
    _require_value(route_cfg, "moe_ffn", False)
    _require_value(route_cfg, "token_type_reembedding", False)
    _require_value(route_cfg.get("ffn_only", {}), "shared_qkv", True)
    _require_value(route_cfg.get("ffn_only", {}), "shared_output_projection", True)
    _require_value(route_cfg.get("ffn_only", {}), "routed_pre_attn_norm", True)
    _require_value(route_cfg.get("ffn_only", {}), "routed_post_attn_norm", True)
    _require_value(route_cfg.get("ffn_only", {}), "routed_ffn", True)
    _require_value(route_cfg.get("full", {}), "enabled_as_default", False)
    if "substreams" in route_cfg and tuple(route_cfg["substreams"]) != HOI_SUBSTREAM_NAMES:
        raise ValueError(
            "Stage1DecoderJointStack v1 requires routes.substreams="
            f"{HOI_SUBSTREAM_NAMES}, got {route_cfg['substreams']!r}."
        )
    _require_value(position_cfg, "type", "temporal_hoi_rope")
    _require_value(position_cfg, "apply_to_static_geometry", False)
    _require_value(position_cfg, "categorical_ids_in_rope", False)


def _require_value(cfg: dict, key: str, expected) -> None:
    if key in cfg and cfg[key] != expected:
        raise ValueError(f"Stage1DecoderJointStack only supports {key}={expected!r} for v1, got {cfg[key]!r}.")
