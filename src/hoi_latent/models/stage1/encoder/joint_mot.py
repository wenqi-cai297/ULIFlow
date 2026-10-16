from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from hoi_latent.models.stage1.encoder.hoi_packing import (
    HOI_SUBSTREAM_NAMES,
    HOISubstreamId,
    Stage1HOIPack,
    Stage1HOISequencePacker,
    build_pack_metadata,
)
from hoi_latent.models.stage1.encoder.hoi_position_encoding import HOIRotaryPositionEncoding
from hoi_latent.models.stage1.encoder.routed_layers import (
    RoutedAttentionProjection,
    RoutedFeedForward,
    RoutedLayerNorm,
    SharedAttentionProjection,
    SubstreamEmbedding,
)
from hoi_latent.models.stage1.encoder.transformer_blocks import FeedForward
from hoi_latent.models.stage1.mask_utils import safe_masked_softmax, zero_invalid_tokens
from hoi_latent.models.stage1.tokenizer_types import TokenizerOutput, validate_tokenizer_output


class HOIMoTAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        route_decoupling: str = "ffn_only",
        full_qkv_init: str = "shared_plus_delta",
        route_delta_gamma_init: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        if route_decoupling not in {"embedding_only", "ffn_only", "full"}:
            raise ValueError(f"Unknown route_decoupling={route_decoupling!r}.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.dropout = float(dropout)
        self.route_decoupling = route_decoupling
        self.projection: SharedAttentionProjection | RoutedAttentionProjection
        if route_decoupling == "full":
            self.projection = RoutedAttentionProjection(
                d_model=d_model,
                init_mode=full_qkv_init,
                gamma_init=route_delta_gamma_init,
            )
        else:
            self.projection = SharedAttentionProjection(d_model)
        self.rope = HOIRotaryPositionEncoding(self.head_dim, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        route_ids: torch.Tensor,
        mask: torch.Tensor,
        time_ids: torch.Tensor,
        apply_temporal_rope: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        if isinstance(self.projection, RoutedAttentionProjection):
            q = self.projection.q(x, route_ids)
            k = self.projection.k(x, route_ids)
            v = self.projection.v(x, route_ids)
        else:
            q = self.projection.q(x)
            k = self.projection.k(x)
            v = self.projection.v(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        q, k = self.rope(q, k, time_ids, apply_temporal_rope)
        diagnostics = _attention_diagnostics(q, k, mask, route_ids) if return_diagnostics else None
        out = safe_full_bidirectional_attention(
            q=q,
            k=k,
            v=v,
            key_mask=mask,
            query_mask=mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        out = self._merge_heads(out)
        if isinstance(self.projection, RoutedAttentionProjection):
            out = self.projection.o(out, route_ids)
        else:
            out = self.projection.o(out)
        return out, diagnostics

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(b, n, self.d_model)


class HOIMoTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        ffn_mult: int = 4,
        route_decoupling: str = "ffn_only",
        full_qkv_init: str = "shared_plus_delta",
        route_delta_gamma_init: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if route_decoupling not in {"embedding_only", "ffn_only", "full"}:
            raise ValueError(f"Unknown route_decoupling={route_decoupling!r}.")
        self.route_decoupling = route_decoupling
        if route_decoupling == "embedding_only":
            self.pre_attn_norm: nn.Module = nn.LayerNorm(d_model)
            self.post_attn_norm: nn.Module = nn.LayerNorm(d_model)
            self.ffn: nn.Module = FeedForward(d_model=d_model, ffn_mult=ffn_mult, dropout=dropout)
        else:
            self.pre_attn_norm = RoutedLayerNorm(d_model)
            self.post_attn_norm = RoutedLayerNorm(d_model)
            self.ffn = RoutedFeedForward(d_model=d_model, ffn_mult=ffn_mult, dropout=dropout)
        self.attn = HOIMoTAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            route_decoupling=route_decoupling,
            full_qkv_init=full_qkv_init,
            route_delta_gamma_init=route_delta_gamma_init,
            rope_base=rope_base,
        )

    def forward(self, x: torch.Tensor, pack: Stage1HOIPack, return_diagnostics: bool = False) -> tuple[torch.Tensor, dict]:
        if self.route_decoupling == "embedding_only":
            x_norm = self.pre_attn_norm(x)
        else:
            x_norm = self.pre_attn_norm(x, pack.substream_ids)
        attn_out, attention_diag = self.attn(
            x_norm,
            pack.substream_ids,
            pack.mask,
            pack.time_ids,
            pack.apply_temporal_rope,
            return_diagnostics=return_diagnostics,
        )
        x = zero_invalid_tokens(x + attn_out, pack.mask)

        if self.route_decoupling == "embedding_only":
            y = self.ffn(self.post_attn_norm(x))
        else:
            y = self.ffn(self.post_attn_norm(x, pack.substream_ids), pack.substream_ids)
        x = zero_invalid_tokens(x + y, pack.mask)
        return x, {"attention": attention_diag} if return_diagnostics else {}


class Stage1JointMoTEncoderModule(nn.Module):
    checkpoint_key = "joint_encoder"
    input_keys = ("human", "object", "relation")
    output_key = "joint"

    def __init__(
        self,
        d_model: int = 256,
        depth: int = 6,
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
            raise ValueError(f"JointMoT depth must be positive, got {depth}.")
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
        self.packer = Stage1HOISequencePacker()
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
        compressed_outputs: dict[str, TokenizerOutput],
        return_split: bool = False,
        return_diagnostics: bool = False,
        validate: bool = False,
    ) -> dict[str, TokenizerOutput]:
        pack, concat_input = self.packer(compressed_outputs)
        if not self.apply_temporal_rope_to_dynamic_tokens:
            pack.apply_temporal_rope = torch.zeros_like(pack.apply_temporal_rope)
        x = pack.tokens
        if x.shape[-1] != self.d_model:
            raise ValueError(f"JointMoT d_model={self.d_model} but input token dim is {x.shape[-1]}.")
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

        joint_aux = {
            **concat_input.aux,
            "pack": build_pack_metadata(pack, include_diagnostics=return_diagnostics),
            "joint_encoder": (
                self._diagnostics(x, pack, layer_diagnostics, collect_attention_diagnostics)
                if return_diagnostics
                else self._minimal_diagnostics()
            ),
        }
        joint = TokenizerOutput(
            tokens=x,
            mask=concat_input.mask,
            modality_ids=concat_input.modality_ids,
            token_type_ids=concat_input.token_type_ids,
            time_ids=concat_input.time_ids,
            entity_ids=concat_input.entity_ids,
            time_mask=pack.time_mask,
            aux=joint_aux,
        )
        if validate:
            validate_tokenizer_output(joint, "Stage1JointMoTEncoder.joint")
        outputs = {self.output_key: joint}
        if return_split:
            for key, stream_slice in pack.stream_slices.items():
                outputs[key] = _slice_tokenizer_output(joint, stream_slice, aux_key=f"joint_split_{key}")
        return outputs

    def _minimal_diagnostics(self) -> dict:
        return {
            "route_decoupling": self.route_decoupling,
            "diagnostics_enabled": False,
        }

    def _diagnostics(
        self,
        x: torch.Tensor,
        pack: Stage1HOIPack,
        layer_diagnostics: list[dict],
        include_attention: bool,
    ) -> dict:
        diagnostics = {
            "route_decoupling": self.route_decoupling,
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


def safe_full_bidirectional_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
    query_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B,H,N,D].")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shapes must match, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}.")
    b, _, n, _ = q.shape
    key_mask = key_mask.to(device=q.device, dtype=torch.bool)
    query_mask = query_mask.to(device=q.device, dtype=torch.bool)
    if key_mask.shape != (b, n) or query_mask.shape != (b, n):
        raise ValueError(f"key_mask and query_mask must have shape {(b, n)}.")

    attn_mask = key_mask[:, None, None, :].expand(b, 1, n, n).clone()
    no_valid_keys = ~key_mask.any(dim=1)
    if no_valid_keys.any():
        eye = torch.eye(n, dtype=torch.bool, device=q.device)[None, None, :, :]
        attn_mask[no_valid_keys] = eye.expand(int(no_valid_keys.sum().item()), 1, n, n)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=float(dropout_p) if training else 0.0,
        is_causal=False,
    )
    has_valid_key = key_mask.any(dim=1, keepdim=True)
    valid_output = query_mask & has_valid_key
    return out * valid_output[:, None, :, None].to(dtype=out.dtype)


def build_stage1_joint_mot_encoder_from_config(config: dict) -> Stage1JointMoTEncoderModule:
    stage1_cfg = config.get("model", {}).get("stage1", {})
    joint_cfg = stage1_cfg.get("joint_encoder", {})
    attention_cfg = joint_cfg.get("attention", {})
    route_cfg = joint_cfg.get("routes", {})
    full_cfg = route_cfg.get("full", {})
    position_cfg = joint_cfg.get("position_encoding", {})
    final_norm_cfg = joint_cfg.get("final_norm", {})
    diagnostics_cfg = joint_cfg.get("diagnostics", {})
    _validate_joint_config_switches(joint_cfg, attention_cfg, route_cfg, position_cfg)
    return Stage1JointMoTEncoderModule(
        d_model=int(joint_cfg.get("d_model", stage1_cfg.get("d_model", 256))),
        depth=int(joint_cfg.get("depth", 6)),
        num_heads=int(joint_cfg.get("num_heads", stage1_cfg.get("num_heads", 4))),
        dropout=float(joint_cfg.get("dropout", stage1_cfg.get("dropout", 0.0))),
        ffn_mult=int(joint_cfg.get("ffn_mult", 4)),
        route_decoupling=str(route_cfg.get("route_decoupling", "ffn_only")),
        substream_embedding=bool(route_cfg.get("substream_embedding", True)),
        full_qkv_init=str(full_cfg.get("full_qkv_init", "shared_plus_delta")),
        route_delta_gamma_init=float(full_cfg.get("route_delta_gamma_init", 0.0)),
        rope_base=float(position_cfg.get("rope_base", 10000.0)),
        apply_temporal_rope_to_dynamic_tokens=bool(position_cfg.get("apply_to_dynamic_tokens", True)),
        routed_final_norm=bool(final_norm_cfg.get("routed_final_norm", True)),
        diagnostics_cross_substream_attention=bool(diagnostics_cfg.get("cross_substream_attention", True)),
        diagnostics_qk_stats=bool(diagnostics_cfg.get("qk_stats", True)),
        diagnostics_invalid_token_check=bool(diagnostics_cfg.get("invalid_token_check", True)),
    )


def _validate_joint_config_switches(joint_cfg: dict, attention_cfg: dict, route_cfg: dict, position_cfg: dict) -> None:
    _require_value(joint_cfg, "activation", "gelu")
    _require_value(attention_cfg, "mode", "full_bidirectional")
    _require_value(attention_cfg, "implementation", "safe_sdpa")
    _require_value(attention_cfg, "diagnostics_attention", "manual_debug_only")
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
            "Stage1JointMoTEncoder v1 requires routes.substreams="
            f"{HOI_SUBSTREAM_NAMES}, got {route_cfg['substreams']!r}."
        )
    _require_value(position_cfg, "type", "temporal_hoi_rope")
    _require_value(position_cfg, "apply_to_static_geometry", False)
    _require_value(position_cfg, "categorical_ids_in_rope", False)


def _require_value(cfg: dict, key: str, expected) -> None:
    if key in cfg and cfg[key] != expected:
        raise ValueError(f"Stage1JointMoTEncoder only supports {key}={expected!r} for v1, got {cfg[key]!r}.")


def _attention_diagnostics(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor, route_ids: torch.Tensor) -> dict:
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(q.shape[-1]))
    attn = safe_masked_softmax(scores, mask[:, None, None, :], dim=-1)
    attn = attn * mask[:, None, :, None].to(dtype=attn.dtype)
    eps = torch.finfo(attn.dtype).eps
    valid_queries = mask[:, None, :].expand(mask.shape[0], q.shape[1], mask.shape[1])
    entropy = -(attn * (attn + eps).log()).sum(dim=-1)
    entropy = _masked_average(entropy, valid_queries)
    mass = _cross_substream_attention_mass(attn, mask, route_ids)
    q_norm = _head_substream_norm(q, mask, route_ids)
    k_norm = _head_substream_norm(k, mask, route_ids)
    qk_similarity = _qk_similarity_by_substream_pair(q, k, mask, route_ids)
    geometry_idx = int(HOISubstreamId.OBJECT_GEOMETRY)
    relation_idx = int(HOISubstreamId.RELATION_DYNAMIC)
    human_idx = int(HOISubstreamId.HUMAN_DYNAMIC)
    object_pose_idx = int(HOISubstreamId.OBJECT_POSE)
    return {
        "cross_substream_attention_mass": mass.detach(),
        "attention_entropy": entropy.detach(),
        "object_geometry_received_attention_mass": mass[:, :, geometry_idx].mean().detach(),
        "relation_to_human_attention_mass": mass[:, relation_idx, human_idx].mean().detach(),
        "relation_to_object_pose_attention_mass": mass[:, relation_idx, object_pose_idx].mean().detach(),
        "q_norm_by_substream": q_norm.detach(),
        "k_norm_by_substream": k_norm.detach(),
        "qk_similarity_by_substream_pair": qk_similarity.detach(),
    }


def _cross_substream_attention_mass(attn: torch.Tensor, mask: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
    _, heads, _, _ = attn.shape
    out = attn.new_zeros(heads, len(HOI_SUBSTREAM_NAMES), len(HOI_SUBSTREAM_NAMES))
    for q_idx in range(len(HOI_SUBSTREAM_NAMES)):
        q_mask = (route_ids == q_idx) & mask
        q_count = q_mask.sum().clamp_min(1).to(dtype=attn.dtype)
        for k_idx in range(len(HOI_SUBSTREAM_NAMES)):
            k_mask = (route_ids == k_idx) & mask
            mass = attn * q_mask[:, None, :, None].to(dtype=attn.dtype) * k_mask[:, None, None, :].to(dtype=attn.dtype)
            out[:, q_idx, k_idx] = mass.sum(dim=(0, 2, 3)) / q_count
    return out


def _head_substream_norm(x: torch.Tensor, mask: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
    _, heads, _, _ = x.shape
    out = x.new_zeros(heads, len(HOI_SUBSTREAM_NAMES))
    norm = x.norm(dim=-1)
    for idx in range(len(HOI_SUBSTREAM_NAMES)):
        route_mask = ((route_ids == idx) & mask).to(dtype=norm.dtype)
        denom = route_mask.sum().clamp_min(1.0)
        out[:, idx] = (norm * route_mask[:, None, :]).sum(dim=(0, 2)) / denom
    return out


def _qk_similarity_by_substream_pair(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
    q_mean = _head_route_mean(q, mask, route_ids)
    k_mean = _head_route_mean(k, mask, route_ids)
    q_norm = F.normalize(q_mean, dim=-1)
    k_norm = F.normalize(k_mean, dim=-1)
    return torch.einsum("hrd,hsd->hrs", q_norm, k_norm)


def _head_route_mean(x: torch.Tensor, mask: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
    _, heads, _, dim = x.shape
    out = x.new_zeros(heads, len(HOI_SUBSTREAM_NAMES), dim)
    for idx in range(len(HOI_SUBSTREAM_NAMES)):
        route_mask = ((route_ids == idx) & mask)[:, None, :, None].to(dtype=x.dtype)
        denom = route_mask.sum(dim=(0, 2, 3)).clamp_min(1.0)
        out[:, idx, :] = (x * route_mask).sum(dim=(0, 2)) / denom[:, None]
    return out


def _masked_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = values * mask.to(dtype=values.dtype)
    denom = mask.sum().clamp_min(1).to(dtype=values.dtype)
    return weighted.sum() / denom


def _substream_stat(x: torch.Tensor, route_ids: torch.Tensor, mask: torch.Tensor, stat: str) -> dict[str, torch.Tensor]:
    result = {}
    for idx, name in enumerate(HOI_SUBSTREAM_NAMES):
        route_mask = (route_ids == idx) & mask
        if route_mask.any():
            values = x[route_mask]
            if stat == "mean":
                result[name] = values.mean().detach()
            elif stat == "std":
                result[name] = values.std(unbiased=False).detach()
            else:
                raise ValueError(f"Unknown stat {stat!r}.")
        else:
            result[name] = x.new_tensor(0.0)
    return result


def _merge_layer_attention_diagnostics(
    layer_diagnostics: list[dict],
    include_cross_substream: bool,
    include_qk_stats: bool,
) -> dict:
    attention = [item.get("attention") for item in layer_diagnostics if item.get("attention") is not None]
    if not attention:
        return {}
    keys = []
    if include_cross_substream:
        keys.extend(
            [
                "cross_substream_attention_mass",
                "attention_entropy",
                "object_geometry_received_attention_mass",
                "relation_to_human_attention_mass",
                "relation_to_object_pose_attention_mass",
            ]
        )
    if include_qk_stats:
        keys.extend(["q_norm_by_substream", "k_norm_by_substream", "qk_similarity_by_substream_pair"])
    merged = {}
    for key in keys:
        values = [item[key] for item in attention]
        if values[0].ndim == 0:
            merged[key] = torch.stack(values)
        else:
            merged[key] = torch.stack(values, dim=0)
    return merged


def _slice_tokenizer_output(out: TokenizerOutput, stream_slice: slice, aux_key: str) -> TokenizerOutput:
    return TokenizerOutput(
        tokens=out.tokens[:, stream_slice, :],
        mask=out.mask[:, stream_slice],
        modality_ids=out.modality_ids[:, stream_slice],
        token_type_ids=out.token_type_ids[:, stream_slice],
        time_ids=out.time_ids[:, stream_slice],
        entity_ids=out.entity_ids[:, stream_slice],
        time_mask=None if out.time_mask is None else out.time_mask[:, stream_slice],
        aux={"source": aux_key, "joint_encoder": out.aux.get("joint_encoder", {})},
    )
