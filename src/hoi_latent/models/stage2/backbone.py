from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.models.stage2.flow_time import Stage2FlowTimeField
from hoi_latent.models.stage2.mot_blocks import (
    Stage2LatentVelocityHead,
    Stage2TimeEmbedder,
    TriStreamMoTBlock,
)
from hoi_latent.models.stage2.stem_types import (
    NUM_STAGE2_FLOW_ROUTES,
    Stage2FlowRouteId,
    Stage2PackedStreams,
    sanitize_ids_for_embedding,
)


@dataclass
class Stage2TriStreamMoTOutput:
    tokens: torch.Tensor
    latent_hidden: torch.Tensor
    velocity: torch.Tensor
    attention_mask: torch.Tensor
    flow_route_ids: torch.Tensor
    time_ids: torch.Tensor
    apply_temporal_rope: torch.Tensor
    latent_slice: slice
    text_slice: slice
    geometry_slice: slice
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage2BlockCausalAttentionMask:
    causal_order: torch.Tensor
    pairwise_mask: torch.Tensor
    sdpa_mask: torch.Tensor
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


class Stage2TriStreamMoTBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        latent_dim: int = 16,
        depth: int = 12,
        num_heads: int = 6,
        head_dim: int | None = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        qk_norm: bool = True,
        qkvo_routing: str = "shared_plus_delta",
        route_delta_gamma_init: float = 0.0,
        time_scale: float = 1000.0,
        rope_base: float = 10000.0,
        use_rollout_progress: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}.")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        derived_head_dim = int(d_model) // int(num_heads)
        if head_dim is not None and int(head_dim) != derived_head_dim:
            raise ValueError(
                f"head_dim={head_dim} must equal d_model//num_heads={derived_head_dim} "
                f"(d_model={d_model}, num_heads={num_heads})."
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}.")
        if qkvo_routing != "shared_plus_delta":
            raise ValueError(f"Stage2TriStreamMoTBackbone v1 requires qkvo_routing='shared_plus_delta', got {qkvo_routing!r}.")

        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.head_dim = derived_head_dim
        self.dropout = float(dropout)
        self.qk_norm = bool(qk_norm)
        self.qkvo_routing = qkvo_routing
        self.use_rollout_progress = bool(use_rollout_progress)

        self.time_embedder = Stage2TimeEmbedder(d_model=d_model, time_scale=time_scale)
        self.text_pool_proj = nn.Linear(d_model, d_model)
        self.geometry_pool_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [
                TriStreamMoTBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    head_dim=self.head_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    route_delta_gamma_init=route_delta_gamma_init,
                    rope_base=rope_base,
                )
                for _ in range(depth)
            ]
        )
        self.velocity_head = Stage2LatentVelocityHead(d_model=d_model, latent_dim=latent_dim)

    def forward(
        self,
        packed: Stage2PackedStreams,
        flow_time: Stage2FlowTimeField,
        return_diagnostics: bool = False,
    ) -> Stage2TriStreamMoTOutput:
        self._validate_packed(packed)
        self._validate_flow_time(packed, flow_time)
        route_index = sanitize_ids_for_embedding(
            packed.flow_route_ids.to(device=packed.tokens.device),
            packed.attention_mask.to(device=packed.tokens.device),
            pad_id=0,
            num_embeddings=NUM_STAGE2_FLOW_ROUTES,
            name="flow_route_ids",
        )
        x = zero_invalid_tokens(packed.tokens, packed.attention_mask)
        cond_tokens, cond_latent = self._build_condition_tokens(packed, flow_time, dtype=x.dtype)
        attention_mask = build_stage2_block_causal_attention_mask(packed, flow_time)
        block_diags = []
        for block in self.blocks:
            x, diag = block(
                x,
                packed,
                route_index,
                cond_tokens,
                attention_mask.pairwise_mask,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                block_diags.append(diag)
        x = zero_invalid_tokens(x, packed.attention_mask)

        latent_mask = packed.attention_mask[:, packed.latent_slice]
        latent_hidden = zero_invalid_tokens(x[:, packed.latent_slice], latent_mask)
        velocity = self.velocity_head(latent_hidden, cond_latent)
        velocity = zero_invalid_tokens(velocity, latent_mask)
        return Stage2TriStreamMoTOutput(
            tokens=x,
            latent_hidden=latent_hidden,
            velocity=velocity,
            attention_mask=packed.attention_mask,
            flow_route_ids=packed.flow_route_ids,
            time_ids=packed.time_ids,
            apply_temporal_rope=packed.apply_temporal_rope,
            latent_slice=packed.latent_slice,
            text_slice=packed.text_slice,
            geometry_slice=packed.geometry_slice,
            diagnostics=self._diagnostics(x, velocity, packed, cond_tokens, attention_mask, block_diags) if return_diagnostics else {},
        )

    def _build_condition_tokens(
        self,
        packed: Stage2PackedStreams,
        flow_time: Stage2FlowTimeField,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = packed.tokens.device
        latent_len = packed.latent_slice.stop - packed.latent_slice.start
        text_len = packed.text_slice.stop - packed.text_slice.start
        geometry_len = packed.geometry_slice.stop - packed.geometry_slice.start
        t_latent = flow_time.t_latent_token.to(device=device)
        rollout_progress = flow_time.rollout_progress.to(device=device)
        t_latent_emb = self.time_embedder(t_latent.reshape(-1), dtype=dtype).reshape(
            packed.tokens.shape[0],
            latent_len,
            self.d_model,
        )
        if self.use_rollout_progress:
            progress_emb = self.time_embedder(rollout_progress, dtype=dtype)
        else:
            progress_emb = torch.zeros(
                packed.tokens.shape[0],
                self.d_model,
                device=device,
                dtype=dtype,
            )
        pooled_text = packed.pooled_text.to(device=packed.tokens.device, dtype=dtype)
        pooled_geometry = packed.pooled_geometry.to(device=packed.tokens.device, dtype=dtype)
        context = self.text_pool_proj(pooled_text) + self.geometry_pool_proj(pooled_geometry)
        cond_latent = t_latent_emb + context[:, None, :]
        cond_text = progress_emb[:, None, :].expand(-1, text_len, -1)
        cond_geometry = progress_emb[:, None, :].expand(-1, geometry_len, -1)
        cond_tokens = torch.cat([cond_latent, cond_text, cond_geometry], dim=1)
        cond_tokens = zero_invalid_tokens(cond_tokens, packed.attention_mask)
        return cond_tokens, cond_latent

    def _validate_packed(self, packed: Stage2PackedStreams) -> None:
        if packed.tokens.ndim != 3:
            raise ValueError(f"packed.tokens must have shape [B,N,D], got {tuple(packed.tokens.shape)}.")
        if packed.tokens.shape[-1] != self.d_model:
            raise ValueError(f"Backbone d_model={self.d_model} but packed token dim is {packed.tokens.shape[-1]}.")
        expected = packed.tokens.shape[:2]
        for name in ["attention_mask", "flow_route_ids", "time_ids", "apply_temporal_rope"]:
            value = getattr(packed, name)
            if value.shape != expected:
                raise ValueError(f"packed.{name} must have shape {expected}, got {tuple(value.shape)}.")
        if packed.pooled_text.shape != (packed.tokens.shape[0], self.d_model):
            raise ValueError(f"packed.pooled_text must have shape {(packed.tokens.shape[0], self.d_model)}.")
        if packed.pooled_geometry.shape != (packed.tokens.shape[0], self.d_model):
            raise ValueError(f"packed.pooled_geometry must have shape {(packed.tokens.shape[0], self.d_model)}.")

    def _validate_flow_time(self, packed: Stage2PackedStreams, flow_time: Stage2FlowTimeField) -> None:
        latent_shape = packed.attention_mask[:, packed.latent_slice].shape
        if flow_time.t_latent_token.shape != latent_shape:
            raise ValueError(f"flow_time.t_latent_token must have shape {latent_shape}, got {tuple(flow_time.t_latent_token.shape)}.")
        layout = flow_time.layout
        for name in ["token_anchor_index", "token_block_index", "dynamic_token_mask", "static_geometry_token_mask"]:
            if getattr(layout, name).shape != latent_shape:
                raise ValueError(f"flow_time.layout.{name} must have shape {latent_shape}.")
        if flow_time.t_static.shape != (packed.tokens.shape[0],):
            raise ValueError(f"flow_time.t_static must have shape {(packed.tokens.shape[0],)}, got {tuple(flow_time.t_static.shape)}.")
        if flow_time.rollout_progress.shape != (packed.tokens.shape[0],):
            raise ValueError(
                f"flow_time.rollout_progress must have shape {(packed.tokens.shape[0],)}, "
                f"got {tuple(flow_time.rollout_progress.shape)}."
            )

    def _diagnostics(
        self,
        tokens: torch.Tensor,
        velocity: torch.Tensor,
        packed: Stage2PackedStreams,
        cond_tokens: torch.Tensor,
        attention_mask: Stage2BlockCausalAttentionMask,
        block_diags: list[dict[str, Any]],
    ) -> dict[str, Any]:
        invalid = ~packed.attention_mask
        invalid_max = tokens[invalid].abs().max() if invalid.any() else tokens.new_tensor(0.0)
        diagnostics: dict[str, Any] = {
            "route_token_counts": {
                str(route): ((packed.flow_route_ids == route) & packed.attention_mask).sum(dim=1).detach()
                for route in range(NUM_STAGE2_FLOW_ROUTES)
            },
            "invalid_token_max_abs": invalid_max.detach(),
            "latent_velocity_rms": velocity.detach().pow(2).mean().sqrt(),
            "cond_tokens_rms": cond_tokens.detach().pow(2).mean().sqrt(),
            "causal_order_min": attention_mask.causal_order.detach().float().min(),
            "causal_order_max": attention_mask.causal_order.detach().float().max(),
        }
        attention = [diag.get("attention", {}) for diag in block_diags if diag.get("attention")]
        if attention:
            diagnostics["attention_entropy"] = torch.stack([item["attention_entropy"] for item in attention])
            if "cross_route_attention_mass" in attention[0]:
                diagnostics["cross_route_attention_mass"] = torch.stack(
                    [item["cross_route_attention_mass"] for item in attention],
                    dim=0,
                )
        residual_attn = [diag.get("residual_attn_rms") for diag in block_diags if "residual_attn_rms" in diag]
        residual_ffn = [diag.get("residual_ffn_rms") for diag in block_diags if "residual_ffn_rms" in diag]
        if residual_attn:
            diagnostics["residual_attn_rms"] = torch.stack(residual_attn)
        if residual_ffn:
            diagnostics["residual_ffn_rms"] = torch.stack(residual_ffn)
        return diagnostics


def build_stage2_block_causal_attention_mask(
    packed: Stage2PackedStreams,
    flow_time: Stage2FlowTimeField,
) -> Stage2BlockCausalAttentionMask:
    batch, total = packed.attention_mask.shape
    device = packed.attention_mask.device
    causal_order = torch.zeros(batch, total, device=device, dtype=torch.long)
    causal_order[:, packed.text_slice] = -1
    causal_order[:, packed.geometry_slice] = -1

    latent_mask = packed.attention_mask[:, packed.latent_slice].to(dtype=torch.bool)
    layout = flow_time.layout
    if layout.token_block_index.shape != latent_mask.shape:
        raise ValueError(
            f"flow_time.layout token shape must match packed latent slice {tuple(latent_mask.shape)}, "
            f"got {tuple(layout.token_block_index.shape)}."
        )
    latent_order = torch.zeros_like(layout.token_block_index, device=device, dtype=torch.long)
    dynamic = layout.dynamic_token_mask.to(device=device, dtype=torch.bool) & latent_mask
    static = layout.static_geometry_token_mask.to(device=device, dtype=torch.bool) & latent_mask
    latent_order[dynamic] = layout.token_block_index.to(device=device, dtype=torch.long)[dynamic]
    latent_order[static] = 0
    causal_order[:, packed.latent_slice] = latent_order

    valid = packed.attention_mask.to(dtype=torch.bool)
    max_valid_order = causal_order.masked_fill(~valid, -1).amax(dim=1, keepdim=True)
    causal_order = torch.where(valid, causal_order, max_valid_order + 1)

    key_valid = valid[:, None, :]
    query_order = causal_order[:, :, None]
    key_order = causal_order[:, None, :]
    pairwise_mask = key_valid & (key_order <= query_order)
    if not pairwise_mask.any(dim=-1).all():
        raise ValueError("Stage2 block-causal pairwise mask produced an all-masked query row.")
    return Stage2BlockCausalAttentionMask(
        causal_order=causal_order,
        pairwise_mask=pairwise_mask,
        sdpa_mask=pairwise_mask[:, None, :, :],
        diagnostics={
            "causal_order_min": causal_order.detach().float().min(),
            "causal_order_max": causal_order.detach().float().max(),
        },
    )


def build_stage2_tri_stream_mot_backbone_from_config(config: Mapping[str, Any]) -> Stage2TriStreamMoTBackbone:
    stage2_cfg = config.get("model", {}).get("stage2", {})
    backbone_cfg = stage2_cfg.get("backbone", {})
    d_model = int(stage2_cfg.get("d_model", 384))
    latent_dim = int(stage2_cfg.get("latent_dim", 16))
    _assert_shared_override_matches(backbone_cfg, "d_model", d_model)
    _assert_shared_override_matches(backbone_cfg, "latent_dim", latent_dim)
    stage2_attention_cfg = _nested(stage2_cfg, "attention", {})
    backbone_attention_cfg = _nested(backbone_cfg, "attention", {})
    attention_cfg = {**stage2_attention_cfg, **backbone_attention_cfg}
    final_head_cfg = _nested(backbone_cfg, "final_head", {})
    conditioning_cfg = {
        **_nested(stage2_cfg, "conditioning", {}),
        **_nested(backbone_cfg, "conditioning", {}),
    }
    _require_value(attention_cfg, "mode", "block_causal")
    _require_value(final_head_cfg, "adaln", True)
    _require_value(final_head_cfg, "zero_init_velocity", True)
    return Stage2TriStreamMoTBackbone(
        d_model=d_model,
        latent_dim=latent_dim,
        depth=int(_cfg_get(stage2_cfg, backbone_cfg, "depth", 12)),
        num_heads=int(_cfg_get(stage2_cfg, backbone_cfg, "num_heads", 6)),
        head_dim=_optional_int(_cfg_get(stage2_cfg, backbone_cfg, "head_dim", 64)),
        mlp_ratio=int(_cfg_get(stage2_cfg, backbone_cfg, "mlp_ratio", 4)),
        dropout=float(_cfg_get(stage2_cfg, backbone_cfg, "dropout", 0.0)),
        qk_norm=bool(_cfg_get(stage2_cfg, backbone_cfg, "qk_norm", True)),
        qkvo_routing=str(attention_cfg.get("qkvo_routing", _cfg_get(stage2_cfg, backbone_cfg, "qkvo_routing", "shared_plus_delta"))),
        route_delta_gamma_init=float(
            attention_cfg.get(
                "route_delta_gamma_init",
                _cfg_get(stage2_cfg, backbone_cfg, "route_delta_gamma_init", 0.0),
            )
        ),
        time_scale=float(_cfg_get(stage2_cfg, backbone_cfg, "time_scale", 1000.0)),
        rope_base=float(_cfg_get(stage2_cfg, backbone_cfg, "rope_base", 10000.0)),
        use_rollout_progress=bool(conditioning_cfg.get("use_rollout_progress", True)),
    )


def _cfg_get(stage2_cfg: Mapping[str, Any], backbone_cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    if key in backbone_cfg:
        return backbone_cfg[key]
    return stage2_cfg.get(key, default)


def _nested(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return value if value is not None else default


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _assert_shared_override_matches(config: Mapping[str, Any], key: str, expected: int) -> None:
    if key in config and int(config[key]) != int(expected):
        raise ValueError(
            f"model.stage2.backbone.{key}={config[key]} must match shared model.stage2.{key}={expected}."
        )


def _require_value(config: Mapping[str, Any], key: str, expected: Any) -> None:
    if key in config and config[key] != expected:
        raise ValueError(f"Stage2TriStreamMoTBackbone only supports {key}={expected!r}, got {config[key]!r}.")
