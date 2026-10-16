from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hoi_latent.models.stage1.mask_utils import safe_masked_softmax, zero_invalid_tokens
from hoi_latent.models.stage2.position_encoding import Stage2TemporalRoPE
from hoi_latent.models.stage2.stem_types import NUM_STAGE2_FLOW_ROUTES, Stage2FlowRouteId, Stage2PackedStreams


ROUTE_IDS: tuple[int, ...] = tuple(int(route) for route in Stage2FlowRouteId)


def _route_key(route_id: int) -> str:
    return str(int(route_id))


def apply_routed_token_module(
    modules_by_route: nn.ModuleDict,
    x: torch.Tensor,
    route_index: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_routes: int = NUM_STAGE2_FLOW_ROUTES,
    out_dim: int | None = None,
) -> torch.Tensor:
    """Apply per-route token modules by gather/scatter.

    Supports `[B,N,D]` tensors and `[B,H,N,D]` tensors. The route and valid masks
    live on the token axis `[B,N]`; padding tokens are skipped and remain zero.
    """

    if route_index.shape != valid_mask.shape:
        raise ValueError(
            f"route_index and valid_mask must have the same shape, got {tuple(route_index.shape)} and {tuple(valid_mask.shape)}."
        )
    if x.ndim not in {3, 4}:
        raise ValueError(f"x must have rank 3 or 4, got {tuple(x.shape)}.")
    if x.shape[0] != route_index.shape[0] or x.shape[-2 if x.ndim == 4 else 1] != route_index.shape[1]:
        raise ValueError("x token axis does not match route_index shape.")

    route_flat = route_index.reshape(-1).to(dtype=torch.long)
    valid_flat = valid_mask.reshape(-1).to(dtype=torch.bool)
    if x.ndim == 3:
        b, n, d = x.shape
        out_d = d if out_dim is None else int(out_dim)
        flat_x = x.reshape(b * n, d)
        flat_out = x.new_zeros(b * n, out_d)
        out_shape = (b, n, out_d)
        restore = lambda value: value.reshape(out_shape)
    else:
        if out_dim is not None and int(out_dim) != x.shape[-1]:
            raise ValueError("out_dim for rank-4 routed modules must match the input head dimension.")
        b, h, n, d = x.shape
        flat_x = x.permute(0, 2, 1, 3).contiguous().reshape(b * n, h, d)
        flat_out = torch.zeros_like(flat_x)

        def restore(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(b, n, h, d).permute(0, 2, 1, 3).contiguous()

    for route in range(num_routes):
        key = _route_key(route)
        if key not in modules_by_route:
            raise KeyError(f"Missing routed module for route {route}.")
        idx = (route_flat == route) & valid_flat
        if idx.any():
            flat_out[idx] = modules_by_route[key](flat_x[idx]).to(dtype=flat_out.dtype)
    return restore(flat_out)


class Stage2TimeEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        frequency_embedding_size: int = 256,
        max_period: float = 10000.0,
        time_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if frequency_embedding_size <= 0:
            raise ValueError(f"frequency_embedding_size must be positive, got {frequency_embedding_size}.")
        self.d_model = int(d_model)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.max_period = float(max_period)
        self.time_scale = float(time_scale)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)}.")
        emb = self.timestep_embedding(t)
        out = self.mlp(emb.to(device=t.device, dtype=self.mlp[0].weight.dtype))
        return out if dtype is None else out.to(dtype=dtype)

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half, 1)
        )
        args = (t.float() * self.time_scale)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class TriStreamAdaLNZero(nn.Module):
    def __init__(self, d_model: int, num_routes: int = NUM_STAGE2_FLOW_ROUTES, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_routes = int(num_routes)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
        self.modulation_by_route = nn.ModuleDict(
            {_route_key(route): nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model)) for route in range(num_routes)}
        )
        for module in self.modulation_by_route.values():
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

    def compute_modulations(
        self,
        cond_tokens: torch.Tensor,
        route_index: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if cond_tokens.ndim != 3 or cond_tokens.shape[-1] != self.d_model:
            raise ValueError(f"cond_tokens must have shape [B,N,{self.d_model}], got {tuple(cond_tokens.shape)}.")
        if route_index.shape != cond_tokens.shape[:2] or valid_mask.shape != cond_tokens.shape[:2]:
            raise ValueError("route_index and valid_mask must match cond_tokens token shape.")
        return apply_routed_token_module(
            self.modulation_by_route,
            cond_tokens,
            route_index,
            valid_mask,
            num_routes=self.num_routes,
            out_dim=6 * self.d_model,
        )

    def apply(
        self,
        x: torch.Tensor,
        token_mods: torch.Tensor,
        branch: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if branch not in {"attn", "ffn"}:
            raise ValueError(f"branch must be 'attn' or 'ffn', got {branch!r}.")
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B,N,D], got {tuple(x.shape)}.")
        b, n, d = x.shape
        if token_mods.shape != (b, n, 6 * d):
            raise ValueError(f"token_mods must have shape {(b, n, 6 * d)}, got {tuple(token_mods.shape)}.")

        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = token_mods.chunk(6, dim=-1)
        if branch == "attn":
            shift, scale, gate = shift_attn, scale_attn, gate_attn
        else:
            shift, scale, gate = shift_ffn, scale_ffn, gate_ffn
        normed = self.norm(x)
        modulated = normed * (1.0 + scale.to(dtype=normed.dtype)) + shift.to(dtype=normed.dtype)
        return modulated, gate


class SharedPlusDeltaQKVOProjection(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_routes: int = NUM_STAGE2_FLOW_ROUTES,
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_routes = int(num_routes)
        self.q_shared = nn.Linear(d_model, d_model)
        self.k_shared = nn.Linear(d_model, d_model)
        self.v_shared = nn.Linear(d_model, d_model)
        self.o_shared = nn.Linear(d_model, d_model)
        self.delta_q = nn.ModuleDict({_route_key(route): nn.Linear(d_model, d_model) for route in range(num_routes)})
        self.delta_k = nn.ModuleDict({_route_key(route): nn.Linear(d_model, d_model) for route in range(num_routes)})
        self.delta_v = nn.ModuleDict({_route_key(route): nn.Linear(d_model, d_model) for route in range(num_routes)})
        self.delta_o = nn.ModuleDict({_route_key(route): nn.Linear(d_model, d_model) for route in range(num_routes)})
        self.gamma_q = nn.ParameterDict(
            {_route_key(route): nn.Parameter(torch.tensor(float(gamma_init))) for route in range(num_routes)}
        )
        self.gamma_k = nn.ParameterDict(
            {_route_key(route): nn.Parameter(torch.tensor(float(gamma_init))) for route in range(num_routes)}
        )
        self.gamma_v = nn.ParameterDict(
            {_route_key(route): nn.Parameter(torch.tensor(float(gamma_init))) for route in range(num_routes)}
        )
        self.gamma_o = nn.ParameterDict(
            {_route_key(route): nn.Parameter(torch.tensor(float(gamma_init))) for route in range(num_routes)}
        )

    def q(self, x: torch.Tensor, route_index: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self._project(x, route_index, valid_mask, "q")

    def k(self, x: torch.Tensor, route_index: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self._project(x, route_index, valid_mask, "k")

    def v(self, x: torch.Tensor, route_index: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self._project(x, route_index, valid_mask, "v")

    def o(self, x: torch.Tensor, route_index: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self._project(x, route_index, valid_mask, "o")

    def _project(self, x: torch.Tensor, route_index: torch.Tensor, valid_mask: torch.Tensor, kind: str) -> torch.Tensor:
        shared = getattr(self, f"{kind}_shared")(x)
        delta_modules = getattr(self, f"delta_{kind}")
        gamma_params = getattr(self, f"gamma_{kind}")
        delta = self._delta(delta_modules, gamma_params, x, route_index, valid_mask)
        return shared + delta.to(dtype=shared.dtype)

    def _delta(
        self,
        modules_by_route: nn.ModuleDict,
        gamma_by_route: nn.ParameterDict,
        x: torch.Tensor,
        route_index: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = x.shape
        flat_x = x.reshape(b * n, d)
        flat_out = torch.zeros_like(flat_x)
        flat_route = route_index.reshape(-1).to(dtype=torch.long)
        flat_valid = valid_mask.reshape(-1).to(dtype=torch.bool)
        for route in range(self.num_routes):
            key = _route_key(route)
            idx = (flat_route == route) & flat_valid
            if idx.any():
                gamma = gamma_by_route[key].to(device=x.device, dtype=flat_out.dtype)
                value = modules_by_route[key](flat_x[idx]).to(dtype=flat_out.dtype)
                flat_out[idx] = gamma * value
        return flat_out.reshape(b, n, d)


class RoutedQKNorm(nn.Module):
    def __init__(self, head_dim: int, num_routes: int = NUM_STAGE2_FLOW_ROUTES, eps: float = 1.0e-6) -> None:
        super().__init__()
        norm_cls = getattr(nn, "RMSNorm", _RMSNorm)
        self.q_norms = nn.ModuleDict({_route_key(route): norm_cls(head_dim, eps=eps) for route in range(num_routes)})
        self.k_norms = nn.ModuleDict({_route_key(route): norm_cls(head_dim, eps=eps) for route in range(num_routes)})
        self.num_routes = int(num_routes)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        route_index: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_out = apply_routed_token_module(self.q_norms, q, route_index, valid_mask, num_routes=self.num_routes)
        k_out = apply_routed_token_module(self.k_norms, k, route_index, valid_mask, num_routes=self.num_routes)
        return q_out.to(dtype=q.dtype), k_out.to(dtype=k.dtype)


class Stage2FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(d_model) * int(mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        out = x.float() * torch.rsqrt(variance + self.eps)
        return (out.to(dtype=x.dtype) * self.weight.to(device=x.device, dtype=x.dtype))


class SafeTriStreamJointAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = float(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_mask: torch.Tensor,
        pairwise_mask: torch.Tensor,
        route_index: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        force_manual: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self._validate(q, k, v, query_mask, pairwise_mask)
        if force_manual or return_diagnostics:
            out, attn = self.manual_attention(q, k, v, query_mask=query_mask, pairwise_mask=pairwise_mask)
            diagnostics = self._diagnostics(attn, query_mask, pairwise_mask, route_index) if return_diagnostics else {}
            return out, diagnostics

        pairwise_bool = pairwise_mask.to(device=q.device, dtype=torch.bool)
        if not pairwise_bool.any(dim=-1).all():
            raise ValueError("Stage2 pairwise attention mask contains an all-masked query row.")
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=pairwise_bool[:, None, :, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out * query_mask[:, None, :, None].to(device=out.device, dtype=out.dtype)
        return out, {}

    @staticmethod
    def manual_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_mask: torch.Tensor,
        pairwise_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = safe_masked_softmax(scores, pairwise_mask[:, None, :, :].to(device=q.device), dim=-1)
        out = torch.matmul(attn.to(dtype=v.dtype), v)
        out = out * query_mask[:, None, :, None].to(device=out.device, dtype=out.dtype)
        return out, attn.to(dtype=q.dtype)

    @staticmethod
    def _validate(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_mask: torch.Tensor,
        pairwise_mask: torch.Tensor,
    ) -> None:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must have shape [B,H,N,D].")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"q/k/v shapes must match, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}.")
        expected = (q.shape[0], q.shape[2])
        if query_mask.shape != expected:
            raise ValueError(f"query_mask must have shape {expected}.")
        if pairwise_mask.shape != (q.shape[0], q.shape[2], q.shape[2]):
            raise ValueError(f"pairwise_mask must have shape {(q.shape[0], q.shape[2], q.shape[2])}.")

    def _diagnostics(
        self,
        attn: torch.Tensor,
        query_mask: torch.Tensor,
        pairwise_mask: torch.Tensor,
        route_index: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        eps = torch.finfo(attn.dtype).eps
        valid_queries = query_mask[:, None, :].expand(attn.shape[0], attn.shape[1], attn.shape[2])
        entropy = -(attn * (attn + eps).log()).sum(dim=-1)
        entropy = _masked_average(entropy, valid_queries)
        diagnostics: dict[str, torch.Tensor] = {"attention_entropy": entropy.detach()}
        if route_index is not None:
            diagnostics["cross_route_attention_mass"] = _cross_route_attention_mass(
                attn, query_mask, pairwise_mask, route_index
            ).detach()
        return diagnostics


class TriStreamMoTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        qk_norm: bool = True,
        route_delta_gamma_init: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.adaln = TriStreamAdaLNZero(d_model)
        self.qkv = SharedPlusDeltaQKVOProjection(d_model, gamma_init=route_delta_gamma_init)
        self.qk_norm = RoutedQKNorm(head_dim) if qk_norm else None
        self.rope = Stage2TemporalRoPE(head_dim=head_dim, base=rope_base)
        self.attention = SafeTriStreamJointAttention(dropout=dropout)
        self.ffn = nn.ModuleDict(
            {_route_key(route): Stage2FeedForward(d_model, mlp_ratio=mlp_ratio, dropout=dropout) for route in ROUTE_IDS}
        )

    def forward(
        self,
        x: torch.Tensor,
        packed: Stage2PackedStreams,
        route_index: torch.Tensor,
        cond_tokens: torch.Tensor,
        pairwise_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        token_mods = self.adaln.compute_modulations(
            cond_tokens.to(device=x.device, dtype=x.dtype),
            route_index=route_index,
            valid_mask=packed.attention_mask,
        )
        x_attn, gate_attn = self.adaln.apply(x, token_mods, branch="attn")
        q = self._split_heads(self.qkv.q(x_attn, route_index, packed.attention_mask))
        k = self._split_heads(self.qkv.k(x_attn, route_index, packed.attention_mask))
        v = self._split_heads(self.qkv.v(x_attn, route_index, packed.attention_mask))
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k, route_index, packed.attention_mask)
        q, k = self.rope(q, k, packed.time_ids, packed.apply_temporal_rope)
        attn_out, attn_diag = self.attention(
            q,
            k,
            v,
            query_mask=packed.attention_mask,
            pairwise_mask=pairwise_mask,
            route_index=route_index,
            return_diagnostics=return_diagnostics,
        )
        attn_out = self._merge_heads(attn_out)
        attn_out = self.qkv.o(attn_out, route_index, packed.attention_mask)
        x = zero_invalid_tokens(
            x + gate_attn.to(dtype=x.dtype) * attn_out.to(dtype=x.dtype),
            packed.attention_mask,
        )

        x_ffn, gate_ffn = self.adaln.apply(x, token_mods, branch="ffn")
        ffn_out = apply_routed_token_module(self.ffn, x_ffn, route_index, packed.attention_mask)
        x = zero_invalid_tokens(
            x + gate_ffn.to(dtype=x.dtype) * ffn_out.to(dtype=x.dtype),
            packed.attention_mask,
        )
        diagnostics = {}
        if return_diagnostics:
            diagnostics = {
                "attention": attn_diag,
                "residual_attn_rms": attn_out.detach().pow(2).mean().sqrt(),
                "residual_ffn_rms": ffn_out.detach().pow(2).mean().sqrt(),
            }
        return x, diagnostics

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(b, n, self.d_model)


class Stage2LatentVelocityHead(nn.Module):
    def __init__(self, d_model: int, latent_dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.out_proj = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, latent_hidden: torch.Tensor, cond_latent: torch.Tensor) -> torch.Tensor:
        if latent_hidden.ndim != 3:
            raise ValueError(f"latent_hidden must have shape [B,N,D], got {tuple(latent_hidden.shape)}.")
        if cond_latent.shape != latent_hidden.shape:
            raise ValueError(
                f"cond_latent must have shape {tuple(latent_hidden.shape)}, "
                f"got {tuple(cond_latent.shape)}."
            )
        shift, scale = self.modulation(cond_latent).chunk(2, dim=-1)
        h = self.norm(latent_hidden)
        h = h * (1.0 + scale.to(dtype=h.dtype)) + shift.to(dtype=h.dtype)
        return self.out_proj(h)


def _masked_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = values * mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum().clamp_min(1).to(device=values.device, dtype=values.dtype)
    return weighted.sum() / denom


def _cross_route_attention_mass(
    attn: torch.Tensor,
    query_mask: torch.Tensor,
    pairwise_mask: torch.Tensor,
    route_index: torch.Tensor,
    num_routes: int = NUM_STAGE2_FLOW_ROUTES,
) -> torch.Tensor:
    heads = attn.shape[1]
    out = attn.new_zeros(heads, num_routes, num_routes)
    valid_query = query_mask.to(device=attn.device, dtype=torch.bool)
    valid_key = pairwise_mask.to(device=attn.device, dtype=torch.bool).any(dim=1)
    route_index = route_index.to(device=attn.device, dtype=torch.long)
    for q_route in range(num_routes):
        q_mask = (route_index == q_route) & valid_query
        q_count = q_mask.sum().clamp_min(1).to(dtype=attn.dtype)
        for k_route in range(num_routes):
            k_mask = (route_index == k_route) & valid_key
            mass = attn * q_mask[:, None, :, None].to(dtype=attn.dtype) * k_mask[:, None, None, :].to(dtype=attn.dtype)
            out[:, q_route, k_route] = mass.sum(dim=(0, 2, 3)) / q_count
    return out
