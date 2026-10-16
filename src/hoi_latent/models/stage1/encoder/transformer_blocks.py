from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from hoi_latent.models.stage1.encoder.position_encoding import TemporalRotaryEmbedding
from hoi_latent.models.stage1.mask_utils import safe_masked_softmax, zero_invalid_tokens


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, use_rope: bool = False) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.use_rope = bool(use_rope)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = TemporalRotaryEmbedding(self.head_dim) if self.use_rope else None

    def forward(
        self,
        query: torch.Tensor,
        key_value: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        q_position_ids: Optional[torch.Tensor] = None,
        k_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query.ndim != 3:
            raise ValueError(f"query must have shape [B,Q,D], got {tuple(query.shape)}.")
        if key_value is None:
            key_value = query
        if key_value.ndim != 3:
            raise ValueError(f"key_value must have shape [B,K,D], got {tuple(key_value.shape)}.")
        if query.shape[0] != key_value.shape[0] or query.shape[-1] != self.d_model or key_value.shape[-1] != self.d_model:
            raise ValueError("query and key_value batch/d_model dimensions are incompatible.")
        b, q_len, _ = query.shape
        k_len = key_value.shape[1]
        if key_mask is None:
            key_mask = torch.ones(b, k_len, dtype=torch.bool, device=query.device)
        else:
            key_mask = key_mask.to(device=query.device, dtype=torch.bool)
        if query_mask is None:
            query_mask = torch.ones(b, q_len, dtype=torch.bool, device=query.device)
        else:
            query_mask = query_mask.to(device=query.device, dtype=torch.bool)
        if key_mask.shape != (b, k_len):
            raise ValueError(f"key_mask must have shape {(b, k_len)}, got {tuple(key_mask.shape)}.")
        if query_mask.shape != (b, q_len):
            raise ValueError(f"query_mask must have shape {(b, q_len)}, got {tuple(query_mask.shape)}.")

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))
        if self.rope is not None:
            if q_position_ids is None:
                raise ValueError("q_position_ids is required when use_rope=True.")
            q, k = self.rope(q, k, q_position_ids, k_position_ids)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        attn = safe_masked_softmax(scores, key_mask[:, None, None, :], dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        valid_output = key_mask.any(dim=1, keepdim=True).expand(b, q_len) & query_mask
        return out * valid_output.to(dtype=out.dtype).unsqueeze(-1)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(b, n, self.d_model)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(d_model) * int(ffn_mult)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreNormSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, ffn_mult: int = 4, use_rope: bool = False):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, use_rope=use_rope)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult=ffn_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            key_mask=mask,
            query_mask=mask,
            q_position_ids=position_ids,
            k_position_ids=position_ids,
        )
        x = zero_invalid_tokens(x, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return zero_invalid_tokens(x, mask)


class PreNormAttentionOnlyBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, use_rope: bool = False):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, use_rope=use_rope)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            key_mask=mask,
            query_mask=mask,
            q_position_ids=position_ids,
            k_position_ids=position_ids,
        )
        return zero_invalid_tokens(x, mask)


class PreNormCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, ffn_mult: int = 4):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.key_value_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, use_rope=False)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult=ffn_mult, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: torch.Tensor,
        key_value_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = query + self.attn(
            self.query_norm(query),
            key_value=self.key_value_norm(key_value),
            key_mask=key_value_mask,
            query_mask=query_mask,
        )
        query = zero_invalid_tokens(query, query_mask)
        query = query + self.ffn(self.ffn_norm(query))
        return zero_invalid_tokens(query, query_mask)


class FactorizedTemporalSlotBlock(nn.Module):
    """Frame-slot attention without RoPE, then temporal attention with RoPE."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, ffn_mult: int = 4) -> None:
        super().__init__()
        self.frame_attn = PreNormAttentionOnlyBlock(d_model, num_heads, dropout, use_rope=False)
        self.temporal_attn = PreNormAttentionOnlyBlock(d_model, num_heads, dropout, use_rope=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult=ffn_mult, dropout=dropout)

    def forward(self, x_ts: torch.Tensor, mask_ts: torch.Tensor, time_ids_ts: torch.Tensor) -> torch.Tensor:
        if x_ts.ndim != 4:
            raise ValueError(f"x_ts must have shape [B,T,S,D], got {tuple(x_ts.shape)}.")
        if mask_ts.shape != x_ts.shape[:3]:
            raise ValueError(f"mask_ts must have shape {tuple(x_ts.shape[:3])}, got {tuple(mask_ts.shape)}.")
        if time_ids_ts.shape != x_ts.shape[:3]:
            raise ValueError(f"time_ids_ts must have shape {tuple(x_ts.shape[:3])}, got {tuple(time_ids_ts.shape)}.")
        b, t, s, d = x_ts.shape
        frame_x = x_ts.reshape(b * t, s, d)
        frame_mask = mask_ts.reshape(b * t, s)
        frame_x = self.frame_attn(frame_x, frame_mask)
        x_ts = frame_x.reshape(b, t, s, d)

        temporal_x = x_ts.transpose(1, 2).reshape(b * s, t, d)
        temporal_mask = mask_ts.transpose(1, 2).reshape(b * s, t)
        temporal_time_ids = time_ids_ts.transpose(1, 2).reshape(b * s, t)
        temporal_x = self.temporal_attn(temporal_x, temporal_mask, position_ids=temporal_time_ids)
        x_ts = temporal_x.reshape(b, s, t, d).transpose(1, 2)

        flat_x = x_ts.reshape(b, t * s, d)
        flat_mask = mask_ts.reshape(b, t * s)
        flat_x = flat_x + self.ffn(self.ffn_norm(flat_x))
        flat_x = zero_invalid_tokens(flat_x, flat_mask)
        return flat_x.reshape(b, t, s, d)
