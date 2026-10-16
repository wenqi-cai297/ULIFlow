from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

from hoi_latent.models.stage1.mask_utils import safe_masked_softmax


class CrossAttentionPool(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_output_proj: bool = True,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model) if use_output_proj else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_mask: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query.ndim != 3:
            raise ValueError(f"query must have shape [B,Q,D], got {tuple(query.shape)}.")
        if key_value.ndim != 3:
            raise ValueError(f"key_value must have shape [B,K,D], got {tuple(key_value.shape)}.")
        if query.shape[0] != key_value.shape[0] or query.shape[-1] != self.d_model:
            raise ValueError("query and key_value batch/d_model dimensions are incompatible.")
        if key_value.shape[-1] != self.d_model:
            raise ValueError(f"key_value last dim must be {self.d_model}, got {key_value.shape[-1]}.")
        if key_mask.shape != key_value.shape[:2]:
            raise ValueError(f"key_mask must have shape {tuple(key_value.shape[:2])}, got {tuple(key_mask.shape)}.")
        if query_mask is not None and query_mask.shape != query.shape[:2]:
            raise ValueError(f"query_mask must have shape {tuple(query.shape[:2])}, got {tuple(query_mask.shape)}.")

        b, q_len, _ = query.shape
        k_len = key_value.shape[1]
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        if attn_bias is not None:
            bias = _normalize_attn_bias(attn_bias, b, self.num_heads, q_len, k_len)
            scores = scores + bias.to(dtype=scores.dtype, device=scores.device)

        attn = safe_masked_softmax(scores, key_mask[:, None, None, :], dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        valid_output = key_mask.any(dim=1, keepdim=True).expand(b, q_len)
        if query_mask is not None:
            valid_output = valid_output & query_mask
        out = out * valid_output.to(dtype=out.dtype).unsqueeze(-1)
        return out, attn if need_weights else None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(b, n, self.d_model)


def _normalize_attn_bias(
    attn_bias: torch.Tensor,
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
) -> torch.Tensor:
    if attn_bias.ndim == 3:
        attn_bias = attn_bias[:, None, :, :]
    if attn_bias.ndim != 4:
        raise ValueError(f"attn_bias must have rank 3 or 4, got {tuple(attn_bias.shape)}.")
    b, h, q, k = attn_bias.shape
    if b != batch_size or q not in (1, query_len) or k != key_len:
        raise ValueError(
            "attn_bias shape must be [B,H,Q,K] or [B,1,Q,K], "
            f"got {tuple(attn_bias.shape)} for B={batch_size}, Q={query_len}, K={key_len}."
        )
    if h not in (1, num_heads):
        raise ValueError(f"attn_bias head dimension must be 1 or {num_heads}, got {h}.")
    return attn_bias
