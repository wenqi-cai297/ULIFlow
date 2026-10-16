from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class TemporalRotaryEmbedding(nn.Module):
    """1D RoPE applied to attention Q/K tensors along the temporal axis."""

    def __init__(self, head_dim: int, base: float = 10000.0, rotary_dim: Optional[int] = None) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}.")
        dim = head_dim if rotary_dim is None else min(int(rotary_dim), int(head_dim))
        dim = (dim // 2) * 2
        self.head_dim = int(head_dim)
        self.rotary_dim = int(dim)
        if self.rotary_dim > 0:
            inv_freq = 1.0 / (float(base) ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        else:
            inv_freq = torch.empty(0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_position_ids: torch.Tensor,
        k_position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_dim == 0:
            return q, k
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(f"q and k must be [B,H,N,D], got {tuple(q.shape)} and {tuple(k.shape)}.")
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError("q/k head dimension does not match TemporalRotaryEmbedding.head_dim.")
        if q_position_ids.shape != (q.shape[0], q.shape[2]):
            raise ValueError(
                f"q_position_ids must have shape {(q.shape[0], q.shape[2])}, got {tuple(q_position_ids.shape)}."
            )
        if k_position_ids is None:
            k_position_ids = q_position_ids
        if k_position_ids.shape != (k.shape[0], k.shape[2]):
            raise ValueError(
                f"k_position_ids must have shape {(k.shape[0], k.shape[2])}, got {tuple(k_position_ids.shape)}."
            )
        q = _apply_rope(q, q_position_ids, self.inv_freq, self.rotary_dim)
        k = _apply_rope(k, k_position_ids, self.inv_freq, self.rotary_dim)
        return q, k


def _apply_rope(x: torch.Tensor, position_ids: torch.Tensor, inv_freq: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    freqs = position_ids.to(dtype=inv_freq.dtype, device=x.device).unsqueeze(-1) * inv_freq.to(device=x.device)
    cos = freqs.cos().to(dtype=x.dtype).unsqueeze(1)
    sin = freqs.sin().to(dtype=x.dtype).unsqueeze(1)
    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]
    x_rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)
    return torch.cat((x_rotated, x_pass), dim=-1)
