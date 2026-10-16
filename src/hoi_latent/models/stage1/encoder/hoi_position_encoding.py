from __future__ import annotations

import torch
from torch import nn

from hoi_latent.models.stage1.encoder.position_encoding import TemporalRotaryEmbedding


class HOIRotaryPositionEncoding(nn.Module):
    """Temporal-only HOI RoPE, gated per token so static geometry remains unrotated."""

    def __init__(self, head_dim: int, base: float = 10000.0, rotary_dim: int | None = None) -> None:
        super().__init__()
        self.temporal_rope = TemporalRotaryEmbedding(head_dim=head_dim, base=base, rotary_dim=rotary_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        apply_temporal_rope: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(f"q and k must be [B,H,N,D], got {tuple(q.shape)} and {tuple(k.shape)}.")
        if position_ids.shape != (q.shape[0], q.shape[2]):
            raise ValueError(f"position_ids must have shape {(q.shape[0], q.shape[2])}, got {tuple(position_ids.shape)}.")
        if apply_temporal_rope.shape != position_ids.shape:
            raise ValueError(
                f"apply_temporal_rope must have shape {tuple(position_ids.shape)}, "
                f"got {tuple(apply_temporal_rope.shape)}."
            )
        q_rot, k_rot = self.temporal_rope(q, k, position_ids, position_ids)
        apply = apply_temporal_rope.to(device=q.device, dtype=torch.bool)[:, None, :, None]
        q = torch.where(apply, q_rot, q)
        k = torch.where(apply, k_rot, k)
        return q, k
