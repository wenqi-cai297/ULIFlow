from __future__ import annotations

import torch
from torch import nn

from hoi_latent.models.stage1.encoder.position_encoding import TemporalRotaryEmbedding


class Stage2TemporalRoPE(nn.Module):
    """Temporal-only RoPE for Stage2 packed streams.

    The caller supplies an explicit mask for tokens that should receive temporal
    rotation. Text and object-geometry condition tokens remain unrotated even
    though they are present in the packed attention field.
    """

    def __init__(self, head_dim: int, base: float = 10000.0, rotary_dim: int | None = None) -> None:
        super().__init__()
        self.temporal_rope = TemporalRotaryEmbedding(head_dim=head_dim, base=base, rotary_dim=rotary_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        time_ids: torch.Tensor,
        apply_temporal_rope: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(f"q and k must have shape [B,H,N,D], got {tuple(q.shape)} and {tuple(k.shape)}.")
        if q.shape != k.shape:
            raise ValueError(f"q and k shapes must match, got {tuple(q.shape)} and {tuple(k.shape)}.")
        expected = (q.shape[0], q.shape[2])
        if time_ids.shape != expected:
            raise ValueError(f"time_ids must have shape {expected}, got {tuple(time_ids.shape)}.")
        if apply_temporal_rope.shape != expected:
            raise ValueError(
                f"apply_temporal_rope must have shape {expected}, got {tuple(apply_temporal_rope.shape)}."
            )

        q_rot, k_rot = self.temporal_rope(q.float(), k.float(), time_ids, time_ids)
        q_rot = q_rot.to(dtype=q.dtype)
        k_rot = k_rot.to(dtype=k.dtype)
        apply = apply_temporal_rope.to(device=q.device, dtype=torch.bool)[:, None, :, None]
        q = torch.where(apply, q_rot, q)
        k = torch.where(apply, k_rot, k)
        return q, k
