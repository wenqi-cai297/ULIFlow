from __future__ import annotations

import torch
from torch import nn


class RelativeTimeBias(nn.Module):
    """Learned soft preference for nearby query/key time ids."""

    def __init__(self, alpha_init: float = 0.1) -> None:
        super().__init__()
        if alpha_init <= 0:
            raise ValueError(f"alpha_init must be positive, got {alpha_init}.")
        self.relative_time_alpha = nn.Parameter(_inverse_softplus(torch.tensor(float(alpha_init))))

    def forward(
        self,
        query_time_ids: torch.Tensor,
        key_time_ids: torch.Tensor,
        dtype: torch.dtype | None = None,
        key_time_mask: torch.Tensor | None = None,
        zero_bias_key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_time_ids.ndim != 2:
            raise ValueError(f"query_time_ids must have shape [B,Q], got {tuple(query_time_ids.shape)}.")
        if key_time_ids.ndim != 2:
            raise ValueError(f"key_time_ids must have shape [B,K], got {tuple(key_time_ids.shape)}.")
        if query_time_ids.shape[0] != key_time_ids.shape[0]:
            raise ValueError("query_time_ids and key_time_ids must have the same batch size.")
        device = query_time_ids.device
        key_time_ids = key_time_ids.to(device=device)
        q = query_time_ids.to(dtype=torch.float32)
        k = key_time_ids.to(dtype=torch.float32)
        alpha = torch.nn.functional.softplus(self.relative_time_alpha).to(device=device, dtype=torch.float32)
        bias = -alpha * (q[..., None] - k[:, None, :]).abs()

        zero_mask = None
        if key_time_mask is not None:
            if key_time_mask.shape != key_time_ids.shape:
                raise ValueError(f"key_time_mask must have shape {tuple(key_time_ids.shape)}.")
            zero_mask = ~key_time_mask.to(device=device, dtype=torch.bool)
        if zero_bias_key_mask is not None:
            if zero_bias_key_mask.shape != key_time_ids.shape:
                raise ValueError(f"zero_bias_key_mask must have shape {tuple(key_time_ids.shape)}.")
            current = zero_bias_key_mask.to(device=device, dtype=torch.bool)
            zero_mask = current if zero_mask is None else (zero_mask | current)
        if zero_mask is not None:
            bias = bias.masked_fill(zero_mask[:, None, :], 0.0)

        if dtype is not None:
            bias = bias.to(dtype=dtype)
        return bias[:, None, :, :]


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1e-6)
    return torch.log(torch.expm1(value))
