from __future__ import annotations

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """Learned absolute time embedding.

    Static tokens should use time_ids=0 with time_mask=False outside this
    module. Negative time ids are never accepted.
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}.")
        self.max_len = int(max_len)
        self.embedding = nn.Embedding(self.max_len, d_model)

    def forward(self, time_ids: torch.Tensor) -> torch.Tensor:
        time_ids = time_ids.to(dtype=torch.long)
        if time_ids.numel() > 0:
            min_id = int(time_ids.min().item())
            max_id = int(time_ids.max().item())
            if min_id < 0 or max_id >= self.max_len:
                raise ValueError(
                    f"time_ids must be in [0, {self.max_len - 1}], got min={min_id}, max={max_id}."
                )
        return self.embedding(time_ids)


class TokenTypeEmbedding(nn.Module):
    def __init__(self, num_token_types: int, d_model: int):
        super().__init__()
        if num_token_types <= 0:
            raise ValueError(f"num_token_types must be positive, got {num_token_types}.")
        self.embedding = nn.Embedding(num_token_types, d_model)

    def forward(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_type_ids.to(dtype=torch.long))
