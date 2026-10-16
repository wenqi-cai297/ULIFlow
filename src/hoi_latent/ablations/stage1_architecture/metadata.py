from __future__ import annotations

from torch import nn


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if not trainable_only or parameter.requires_grad)
