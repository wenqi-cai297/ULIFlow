from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from hoi_latent.utils.hash_utils import stable_hash_int


@dataclass(frozen=True)
class WindowSpec:
    start: int
    end: int
    real_length: int
    pad_before: int = 0
    pad_after: int = 0


class FixedWindowSampler:
    def __init__(self, window_size: Optional[int], mode: str, seed: int = 42):
        self.window_size = window_size
        self.mode = mode
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        supported = {"random", "center", "first", "deterministic_random", "full"}
        if mode not in supported:
            raise ValueError(f"Unsupported window mode {mode!r}; expected one of {sorted(supported)}.")

    def sample(
        self, length: int, *, index: Optional[int] = None, epoch: Optional[int] = None
    ) -> WindowSpec:
        if length <= 0:
            raise ValueError(f"Sequence length must be positive, got {length}.")

        if self.mode == "full" or self.window_size is None:
            return WindowSpec(start=0, end=length, real_length=length)

        window_size = int(self.window_size)
        if window_size <= 0:
            raise ValueError(f"window_size must be positive or None, got {self.window_size}.")

        if length <= window_size:
            return WindowSpec(
                start=0,
                end=length,
                real_length=length,
                pad_after=window_size - length,
            )

        max_start = length - window_size
        if self.mode == "first":
            start = 0
        elif self.mode == "center":
            start = max_start // 2
        elif self.mode == "deterministic_random":
            key = f"{index if index is not None else 0}:{epoch if epoch is not None else 0}:{length}"
            start = stable_hash_int(key, seed=self.seed) % (max_start + 1)
        else:
            start = int(self._rng.integers(0, max_start + 1))

        return WindowSpec(start=start, end=start + window_size, real_length=window_size)
