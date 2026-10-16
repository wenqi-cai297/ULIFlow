from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hoi_latent.data.representations.interact_motion_962 import InterActMotion962
from hoi_latent.utils.filesystem import ensure_dir
from hoi_latent.utils.json_io import save_json, load_json


class FeatureNormalizer:
    def __init__(self, mean: Any, std: Any, eps: float = 1e-6):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.eps = float(eps)
        if self.mean.shape != (InterActMotion962.DIM,):
            raise ValueError(f"mean must have shape ({InterActMotion962.DIM},), got {self.mean.shape}.")
        if self.std.shape != (InterActMotion962.DIM,):
            raise ValueError(f"std must have shape ({InterActMotion962.DIM},), got {self.std.shape}.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("mean/std must be finite.")
        if np.any(self.std < self.eps):
            raise ValueError("std contains values below eps; compute/clamp stats before loading.")

    def transform(self, x: Any) -> Any:
        return (x - self._mean_like(x)) / self._std_like(x)

    def inverse_transform(self, x: Any) -> Any:
        return x * self._std_like(x) + self._mean_like(x)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: dict) -> "FeatureNormalizer":
        return cls(state["mean"], state["std"], eps=state.get("eps", 1e-6))

    def save(self, directory: str | Path, stats: dict | None = None) -> None:
        out_dir = ensure_dir(directory)
        np.save(out_dir / "mean.npy", self.mean)
        np.save(out_dir / "std.npy", self.std)
        if stats is not None:
            save_json(stats, out_dir / "stats.json")

    @classmethod
    def load(cls, directory: str | Path) -> "FeatureNormalizer":
        in_dir = Path(directory)
        stats_path = in_dir / "stats.json"
        eps = load_json(stats_path).get("eps", 1e-6) if stats_path.exists() else 1e-6
        return cls(np.load(in_dir / "mean.npy"), np.load(in_dir / "std.npy"), eps=eps)

    def _mean_like(self, x: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        except ImportError:
            pass
        return self.mean.astype(getattr(x, "dtype", np.float32), copy=False)

    def _std_like(self, x: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        except ImportError:
            pass
        return self.std.astype(getattr(x, "dtype", np.float32), copy=False)


def compute_streaming_normalizer(
    records,
    *,
    eps: float = 1e-6,
    datasets: list[str] | None = None,
    split: str = "train",
) -> tuple[FeatureNormalizer, dict]:
    count = 0
    sum_x = np.zeros((InterActMotion962.DIM,), dtype=np.float64)
    sum_x2 = np.zeros((InterActMotion962.DIM,), dtype=np.float64)
    used_sequences = 0

    for record in records:
        if record.split != split:
            continue
        motion = np.load(record.motion_path, mmap_mode="r")
        InterActMotion962.validate_shape(motion, name=record.motion_path)
        x = np.asarray(motion, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite motion values in {record.motion_path}.")
        count += int(x.shape[0])
        used_sequences += 1
        sum_x += x.sum(axis=0)
        sum_x2 += np.square(x).sum(axis=0)

    if count <= 0:
        raise ValueError(f"No frames found for split {split!r}; cannot compute normalization.")

    mean = sum_x / count
    var = np.maximum(sum_x2 / count - mean * mean, 0.0)
    std_before = np.sqrt(var)
    low_std_indices = np.where(std_before < eps)[0].astype(int).tolist()
    std = np.maximum(std_before, eps)
    normalizer = FeatureNormalizer(mean.astype(np.float32), std.astype(np.float32), eps=eps)
    stats = {
        "representation": "interact_motion_962",
        "dim": InterActMotion962.DIM,
        "num_sequences": used_sequences,
        "num_frames": int(count),
        "datasets": datasets or sorted({record.dataset for record in records}),
        "split": split,
        "eps": eps,
        "min_std_before_clamp": float(std_before.min()),
        "max_std_before_clamp": float(std_before.max()),
        "num_std_below_eps": len(low_std_indices),
        "low_std_indices": low_std_indices,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_565d_stats_used": False,
        "note": "Official 565-D stats were not used.",
    }
    return normalizer, stats
