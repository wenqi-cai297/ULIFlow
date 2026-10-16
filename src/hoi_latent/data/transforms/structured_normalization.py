from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hoi_latent.data.representations.interact_stage1_structured import (
    build_stage1_raw_groups_from_motion_window,
    derivative_mask,
)
from hoi_latent.utils.filesystem import ensure_dir
from hoi_latent.utils.json_io import load_json, save_json


STAGE1_STRUCTURED_NORMALIZATION_WHITELIST = frozenset(
    {
        ("human", "pos"),
        ("human", "vel"),
        ("human", "acc"),
        ("human", "vel_official"),
        ("human", "root"),
        ("human", "root_vel"),
        ("human", "root_acc"),
        ("human", "pos_rootrel"),
        ("human", "vel_rootrel"),
        ("human", "acc_rootrel"),
        ("human", "left_heights"),
        ("human", "right_heights"),
        ("object", "rot6d"),
        ("object", "trans"),
        ("object", "trans_vel"),
        ("object", "trans_acc"),
        ("object", "vel9_official"),
        ("object", "rot6d_vel_official"),
        ("object", "trans_vel_official"),
        ("relation", "delta_m2o"),
        ("relation", "delta_o2m"),
        ("relation", "point_canon"),
        ("relation", "dist"),
        ("relation", "delta_vel"),
        ("relation", "delta_acc"),
        ("relation", "nearest_point_world"),
        ("relation", "nearest_point_vel"),
        ("relation", "nearest_point_acc"),
        ("relation", "ground_marker"),
        ("relation", "root_object_delta"),
        ("relation", "root_object_dist"),
        ("relation", "root_object_delta_vel"),
        ("relation", "root_object_delta_acc"),
    }
)


class Stage1StructuredNormalizer:
    def __init__(
        self,
        stats: dict[str, dict[str, dict[str, np.ndarray]]],
        eps: float = 1e-6,
        normalization_whitelist=None,
    ):
        self.stats = stats
        self.eps = float(eps)
        self.normalization_whitelist = frozenset(
            normalization_whitelist or STAGE1_STRUCTURED_NORMALIZATION_WHITELIST
        )

    def normalize_group(self, group_name: str, field_name: str, value: np.ndarray) -> np.ndarray:
        value_arr = np.asarray(value, dtype=np.float32)
        if (group_name, field_name) not in self.normalization_whitelist:
            return value_arr.copy()
        group_stats = self.stats.get(group_name, {})
        field_stats = group_stats.get(field_name)
        if field_stats is None:
            return value_arr.copy()
        mean = np.asarray(field_stats["mean"], dtype=np.float32)
        std = np.asarray(field_stats["std"], dtype=np.float32)
        return ((value_arr - mean) / std).astype(np.float32, copy=False)

    def normalize_sample_raw_groups(self, raw: dict) -> dict:
        normalized = {}
        for group_name, group in raw.items():
            normalized[group_name] = {}
            for field_name, value in group.items():
                normalized[group_name][field_name] = self.normalize_group(group_name, field_name, value)
        return normalized

    def denormalize_group_torch(
        self,
        group_name: str,
        field_name: str,
        value: torch.Tensor,
        *,
        require_stats: bool = False,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"value must be a torch.Tensor, got {type(value).__name__}.")
        if (group_name, field_name) not in self.normalization_whitelist:
            if require_stats:
                raise KeyError(f"Structured normalizer does not whitelist {group_name}.{field_name}.")
            return value.clone()
        group_stats = self.stats.get(group_name, {})
        field_stats = group_stats.get(field_name)
        if field_stats is None:
            if require_stats:
                raise KeyError(f"Structured normalizer stats missing {group_name}.{field_name}.")
            return value.clone()
        mean = torch.as_tensor(field_stats["mean"], device=value.device, dtype=value.dtype)
        std = torch.as_tensor(field_stats["std"], device=value.device, dtype=value.dtype)
        return value * std + mean

    def save(self, directory: str | Path, metadata: dict | None = None) -> None:
        out_dir = ensure_dir(directory)
        fields = {}
        for group_name, group in self.stats.items():
            group_dir = ensure_dir(out_dir / group_name)
            fields[group_name] = {}
            for field_name, field_stats in group.items():
                np.save(group_dir / f"{field_name}_mean.npy", field_stats["mean"])
                np.save(group_dir / f"{field_name}_std.npy", field_stats["std"])
                fields[group_name][field_name] = {
                    "mean": str(Path(group_name) / f"{field_name}_mean.npy"),
                    "std": str(Path(group_name) / f"{field_name}_std.npy"),
                    "shape": list(field_stats["mean"].shape),
                    "count": int(field_stats["count"]),
                    "mask": field_stats.get("mask", "valid"),
                    "min_std_before_clamp": float(field_stats["min_std_before_clamp"]),
                    "max_std_before_clamp": float(field_stats["max_std_before_clamp"]),
                    "num_std_below_eps": int(field_stats["num_std_below_eps"]),
                }
        save_json(
            {
                "version": "interact_stage1_structured_v1",
                "eps": self.eps,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "fields": fields,
                "normalization_whitelist": [
                    f"{group_name}.{field_name}"
                    for group_name, field_name in sorted(self.normalization_whitelist)
                ],
                "metadata": metadata or {},
            },
            out_dir / "stats.json",
        )
        save_json(
            {
                "message": "Structured H/O/R stats are separate from the legacy 962-D mean/std.",
                "version": "interact_stage1_structured_v1",
            },
            out_dir / "README_stats.json",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "Stage1StructuredNormalizer":
        in_dir = Path(directory)
        stats_meta = load_json(in_dir / "stats.json")
        loaded = {}
        for group_name, group in stats_meta["fields"].items():
            loaded[group_name] = {}
            for field_name, field in group.items():
                loaded[group_name][field_name] = {
                    "mean": np.load(in_dir / field["mean"]).astype(np.float32),
                    "std": np.load(in_dir / field["std"]).astype(np.float32),
                    "count": field.get("count", 0),
                    "mask": field.get("mask", "valid"),
                    "min_std_before_clamp": field.get("min_std_before_clamp", 0.0),
                    "max_std_before_clamp": field.get("max_std_before_clamp", 0.0),
                    "num_std_below_eps": field.get("num_std_below_eps", 0),
                }
        whitelist = _parse_whitelist(
            stats_meta.get("normalization_whitelist"),
            default=STAGE1_STRUCTURED_NORMALIZATION_WHITELIST,
        )
        return cls(loaded, eps=stats_meta.get("eps", 1e-6), normalization_whitelist=whitelist)


def zero_stage1_model_groups_by_masks(
    model_groups: dict,
    valid_mask: np.ndarray,
    vel_mask: np.ndarray | None = None,
    acc_mask: np.ndarray | None = None,
) -> dict:
    valid = np.asarray(valid_mask, dtype=bool)
    vel = derivative_mask(valid, order=1) if vel_mask is None else np.asarray(vel_mask, dtype=bool)
    acc = derivative_mask(valid, order=2) if acc_mask is None else np.asarray(acc_mask, dtype=bool)
    out = {}
    for group_name, group in model_groups.items():
        out[group_name] = {}
        for field_name, value in group.items():
            mask = _field_time_mask(field_name, valid, vel, acc)
            out[group_name][field_name] = _zero_by_time_mask(value, mask)
    return out


def compute_streaming_structured_normalizer(
    records,
    *,
    eps: float = 1e-6,
    split: str = "train",
    root_method: str = "marker_mean",
) -> tuple[Stage1StructuredNormalizer, dict]:
    accum: dict[str, dict[str, _RunningStats]] = defaultdict(dict)
    used_sequences = 0
    used_frames = 0

    for record in records:
        if record.split != split:
            continue
        motion = np.load(record.motion_path, mmap_mode="r")
        raw_motion = np.asarray(motion, dtype=np.float32)
        valid_mask = np.ones((raw_motion.shape[0],), dtype=bool)
        raw = build_stage1_raw_groups_from_motion_window(
            raw_motion,
            valid_mask,
            root_method=root_method,
        )
        vel_mask = derivative_mask(valid_mask, order=1)
        acc_mask = derivative_mask(valid_mask, order=2)
        used_sequences += 1
        used_frames += int(raw_motion.shape[0])
        for group_name, group in raw.items():
            for field_name, value in group.items():
                if (group_name, field_name) not in STAGE1_STRUCTURED_NORMALIZATION_WHITELIST:
                    continue
                value_arr = np.asarray(value, dtype=np.float64)
                if not np.isfinite(value_arr).all():
                    raise ValueError(
                        f"Non-finite structured field {group_name}.{field_name} in {record.motion_path}."
                    )
                if field_name not in accum[group_name]:
                    accum[group_name][field_name] = _RunningStats(value_arr.shape[1:])
                mask_name = _field_mask_name(field_name)
                field_mask = _field_time_mask(field_name, valid_mask, vel_mask, acc_mask)
                accum[group_name][field_name].update(value_arr, mask=field_mask, mask_name=mask_name)

    if used_frames <= 0:
        raise ValueError(f"No frames found for split {split!r}; cannot compute structured normalization.")

    stats = {}
    metadata_fields = {}
    for group_name, group in accum.items():
        stats[group_name] = {}
        metadata_fields[group_name] = {}
        for field_name, running in group.items():
            field_stats = running.finalize(eps=eps)
            stats[group_name][field_name] = field_stats
            metadata_fields[group_name][field_name] = {
                "shape": list(field_stats["mean"].shape),
                "count": int(field_stats["count"]),
                "mask": field_stats["mask"],
                "min_std_before_clamp": float(field_stats["min_std_before_clamp"]),
                "max_std_before_clamp": float(field_stats["max_std_before_clamp"]),
                "num_std_below_eps": int(field_stats["num_std_below_eps"]),
            }

    normalizer = Stage1StructuredNormalizer(stats, eps=eps)
    metadata = {
        "split": split,
        "num_sequences": used_sequences,
        "num_frames": used_frames,
        "root_method": root_method,
        "normalization_whitelist": [
            f"{group_name}.{field_name}"
            for group_name, field_name in sorted(STAGE1_STRUCTURED_NORMALIZATION_WHITELIST)
        ],
        "fields": metadata_fields,
    }
    return normalizer, metadata


class _RunningStats:
    def __init__(self, feature_shape: tuple[int, ...]):
        self.count = 0
        self.sum_x = np.zeros(feature_shape, dtype=np.float64)
        self.sum_x2 = np.zeros(feature_shape, dtype=np.float64)
        self.mask = "valid"

    def update(self, value: np.ndarray, mask: np.ndarray | None = None, mask_name: str = "valid") -> None:
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (value.shape[0],):
                raise ValueError(f"mask must have shape ({value.shape[0]},), got {mask.shape}.")
            value = value[mask]
        if value.shape[0] == 0:
            return
        self.mask = mask_name
        self.count += int(value.shape[0])
        self.sum_x += value.sum(axis=0)
        self.sum_x2 += np.square(value).sum(axis=0)

    def finalize(self, eps: float) -> dict[str, Any]:
        if self.count <= 0:
            raise ValueError("Cannot finalize structured stats with zero valid frames.")
        mean = self.sum_x / self.count
        var = np.maximum(self.sum_x2 / self.count - mean * mean, 0.0)
        std_before = np.sqrt(var)
        std = np.maximum(std_before, eps)
        return {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "count": self.count,
            "mask": self.mask,
            "min_std_before_clamp": float(std_before.min()),
            "max_std_before_clamp": float(std_before.max()),
            "num_std_below_eps": int(np.sum(std_before < eps)),
        }


def _field_mask_name(field_name: str) -> str:
    lower = field_name.lower()
    if "acc" in lower:
        return "acc"
    if "vel" in lower:
        return "vel"
    return "valid"


def _field_time_mask(
    field_name: str,
    valid_mask: np.ndarray,
    vel_mask: np.ndarray,
    acc_mask: np.ndarray,
) -> np.ndarray:
    mask_name = _field_mask_name(field_name)
    if mask_name == "acc":
        return np.asarray(acc_mask, dtype=bool)
    if mask_name == "vel":
        return np.asarray(vel_mask, dtype=bool)
    return np.asarray(valid_mask, dtype=bool)


def _zero_by_time_mask(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32).copy()
    if out.ndim == 0 or out.shape[0] != mask.shape[0]:
        return out
    expand_shape = (mask.shape[0],) + (1,) * (out.ndim - 1)
    out *= mask.reshape(expand_shape).astype(np.float32)
    return out


def _parse_whitelist(value, default) -> frozenset:
    if not value:
        return frozenset(default)
    parsed = []
    for item in value:
        if isinstance(item, str) and "." in item:
            group_name, field_name = item.split(".", 1)
            parsed.append((group_name, field_name))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            parsed.append((str(item[0]), str(item[1])))
    return frozenset(parsed)
