from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from hoi_latent.data.representations.interact_marker_metadata import (
    build_marker_part_diagnostics,
    build_stage1_ids,
)
from hoi_latent.data.representations.interact_motion_962 import InterActMotion962
from hoi_latent.data.representations.interact_stage1_structured import (
    build_stage1_raw_groups_from_motion_window,
    derivative_mask,
)
from hoi_latent.data.schemas import InterActSequenceRecord
from hoi_latent.data.splits.interact_splits import InterActSplitAssigner
from hoi_latent.data.transforms.normalization import FeatureNormalizer
from hoi_latent.data.transforms.structured_normalization import zero_stage1_model_groups_by_masks
from hoi_latent.data.transforms.temporal_window import FixedWindowSampler
from hoi_latent.data.text_utils import clean_interact_caption
from hoi_latent.utils.json_io import load_jsonl
from hoi_latent.utils.filesystem import read_text


def build_interact_stage1_index(
    data_root: str | Path,
    datasets: list[str],
    exclude_datasets: Optional[list[str]] = None,
    require_motion: bool = True,
    load_object_metadata: bool = True,
    validate_motion_shape: bool = True,
    max_shape_checks: Optional[int] = None,
) -> tuple[list[InterActSequenceRecord], dict]:
    exclude = set(exclude_datasets or [])
    requested = list(datasets)
    if "omomo" in requested and "omomo_correct_v2" in requested and "omomo" not in exclude:
        raise ValueError(
            "Both omomo and omomo_correct_v2 were requested. Exclude omomo or run an explicit ablation."
        )

    root = Path(data_root)
    records: list[InterActSequenceRecord] = []
    invalid_records = []
    missing_optional = Counter()
    per_dataset_counts = Counter()
    motion_dim_counts = Counter()
    dtype_counts = Counter()
    lengths = []
    shape_checks = 0

    for dataset in requested:
        if dataset in exclude:
            continue
        sequences_root = root / dataset / "sequences_canonical"
        if not sequences_root.exists():
            invalid_records.append(
                {"dataset": dataset, "reason": "missing_sequences_canonical", "path": str(sequences_root)}
            )
            continue
        object_names = _discover_object_names(root / dataset / "objects")
        for sequence_dir in sorted(p for p in sequences_root.iterdir() if p.is_dir()):
            motion_path = sequence_dir / "motion.npy"
            if require_motion and not motion_path.exists():
                invalid_records.append(
                    {
                        "dataset": dataset,
                        "sequence_name": sequence_dir.name,
                        "reason": "missing_motion",
                        "path": str(motion_path),
                    }
                )
                continue
            try:
                motion = np.load(motion_path, mmap_mode="r")
                if validate_motion_shape:
                    if max_shape_checks is None or shape_checks < max_shape_checks:
                        if motion.ndim != 2 or int(motion.shape[1]) != InterActMotion962.DIM:
                            raise ValueError(f"invalid motion shape {tuple(motion.shape)}")
                        shape_checks += 1
                length = int(motion.shape[0])
                motion_dim = int(motion.shape[1]) if motion.ndim >= 2 else -1
                dtype = str(motion.dtype)
            except Exception as exc:
                invalid_records.append(
                    {
                        "dataset": dataset,
                        "sequence_name": sequence_dir.name,
                        "reason": "invalid_motion",
                        "error": str(exc),
                        "path": str(motion_path),
                    }
                )
                continue

            human_path = _path_if_exists(sequence_dir / "human.npz")
            object_path = _path_if_exists(sequence_dir / "object.npz")
            text_path = _path_if_exists(sequence_dir / "text.txt")
            if human_path is None:
                missing_optional["human_npz"] += 1
            if object_path is None:
                missing_optional["object_npz"] += 1
            if text_path is None:
                missing_optional["text_txt"] += 1

            object_name = None
            if load_object_metadata and object_path is not None:
                object_name = _load_object_name(object_path)
            if object_name is None:
                object_name = _infer_object_name_from_sequence(sequence_dir.name, object_names)

            object_dir = None
            sample_points_path = None
            bps_path = None
            if object_name:
                candidate_object_dir = root / dataset / "objects" / object_name
                if candidate_object_dir.exists():
                    object_dir = str(candidate_object_dir)
                    sample_points_path = _path_if_exists(candidate_object_dir / "sample_points.npy")
                bps_dir = root / dataset / "objects_bps" / object_name
                preferred_bps = bps_dir / f"{object_name}_1024.npy"
                if preferred_bps.exists():
                    bps_path = str(preferred_bps)
                elif bps_dir.exists():
                    first_bps = next(iter(sorted(bps_dir.glob("*.npy"))), None)
                    bps_path = str(first_bps) if first_bps else None
            if sample_points_path is None:
                missing_optional["sample_points"] += 1
            if bps_path is None:
                missing_optional["bps"] += 1

            record = InterActSequenceRecord(
                dataset=dataset,
                sequence_name=sequence_dir.name,
                sequence_dir=str(sequence_dir),
                motion_path=str(motion_path),
                human_path=human_path,
                object_path=object_path,
                text_path=text_path,
                object_name=object_name,
                object_dir=object_dir,
                sample_points_path=sample_points_path,
                bps_path=bps_path,
                length=length,
                motion_dim=motion_dim,
                dtype=dtype,
            )
            records.append(record)
            per_dataset_counts[dataset] += 1
            motion_dim_counts[str(motion_dim)] += 1
            dtype_counts[dtype] += 1
            lengths.append(length)

    length_stats = _length_stats(lengths)
    diagnostics = {
        "data_root": str(root),
        "requested_datasets": requested,
        "excluded_datasets": sorted(exclude),
        "num_records": len(records),
        "per_dataset_counts": dict(per_dataset_counts),
        "length": length_stats,
        "motion_dim_counts": dict(motion_dim_counts),
        "dtype_counts": dict(dtype_counts),
        "missing_optional": {
            "human_npz": int(missing_optional["human_npz"]),
            "object_npz": int(missing_optional["object_npz"]),
            "text_txt": int(missing_optional["text_txt"]),
            "sample_points": int(missing_optional["sample_points"]),
            "bps": int(missing_optional["bps"]),
        },
        "invalid_records": invalid_records,
    }
    return records, diagnostics


class InterActStage1Dataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        datasets: list[str],
        split: str,
        split_config: dict,
        normalizer: Optional[FeatureNormalizer] = None,
        window_size: Optional[int] = 128,
        window_mode: str = "random",
        representation: str = "interact_motion_962",
        load_text: bool = True,
        load_object_points: bool = False,
        load_object_bps: bool = False,
        return_raw_motion: bool = False,
        return_structured: bool = False,
        structured_normalizer: Optional[Any] = None,
        require_object_geometry: Optional[bool] = None,
        text_filter: Optional[dict[str, Any]] = None,
        exclude_datasets: Optional[list[str]] = None,
        index_cache_path: Optional[str] = None,
        object_geometry_cache_size: int = 256,
        marker_part_map_path: Optional[str] = "assets/interact/marker_part_map_v1.json",
        root_method: str = "marker_mean",
        seed: int = 42,
    ):
        if representation != "interact_motion_962":
            raise ValueError(f"Unsupported representation {representation!r}.")
        self.data_root = Path(data_root)
        self.datasets = list(datasets)
        self.split = split
        self.normalizer = normalizer
        self.window_size = window_size
        self.window_mode = window_mode
        self.load_text = load_text
        self.return_raw_motion = return_raw_motion
        self.return_structured = return_structured
        self.structured_normalizer = structured_normalizer
        self.require_object_geometry = return_structured if require_object_geometry is None else bool(require_object_geometry)
        self.text_filter_config = dict(text_filter or {})
        self.load_object_points = bool(load_object_points or (return_structured and self.require_object_geometry))
        self.load_object_bps = bool(load_object_bps or (return_structured and self.require_object_geometry))
        self.exclude_datasets = exclude_datasets or []
        self.object_geometry_cache_size = int(object_geometry_cache_size)
        self.marker_part_map_path = _resolve_optional_path(marker_part_map_path)
        self.root_method = root_method
        self.seed = seed
        self.sampler = FixedWindowSampler(window_size=window_size, mode=window_mode, seed=seed)
        self._geometry_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        if index_cache_path and Path(index_cache_path).exists():
            all_records = [_record_from_dict(item) for item in load_jsonl(index_cache_path)]
            index_diag = {"loaded_index_cache_path": index_cache_path, "num_records": len(all_records)}
        else:
            all_records, index_diag = build_interact_stage1_index(
                self.data_root,
                self.datasets,
                exclude_datasets=self.exclude_datasets,
                require_motion=True,
                load_object_metadata=True,
                validate_motion_shape=True,
            )

        assigner = InterActSplitAssigner(
            test_seq_file=split_config.get("test_seq_file"),
            use_official_test_seq=split_config.get("use_official_test_seq", True),
            val_ratio_from_train=split_config.get("val_ratio_from_train", 0.05),
            val_hash_seed=split_config.get("val_hash_seed", 42),
        )
        assigned, split_diag = assigner.assign(all_records)
        split_records = [record for record in assigned if record.split == split]
        self.index_diagnostics = index_diag
        self.split_diagnostics = split_diag
        if not split_records:
            raise ValueError(f"No InterAct Stage 1 records found for split {split!r}.")
        self.records, self.text_filter_diagnostics = _apply_text_token_filter(split_records, self.text_filter_config)
        if not self.records:
            raise ValueError(
                f"No InterAct Stage 1 records found for split {split!r} after text_filter. "
                f"Diagnostics: {self.text_filter_diagnostics}"
            )
        self.dataset_to_id = {name: idx for idx, name in enumerate(sorted({record.dataset for record in assigned}))}
        self.object_to_id = {
            name: idx
            for idx, name in enumerate(sorted({record.object_name for record in assigned if record.object_name}))
        }
        self.static_ids = build_stage1_ids(self.marker_part_map_path)
        self.marker_part_diagnostics = build_marker_part_diagnostics(
            self.marker_part_map_path,
            self.static_ids["marker_part_ids"],
        )
        self.object_geometry_diagnostics = self._build_object_geometry_diagnostics()
        self.structured_diagnostics = {
            "return_structured": self.return_structured,
            "require_object_geometry": self.require_object_geometry,
            "load_object_points": self.load_object_points,
            "load_object_bps": self.load_object_bps,
            "text_filter": self.text_filter_diagnostics,
            "marker_parts": self.marker_part_diagnostics,
            "object_geometry": self.object_geometry_diagnostics,
        }
        if self.return_structured and self.require_object_geometry:
            _raise_if_missing_required_geometry(self.object_geometry_diagnostics)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        motion = np.load(record.motion_path)
        InterActMotion962.validate_shape(motion, name=record.motion_path)
        motion = np.asarray(motion, dtype=np.float32)
        if not np.isfinite(motion).all():
            raise ValueError(f"Non-finite values in {record.motion_path}.")

        window = self.sampler.sample(len(motion), index=index)
        raw_crop = motion[window.start : window.end].copy()
        real_length = int(raw_crop.shape[0])
        if self.normalizer is not None:
            crop = self.normalizer.transform(raw_crop).astype(np.float32, copy=False)
        else:
            crop = raw_crop.astype(np.float32, copy=False)

        valid_mask = np.ones((real_length,), dtype=bool)
        raw_motion_padded = raw_crop.astype(np.float32, copy=True)
        if window.pad_after:
            crop = np.concatenate(
                [crop, np.zeros((window.pad_after, InterActMotion962.DIM), dtype=np.float32)], axis=0
            )
            raw_motion_padded = np.concatenate(
                [
                    raw_motion_padded,
                    np.zeros((window.pad_after, InterActMotion962.DIM), dtype=np.float32),
                ],
                axis=0,
            )
            valid_mask = np.concatenate([valid_mask, np.zeros((window.pad_after,), dtype=bool)], axis=0)

        text = None
        if self.load_text and record.text_path:
            text = clean_interact_caption(read_text(record.text_path, default=None))

        sample_points = None
        if self.load_object_points and record.sample_points_path:
            sample_points = self._load_sample_points(record.sample_points_path)

        bps = None
        if self.load_object_bps and record.bps_path:
            bps = self._load_bps(record.bps_path)
        if self.return_structured and self.require_object_geometry:
            if sample_points is None:
                raise ValueError(
                    f"Structured Stage1 sample requires object sample points, but none were loaded for "
                    f"{record.dataset}/{record.sequence_name}."
                )
            if bps is None:
                raise ValueError(
                    f"Structured Stage1 sample requires object BPS, but none were loaded for "
                    f"{record.dataset}/{record.sequence_name}."
                )

        sample = {
            "motion": torch.from_numpy(crop).float(),
            "target": torch.from_numpy(crop.copy()).float(),
            "valid_mask": torch.from_numpy(valid_mask).bool(),
            "length": real_length,
            "raw_motion": _optional_tensor(
                raw_motion_padded if self.return_structured else raw_crop,
                self.return_raw_motion or self.return_structured,
            ),
            "raw_motion_unpadded": torch.from_numpy(raw_crop).float() if self.return_structured else None,
            "meta": {
                "dataset": record.dataset,
                "sequence_name": record.sequence_name,
                "sequence_dir": record.sequence_dir,
                "object_name": record.object_name,
                "text": text,
                "split": record.split,
                "start": window.start,
                "end": window.end,
                "sequence_length": record.length,
            },
            "object": {
                "sample_points": torch.from_numpy(sample_points).float() if sample_points is not None else None,
                "sample_points_path": record.sample_points_path,
                "bps": torch.from_numpy(bps).float() if bps is not None else None,
                "bps_path": record.bps_path,
            },
        }
        if self.return_structured:
            raw_groups = build_stage1_raw_groups_from_motion_window(
                raw_motion_padded,
                valid_mask,
                root_method=self.root_method,
            )
            if self.structured_normalizer is not None:
                model_groups = self.structured_normalizer.normalize_sample_raw_groups(raw_groups)
            else:
                model_groups = raw_groups
            masks = _build_masks(valid_mask)
            model_groups = zero_stage1_model_groups_by_masks(
                model_groups,
                masks["valid"],
                masks["vel"],
                masks["acc"],
            )
            sample["raw"] = _to_tensor_tree(raw_groups)
            sample["model"] = _to_tensor_tree(model_groups)
            sample["geometry"] = _to_tensor_tree(
                _build_geometry(
                    object_points=sample_points,
                    object_bps=bps,
                )
            )
            sample["ids"] = _to_tensor_tree(
                {
                    **self.static_ids,
                    "dataset_id": np.asarray(self.dataset_to_id[record.dataset], dtype=np.int64),
                    "object_id": np.asarray(
                        self.object_to_id.get(record.object_name, -1),
                        dtype=np.int64,
                    ),
                }
            )
            sample["masks"] = _to_tensor_tree(masks)
        return sample

    def _build_object_geometry_diagnostics(self) -> dict:
        missing_points = [
            _geometry_missing_record(record, "sample_points_path")
            for record in self.records
            if not record.sample_points_path
        ]
        missing_bps = [
            _geometry_missing_record(record, "bps_path")
            for record in self.records
            if not record.bps_path
        ]
        return {
            "num_records": len(self.records),
            "missing_sample_points": len(missing_points),
            "missing_bps": len(missing_bps),
            "example_missing_sample_points": missing_points[:5],
            "example_missing_bps": missing_bps[:5],
        }

    def _load_sample_points(self, path: str) -> np.ndarray:
        points = self._load_cached_array(path).astype(np.float32, copy=False)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"sample_points must have shape [N, 3], got {points.shape} at {path}.")
        return points

    def _load_bps(self, path: str) -> np.ndarray:
        bps = self._load_cached_array(path).astype(np.float32, copy=False)
        if bps.ndim == 3 and bps.shape[0] == 1:
            bps = bps[0]
        if bps.ndim != 2 or bps.shape[-1] != 3:
            raise ValueError(f"object BPS must have shape [N, 3] or [1, N, 3], got {bps.shape} at {path}.")
        return bps

    def _load_cached_array(self, path: str) -> np.ndarray:
        key = str(Path(path).resolve())
        if self.object_geometry_cache_size > 0 and key in self._geometry_cache:
            value = self._geometry_cache.pop(key)
            self._geometry_cache[key] = value
            return value.copy()
        value = np.load(path).astype(np.float32, copy=False)
        if self.object_geometry_cache_size > 0:
            self._geometry_cache[key] = value.copy()
            while len(self._geometry_cache) > self.object_geometry_cache_size:
                self._geometry_cache.popitem(last=False)
        return value.copy()


def _path_if_exists(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _discover_object_names(objects_root: Path) -> list[str]:
    if not objects_root.exists():
        return []
    return sorted([path.name for path in objects_root.iterdir() if path.is_dir()], key=len, reverse=True)


def _load_object_name(object_path: str) -> Optional[str]:
    try:
        with np.load(object_path, allow_pickle=True) as obj:
            if "name" not in obj:
                return None
            value = obj["name"]
            if hasattr(value, "shape") and value.shape == ():
                value = value.item()
            text = str(value)
            return text if text else None
    except Exception:
        return None


def _infer_object_name_from_sequence(sequence_name: str, object_names: list[str]) -> Optional[str]:
    lowered = sequence_name.lower()
    for object_name in object_names:
        if object_name.lower() in lowered:
            return object_name
    return None


def _length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    values = np.asarray(lengths, dtype=np.float64)
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def _record_from_dict(item: dict) -> InterActSequenceRecord:
    fields = {field: item.get(field) for field in InterActSequenceRecord.__dataclass_fields__}
    return InterActSequenceRecord(**fields)


def records_to_dicts(records: list[InterActSequenceRecord]) -> list[dict]:
    return [asdict(record) for record in records]


def _apply_text_token_filter(
    records: list[InterActSequenceRecord],
    config: Optional[dict[str, Any]],
) -> tuple[list[InterActSequenceRecord], dict[str, Any]]:
    before_count = len(records)
    diagnostics: dict[str, Any] = {
        "enabled": False,
        "before_count": before_count,
        "after_count": before_count,
        "dropped_count": 0,
        "dropped_too_long_count": 0,
        "dropped_missing_text_count": 0,
        "max_seen_tokens": None,
        "max_kept_tokens": None,
        "example_dropped_records": [],
    }
    if not config or not bool(config.get("enabled", False)):
        return records, diagnostics

    tokenizer_name = str(config.get("tokenizer") or "")
    if not tokenizer_name:
        raise ValueError("text_filter.enabled=True requires text_filter.tokenizer.")
    max_tokens = int(config.get("max_tokens", 77))
    if max_tokens <= 0:
        raise ValueError(f"text_filter.max_tokens must be positive, got {max_tokens}.")
    add_special_tokens = bool(config.get("add_special_tokens", True))
    require_text = bool(config.get("require_text", False))
    tokenizer = _load_text_filter_tokenizer(tokenizer_name)

    kept: list[InterActSequenceRecord] = []
    dropped_too_long = 0
    dropped_missing_text = 0
    max_seen_tokens: int | None = None
    max_kept_tokens: int | None = None
    examples: list[dict[str, Any]] = []

    for record in records:
        text = clean_interact_caption(read_text(record.text_path, default=None)) if record.text_path else None
        if not text:
            if require_text:
                dropped_missing_text += 1
                if len(examples) < 10:
                    examples.append(_text_filter_example(record, reason="missing_text"))
                continue
            kept.append(record)
            continue

        token_count = _count_text_tokens(tokenizer, text, add_special_tokens=add_special_tokens)
        max_seen_tokens = token_count if max_seen_tokens is None else max(max_seen_tokens, token_count)
        if token_count > max_tokens:
            dropped_too_long += 1
            if len(examples) < 10:
                examples.append(
                    _text_filter_example(
                        record,
                        reason="too_long",
                        token_count=token_count,
                        text=text,
                    )
                )
            continue

        max_kept_tokens = token_count if max_kept_tokens is None else max(max_kept_tokens, token_count)
        kept.append(record)

    diagnostics.update(
        {
            "enabled": True,
            "tokenizer": tokenizer_name,
            "max_tokens": max_tokens,
            "add_special_tokens": add_special_tokens,
            "require_text": require_text,
            "before_count": before_count,
            "after_count": len(kept),
            "dropped_count": before_count - len(kept),
            "dropped_too_long_count": dropped_too_long,
            "dropped_missing_text_count": dropped_missing_text,
            "max_seen_tokens": max_seen_tokens,
            "max_kept_tokens": max_kept_tokens,
            "example_dropped_records": examples,
        }
    )
    return kept, diagnostics


def _load_text_filter_tokenizer(tokenizer_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("text_filter requires transformers.AutoTokenizer to be installed.") from exc
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def _count_text_tokens(tokenizer: Any, text: str, *, add_special_tokens: bool) -> int:
    tokenized = tokenizer(text, add_special_tokens=add_special_tokens, truncation=False)
    input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    if hasattr(input_ids, "ndim") and int(input_ids.ndim) == 2:
        return int(input_ids.shape[-1])
    if len(input_ids) > 0 and isinstance(input_ids[0], (list, tuple)):
        input_ids = input_ids[0]
    return len(input_ids)


def _text_filter_example(
    record: InterActSequenceRecord,
    *,
    reason: str,
    token_count: int | None = None,
    text: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset": record.dataset,
        "sequence_name": record.sequence_name,
        "object_name": record.object_name,
        "reason": reason,
        "token_count": token_count,
        "text_path": record.text_path,
        "text_preview": text[:160] if text else None,
    }


def _geometry_missing_record(record: InterActSequenceRecord, field_name: str) -> dict:
    return {
        "dataset": record.dataset,
        "sequence_name": record.sequence_name,
        "object_name": record.object_name,
        "field": field_name,
    }


def _raise_if_missing_required_geometry(diagnostics: dict) -> None:
    missing_points = int(diagnostics["missing_sample_points"])
    missing_bps = int(diagnostics["missing_bps"])
    if missing_points == 0 and missing_bps == 0:
        return
    raise ValueError(
        "Structured Stage1 dataset requires object geometry when require_object_geometry=True. "
        f"Missing sample_points for {missing_points} records and BPS for {missing_bps} records. "
        f"Example missing sample_points: {diagnostics['example_missing_sample_points']}. "
        f"Example missing BPS: {diagnostics['example_missing_bps']}."
    )


def _optional_tensor(array: np.ndarray, enabled: bool) -> Optional[torch.Tensor]:
    return torch.from_numpy(array).float() if enabled else None


def _resolve_optional_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((Path.cwd() / path_obj).resolve())


def _build_geometry(
    *,
    object_points: Optional[np.ndarray],
    object_bps: Optional[np.ndarray],
) -> dict:
    points = object_points if object_points is not None else np.zeros((0, 3), dtype=np.float32)
    bps = object_bps if object_bps is not None else np.zeros((0, 3), dtype=np.float32)
    return {
        "object_points_local": points.astype(np.float32, copy=False),
        "object_points_mask": np.ones((points.shape[0],), dtype=bool),
        "object_bps": bps.astype(np.float32, copy=False),
        "object_bps_mask": np.ones((bps.shape[0],), dtype=bool),
        "has_object_points": np.asarray(object_points is not None, dtype=bool),
        "has_object_bps": np.asarray(object_bps is not None, dtype=bool),
    }


def _build_masks(valid_mask: np.ndarray) -> dict:
    valid = np.asarray(valid_mask, dtype=bool)
    return {
        "valid": valid,
        "human": np.broadcast_to(valid[:, None], (len(valid), 77)).copy(),
        "object": valid.copy(),
        "relation": np.broadcast_to(valid[:, None], (len(valid), 78)).copy(),
        "vel": derivative_mask(valid, order=1),
        "acc": derivative_mask(valid, order=2),
    }


def _to_tensor_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        if value.dtype == np.bool_:
            return tensor.bool()
        if np.issubdtype(value.dtype, np.integer):
            return tensor.long()
        return tensor.float()
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.bool_):
            return torch.as_tensor(bool(value), dtype=torch.bool)
        if np.issubdtype(value.dtype, np.integer):
            return torch.as_tensor(int(value), dtype=torch.long)
        return torch.as_tensor(float(value), dtype=torch.float32)
    return value
