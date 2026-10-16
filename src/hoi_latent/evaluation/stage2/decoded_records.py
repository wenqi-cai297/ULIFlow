from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hoi_latent.inference.stage2.decode_render import Stage2WorldRenderArrays


@dataclass
class Stage2WorldMotionArrays:
    human_markers: np.ndarray
    object_points: np.ndarray
    frame_mask: np.ndarray
    object_trans: np.ndarray | None = None
    object_rotmat: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.human_markers = _float32_array(self.human_markers, "human_markers")
        self.object_points = _float32_array(self.object_points, "object_points")
        self.frame_mask = np.asarray(self.frame_mask, dtype=bool)
        if self.human_markers.ndim != 3 or self.human_markers.shape[1:] != (77, 3):
            raise ValueError(f"human_markers must have shape [T,77,3], got {self.human_markers.shape}")
        if self.object_points.ndim != 3 or self.object_points.shape[0] != self.human_markers.shape[0] or self.object_points.shape[-1] != 3:
            raise ValueError(f"object_points must have shape [T,P,3], got {self.object_points.shape}")
        if self.frame_mask.shape != (self.human_markers.shape[0],):
            raise ValueError(f"frame_mask must have shape [{self.human_markers.shape[0]}], got {self.frame_mask.shape}")
        if self.object_trans is not None:
            self.object_trans = _float32_array(self.object_trans, "object_trans")
            if self.object_trans.shape != (self.human_markers.shape[0], 3):
                raise ValueError(f"object_trans must have shape [T,3], got {self.object_trans.shape}")
        if self.object_rotmat is not None:
            self.object_rotmat = _float32_array(self.object_rotmat, "object_rotmat")
            if self.object_rotmat.shape != (self.human_markers.shape[0], 3, 3):
                raise ValueError(f"object_rotmat must have shape [T,3,3], got {self.object_rotmat.shape}")


@dataclass
class Stage2DecodedSampleRecord:
    sample_id: str
    split: str
    dataset: str | None
    sequence_name: str | None
    text: str | None
    object_name: str | None
    requested_frames: int
    fps: float
    seed: int
    condition_index: int
    sample_index: int
    generated: Stage2WorldMotionArrays
    gt: Stage2WorldMotionArrays | None = None
    stage1_ceiling: Stage2WorldMotionArrays | None = None
    rollout_metadata: dict[str, Any] = field(default_factory=dict)
    decode_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.requested_frames <= 0:
            raise ValueError(f"requested_frames must be positive, got {self.requested_frames}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.generated.human_markers.shape[0] < self.requested_frames:
            raise ValueError("generated arrays are shorter than requested_frames")
        for name, arrays in (("gt", self.gt), ("stage1_ceiling", self.stage1_ceiling)):
            if arrays is not None and arrays.human_markers.shape[0] < self.requested_frames:
                raise ValueError(f"{name} arrays are shorter than requested_frames")


def split_batched_world_arrays(arrays: Stage2WorldRenderArrays, sample_index: int) -> Stage2WorldMotionArrays:
    human = np.asarray(arrays.human_markers)
    obj = np.asarray(arrays.object_points)
    frame = np.asarray(arrays.frame_mask)
    if human.ndim != 4 or obj.ndim != 4:
        raise ValueError("batched world arrays must have human [B,T,77,3] and object [B,T,P,3]")
    batch_size = human.shape[0]
    if sample_index < 0 or sample_index >= batch_size:
        raise IndexError(f"sample_index={sample_index} out of range for batch size {batch_size}")
    trans = None if arrays.object_trans is None else np.asarray(arrays.object_trans)[sample_index]
    rotmat = None if arrays.object_rotmat is None else np.asarray(arrays.object_rotmat)[sample_index]
    frame_sample = frame[sample_index] if frame.ndim == 2 else frame
    return Stage2WorldMotionArrays(
        human_markers=human[sample_index],
        object_points=obj[sample_index],
        frame_mask=frame_sample,
        object_trans=trans,
        object_rotmat=rotmat,
    )


def slice_world_motion_arrays(arrays: Stage2WorldMotionArrays, frames: int) -> Stage2WorldMotionArrays:
    """Return a copy truncated to the first ``frames`` frames of every array."""

    frames = int(frames)
    total = int(arrays.human_markers.shape[0])
    if frames <= 0 or frames > total:
        raise ValueError(f"frames must be in [1, {total}], got {frames}")
    if frames == total:
        return arrays
    return Stage2WorldMotionArrays(
        human_markers=arrays.human_markers[:frames],
        object_points=arrays.object_points[:frames],
        frame_mask=arrays.frame_mask[:frames],
        object_trans=None if arrays.object_trans is None else arrays.object_trans[:frames],
        object_rotmat=None if arrays.object_rotmat is None else arrays.object_rotmat[:frames],
    )


def build_stage2_eval128_decoded_record(
    *,
    generated: Stage2WorldRenderArrays,
    gt: Stage2WorldRenderArrays,
    stage1_ceiling: Stage2WorldRenderArrays,
    batch: Mapping[str, Any],
    batch_index: int,
    split: str,
    requested_frames: int,
    fps: float,
    seed: int,
    condition_index: int,
    sample_index: int,
    rollout_metadata: Mapping[str, Any] | None = None,
    decode_metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Stage2DecodedSampleRecord:
    meta = batch.get("meta", {}) if isinstance(batch, Mapping) else {}
    dataset = _meta_value(meta, "dataset", batch_index)
    sequence = _meta_value(meta, "sequence_name", batch_index) or _meta_value(meta, "seq_name", batch_index)
    text = _meta_value(meta, "text", batch_index)
    object_name = _meta_value(meta, "object_name", batch_index) or _meta_value(meta, "object", batch_index)
    sample_id = _make_sample_id(dataset, sequence, condition_index, sample_index)
    return Stage2DecodedSampleRecord(
        sample_id=sample_id,
        split=str(split),
        dataset=_none_or_str(dataset),
        sequence_name=_none_or_str(sequence),
        text=_none_or_str(text),
        object_name=_none_or_str(object_name),
        requested_frames=int(requested_frames),
        fps=float(fps),
        seed=int(seed),
        condition_index=int(condition_index),
        sample_index=int(sample_index),
        generated=split_batched_world_arrays(generated, batch_index),
        gt=split_batched_world_arrays(gt, batch_index),
        stage1_ceiling=split_batched_world_arrays(stage1_ceiling, batch_index),
        rollout_metadata=dict(rollout_metadata or {}),
        decode_metadata=dict(decode_metadata or {}),
        provenance=dict(provenance or {}),
    )


def write_stage2_decoded_record(record: Stage2DecodedSampleRecord, npz_path: str | Path, jsonl_path: str | Path | None = None) -> None:
    target = Path(npz_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    _put_arrays(arrays, "generated", record.generated)
    if record.gt is not None:
        _put_arrays(arrays, "gt", record.gt)
    if record.stage1_ceiling is not None:
        _put_arrays(arrays, "stage1_ceiling", record.stage1_ceiling)
    np.savez_compressed(target, **arrays)
    meta_path = Path(jsonl_path) if jsonl_path is not None else target.with_suffix(".jsonl")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_record_metadata(record), sort_keys=True) + "\n")


def read_stage2_decoded_record(npz_path: str | Path, jsonl_path: str | Path | None = None) -> Stage2DecodedSampleRecord:
    source = Path(npz_path)
    meta_path = Path(jsonl_path) if jsonl_path is not None else source.with_suffix(".jsonl")
    with np.load(source) as payload:
        arrays = {key: payload[key] for key in payload.files}
    lines = [line for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{meta_path} must contain exactly one metadata JSON line, got {len(lines)}")
    meta = json.loads(lines[0])
    return Stage2DecodedSampleRecord(
        sample_id=str(meta["sample_id"]),
        split=str(meta["split"]),
        dataset=meta.get("dataset"),
        sequence_name=meta.get("sequence_name"),
        text=meta.get("text"),
        object_name=meta.get("object_name"),
        requested_frames=int(meta["requested_frames"]),
        fps=float(meta["fps"]),
        seed=int(meta["seed"]),
        condition_index=int(meta["condition_index"]),
        sample_index=int(meta["sample_index"]),
        generated=_get_arrays(arrays, "generated"),
        gt=_get_arrays(arrays, "gt") if "gt/human_markers" in arrays else None,
        stage1_ceiling=_get_arrays(arrays, "stage1_ceiling") if "stage1_ceiling/human_markers" in arrays else None,
        rollout_metadata=dict(meta.get("rollout", {})),
        decode_metadata=dict(meta.get("decode", {})),
        provenance=dict(meta.get("provenance", {})),
    )


def _put_arrays(out: dict[str, np.ndarray], prefix: str, arrays: Stage2WorldMotionArrays) -> None:
    out[f"{prefix}/human_markers"] = arrays.human_markers
    out[f"{prefix}/object_points"] = arrays.object_points
    out[f"{prefix}/frame_mask"] = arrays.frame_mask
    if arrays.object_trans is not None:
        out[f"{prefix}/object_trans"] = arrays.object_trans
    if arrays.object_rotmat is not None:
        out[f"{prefix}/object_rotmat"] = arrays.object_rotmat


def _get_arrays(payload: Mapping[str, np.ndarray], prefix: str) -> Stage2WorldMotionArrays:
    return Stage2WorldMotionArrays(
        human_markers=payload[f"{prefix}/human_markers"],
        object_points=payload[f"{prefix}/object_points"],
        frame_mask=payload[f"{prefix}/frame_mask"],
        object_trans=payload.get(f"{prefix}/object_trans"),
        object_rotmat=payload.get(f"{prefix}/object_rotmat"),
    )


def _record_metadata(record: Stage2DecodedSampleRecord) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "split": record.split,
        "dataset": record.dataset,
        "sequence_name": record.sequence_name,
        "text": record.text,
        "object_name": record.object_name,
        "requested_frames": int(record.requested_frames),
        "fps": float(record.fps),
        "seed": int(record.seed),
        "condition_index": int(record.condition_index),
        "sample_index": int(record.sample_index),
        "rollout": _jsonable(record.rollout_metadata),
        "decode": _jsonable(record.decode_metadata),
        "provenance": _jsonable(record.provenance),
    }


def _float32_array(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _meta_value(meta: Any, key: str, index: int) -> Any:
    if not isinstance(meta, Mapping) or key not in meta:
        return None
    value = meta[key]
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    if isinstance(value, np.ndarray):
        return value[index].item() if value.ndim > 0 else value.item()
    return value


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _make_sample_id(dataset: Any, sequence: Any, condition_index: int, sample_index: int) -> str:
    parts = [str(v) for v in (dataset, sequence) if v is not None]
    parts.extend([f"cond{int(condition_index):06d}", f"sample{int(sample_index):02d}"])
    return "/".join(parts)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value
