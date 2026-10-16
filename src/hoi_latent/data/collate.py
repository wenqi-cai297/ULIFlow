from __future__ import annotations

import torch


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _pad_to(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[0] == length:
        return x
    out = x.new_zeros((length,) + tuple(x.shape[1:]))
    out[: x.shape[0]] = x
    return out


def collate_stage1_motion962(batch: list[dict]) -> dict:
    max_len = max(int(sample["motion"].shape[0]) for sample in batch)
    motions = []
    targets = []
    masks = []
    lengths = []
    for sample in batch:
        motion = _as_tensor(sample["motion"]).float()
        target = _as_tensor(sample["target"]).float()
        valid_mask = _as_tensor(sample["valid_mask"]).bool()
        motions.append(_pad_to(motion, max_len))
        targets.append(_pad_to(target, max_len))
        masks.append(_pad_to(valid_mask, max_len))
        lengths.append(int(sample["length"]))

    meta = {
        "dataset": [sample["meta"]["dataset"] for sample in batch],
        "sequence_name": [sample["meta"]["sequence_name"] for sample in batch],
        "sequence_dir": [sample["meta"]["sequence_dir"] for sample in batch],
        "object_name": [sample["meta"].get("object_name") for sample in batch],
        "text": [sample["meta"].get("text") for sample in batch],
        "split": [sample["meta"].get("split") for sample in batch],
        "start": torch.as_tensor([sample["meta"]["start"] for sample in batch], dtype=torch.long),
        "end": torch.as_tensor([sample["meta"]["end"] for sample in batch], dtype=torch.long),
        "sequence_length": torch.as_tensor(
            [sample["meta"]["sequence_length"] for sample in batch],
            dtype=torch.long,
        ),
    }
    out = {
        "motion": torch.stack(motions, dim=0),
        "target": torch.stack(targets, dim=0),
        "valid_mask": torch.stack(masks, dim=0),
        "lengths": torch.as_tensor(lengths, dtype=torch.long),
        "meta": meta,
        "object": _collate_nested([sample.get("object", {}) for sample in batch]),
    }
    optional_keys = [
        "raw_motion",
        "raw_motion_unpadded",
        "raw",
        "model",
        "geometry",
        "ids",
        "masks",
    ]
    for key in optional_keys:
        if any(key in sample and sample[key] is not None for sample in batch):
            out[key] = _collate_nested([sample.get(key) for sample in batch])
    if "ids" in out:
        for static_key in [
            "marker_ids",
            "marker_part_ids",
            "relation_ids",
            "relation_marker_ids",
            "relation_part_ids",
            "relation_type_ids",
        ]:
            value = out["ids"].get(static_key)
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                out["ids"][static_key] = value[0]
    return out


def _collate_nested(values: list):
    if all(value is None for value in values):
        return None
    non_none = [value for value in values if value is not None]
    first = non_none[0]
    if isinstance(first, dict):
        keys = sorted({key for value in non_none for key in value.keys()})
        return {
            key: _collate_nested(
                [value.get(key) if isinstance(value, dict) else None for value in values]
            )
            for key in keys
        }
    if isinstance(first, torch.Tensor):
        return _collate_tensors_or_none(values)
    if isinstance(first, (int, bool)):
        return torch.as_tensor([value if value is not None else 0 for value in values], dtype=torch.long)
    if isinstance(first, float):
        return torch.as_tensor([value if value is not None else 0.0 for value in values], dtype=torch.float32)
    return values


def _collate_tensors_or_none(values: list) -> torch.Tensor:
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    target_shape = _max_compatible_shape([tuple(t.shape) for t in tensors])
    padded = []
    for value in values:
        if value is None:
            padded.append(torch.zeros(target_shape, dtype=tensors[0].dtype, device=tensors[0].device))
        else:
            padded.append(_pad_tensor_to_shape(value, target_shape))
    return torch.stack(padded, dim=0)


def _max_compatible_shape(shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    if not shapes:
        return ()
    first = shapes[0]
    if all(shape == first for shape in shapes):
        return first
    if all(len(shape) == len(first) and shape[1:] == first[1:] for shape in shapes):
        return (max(shape[0] for shape in shapes),) + first[1:]
    raise ValueError(f"Cannot collate incompatible tensor shapes: {shapes}.")


def _pad_tensor_to_shape(value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(value.shape) == shape:
        return value
    out = value.new_zeros(shape)
    slices = tuple(slice(0, size) for size in value.shape)
    out[slices] = value
    return out
