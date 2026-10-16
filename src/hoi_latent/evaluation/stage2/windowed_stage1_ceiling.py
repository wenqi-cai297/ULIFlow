from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from hoi_latent.data.representations.interact_stage1_structured import (
    build_stage1_raw_groups_from_motion_window,
    derivative_mask,
)
from hoi_latent.data.transforms.structured_normalization import zero_stage1_model_groups_by_masks
from hoi_latent.inference.stage2.decode_render import (
    Stage2WorldRenderArrays,
    prediction_to_world_markers_and_object_points,
    stitch_stage1_predictions_preferred_slot,
)
from hoi_latent.inference.stage2.flow_forcing_sampler import build_preferred_decode_plan
from hoi_latent.inference.stage2.latent_template import Stage2InferenceLatentTemplateSpec
from hoi_latent.models.stage1.decoder.readout_heads import Stage1CoarsePrediction

from .stage1_ceiling import compute_stage1_reconstruction_ceiling_prediction


@dataclass
class WindowedStage1CeilingOutput:
    prediction: Stage1CoarsePrediction
    world: Stage2WorldRenderArrays
    decode_metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


@torch.no_grad()
def compute_windowed_stage1_reconstruction_ceiling(
    *,
    stage1_model: torch.nn.Module,
    whitening: Any,
    batch: Mapping[str, Any],
    structured_normalizer: Any,
    template_spec: Stage2InferenceLatentTemplateSpec,
    requested_frames: int,
    object_points_local: torch.Tensor | None = None,
    clean_history_blocks: int = 1,
    root_method: str = "marker_mean",
    validate: bool = False,
) -> WindowedStage1CeilingOutput:
    """Run the Stage1 reconstruction ceiling through 128-frame decode windows.

    This mirrors the Stage2 rollout decode path for GT-length evaluation.  The
    frozen Stage1 decoder is still a 128-frame model, so sequences longer than a
    single window must be encoded, decoded, and stitched window by window.
    """
    if requested_frames <= 0:
        raise ValueError(f"requested_frames must be positive, got {requested_frames}.")
    window_frames = int(template_spec.window_frames)
    block_size_frames = int(template_spec.block_size_anchors) * int(template_spec.anchor_stride_frames)
    if window_frames <= 0 or block_size_frames <= 0:
        raise ValueError("template_spec must define positive window and block sizes.")
    window_blocks = int(
        math.ceil(float(int(template_spec.anchors_per_window)) / float(int(template_spec.block_size_anchors)))
    )
    num_blocks = int(math.ceil(float(requested_frames) / float(block_size_frames)))
    raw_motion = _require_raw_motion(batch)
    device = raw_motion.device
    batch_size = int(raw_motion.shape[0])

    plan = build_preferred_decode_plan(
        num_blocks=num_blocks,
        window_blocks=window_blocks,
        clean_history_blocks=int(clean_history_blocks),
        device=device,
    )
    valid_mask = _frame_valid_mask(batch, requested_frames=requested_frames, device=device)
    predictions: list[Stage1CoarsePrediction] = []
    diagnostics: dict[str, torch.Tensor] = {}
    for window_index, start_block_t in enumerate(plan.decode_window_starts.detach().cpu()):
        start_block = int(start_block_t.item())
        start_frame = start_block * block_size_frames
        local_batch = build_local_stage1_window_batch(
            batch,
            structured_normalizer=structured_normalizer,
            valid_mask=valid_mask,
            start_frame=start_frame,
            window_frames=window_frames,
            root_method=root_method,
        )
        target_frame_mask = local_batch["masks"]["valid"].to(device=device, dtype=torch.bool)
        prediction, diag = compute_stage1_reconstruction_ceiling_prediction(
            stage1_model=stage1_model,
            whitening=whitening,
            batch=local_batch,
            target_frame_mask=target_frame_mask,
            validate=validate,
        )
        predictions.append(prediction)
        for key, value in diag.items():
            diagnostics[f"window_{window_index}/{key}"] = value

    stitched = stitch_stage1_predictions_preferred_slot(
        window_predictions=predictions,
        decode_window_starts=plan.decode_window_starts,
        preferred_decode_window_start=plan.preferred_decode_window_start,
        preferred_local_slot=plan.preferred_local_slot,
        requested_frames=int(requested_frames),
        block_size_frames=block_size_frames,
    )
    points = object_points_local if object_points_local is not None else _object_points_local_from_batch(batch)
    world = prediction_to_world_markers_and_object_points(
        stitched,
        structured_normalizer=structured_normalizer,
        object_points_local=points,
    )
    diagnostics.update(
        {
            "stage1_ceiling_window_count": torch.tensor(len(predictions), device=device),
            "stage1_ceiling_requested_frames": torch.tensor(int(requested_frames), device=device),
            "stage1_ceiling_window_frames": torch.tensor(window_frames, device=device),
        }
    )
    decode_metadata = {
        "stage1_ceiling_windowed": True,
        "stitch_strategy": "preferred_slot",
        "requested_frames": int(requested_frames),
        "window_frames": int(window_frames),
        "block_size_frames": int(block_size_frames),
        "root_method": str(root_method),
        "decode_window_starts": plan.decode_window_starts.detach().cpu().tolist(),
        "preferred_decode_window_start": plan.preferred_decode_window_start.detach().cpu().tolist(),
        "preferred_local_slot": plan.preferred_local_slot.detach().cpu().tolist(),
    }
    return WindowedStage1CeilingOutput(
        prediction=stitched,
        world=world,
        decode_metadata=decode_metadata,
        diagnostics=diagnostics,
    )


def build_local_stage1_window_batch(
    batch: Mapping[str, Any],
    *,
    structured_normalizer: Any,
    valid_mask: torch.Tensor,
    start_frame: int,
    window_frames: int,
    root_method: str,
) -> dict[str, Any]:
    raw_motion = _require_raw_motion(batch)
    batch_size = int(raw_motion.shape[0])
    raw_np = raw_motion.detach().cpu().numpy().astype(np.float32, copy=False)
    valid_np = valid_mask.detach().cpu().numpy().astype(bool, copy=False)

    raw_groups_per_sample = []
    model_groups_per_sample = []
    masks_per_sample = []
    raw_windows = []
    valid_counts = []
    for sample_idx in range(batch_size):
        raw_window, local_valid = _slice_pad_window(
            raw_np[sample_idx],
            valid_np[sample_idx],
            start_frame=start_frame,
            window_frames=window_frames,
        )
        raw_windows.append(raw_window)
        raw_groups = build_stage1_raw_groups_from_motion_window(
            raw_window,
            local_valid,
            root_method=root_method,
        )
        model_groups = structured_normalizer.normalize_sample_raw_groups(raw_groups)
        masks = _build_masks(local_valid)
        model_groups = zero_stage1_model_groups_by_masks(
            model_groups,
            masks["valid"],
            masks["vel"],
            masks["acc"],
        )
        raw_groups_per_sample.append(raw_groups)
        model_groups_per_sample.append(model_groups)
        masks_per_sample.append(masks)
        valid_counts.append(int(local_valid.sum()))

    local = {
        "raw_motion": torch.from_numpy(np.stack(raw_windows, axis=0)).to(device=raw_motion.device),
        "raw": _stack_tensor_tree(raw_groups_per_sample, device=raw_motion.device),
        "model": _stack_tensor_tree(model_groups_per_sample, device=raw_motion.device),
        "masks": _stack_tensor_tree(masks_per_sample, device=raw_motion.device),
        "meta": _local_meta(batch.get("meta", {}), start_frame=start_frame, valid_counts=valid_counts),
    }
    for key in ("geometry", "ids", "object"):
        value = batch.get(key)
        if isinstance(value, Mapping):
            local[key] = _clone_tree(value)
    return local


def _slice_pad_window(
    raw_motion: np.ndarray,
    valid_mask: np.ndarray,
    *,
    start_frame: int,
    window_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((window_frames, raw_motion.shape[-1]), dtype=np.float32)
    valid = np.zeros((window_frames,), dtype=bool)
    if start_frame < raw_motion.shape[0]:
        end = min(raw_motion.shape[0], start_frame + window_frames)
        length = max(0, end - start_frame)
        if length:
            out[:length] = raw_motion[start_frame:end]
            valid[:length] = valid_mask[start_frame:end]
    return out, valid


def _build_masks(valid_mask: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.asarray(valid_mask, dtype=bool)
    return {
        "valid": valid,
        "frame": valid.copy(),
        "human": np.broadcast_to(valid[:, None], (len(valid), 77)).copy(),
        "object": valid.copy(),
        "relation": np.broadcast_to(valid[:, None], (len(valid), 78)).copy(),
        "vel": derivative_mask(valid, order=1),
        "acc": derivative_mask(valid, order=2),
    }


def _frame_valid_mask(batch: Mapping[str, Any], *, requested_frames: int, device: torch.device) -> torch.Tensor:
    masks = batch.get("masks")
    if not isinstance(masks, Mapping):
        raise KeyError("Windowed Stage1 ceiling requires batch['masks'].")
    frame = masks.get("frame")
    if frame is None:
        frame = masks.get("valid")
    if not isinstance(frame, torch.Tensor):
        raise KeyError("Windowed Stage1 ceiling requires batch['masks']['frame'] or ['valid'].")
    frame = frame.to(device=device, dtype=torch.bool)
    capped = torch.zeros_like(frame)
    limit = min(int(requested_frames), int(frame.shape[1]))
    capped[:, :limit] = frame[:, :limit]
    return capped


def _require_raw_motion(batch: Mapping[str, Any]) -> torch.Tensor:
    value = batch.get("raw_motion")
    if not isinstance(value, torch.Tensor):
        raise KeyError("Windowed Stage1 ceiling requires batch['raw_motion'] from the structured InterAct dataset.")
    if value.ndim != 3 or value.shape[-1] != 962:
        raise ValueError(f"batch['raw_motion'] must have shape [B,T,962], got {tuple(value.shape)}.")
    return value.float()


def _object_points_local_from_batch(batch: Mapping[str, Any]) -> torch.Tensor:
    geometry = batch.get("geometry")
    if not isinstance(geometry, Mapping) or "object_points_local" not in geometry:
        raise KeyError("Windowed Stage1 ceiling requires batch['geometry']['object_points_local'].")
    value = geometry["object_points_local"]
    if not isinstance(value, torch.Tensor):
        raise TypeError("batch['geometry']['object_points_local'] must be a tensor.")
    return value


def _local_meta(meta: Any, *, start_frame: int, valid_counts: list[int]) -> dict[str, Any]:
    if not isinstance(meta, Mapping):
        starts = torch.full((len(valid_counts),), int(start_frame), dtype=torch.long)
        return {"start": starts, "end": starts + torch.as_tensor(valid_counts, dtype=torch.long)}
    out: dict[str, Any] = {key: _clone_tree(value) for key, value in meta.items()}
    base_start = meta.get("start", 0)
    start = _add_meta_offset(base_start, start_frame)
    out["start"] = start
    counts = torch.as_tensor(valid_counts, dtype=torch.long)
    if isinstance(start, torch.Tensor):
        out["end"] = start + counts.to(device=start.device, dtype=start.dtype)
    elif isinstance(start, np.ndarray):
        out["end"] = start + counts.cpu().numpy().astype(start.dtype, copy=False)
    elif isinstance(start, list):
        out["end"] = [int(s) + int(c) for s, c in zip(start, valid_counts)]
    else:
        out["end"] = int(start) + int(valid_counts[0] if valid_counts else 0)
    return out


def _add_meta_offset(value: Any, offset: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone() + int(offset)
    if isinstance(value, np.ndarray):
        return value.copy() + int(offset)
    if isinstance(value, (list, tuple)):
        return [int(v) + int(offset) for v in value]
    try:
        return int(value) + int(offset)
    except Exception:
        return int(offset)


def _stack_tensor_tree(items: list[Any], *, device: torch.device) -> Any:
    first = items[0]
    if isinstance(first, Mapping):
        return {key: _stack_tensor_tree([item[key] for item in items], device=device) for key in first}
    arrays = [np.asarray(item) for item in items]
    stacked = np.stack(arrays, axis=0)
    tensor = torch.from_numpy(stacked)
    if stacked.dtype == np.bool_:
        tensor = tensor.bool()
    elif np.issubdtype(stacked.dtype, np.integer):
        tensor = tensor.long()
    else:
        tensor = tensor.float()
    return tensor.to(device=device)


def _clone_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value
