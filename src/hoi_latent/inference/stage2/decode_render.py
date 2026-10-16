from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from hoi_latent.inference.stage2.flow_forcing_sampler import Stage2FlowForcingRolloutOutput
from hoi_latent.inference.stage2.latent_template import (
    Stage2InferenceLatentTemplateSpec,
    build_stage2_inference_latent_template,
)
from hoi_latent.models.stage1.autoencoder.latent_bottleneck import Stage1LatentFieldOutput
from hoi_latent.models.stage1.decoder.readout_heads import Stage1CoarsePrediction
from hoi_latent.training.stage1.source_autoencoder import decode_stage1_source_prediction


_REQUIRED_GT_MODEL_KEYS: dict[str, tuple[str, ...]] = {
    "human": (
        "pos_rootrel",
        "vel_rootrel",
        "acc_rootrel",
        "root",
        "root_vel",
        "root_acc",
        "left_heights",
        "right_heights",
    ),
    "object": ("rot6d", "trans", "trans_vel", "trans_acc"),
    "relation": (
        "delta_m2o",
        "dist",
        "delta_vel",
        "delta_acc",
        "point_canon",
        "root_object_delta",
        "root_object_dist",
        "root_object_delta_vel",
        "root_object_delta_acc",
    ),
}
_REQUIRED_MASK_KEYS = ("frame", "human", "object", "relation")


@dataclass
class Stage2DecodedWindowBatch:
    predictions: list[Stage1CoarsePrediction]
    decode_window_starts: torch.Tensor
    target_frame_masks: torch.Tensor
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class Stage2WorldRenderArrays:
    human_markers: np.ndarray
    object_points: np.ndarray
    frame_mask: np.ndarray
    object_trans: np.ndarray | None = None
    object_rotmat: np.ndarray | None = None


def build_decode_target_frame_mask(
    *,
    decode_window_start: int,
    requested_frames: int,
    batch_size: int,
    window_blocks: int = 4,
    block_size_frames: int = 32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if requested_frames <= 0:
        raise ValueError(f"requested_frames must be positive, got {requested_frames}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if window_blocks <= 0 or block_size_frames <= 0:
        raise ValueError("window_blocks and block_size_frames must be positive.")
    device = torch.device("cpu") if device is None else torch.device(device)
    target_num_frames = int(window_blocks) * int(block_size_frames)
    local_frames = torch.arange(target_num_frames, device=device, dtype=torch.long)
    local_block = local_frames // int(block_size_frames)
    frame_in_block = local_frames % int(block_size_frames)
    global_frame = (int(decode_window_start) + local_block) * int(block_size_frames) + frame_in_block
    valid = global_frame < int(requested_frames)
    return valid.unsqueeze(0).expand(int(batch_size), -1).clone()


@torch.no_grad()
def decode_stage2_z_white_windows(
    *,
    whitening: Any,
    rollout: Stage2FlowForcingRolloutOutput,
    template_spec: Stage2InferenceLatentTemplateSpec,
    requested_frames: int,
    stage1_decoder: torch.nn.Module | None = None,
    stage1_model: torch.nn.Module | None = None,
    validate: bool = False,
) -> Stage2DecodedWindowBatch:
    if (stage1_decoder is None) == (stage1_model is None):
        raise ValueError("Provide exactly one of stage1_decoder or stage1_model.")
    z_windows = rollout.decode_z_white_windows
    if z_windows.ndim != 4:
        raise ValueError(f"decode_z_white_windows must have shape [B,W,N,d], got {tuple(z_windows.shape)}.")
    batch_size, num_windows, latent_tokens, _ = z_windows.shape
    device = z_windows.device
    window_blocks = _window_blocks(template_spec)
    block_size_frames = _block_size_frames(template_spec)
    predictions: list[Stage1CoarsePrediction] = []
    masks: list[torch.Tensor] = []
    finite_flags = []
    for window_idx in range(num_windows):
        start = int(rollout.decode_window_starts[window_idx].item())
        local_to_global = torch.arange(start, start + window_blocks, device=device, dtype=torch.long)
        template = build_stage2_inference_latent_template(
            template_spec,
            batch_size=batch_size,
            requested_frames=int(requested_frames),
            local_to_global_block=local_to_global,
            device=device,
        )
        z_white = z_windows[:, window_idx].float()
        if z_white.shape[1] != latent_tokens:
            raise ValueError("Unexpected latent token count mismatch while decoding Stage2 windows.")
        z_rec = whitening.unwhiten(z_white, template.latent_mask.bool()).float()
        latent = Stage1LatentFieldOutput(
            z_rec=z_rec,
            z_raw=z_rec,
            mask=template.latent_mask,
            modality_ids=template.modality_ids,
            token_type_ids=template.token_type_ids,
            time_ids=template.time_ids,
            entity_ids=template.entity_ids,
            time_mask=template.time_mask,
        )
        target_frame_mask = build_decode_target_frame_mask(
            decode_window_start=start,
            requested_frames=int(requested_frames),
            batch_size=batch_size,
            window_blocks=window_blocks,
            block_size_frames=block_size_frames,
            device=device,
        )
        if stage1_model is not None:
            pred = decode_stage1_source_prediction(
                stage1_model,
                latent,
                {"masks": {"frame": target_frame_mask}},
                z_override=z_rec,
                target_num_frames=int(template_spec.window_frames),
                target_frame_mask=target_frame_mask,
                return_diagnostics=False,
                validate=validate,
            )
        else:
            pred = stage1_decoder(
                latent,
                z_override=z_rec,
                target_num_frames=int(template_spec.window_frames),
                target_frame_mask=target_frame_mask,
                return_diagnostics=False,
                validate=validate,
            )
        predictions.append(pred)
        masks.append(target_frame_mask)
        finite_flags.append(_prediction_is_finite(pred))
    return Stage2DecodedWindowBatch(
        predictions=predictions,
        decode_window_starts=rollout.decode_window_starts.detach().clone(),
        target_frame_masks=torch.stack(masks, dim=1),
        diagnostics={
            "decoded_window_count": torch.tensor(num_windows, device=device),
            "decoded_all_finite": torch.tensor(bool(all(finite_flags)), device=device),
        },
    )


def stitch_stage1_predictions_preferred_slot(
    *,
    window_predictions: Sequence[Stage1CoarsePrediction],
    decode_window_starts: torch.Tensor,
    preferred_decode_window_start: torch.Tensor,
    preferred_local_slot: torch.Tensor,
    requested_frames: int,
    block_size_frames: int = 32,
) -> Stage1CoarsePrediction:
    if not window_predictions:
        raise ValueError("window_predictions must be non-empty.")
    if requested_frames <= 0:
        raise ValueError(f"requested_frames must be positive, got {requested_frames}.")
    decode_start_to_index = {int(start.item()): idx for idx, start in enumerate(decode_window_starts.detach().cpu())}
    num_blocks = int(preferred_decode_window_start.numel())
    total_frames = num_blocks * int(block_size_frames)
    first = window_predictions[0]
    stitched_human = _allocate_stitched_group(first.human, total_frames)
    stitched_object = _allocate_stitched_group(first.object, total_frames)
    stitched_relation = _allocate_stitched_group(first.relation, total_frames)
    stitched_masks = _allocate_stitched_group(first.masks, total_frames)

    for global_block in range(num_blocks):
        window_start = int(preferred_decode_window_start[global_block].item())
        local_slot = int(preferred_local_slot[global_block].item())
        if window_start not in decode_start_to_index:
            raise ValueError(f"Preferred decode window start {window_start} is not in decode_window_starts.")
        window_pred = window_predictions[decode_start_to_index[window_start]]
        global_start = global_block * int(block_size_frames)
        global_end = min(global_start + int(block_size_frames), total_frames)
        local_start = local_slot * int(block_size_frames)
        local_end = local_start + (global_end - global_start)
        _copy_group_slice(stitched_human, window_pred.human, global_start, global_end, local_start, local_end)
        _copy_group_slice(stitched_object, window_pred.object, global_start, global_end, local_start, local_end)
        _copy_group_slice(stitched_relation, window_pred.relation, global_start, global_end, local_start, local_end)
        _copy_group_slice(stitched_masks, window_pred.masks, global_start, global_end, local_start, local_end)

    stitched = Stage1CoarsePrediction(
        human=_crop_group(stitched_human, requested_frames),
        object=_crop_group(stitched_object, requested_frames),
        relation=_crop_group(stitched_relation, requested_frames),
        masks=_crop_group(stitched_masks, requested_frames),
        aux={
            "stage2_decode_stitch": {
                "stitch_strategy": "preferred_slot",
                "requested_frames": int(requested_frames),
                "block_size_frames": int(block_size_frames),
                "decode_window_starts": decode_window_starts.detach().cpu(),
                "preferred_decode_window_start": preferred_decode_window_start.detach().cpu(),
                "preferred_local_slot": preferred_local_slot.detach().cpu(),
            }
        },
    )
    stitched.aux["stage2_decode_stitch"]["boundary_diagnostics"] = compute_stage2_stitch_boundary_diagnostics(
        stitched,
        block_size_frames=block_size_frames,
    )
    return stitched


def build_gt_stage1_coarse_prediction_from_batch(batch: Mapping[str, Any]) -> Stage1CoarsePrediction:
    model = batch.get("model")
    masks = batch.get("masks")
    if not isinstance(model, Mapping):
        raise KeyError("GT construction requires batch['model'].")
    if not isinstance(masks, Mapping):
        raise KeyError("GT construction requires batch['masks'].")
    out: dict[str, dict[str, torch.Tensor]] = {}
    frame_dim: int | None = None
    for group, keys in _REQUIRED_GT_MODEL_KEYS.items():
        source = model.get(group)
        if not isinstance(source, Mapping):
            raise KeyError(f"GT construction requires batch['model'][{group!r}].")
        out[group] = {}
        for key in keys:
            if key not in source:
                raise KeyError(f"GT construction missing batch['model'][{group!r}][{key!r}].")
            value = source[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"batch['model'][{group!r}][{key!r}] must be a tensor.")
            if value.ndim < 2:
                raise ValueError(f"batch['model'][{group!r}][{key!r}] must have a frame dimension.")
            frame_dim = int(value.shape[1]) if frame_dim is None else frame_dim
            if int(value.shape[1]) != frame_dim:
                raise ValueError("All GT model fields must share the same frame dimension.")
            out[group][key] = value
    mask_out: dict[str, torch.Tensor] = {}
    for key in _REQUIRED_MASK_KEYS:
        if key not in masks:
            raise KeyError(f"GT construction missing batch['masks'][{key!r}].")
        value = masks[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch['masks'][{key!r}] must be a tensor.")
        if value.ndim < 2 or int(value.shape[1]) != frame_dim:
            raise ValueError(f"batch['masks'][{key!r}] has incompatible frame dimension.")
        mask_out[key] = value
    return Stage1CoarsePrediction(
        human=out["human"],
        object=out["object"],
        relation=out["relation"],
        masks=mask_out,
        aux={"source": "dataset_gt_model_space"},
    )


def prediction_to_world_markers_and_object_points(
    prediction: Stage1CoarsePrediction,
    *,
    structured_normalizer: Any,
    object_points_local: torch.Tensor | np.ndarray,
) -> Stage2WorldRenderArrays:
    pos_rootrel = structured_normalizer.denormalize_group_torch(
        "human",
        "pos_rootrel",
        prediction.human["pos_rootrel"].detach().cpu(),
        require_stats=True,
    )
    root = structured_normalizer.denormalize_group_torch(
        "human",
        "root",
        prediction.human["root"].detach().cpu(),
        require_stats=True,
    )
    human = (pos_rootrel + root[:, :, None, :]).detach().cpu().numpy().astype(np.float32)
    trans = structured_normalizer.denormalize_group_torch(
        "object",
        "trans",
        prediction.object["trans"].detach().cpu(),
        require_stats=True,
    )
    rot6d = structured_normalizer.denormalize_group_torch(
        "object",
        "rot6d",
        prediction.object["rot6d"].detach().cpu(),
        require_stats=True,
    )
    rotmat = rot6d_to_matrix_np(rot6d.detach().cpu().numpy()).astype(np.float32)
    trans_np = trans.detach().cpu().numpy().astype(np.float32)
    points_np = _as_numpy(object_points_local).astype(np.float32)
    if points_np.ndim == 2:
        points_np = np.broadcast_to(points_np[None], (human.shape[0],) + points_np.shape).copy()
    if points_np.ndim != 3 or points_np.shape[0] != human.shape[0] or points_np.shape[-1] != 3:
        raise ValueError("object_points_local must have shape [B,P,3] or [P,3].")
    obj = np.stack(
        [object_points_world(points_np[b], rotmat[b], trans_np[b]) for b in range(human.shape[0])],
        axis=0,
    )
    frame_mask = prediction.masks.get("frame")
    if frame_mask is None:
        frame = np.ones(human.shape[:2], dtype=bool)
    else:
        frame = frame_mask.detach().cpu().numpy().astype(bool)
    return Stage2WorldRenderArrays(
        human_markers=human,
        object_points=obj.astype(np.float32),
        frame_mask=frame,
        object_trans=trans_np,
        object_rotmat=rotmat,
    )


def batch_gt_to_world_markers_and_object_points(batch: Mapping[str, Any]) -> Stage2WorldRenderArrays:
    raw = batch.get("raw")
    geometry = batch.get("geometry")
    masks = batch.get("masks", {})
    if not isinstance(raw, Mapping):
        raise KeyError("GT render requires batch['raw'].")
    if not isinstance(geometry, Mapping):
        raise KeyError("GT render requires batch['geometry'].")
    human = _require_nested_tensor(raw, ("human", "pos")).detach().cpu().numpy().astype(np.float32)
    rotmat = _require_nested_tensor(raw, ("object", "rotmat")).detach().cpu().numpy().astype(np.float32)
    trans = _require_nested_tensor(raw, ("object", "trans")).detach().cpu().numpy().astype(np.float32)
    points_local = _require_tensor(geometry, "object_points_local").detach().cpu().numpy().astype(np.float32)
    if points_local.ndim != 3 or points_local.shape[0] != human.shape[0] or points_local.shape[-1] != 3:
        raise ValueError("batch['geometry']['object_points_local'] must have shape [B,P,3].")
    obj = np.stack(
        [object_points_world(points_local[b], rotmat[b], trans[b]) for b in range(human.shape[0])],
        axis=0,
    )
    frame_mask = masks.get("frame") if isinstance(masks, Mapping) else None
    if isinstance(frame_mask, torch.Tensor):
        frame = frame_mask.detach().cpu().numpy().astype(bool)
    else:
        frame = np.ones(human.shape[:2], dtype=bool)
    return Stage2WorldRenderArrays(
        human_markers=human,
        object_points=obj.astype(np.float32),
        frame_mask=frame,
        object_trans=trans,
        object_rotmat=rotmat,
    )


def compute_stage2_stitch_boundary_diagnostics(
    stitched: Stage1CoarsePrediction,
    *,
    block_size_frames: int = 32,
) -> dict[str, Any]:
    if block_size_frames <= 0:
        raise ValueError(f"block_size_frames must be positive, got {block_size_frames}.")
    frame_count = _frame_count_from_prediction(stitched)
    boundaries = [idx for idx in range(int(block_size_frames), frame_count, int(block_size_frames))]
    device = _first_tensor(stitched.human).device
    if not boundaries:
        empty = torch.empty(0, device=device)
        return {
            "global_boundary_block": torch.empty(0, dtype=torch.long, device=device),
            "human_root_position_jump": empty,
            "human_root_velocity_jump": empty,
            "object_translation_jump": empty,
            "object_velocity_jump": empty,
            "object_rotation_geodesic_jump": empty,
            "relation_distance_jump": empty,
        }
    boundary_tensor = torch.tensor([b // int(block_size_frames) for b in boundaries], dtype=torch.long, device=device)
    left = torch.tensor([b - 1 for b in boundaries], dtype=torch.long, device=device)
    right = torch.tensor(boundaries, dtype=torch.long, device=device)
    valid = _boundary_valid_mask(stitched, left, right)

    root_jump = _masked_norm_jump(stitched.human["root"], left, right, valid)
    root_vel_jump = _masked_norm_jump(stitched.human["root_vel"], left, right, valid)
    trans_jump = _masked_norm_jump(stitched.object["trans"], left, right, valid)
    trans_vel_jump = _masked_norm_jump(stitched.object["trans_vel"], left, right, valid)
    rot_jump = _rotation_geodesic_from_rot6d(stitched.object["rot6d"][:, right], stitched.object["rot6d"][:, left])
    rot_jump = torch.where(valid, rot_jump, torch.zeros_like(rot_jump))
    rel = stitched.relation["dist"]
    rel_jump = (rel[:, right].float() - rel[:, left].float()).abs()
    rel_jump = _reduce_nonbatch_dims(rel_jump)
    rel_jump = torch.where(valid, rel_jump, torch.zeros_like(rel_jump))
    return {
        "global_boundary_block": boundary_tensor,
        "valid_boundary_mask": valid,
        "human_root_position_jump": root_jump,
        "human_root_velocity_jump": root_vel_jump,
        "object_translation_jump": trans_jump,
        "object_velocity_jump": trans_vel_jump,
        "object_rotation_geodesic_jump": rot_jump,
        "relation_distance_jump": rel_jump,
        "human_root_position_jump_mean": _valid_mean(root_jump, valid),
        "object_translation_jump_mean": _valid_mean(trans_jump, valid),
        "object_rotation_geodesic_jump_mean": _valid_mean(rot_jump, valid),
        "relation_distance_jump_mean": _valid_mean(rel_jump, valid),
    }


def compute_overlap_candidate_disagreement(
    *,
    window_predictions: Sequence[Stage1CoarsePrediction],
    decode_window_starts: torch.Tensor,
    preferred_decode_window_start: torch.Tensor,
    preferred_local_slot: torch.Tensor,
    block_size_frames: int = 32,
) -> dict[str, Any]:
    if not window_predictions:
        raise ValueError("window_predictions must be non-empty.")
    start_to_index = {int(start.item()): idx for idx, start in enumerate(decode_window_starts.detach().cpu())}
    num_blocks = int(preferred_decode_window_start.numel())
    records = []
    for global_block in range(num_blocks):
        pref_start = int(preferred_decode_window_start[global_block].item())
        pref_slot = int(preferred_local_slot[global_block].item())
        if pref_start not in start_to_index:
            continue
        pref = _block_slice(window_predictions[start_to_index[pref_start]], pref_slot, block_size_frames)
        for start, idx in start_to_index.items():
            local_slot = global_block - start
            if local_slot < 0:
                continue
            if local_slot * block_size_frames >= _frame_count_from_prediction(window_predictions[idx]):
                continue
            other = _block_slice(window_predictions[idx], local_slot, block_size_frames)
            if start == pref_start:
                continue
            records.append(
                {
                    "global_block": global_block,
                    "preferred_start": pref_start,
                    "other_start": start,
                    "human_root_l2": _mean_l2(pref.human["root"] - other.human["root"]),
                    "object_trans_l2": _mean_l2(pref.object["trans"] - other.object["trans"]),
                    "relation_dist_l1": (pref.relation["dist"].float() - other.relation["dist"].float()).abs().mean(),
                }
            )
    return {"candidate_count": len(records), "records": records}


def object_points_world(points_local: np.ndarray, rotmat: np.ndarray, trans: np.ndarray) -> np.ndarray:
    points_local = np.asarray(points_local, dtype=np.float32)
    rotmat = np.asarray(rotmat, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    if points_local.ndim != 2 or points_local.shape[-1] != 3:
        raise ValueError(f"points_local must have shape [P,3], got {points_local.shape}.")
    if rotmat.ndim != 3 or rotmat.shape[-2:] != (3, 3):
        raise ValueError(f"rotmat must have shape [T,3,3], got {rotmat.shape}.")
    if trans.shape != (rotmat.shape[0], 3):
        raise ValueError(f"trans must have shape [T,3], got {trans.shape}.")
    # InterAct treats canonical object points as row vectors and applies
    # points @ R.T + trans. Official visualization/eval code uses
    # np.matmul(points, np.transpose(rotmat, (0, 2, 1))) for objects.
    return np.einsum("pc,tjc->tpj", points_local, rotmat) + trans[:, None, :]


def rot6d_to_matrix_np(rot6d: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d last dimension must be 6, got {rot6d.shape}.")
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = _normalize_np(a1, eps=eps)
    b2 = _normalize_np(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1, eps=eps)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-2).astype(np.float32, copy=False)


def map_xyz_for_render(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.shape[-1] != 3:
        raise ValueError(f"points last dimension must be 3, got {points.shape}.")
    return np.stack([points[..., 0], points[..., 2], points[..., 1]], axis=-1)


def render_stage2_marker_comparison_mp4(
    *,
    generated: Stage2WorldRenderArrays,
    gt: Stage2WorldRenderArrays | None,
    condition_text: str,
    video_path: str | Path,
    first_frame_path: str | Path | None = None,
    fps: int = 30,
    stride: int = 1,
    dpi: int = 110,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    first_path = Path(first_frame_path) if first_frame_path is not None else video_path.with_suffix(".frame000.png")
    frame_mask = generated.frame_mask[0] if generated.frame_mask.ndim == 2 else generated.frame_mask
    frame_indices = np.where(frame_mask.astype(bool))[0]
    if frame_indices.size == 0:
        frame_indices = np.arange(generated.human_markers.shape[1])
    frame_indices = frame_indices[:: max(1, int(stride))]
    bounds = _render_bounds(generated, gt, frame_indices)
    has_gt = gt is not None
    fig = plt.figure(figsize=(12.8 if has_gt else 6.4, 5.4), dpi=dpi)
    writer = None
    try:
        for out_idx, frame_idx in enumerate(frame_indices):
            fig.clf()
            panel_count = 2 if has_gt else 1
            if has_gt:
                ax_gt = fig.add_subplot(1, panel_count, 1, projection="3d")
                _draw_marker_scene(
                    ax_gt,
                    gt.human_markers[0, frame_idx],
                    gt.object_points[0, frame_idx],
                    title=f"GT | frame {int(frame_idx)}",
                    bounds=bounds,
                )
                ax_pred = fig.add_subplot(1, panel_count, 2, projection="3d")
            else:
                ax_pred = fig.add_subplot(1, panel_count, 1, projection="3d")
            _draw_marker_scene(
                ax_pred,
                generated.human_markers[0, frame_idx],
                generated.object_points[0, frame_idx],
                title=f"Generated | frame {int(frame_idx)}",
                bounds=bounds,
            )
            title = _render_title(condition_text, metadata)
            fig.suptitle(title, fontsize=8, y=0.98)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
            frame = _figure_to_rgb(fig)
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {video_path}.")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if out_idx == 0:
                cv2.imwrite(str(first_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
        plt.close(fig)


def write_stage2_render_summary(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _allocate_stitched_group(group: Mapping[str, torch.Tensor], total_frames: int) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in group.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim < 2:
            raise ValueError(f"Prediction field {key!r} must have frame dimension at dim=1.")
        shape = list(value.shape)
        shape[1] = int(total_frames)
        out[key] = value.new_zeros(tuple(shape))
    return out


def _copy_group_slice(
    dst: dict[str, torch.Tensor],
    src: Mapping[str, torch.Tensor],
    global_start: int,
    global_end: int,
    local_start: int,
    local_end: int,
) -> None:
    for key, dst_value in dst.items():
        if key not in src:
            raise KeyError(f"Window prediction missing field {key!r}.")
        src_value = src[key]
        if src_value.ndim < 2:
            raise ValueError(f"Prediction field {key!r} must have frame dimension at dim=1.")
        dst_value[:, global_start:global_end, ...] = src_value[:, local_start:local_end, ...].to(dtype=dst_value.dtype)


def _crop_group(group: Mapping[str, torch.Tensor], requested_frames: int) -> dict[str, torch.Tensor]:
    return {key: value[:, : int(requested_frames), ...].clone() for key, value in group.items()}


def _prediction_is_finite(prediction: Stage1CoarsePrediction) -> bool:
    for group in (prediction.human, prediction.object, prediction.relation, prediction.masks):
        for value in group.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value.float()).all():
                return False
    return True


def _require_tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    if key not in mapping:
        raise KeyError(f"Missing required tensor {key!r}.")
    value = mapping[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key!r} must be a torch.Tensor.")
    return value


def _require_nested_tensor(mapping: Mapping[str, Any], path: tuple[str, str]) -> torch.Tensor:
    group = mapping.get(path[0])
    if not isinstance(group, Mapping):
        raise KeyError(f"Missing required mapping {path[0]!r}.")
    return _require_tensor(group, path[1])


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _frame_count_from_prediction(prediction: Stage1CoarsePrediction) -> int:
    return int(_first_tensor(prediction.human).shape[1])


def _first_tensor(group: Mapping[str, torch.Tensor]) -> torch.Tensor:
    for value in group.values():
        if isinstance(value, torch.Tensor):
            return value
    raise ValueError("Group contains no tensors.")


def _boundary_valid_mask(prediction: Stage1CoarsePrediction, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    frame = prediction.masks.get("frame")
    if frame is None:
        batch = _first_tensor(prediction.human).shape[0]
        return torch.ones(batch, left.numel(), dtype=torch.bool, device=left.device)
    valid = frame.to(device=left.device, dtype=torch.bool)
    return valid[:, left] & valid[:, right]


def _masked_norm_jump(value: torch.Tensor, left: torch.Tensor, right: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    jump = (value[:, right].float() - value[:, left].float()).norm(dim=-1)
    jump = _reduce_nonbatch_dims(jump)
    return torch.where(valid, jump, torch.zeros_like(jump))


def _reduce_nonbatch_dims(value: torch.Tensor) -> torch.Tensor:
    if value.ndim <= 2:
        return value
    dims = tuple(range(2, value.ndim))
    return value.float().mean(dim=dims)


def _rotation_geodesic_from_rot6d(right: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    r_right = _rot6d_to_matrix_torch(right.float())
    r_left = _rot6d_to_matrix_torch(left.float())
    relative = r_right.transpose(-1, -2) @ r_left
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.arccos(cos_theta)


def _rot6d_to_matrix_torch(rot6d: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=eps)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def _valid_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return value.new_zeros(())
    return value[valid].float().mean()


def _block_slice(prediction: Stage1CoarsePrediction, local_slot: int, block_size_frames: int) -> Stage1CoarsePrediction:
    start = int(local_slot) * int(block_size_frames)
    end = start + int(block_size_frames)
    return Stage1CoarsePrediction(
        human={key: value[:, start:end, ...] for key, value in prediction.human.items()},
        object={key: value[:, start:end, ...] for key, value in prediction.object.items()},
        relation={key: value[:, start:end, ...] for key, value in prediction.relation.items()},
        masks={key: value[:, start:end, ...] for key, value in prediction.masks.items()},
        aux={},
    )


def _mean_l2(value: torch.Tensor) -> torch.Tensor:
    return value.float().norm(dim=-1).mean()


def _normalize_np(value: np.ndarray, *, eps: float) -> np.ndarray:
    denom = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(denom, eps)


def _render_bounds(
    generated: Stage2WorldRenderArrays,
    gt: Stage2WorldRenderArrays | None,
    frame_indices: np.ndarray,
    *,
    min_radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    sets = [
        generated.human_markers[0, frame_indices].reshape(-1, 3),
        generated.object_points[0, frame_indices].reshape(-1, 3),
    ]
    if gt is not None:
        sets.extend(
            [
                gt.human_markers[0, frame_indices].reshape(-1, 3),
                gt.object_points[0, frame_indices].reshape(-1, 3),
            ]
        )
    mapped = []
    for points in sets:
        finite = points[np.isfinite(points).all(axis=-1)]
        if finite.size:
            mapped.append(map_xyz_for_render(finite))
    if not mapped:
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    pts = np.concatenate(mapped, axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) * 0.5
    radius = max(float(np.max(hi - lo) * 0.58), float(min_radius))
    return center - radius, center + radius


def _draw_marker_scene(ax, human: np.ndarray, object_points: np.ndarray, *, title: str, bounds: tuple[np.ndarray, np.ndarray]) -> None:
    h = map_xyz_for_render(human)
    o = map_xyz_for_render(object_points)
    ax.scatter(o[:, 0], o[:, 1], o[:, 2], c="#f97316", s=5, depthshade=False, alpha=0.85)
    ax.scatter(h[:, 0], h[:, 1], h[:, 2], c="#2563eb", s=10, depthshade=False)
    root = h.mean(axis=0)
    ax.scatter([root[0]], [root[1]], [root[2]], c="#111827", s=24, depthshade=False)
    lo, hi = bounds
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    span = hi - lo
    ax.set_box_aspect((float(span[0]), float(span[1]), float(span[2])))
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.view_init(elev=18.0, azim=-68.0)


def _figure_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[..., :3])


def _render_title(condition_text: str, metadata: Mapping[str, Any] | None) -> str:
    text = str(condition_text) if condition_text else "<empty condition text>"
    if len(text) > 160:
        text = text[:157] + "..."
    condition = textwrap.fill(text, width=78)
    meta_parts = []
    if metadata:
        for key in ("sequence_name", "requested_frames", "seed", "solver", "steps_per_block", "guidance"):
            if key in metadata and metadata[key] is not None:
                meta_parts.append(f"{key}={metadata[key]}")
    if not meta_parts:
        return condition
    return condition + "\n" + textwrap.fill(" | ".join(meta_parts), width=96)


def _window_blocks(spec: Stage2InferenceLatentTemplateSpec) -> int:
    return int(math.ceil(float(spec.anchors_per_window) / float(spec.block_size_anchors)))


def _block_size_frames(spec: Stage2InferenceLatentTemplateSpec) -> int:
    return int(spec.anchor_stride_frames) * int(spec.block_size_anchors)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
