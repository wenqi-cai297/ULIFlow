from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from hoi_latent.models.stage1.autoencoder.latent_bottleneck import Stage1LatentFieldOutput
from hoi_latent.models.stage1.tokenizer_types import TokenTypeId
from hoi_latent.models.stage2.flow_time import build_stage2_flow_anchor_layout, temporal_dynamic_token_mask


_METADATA_KEYS = ("latent_mask", "modality_ids", "token_type_ids", "time_ids", "entity_ids", "time_mask")
_TOPOLOGY_KEYS = ("modality_ids", "token_type_ids", "time_ids", "entity_ids", "time_mask")


@dataclass(frozen=True)
class Stage2InferenceLatentTemplateSpec:
    window_frames: int
    anchor_stride_frames: int
    anchors_per_window: int
    block_size_anchors: int
    latent_token_count: int
    geometry_token_count: int
    metadata_1d: dict[str, torch.Tensor]
    source_stage1_checkpoint_sha256: str
    stage1_anchor_override: dict[str, Any]
    spec_hash: str


@dataclass
class Stage2InferenceLatentTemplate:
    latent_mask: torch.Tensor
    modality_ids: torch.Tensor
    token_type_ids: torch.Tensor
    time_ids: torch.Tensor
    entity_ids: torch.Tensor
    time_mask: torch.Tensor
    block_size_anchors: int
    local_anchor_time_ids: torch.Tensor
    token_anchor_index: torch.Tensor
    token_block_index: torch.Tensor
    dynamic_token_mask: torch.Tensor
    static_geometry_token_mask: torch.Tensor
    local_to_global_block: torch.Tensor
    local_block_mask: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def extract_stage2_inference_latent_template_spec(
    latent: Stage1LatentFieldOutput,
    *,
    window_frames: int = 128,
    anchor_stride_frames: int = 4,
    block_size_anchors: int = 8,
    source_stage1_checkpoint_sha256: str,
    stage1_anchor_override: dict[str, Any],
    sample_index: int = 0,
) -> Stage2InferenceLatentTemplateSpec:
    if window_frames <= 0:
        raise ValueError(f"window_frames must be positive, got {window_frames}.")
    if anchor_stride_frames <= 0:
        raise ValueError(f"anchor_stride_frames must be positive, got {anchor_stride_frames}.")
    if block_size_anchors <= 0:
        raise ValueError(f"block_size_anchors must be positive, got {block_size_anchors}.")
    if window_frames % anchor_stride_frames != 0:
        raise ValueError("window_frames must be divisible by anchor_stride_frames.")
    metadata_1d = _metadata_from_latent(latent, sample_index=sample_index)
    anchors_per_window = int(window_frames // anchor_stride_frames)
    latent_token_count = int(metadata_1d["latent_mask"].numel())
    geometry_token_count = int(_static_geometry_mask(metadata_1d).sum().item())
    payload = {
        "window_frames": int(window_frames),
        "anchor_stride_frames": int(anchor_stride_frames),
        "anchors_per_window": anchors_per_window,
        "block_size_anchors": int(block_size_anchors),
        "latent_token_count": latent_token_count,
        "geometry_token_count": geometry_token_count,
        "metadata_1d": metadata_1d,
        "source_stage1_checkpoint_sha256": str(source_stage1_checkpoint_sha256),
        "stage1_anchor_override": _jsonable(stage1_anchor_override),
    }
    spec_hash = compute_stage2_template_spec_hash(payload)
    return Stage2InferenceLatentTemplateSpec(
        window_frames=int(window_frames),
        anchor_stride_frames=int(anchor_stride_frames),
        anchors_per_window=anchors_per_window,
        block_size_anchors=int(block_size_anchors),
        latent_token_count=latent_token_count,
        geometry_token_count=geometry_token_count,
        metadata_1d=metadata_1d,
        source_stage1_checkpoint_sha256=str(source_stage1_checkpoint_sha256),
        stage1_anchor_override=dict(_jsonable(stage1_anchor_override)),
        spec_hash=spec_hash,
    )


def validate_stage2_template_content_independence(
    latents: Iterable[Stage1LatentFieldOutput],
    *,
    sample_index: int = 0,
) -> dict[str, torch.Tensor]:
    metadata = [_metadata_from_latent(latent, sample_index=sample_index) for latent in latents]
    if len(metadata) < 2:
        raise ValueError("Template content-independence validation requires at least two latent examples.")
    reference = metadata[0]
    for idx, current in enumerate(metadata[1:], start=1):
        for key in _TOPOLOGY_KEYS:
            if current[key].shape != reference[key].shape or not torch.equal(current[key], reference[key]):
                raise ValueError(
                    f"Stage2 template metadata field {key!r} differs between canonical sample 0 and sample {idx}."
                )
    mask_equal = [torch.equal(current["latent_mask"], reference["latent_mask"]) for current in metadata[1:]]
    return {
        "template_content_examples": torch.tensor(len(metadata), dtype=torch.long),
        "template_latent_mask_all_equal": torch.tensor(all(mask_equal), dtype=torch.bool),
    }


def build_stage2_inference_latent_template(
    spec: Stage2InferenceLatentTemplateSpec,
    *,
    batch_size: int,
    requested_frames: int,
    local_to_global_block: torch.Tensor,
    device: torch.device | str,
) -> Stage2InferenceLatentTemplate:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if requested_frames <= 0:
        raise ValueError(f"requested_frames must be positive, got {requested_frames}.")
    device = torch.device(device)
    local_to_global_block = _normalize_local_to_global_block(
        local_to_global_block,
        batch_size=batch_size,
        window_blocks=_window_blocks(spec),
        device=device,
    )
    metadata = {key: _tile_metadata(value, batch_size=batch_size, device=device) for key, value in spec.metadata_1d.items()}
    missing = [key for key in _METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"Stage2 template spec is missing metadata fields: {missing}.")

    base_latent_mask = metadata["latent_mask"].to(dtype=torch.bool)
    base_time_mask = metadata["time_mask"].to(dtype=torch.bool)
    token_type_ids = metadata["token_type_ids"].to(dtype=torch.long)
    time_ids = metadata["time_ids"].to(dtype=torch.long)
    full_layout = build_stage2_flow_anchor_layout(
        latent_mask=base_latent_mask,
        token_type_ids=token_type_ids,
        time_ids=time_ids,
        time_mask=base_time_mask,
        block_size_anchors=spec.block_size_anchors,
    )
    full_dynamic = full_layout.dynamic_token_mask
    static_geometry = full_layout.static_geometry_token_mask
    rebased_time_ids = time_ids.clone()
    local_time_for_token = full_layout.token_anchor_index.clamp_min(0) * int(spec.anchor_stride_frames)
    rebased_time_ids = torch.where(full_dynamic, local_time_for_token.to(dtype=rebased_time_ids.dtype), rebased_time_ids)
    window_blocks = _window_blocks(spec)
    num_blocks = _ceil_div(int(requested_frames), _block_size_frames(spec))
    local_block_mask = local_to_global_block < num_blocks

    token_block = full_layout.token_block_index.clamp_min(0).clamp_max(window_blocks - 1)
    block_valid_for_token = torch.gather(local_block_mask, dim=1, index=token_block)
    global_block_for_token = torch.gather(local_to_global_block, dim=1, index=token_block)
    local_anchor_in_block = full_layout.token_anchor_index.remainder(int(spec.block_size_anchors))
    global_frame = global_block_for_token * _block_size_frames(spec) + local_anchor_in_block * int(spec.anchor_stride_frames)
    dynamic_valid = full_dynamic & block_valid_for_token & (global_frame < int(requested_frames))

    latent_mask = torch.where(full_dynamic, dynamic_valid, base_latent_mask)
    time_mask = base_time_mask & dynamic_valid
    local_anchor_time_ids = (
        torch.arange(int(spec.anchors_per_window), device=device, dtype=torch.long)
        .mul(int(spec.anchor_stride_frames))
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    return Stage2InferenceLatentTemplate(
        latent_mask=latent_mask,
        modality_ids=metadata["modality_ids"].to(dtype=torch.long),
        token_type_ids=token_type_ids,
        time_ids=rebased_time_ids,
        entity_ids=metadata["entity_ids"].to(dtype=torch.long),
        time_mask=time_mask,
        block_size_anchors=int(spec.block_size_anchors),
        local_anchor_time_ids=local_anchor_time_ids,
        token_anchor_index=full_layout.token_anchor_index,
        token_block_index=full_layout.token_block_index,
        dynamic_token_mask=dynamic_valid,
        static_geometry_token_mask=static_geometry & latent_mask,
        local_to_global_block=local_to_global_block,
        local_block_mask=local_block_mask,
        diagnostics={
            "requested_frames": torch.tensor(int(requested_frames), device=device),
            "num_blocks": torch.tensor(int(num_blocks), device=device),
            "valid_dynamic_token_count": dynamic_valid.sum(dim=1),
            "static_geometry_token_count": (static_geometry & latent_mask).sum(dim=1),
        },
    )


def compute_stage2_template_spec_hash(payload_or_spec: Stage2InferenceLatentTemplateSpec | dict[str, Any]) -> str:
    h = hashlib.sha256()
    if isinstance(payload_or_spec, Stage2InferenceLatentTemplateSpec):
        payload: dict[str, Any] = {
            "window_frames": payload_or_spec.window_frames,
            "anchor_stride_frames": payload_or_spec.anchor_stride_frames,
            "anchors_per_window": payload_or_spec.anchors_per_window,
            "block_size_anchors": payload_or_spec.block_size_anchors,
            "latent_token_count": payload_or_spec.latent_token_count,
            "geometry_token_count": payload_or_spec.geometry_token_count,
            "metadata_1d": payload_or_spec.metadata_1d,
            "source_stage1_checkpoint_sha256": payload_or_spec.source_stage1_checkpoint_sha256,
            "stage1_anchor_override": payload_or_spec.stage1_anchor_override,
        }
    else:
        payload = dict(payload_or_spec)
        payload.pop("spec_hash", None)
    for key in sorted(payload):
        value = payload[key]
        h.update(str(key).encode("utf-8"))
        if isinstance(value, dict) and all(isinstance(v, torch.Tensor) for v in value.values()):
            for tensor_key in sorted(value):
                _hash_tensor(h, tensor_key, value[tensor_key])
        else:
            h.update(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def _metadata_from_latent(latent: Stage1LatentFieldOutput, *, sample_index: int) -> dict[str, torch.Tensor]:
    if latent.mask.ndim != 2:
        raise ValueError(f"latent metadata must be rank-2 [B,N], got mask shape {tuple(latent.mask.shape)}.")
    if not 0 <= int(sample_index) < latent.mask.shape[0]:
        raise ValueError(f"sample_index={sample_index} is out of range for batch size {latent.mask.shape[0]}.")
    fields = {
        "latent_mask": latent.mask,
        "modality_ids": latent.modality_ids,
        "token_type_ids": latent.token_type_ids,
        "time_ids": latent.time_ids,
        "entity_ids": latent.entity_ids,
        "time_mask": latent.time_mask,
    }
    expected = tuple(latent.mask.shape)
    out: dict[str, torch.Tensor] = {}
    for key, value in fields.items():
        if tuple(value.shape) != expected:
            raise ValueError(f"latent.{key} must have shape {expected}, got {tuple(value.shape)}.")
        tensor = value[sample_index].detach().cpu().clone()
        if key in {"latent_mask", "time_mask"}:
            tensor = tensor.to(dtype=torch.bool)
        else:
            tensor = tensor.to(dtype=torch.long)
        out[key] = tensor
    return out


def _static_geometry_mask(metadata_1d: dict[str, torch.Tensor]) -> torch.Tensor:
    latent_mask = metadata_1d["latent_mask"].unsqueeze(0).to(dtype=torch.bool)
    time_mask = metadata_1d["time_mask"].unsqueeze(0).to(dtype=torch.bool)
    token_type_ids = metadata_1d["token_type_ids"].unsqueeze(0).to(dtype=torch.long)
    dynamic = temporal_dynamic_token_mask(time_mask, latent_mask, token_type_ids)[0]
    return latent_mask[0] & ~dynamic & (token_type_ids[0] == int(TokenTypeId.OBJECT_GEOMETRY))


def _tile_metadata(value: torch.Tensor, *, batch_size: int, device: torch.device) -> torch.Tensor:
    return value.to(device=device).unsqueeze(0).expand(batch_size, -1).clone()


def _normalize_local_to_global_block(
    value: torch.Tensor,
    *,
    batch_size: int,
    window_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    value = torch.as_tensor(value, device=device, dtype=torch.long)
    if value.ndim == 1:
        value = value.unsqueeze(0).expand(batch_size, -1).clone()
    if value.shape != (batch_size, window_blocks):
        raise ValueError(f"local_to_global_block must have shape {(batch_size, window_blocks)}, got {tuple(value.shape)}.")
    if (value < 0).any():
        raise ValueError("local_to_global_block must not contain negative block ids.")
    return value


def _window_blocks(spec: Stage2InferenceLatentTemplateSpec) -> int:
    return _ceil_div(int(spec.anchors_per_window), int(spec.block_size_anchors))


def _block_size_frames(spec: Stage2InferenceLatentTemplateSpec) -> int:
    return int(spec.block_size_anchors) * int(spec.anchor_stride_frames)


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _hash_tensor(h: hashlib._Hash, key: str, tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous()
    h.update(key.encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(cpu.numpy().tobytes())


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.detach().cpu().tolist()}
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
