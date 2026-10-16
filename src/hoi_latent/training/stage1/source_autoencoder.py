from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from hoi_latent.ablations.stage1_architecture import build_stage1_architecture_ablation_from_config
from hoi_latent.config.io import load_yaml
from hoi_latent.models.stage1 import build_stage1_autoencoder_from_config
from hoi_latent.training.stage1.phase1_warmup import deep_merge, infer_frame_mask


_STAGE1_ARCHITECTURE_ABLATION_VARIANTS = {
    "full_mot_reference",
}


def load_stage1_source_config(
    source_run: str | Path,
    *,
    fallback_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_path = Path(source_run) / "resolved_config.yaml"
    if resolved_path.exists():
        return load_yaml(resolved_path)
    if fallback_config is not None:
        return fallback_config
    raise FileNotFoundError(f"Stage1 source run is missing resolved_config.yaml: {resolved_path}")


def merge_stage1_source_model_contract(
    config: dict[str, Any],
    source_config: dict[str, Any],
) -> dict[str, Any]:
    """Carry the exact Stage1 model/ablation contract into a downstream run config."""

    out = copy.deepcopy(config)
    source_stage1 = source_config.get("model", {}).get("stage1")
    if isinstance(source_stage1, dict):
        out.setdefault("model", {})["stage1"] = copy.deepcopy(source_stage1)
    source_ablation = source_config.get("ablation")
    if isinstance(source_ablation, dict):
        downstream_ablation = out.get("ablation", {})
        if not isinstance(downstream_ablation, dict):
            downstream_ablation = {}
        out["ablation"] = deep_merge(source_ablation, downstream_ablation)
    return out


def is_stage1_architecture_ablation_config(config: dict[str, Any]) -> bool:
    ablation_cfg = config.get("ablation", {})
    if not isinstance(ablation_cfg, dict):
        return False
    variant = _stage1_architecture_ablation_variant(config)
    return (
        str(ablation_cfg.get("name", "")) == "stage1_architecture"
        or variant in _STAGE1_ARCHITECTURE_ABLATION_VARIANTS
    )


def build_stage1_source_autoencoder_from_config(
    config: dict[str, Any],
    *,
    batch: dict | None = None,
) -> torch.nn.Module:
    if is_stage1_architecture_ablation_config(config):
        return build_stage1_architecture_ablation_from_config(config, batch=batch)
    return build_stage1_autoencoder_from_config(config, batch=batch)


def load_stage1_source_component_state_dicts(
    model: torch.nn.Module,
    components: dict[str, dict[str, torch.Tensor]],
) -> None:
    custom_loader = getattr(model, "load_component_state_dicts", None)
    if callable(custom_loader):
        custom_loader(components)
        return

    checkpoint_key = str(getattr(model, "checkpoint_key", ""))
    if checkpoint_key and checkpoint_key in components:
        model.load_state_dict(components[checkpoint_key])
        return

    loaded_any = False
    for attr in ("tokenizer", "encoder", "decoder"):
        module = getattr(model, attr, None)
        if module is None:
            continue
        key = str(getattr(module, "checkpoint_key", attr))
        state = components.get(key, components.get(attr))
        if state is None:
            continue
        module.load_state_dict(state)
        loaded_any = True

    if not loaded_any:
        expected = ["tokenizer", "encoder", "decoder"]
        if checkpoint_key:
            expected.insert(0, checkpoint_key)
        raise KeyError(f"Checkpoint components do not match Stage1 source model. Expected one of {expected}.")


def encode_stage1_source_latent(
    model: torch.nn.Module,
    batch: dict,
    *,
    return_diagnostics: bool = False,
    validate: bool = False,
):
    """Encode a Stage1 source model regardless of production or ablation layout."""

    custom_encode = getattr(model, "encode_for_post_jitter", None)
    if callable(custom_encode):
        return custom_encode(batch, return_diagnostics=return_diagnostics, validate=validate)

    if not hasattr(model, "tokenizer") or not hasattr(model, "encoder"):
        raise AttributeError(
            f"{type(model).__name__} must provide encode_for_post_jitter() or production tokenizer/encoder modules."
        )
    tokenizer_outputs = model.tokenizer(batch, return_aux=False, return_concat=False, validate=validate)
    return model.encoder(tokenizer_outputs, batch=batch, return_diagnostics=return_diagnostics, validate=validate)


def decode_stage1_source_prediction(
    model: torch.nn.Module,
    latent,
    batch: dict,
    *,
    z_override: torch.Tensor | None = None,
    target_num_frames: int | None = None,
    target_frame_mask: torch.Tensor | None = None,
    return_diagnostics: bool = False,
    validate: bool = False,
):
    """Decode a Stage1 source latent regardless of production or ablation layout."""

    if target_frame_mask is None:
        target_frame_mask = infer_frame_mask(batch).to(device=latent.z_rec.device, dtype=torch.bool)
    else:
        target_frame_mask = target_frame_mask.to(device=latent.z_rec.device, dtype=torch.bool)
    if target_num_frames is None:
        target_num_frames = int(target_frame_mask.shape[1])

    custom_decode = getattr(model, "decode_for_post_jitter", None)
    if callable(custom_decode):
        return custom_decode(
            latent,
            batch,
            z_override=z_override,
            target_num_frames=target_num_frames,
            target_frame_mask=target_frame_mask,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )

    if not hasattr(model, "decoder"):
        raise AttributeError(f"{type(model).__name__} must provide decode_for_post_jitter() or a production decoder.")
    return model.decoder(
        latent,
        z_override=z_override,
        target_num_frames=target_num_frames,
        target_frame_mask=target_frame_mask,
        return_diagnostics=return_diagnostics,
        validate=validate,
    )


def _stage1_architecture_ablation_variant(config: dict[str, Any]) -> str:
    ablation_cfg = config.get("ablation", {})
    if not isinstance(ablation_cfg, dict):
        return ""
    variant = str(ablation_cfg.get("variant", ""))
    if variant:
        return variant
    nested = ablation_cfg.get("stage1_architecture", {})
    if isinstance(nested, dict):
        return str(nested.get("variant", ""))
    return ""
