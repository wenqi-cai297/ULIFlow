from __future__ import annotations

from torch import nn

from hoi_latent.ablations.stage1_architecture.reference import build_stage1_reference_from_config


def build_stage1_architecture_ablation_from_config(config: dict, batch: dict | None = None) -> nn.Module:
    ablation = config.get("ablation", {})
    variant = str(ablation.get("variant", ablation.get("name", "")))
    if variant != "full_mot_reference":
        raise ValueError(f"This release supports only the full_mot_reference Stage1 model, got {variant!r}.")
    model = build_stage1_reference_from_config(config, batch=batch)
    model.stage1_architecture_variant = variant
    model.stage1_architecture_resolved_config = config
    return model
