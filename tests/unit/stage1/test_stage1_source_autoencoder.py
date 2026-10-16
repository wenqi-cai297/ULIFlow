from __future__ import annotations

import copy

from hoi_latent.ablations.stage1_architecture import Stage1ReferenceAutoEncoder
from hoi_latent.training.stage1.source_autoencoder import (
    build_stage1_source_autoencoder_from_config,
    is_stage1_architecture_ablation_config,
    load_stage1_source_component_state_dicts,
    merge_stage1_source_model_contract,
)
from test_stage1_architecture_ablation_reference import _reference_config
from test_stage1_tokenizers import make_synthetic_stage1_structured_batch


def test_source_rebuild_preserves_reference_checkpoint_contract() -> None:
    config = _reference_config()
    batch = make_synthetic_stage1_structured_batch(batch_size=1, seq_len=5)
    source = build_stage1_source_autoencoder_from_config(config, batch=batch)
    merged = merge_stage1_source_model_contract({"data": config["data"]}, config)
    rebuilt = build_stage1_source_autoencoder_from_config(merged, batch=batch)

    load_stage1_source_component_state_dicts(rebuilt, source.component_state_dicts())

    assert is_stage1_architecture_ablation_config(merged)
    assert isinstance(source, Stage1ReferenceAutoEncoder)
    assert isinstance(rebuilt, Stage1ReferenceAutoEncoder)
    assert set(rebuilt.component_state_dicts()) == {"tokenizer", "encoder", "decoder"}


def test_source_stage1_contract_overrides_downstream_defaults_without_replacing_stage2() -> None:
    source_config = _reference_config()
    source_config["model"]["stage2"] = {"d_model": 24}
    downstream = {
        "data": source_config["data"],
        "model": {
            "stage1": {"d_model": 256, "latent_bottleneck": {"z_dim": 16}, "stale_only": True},
            "stage2": {"d_model": 384},
        },
    }
    original_source = copy.deepcopy(source_config)
    original_downstream = copy.deepcopy(downstream)

    merged = merge_stage1_source_model_contract(downstream, source_config)
    assert merged["model"]["stage1"] == source_config["model"]["stage1"]
    assert merged["model"]["stage2"] == downstream["model"]["stage2"]
    assert source_config == original_source
    assert downstream == original_downstream

    batch = make_synthetic_stage1_structured_batch(batch_size=1, seq_len=5)
    source = build_stage1_source_autoencoder_from_config(source_config, batch=batch)
    rebuilt = build_stage1_source_autoencoder_from_config(merged, batch=batch)
    load_stage1_source_component_state_dicts(rebuilt, source.component_state_dicts())
