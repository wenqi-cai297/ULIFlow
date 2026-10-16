from __future__ import annotations

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
