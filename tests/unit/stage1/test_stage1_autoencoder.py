from __future__ import annotations

import torch

from hoi_latent.models.stage1 import (
    Stage1AutoencoderModule,
    build_stage1_autoencoder_from_config,
    validate_stage1_autoencoder_output,
)
from test_stage1_tokenizers import make_synthetic_stage1_structured_batch


def test_stage1_autoencoder_full_forward_from_batch_to_prediction():
    batch = make_synthetic_stage1_structured_batch(batch_size=2, seq_len=5)
    module = build_stage1_autoencoder_from_config(_config(), batch=batch)

    out = module(batch, return_tokenizer_outputs=True, return_diagnostics=True, validate=True)

    assert isinstance(module, Stage1AutoencoderModule)
    assert module.checkpoint_key == "stage1_autoencoder"
    assert module.output_key == "stage1_reconstruction"
    assert out.latent.z_rec.shape[-1] == 8
    assert out.prediction.human["pos_rootrel"].shape == (2, 5, 77, 3)
    assert out.coarse is out.prediction
    assert out.refined is out.prediction
    assert out.human is out.prediction.human
    assert out.object is out.prediction.object
    assert out.masks is out.prediction.masks
    assert out.tokenizer_outputs is not None
    assert set(out.tokenizer_outputs) == {"human", "object", "relation"}
    assert out.aux["stage1_autoencoder"]["checkpoint_key"] == "stage1_autoencoder"
    assert out.aux["stage1_autoencoder"]["target_num_frames"] == 5
    assert out.aux["stage1_autoencoder"]["component_checkpoint_keys"] == {
        "tokenizer": "tokenizer",
        "encoder": "encoder",
        "decoder": "decoder",
    }
    assert out.aux["stage1_autoencoder"]["refiner_removed"] is True
    assert out.prediction.human["pos_rootrel"][~out.masks["human"]].abs().max().item() == 0.0
    assert out.prediction.object["trans"][~out.masks["object"]].abs().max().item() == 0.0
    validate_stage1_autoencoder_output(out)


def test_stage1_autoencoder_default_hides_tokenizer_outputs_and_checkpoint_components():
    batch = make_synthetic_stage1_structured_batch(batch_size=1, seq_len=5)
    module = build_stage1_autoencoder_from_config(_config(), batch=batch)

    out = module(batch, validate=True)
    component_state = module.component_state_dicts()

    assert out.tokenizer_outputs is None
    assert set(component_state) == {"tokenizer", "encoder", "decoder"}
    state = module.state_dict()
    assert any(key.startswith("tokenizer.") for key in state)
    assert any(key.startswith("encoder.") for key in state)
    assert any(key.startswith("decoder.") for key in state)
    assert not any(key.startswith("refiner.") for key in state)


def test_stage1_autoencoder_gradients_reach_all_major_submodules():
    batch = make_synthetic_stage1_structured_batch(batch_size=2, seq_len=5)
    module = build_stage1_autoencoder_from_config(_config(), batch=batch)

    out = module(batch, validate=True)
    loss = (
        out.prediction.human["pos_rootrel"][out.masks["human"]].sum()
        + out.prediction.object["trans"][out.masks["object"]].sum()
        + out.latent.z_rec[out.latent.mask].sum()
    )
    loss.backward()

    assert _has_nonzero_grad(module.tokenizer)
    assert _has_nonzero_grad(module.encoder)
    assert _has_nonzero_grad(module.decoder)
    assert not hasattr(module, "refiner")


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    for param in module.parameters():
        if param.grad is not None and torch.isfinite(param.grad).all() and float(param.grad.abs().sum().item()) > 0.0:
            return True
    return False


def _config() -> dict:
    return {
        "data": {
            "window_size": 5,
            "structured": {"marker_part_map_path": "assets/interact/marker_part_map_v1.json"},
        },
        "model": {
            "stage1": {
                "d_model": 32,
                "num_heads": 4,
                "max_seq_len": 5,
                "target_num_frames": 5,
                "num_marker_parts": 100,
                "num_relation_types": 2,
                "tokenizers": {
                    "human": {"num_body_slots": 2},
                    "object": {"num_geometry_slots": 2, "geometry_source": "bps"},
                    "relation": {"num_relation_slots": 2},
                },
                "stem_encoder": {
                    "human": {"depth": 1},
                    "object": {"pose_depth": 1, "geometry_depth": 1, "pose_geometry_cross_depth": 1},
                    "relation": {"depth": 1},
                    "ffn_mult": 2,
                },
                "token_compressor": {
                    "temporal": {"compression_factor": 3, "max_temporal_anchors": 2},
                    "geometry": {"target_tokens": 2},
                    "budget": {"global_cap": 64},
                },
                "joint_encoder": {"depth": 1, "ffn_mult": 2},
                "latent_bottleneck": {
                    "z_dim": 8,
                    "projection": {"taper_dim": 16},
                },
                "decoder": {
                    "up_neck": {"z_dim": 8, "d_model": 32, "taper_dim": 16},
                    "joint_stack": {"d_model": 32, "depth": 1, "num_heads": 4, "ffn_mult": 2},
                    "temporal_densifier": {
                        "d_model": 32,
                        "target_num_frames": 5,
                        "depth": 1,
                        "num_heads": 4,
                        "ffn_mult": 2,
                        "local_temporal_attention": {"radius": 1},
                    },
                    "readout_heads": {
                        "d_model": 32,
                        "num_heads": 4,
                        "target_num_frames": 5,
                        "coupling": {"per_frame_entity_attention_depth": 1},
                    },
                },
            }
        },
    }
