from __future__ import annotations

import torch

from hoi_latent.ablations.stage1_architecture import (
    Stage1ArchitectureAblationOutput,
    Stage1ReferenceAutoEncoder,
    build_stage1_architecture_ablation_from_config,
)
from hoi_latent.losses.stage1 import Stage1Phase1WarmupLoss
from hoi_latent.models.stage1 import validate_stage1_coarse_prediction
from test_stage1_autoencoder import _config as _autoencoder_config
from test_stage1_tokenizers import make_synthetic_stage1_structured_batch


def test_full_mot_reference_builds_production_mot_no_refiner_layout():
    batch = make_synthetic_stage1_structured_batch(batch_size=2, seq_len=5)
    module = build_stage1_architecture_ablation_from_config(_reference_config(), batch=batch)

    out = module(batch, return_tokenizer_outputs=True, return_diagnostics=True, validate=True)

    assert isinstance(module, Stage1ReferenceAutoEncoder)
    assert isinstance(out, Stage1ArchitectureAblationOutput)
    assert module.variant_name == "full_mot_reference"
    assert module.layout == "production_mot_no_refiner"
    assert not hasattr(module, "refiner")
    assert set(module.component_state_dicts()) == {"tokenizer", "encoder", "decoder"}
    assert out.latent.z_rec.shape[-1] == 8
    assert out.prediction.human["pos_rootrel"].shape == (2, 5, 77, 3)
    assert out.coarse is out.prediction
    assert out.refined is out.prediction
    assert out.tokenizer_outputs is not None
    assert out.aux["stage1_architecture_ablation"]["variant"] == "full_mot_reference"
    assert out.aux["stage1_architecture_ablation"]["layout"] == "production_mot_no_refiner"
    assert out.aux["stage1_architecture_ablation"]["route_decoupling"] == "ffn_only"
    validate_stage1_coarse_prediction(out.prediction)


def test_full_mot_reference_is_phase1_loss_compatible():
    batch = make_synthetic_stage1_structured_batch(batch_size=2, seq_len=5)
    module = build_stage1_architecture_ablation_from_config(_reference_config(), batch=batch)
    loss_fn = Stage1Phase1WarmupLoss(lambda_world_marker=0.0, lambda_latent_cov_floor=0.0)

    out = module(batch, validate=True)
    loss = loss_fn(out.prediction, out.latent, batch, global_step=0)

    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert _has_nonzero_grad(module.tokenizer)
    assert _has_nonzero_grad(module.encoder)
    assert _has_nonzero_grad(module.decoder)


def _reference_config() -> dict:
    config = _autoencoder_config()
    config["ablation"] = {
        "name": "stage1_architecture",
        "variant": "full_mot_reference",
        "full_mot_reference": {
            "role": "reference_control",
            "architecture": "production_mot_no_refiner",
            "latent_capacity": {"n_latent": 708, "z_dim": 16, "capacity": 11328},
        },
    }
    return config


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        param.grad is not None
        and torch.isfinite(param.grad).all()
        and float(param.grad.abs().sum().item()) > 0.0
        for param in module.parameters()
    )
