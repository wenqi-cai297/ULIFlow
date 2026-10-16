from __future__ import annotations

import torch

from hoi_latent.losses.stage1 import Stage1LossOutput, Stage1Phase1WarmupLoss
from hoi_latent.models.stage1 import build_stage1_autoencoder_from_config
from hoi_latent.training.stage1.phase1_warmup import (
    Stage1Phase1TrainState,
    Stage1Phase1WarmupModule,
    Stage1Phase1WarmupTrainer,
    apply_cli_overrides,
    deep_merge,
)
from test_stage1_autoencoder import _config as _autoencoder_config
from test_stage1_tokenizers import make_synthetic_stage1_structured_batch


def test_warmup_module_trains_tokenizer_encoder_decoder_wrapper():
    batch = _make_structured_batch()
    stage1 = build_stage1_autoencoder_from_config(_autoencoder_config(), batch=batch)
    loss_fn = Stage1Phase1WarmupLoss(lambda_world_marker=0.0)
    module = Stage1Phase1WarmupModule(stage1, loss_fn)

    out = module(batch, global_step=0)
    assert isinstance(out, Stage1LossOutput)
    assert torch.isfinite(out.total)

    out.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in stage1.decoder.parameters())
    assert any(p.grad is not None for p in stage1.encoder.parameters())
    assert any(p.grad is not None for p in stage1.tokenizer.parameters())
    assert not hasattr(stage1, "refiner")


def test_trainer_single_train_step_cpu_smoke(tmp_path):
    # End-to-end smoke of the real Accelerate train_step on CPU: one optimizer step
    # advances global_step, loss is finite,
    # The append-only train log is written and the no-refiner wrapper remains valid.
    from accelerate import Accelerator

    batch = _make_structured_batch()
    stage1 = build_stage1_autoencoder_from_config(_autoencoder_config(), batch=batch)
    loss_fn = Stage1Phase1WarmupLoss(lambda_world_marker=0.0)
    module = Stage1Phase1WarmupModule(stage1, loss_fn)

    accelerator = Accelerator(cpu=True)
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=1.0e-4)
    prepared_model, prepared_optimizer = accelerator.prepare(module, optimizer)
    state = Stage1Phase1TrainState(
        accelerator=accelerator,
        model=prepared_model,
        loss_fn=loss_fn,
        optimizer=prepared_optimizer,
        scheduler=None,
        global_step=0,
        epoch=0,
    )
    trainer = Stage1Phase1WarmupTrainer(
        state=state,
        run_dir=tmp_path,
        resolved_config={},
        log_interval_steps=1,
    )

    out = trainer.train_step(batch)

    assert torch.isfinite(out.total)
    assert state.global_step == 1
    assert (tmp_path / "logs" / "train.jsonl").exists()
    trainer.assert_frozen_modules_have_no_grad()


def test_deep_merge_preserves_nested_and_overrides():
    base = {"data": {"window_size": 128, "loader": {"batch_size": 4}}, "model": {"x": 1}}
    override = {"data": {"window_size": 64}, "training": {"stage1": {"max_steps": 10}}}

    merged = deep_merge(base, override)

    assert merged["data"]["window_size"] == 64  # overridden
    assert merged["data"]["loader"]["batch_size"] == 4  # deep-preserved
    assert merged["model"]["x"] == 1
    assert merged["training"]["stage1"]["max_steps"] == 10
    assert "training" not in base  # input not mutated


def test_apply_cli_overrides_sets_nested_paths_and_skips_none():
    config = {"training": {"stage1": {"max_steps": 10}}}

    out = apply_cli_overrides(config, {"lr": 5.0e-4, "run_name": "r1", "max_steps": None})

    assert out["training"]["stage1"]["optimizer"]["lr"] == 5.0e-4  # nested path created
    assert out["training"]["stage1"]["run_name"] == "r1"
    assert out["training"]["stage1"]["max_steps"] == 10  # None override is ignored
    assert "optimizer" not in config["training"]["stage1"]  # input not mutated


def _make_structured_batch() -> dict:
    batch = make_synthetic_stage1_structured_batch(batch_size=1, seq_len=5)
    batch["raw"]["human"] = {
        "pos": batch["model"]["human"]["pos_rootrel"] + batch["model"]["human"]["root"][:, :, None, :]
    }
    return batch
