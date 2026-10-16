from __future__ import annotations

import importlib
import random

import numpy as np
import pytest
import torch
from accelerate.utils import set_seed

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


@pytest.mark.parametrize(
    "module_name,entry_name,loader_name,needs_source",
    [
        ("hoi_latent.training.stage1.phase1_warmup", "run_stage1_phase1_warmup_training", "load_phase1_training_config", False),
        ("hoi_latent.training.ablations.stage1_architecture", "run_stage1_architecture_ablation_training", "load_phase1_training_config", False),
        ("hoi_latent.training.stage1.decoder_post_jitter", "run_stage1_decoder_post_jitter_training", "load_decoder_post_jitter_training_config", True),
        ("hoi_latent.training.stage2.rectified_flow", "run_stage2_rectified_flow_training", "load_stage2_rectified_flow_training_config", True),
    ],
)
def test_training_entries_seed_before_setup_with_common_rank_seed(
    monkeypatch, module_name, entry_name, loader_name, needs_source
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, loader_name, lambda *args, **kwargs: ({}, {}, {"seed": kwargs["overrides"]["seed"]}))
    observed = []

    class SeededBeforeSetup(Exception):
        pass

    def capture_seed(seed, *, device_specific):
        assert seed == 123
        assert device_specific is False
        set_seed(seed, device_specific=device_specific)
        observed.append((random.random(), float(np.random.rand()), torch.rand(4)))
        raise SeededBeforeSetup

    monkeypatch.setattr(module, "set_seed", capture_seed)
    kwargs = {"data_config_path": "unused", "train_config_path": "unused", "overrides": {"seed": 123}}
    if needs_source:
        kwargs.update(source_run="unused", whitening_path="unused")
    for rank in (0, 1):
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        monkeypatch.setenv("RANK", str(rank))
        with pytest.raises(SeededBeforeSetup):
            getattr(module, entry_name)(**kwargs)
    assert observed[0][:2] == observed[1][:2]
    assert torch.equal(observed[0][2], observed[1][2])


def _make_structured_batch() -> dict:
    batch = make_synthetic_stage1_structured_batch(batch_size=1, seq_len=5)
    batch["raw"]["human"] = {
        "pos": batch["model"]["human"]["pos_rootrel"] + batch["model"]["human"]["root"][:, :, None, :]
    }
    return batch
