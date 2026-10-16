from __future__ import annotations

import hashlib
import json

import torch
import pytest
from accelerate import Accelerator

from hoi_latent.losses.stage1 import Stage1DecoderPostTrainingLoss
from hoi_latent.models.stage1 import build_stage1_autoencoder_from_config
from hoi_latent.models.stage1.autoencoder.latent_whitening import (
    Stage1LatentWhiteningTransform,
    fit_stage1_latent_whitening_from_values,
)
from hoi_latent.training.stage1.decoder_post_jitter import (
    DecoderPostJitterMetadata,
    Stage1DecoderPostJitterModule,
    Stage1DecoderPostJitterTrainer,
    _validate_post_jitter_artifacts,
    build_decoder_post_jitter_loss_from_config,
    freeze_decoder_post_jitter_parameters,
    sample_per_sample_sigma,
    sigma_cap_schedule,
)
from hoi_latent.training.stage1.phase1_warmup import (
    Stage1Phase1TrainState,
    build_optimizer_from_config,
    load_component_state_dicts,
)
from test_stage1_autoencoder import _config as _autoencoder_config
from test_stage1_phase1_warmup_loss import _identity_normalizer, _make_latent, _make_zero_batch, _make_zero_coarse
from test_stage1_tokenizers import make_synthetic_stage1_structured_batch


def test_sigma_cap_schedule_and_per_sample_sampling():
    assert sigma_cap_schedule(0, warmup_steps=2, ramp_end_step=10, sigma_target=0.125) == 0.0
    assert sigma_cap_schedule(2, warmup_steps=2, ramp_end_step=10, sigma_target=0.125) == 0.0
    assert sigma_cap_schedule(6, warmup_steps=2, ramp_end_step=10, sigma_target=0.125) == pytest.approx(0.0625)
    assert sigma_cap_schedule(10, warmup_steps=2, ramp_end_step=10, sigma_target=0.125) == pytest.approx(0.125)

    sigma = sample_per_sample_sigma(
        16,
        sigma_cap=0.125,
        clean_probability=0.25,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert sigma.shape == (16,)
    assert sigma.dtype == torch.float32
    assert sigma.min().item() >= 0.0
    assert sigma.max().item() <= 0.125
    assert torch.equal(
        sample_per_sample_sigma(
            4,
            sigma_cap=0.125,
            clean_probability=1.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        torch.zeros(4),
    )


def test_decoder_post_loss_disables_cov_and_reports_clean_noisy_groups():
    frame_mask = torch.ones(2, 4, dtype=torch.bool)
    batch = _make_zero_batch(frame_mask)
    coarse = _make_zero_coarse(frame_mask)
    latent = _make_latent(batch_size=2)
    loss_fn = Stage1DecoderPostTrainingLoss(normalizer=_identity_normalizer())

    out = loss_fn(
        coarse,
        latent,
        batch,
        global_step=40000,
        sigma_white=torch.tensor([0.0, 0.1]),
    )

    assert torch.isfinite(out.total)
    assert out.components["latent_cov"].item() == 0.0
    assert out.components["latent_cov_mean"].item() == 0.0
    assert out.components["latent_cov_offdiag"].item() == 0.0
    assert out.components["latent_cov_diag"].item() == 0.0
    assert out.diagnostics["clean_samples.sample_count"].item() == 1
    assert out.diagnostics["noisy_samples.sample_count"].item() == 1
    assert out.diagnostics["rec_loss_clean_samples"] is not None
    assert out.diagnostics["rec_loss_noisy_samples"] is not None
    assert out.diagnostics["sigma_zero_fraction"].item() == pytest.approx(0.5)
    assert out.diagnostics["sigma_nonzero_fraction"].item() == pytest.approx(0.5)


def test_decoder_post_loss_records_none_for_missing_groups():
    frame_mask = torch.ones(2, 4, dtype=torch.bool)
    batch = _make_zero_batch(frame_mask)
    coarse = _make_zero_coarse(frame_mask)
    latent = _make_latent(batch_size=2)
    loss_fn = Stage1DecoderPostTrainingLoss(normalizer=_identity_normalizer())

    out = loss_fn(
        coarse,
        latent,
        batch,
        global_step=40000,
        sigma_white=torch.zeros(2),
    )

    assert out.diagnostics["noisy_samples.sample_count"].item() == 0
    assert out.diagnostics["rec_loss_noisy_samples"] is None


def test_decoder_post_loss_can_skip_expensive_group_diagnostics():
    frame_mask = torch.ones(2, 4, dtype=torch.bool)
    batch = _make_zero_batch(frame_mask)
    coarse = _make_zero_coarse(frame_mask)
    latent = _make_latent(batch_size=2)
    loss_fn = Stage1DecoderPostTrainingLoss(normalizer=_identity_normalizer())
    calls = {"count": 0}
    original_forward = loss_fn.reconstruction_loss.forward

    def _count_forward(*args, **kwargs):
        calls["count"] += 1
        return original_forward(*args, **kwargs)

    loss_fn.reconstruction_loss.forward = _count_forward  # type: ignore[assignment]
    out = loss_fn(
        coarse,
        latent,
        batch,
        global_step=40000,
        sigma_white=torch.tensor([0.0, 0.1]),
        compute_group_diagnostics=False,
    )

    assert torch.isfinite(out.total)
    assert calls["count"] == 1
    assert out.diagnostics["clean_samples.sample_count"].item() == 1
    assert out.diagnostics["noisy_samples.sample_count"].item() == 1
    assert out.diagnostics["rec_loss_clean_samples"] is None
    assert out.diagnostics["rec_loss_noisy_samples"] is None


def test_decoder_post_module_single_decoder_forward_without_refiner():
    module, batch = _make_post_jitter_module(sigma_target=0.0)
    calls = {"decoder": 0}
    original_forward = module.stage1.decoder.forward

    def _count_decoder_forward(*args, **kwargs):
        calls["decoder"] += 1
        return original_forward(*args, **kwargs)

    module.stage1.decoder.forward = _count_decoder_forward  # type: ignore[assignment]

    out = module(batch, global_step=1)

    assert torch.isfinite(out.total)
    assert calls["decoder"] == 1
    assert not hasattr(module.stage1, "refiner")


def test_decoder_post_module_only_decoder_gets_gradients():
    module, batch = _make_post_jitter_module(sigma_target=0.0)

    out = module(batch, global_step=1)
    out.total.backward()

    assert any(_has_grad(param) for param in module.stage1.decoder.parameters())
    for name, submodule in [
        ("tokenizer", module.stage1.tokenizer),
        ("encoder", module.stage1.encoder),
        ("whitening", module.whitening),
    ]:
        assert all(param.grad is None for param in submodule.parameters()), name


def test_decoder_post_module_registers_only_decoder_for_ddp_sync():
    module, _ = _make_post_jitter_module(sigma_target=0.0)

    named_modules = dict(module.named_modules())
    decoder_param_ids = {id(param) for param in module.stage1.decoder.parameters()}
    wrapper_param_ids = {id(param) for param in module.parameters()}

    assert "decoder" in named_modules
    assert module.decoder is module.stage1.decoder
    assert wrapper_param_ids == decoder_param_ids
    assert "whitening" not in named_modules
    assert "loss_fn" not in named_modules
    assert "_stage1" not in named_modules


def test_decoder_post_checkpoint_records_source_and_whitening(tmp_path):
    module, batch = _make_post_jitter_module(sigma_target=0.0)
    whitening_path = tmp_path / "whitening.pt"
    whitening_path.write_bytes(b"whitening artifact")
    cfg = {
        "training": {
            "stage1": {
                "run_name": "unit",
                "output_root": str(tmp_path),
                "best_checkpoint_metric": "eval/T3_drift_at_0.125_cm",
                "best_checkpoint_mode": "min",
                "optimizer": {
                    "type": "adamw",
                    "lr": 2.0e-5,
                    "weight_decay": 1.0e-4,
                    "betas": [0.9, 0.95],
                },
            }
        },
        "loss": {"stage1": {"decoder_post_jitter": {}}},
    }
    optimizer = build_optimizer_from_config(cfg, module)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    module, optimizer = accelerator.prepare(module, optimizer)
    state = Stage1Phase1TrainState(
        accelerator=accelerator,
        model=module,
        loss_fn=accelerator.unwrap_model(module).loss_fn,
        optimizer=optimizer,
        scheduler=None,
        global_step=0,
        epoch=0,
    )
    trainer = Stage1DecoderPostJitterTrainer(
        state=state,
        run_dir=tmp_path,
        resolved_config=cfg,
        grad_clip_norm=1.0,
        log_interval_steps=1,
        checkpoint_interval_steps=1,
        eval_interval_steps=1,
        post_jitter_metadata=DecoderPostJitterMetadata("source/final.pt", str(whitening_path)),
        structured_normalizer=_identity_normalizer(),
    )

    trainer.train_step(batch)
    rows = [json.loads(line) for line in (tmp_path / "logs" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "train/rec_loss_all" in rows[-1]
    assert "train/rec_loss_clean_samples" in rows[-1]
    assert "train/rec_loss_noisy_samples" in rows[-1]
    assert "train/sigma_zero_fraction" in rows[-1]
    assert "train/sigma_mean" in rows[-1]
    path = trainer.save_checkpoint("latest")
    payload = torch.load(path, map_location="cpu")

    assert payload["post_training_phase"] == "decoder_post_jitter"
    assert payload["source_stage1_checkpoint"] == "source/final.pt"
    assert payload["whitening_artifact"] == str(whitening_path)
    expected_hash = hashlib.sha256(whitening_path.read_bytes()).hexdigest()
    assert payload["whitening_artifact_sha256"] == expected_hash

    relocated = tmp_path / "relocated.pt"
    relocated.write_bytes(whitening_path.read_bytes())
    trainer.post_jitter_metadata = DecoderPostJitterMetadata("source/final.pt", str(relocated))
    trainer.load_checkpoint(path)
    assert trainer.post_jitter_metadata.whitening_artifact_sha256 == expected_hash
    relocated.write_bytes(b"different artifact")
    with pytest.raises(ValueError, match="whitening artifact hash"):
        trainer.load_checkpoint(path)

    new_module, _ = _make_post_jitter_module(sigma_target=0.0)
    load_component_state_dicts(new_module.stage1, payload["components"])


def test_post_jitter_forwards_autocast_but_whitening_stays_fp32(tmp_path, monkeypatch):
    module, batch = _make_post_jitter_module(sigma_target=0.125)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    monkeypatch.setattr(accelerator, "autocast", lambda: torch.autocast("cpu", dtype=torch.bfloat16))
    optimizer = accelerator.prepare(torch.optim.AdamW(module.parameters(), lr=1.0e-4))
    state = Stage1Phase1TrainState(
        accelerator=accelerator,
        model=module,
        loss_fn=module.loss_fn,
        optimizer=optimizer,
        scheduler=None,
        global_step=0,
        epoch=0,
    )
    trainer = Stage1DecoderPostJitterTrainer(
        state=state,
        run_dir=tmp_path,
        resolved_config={},
        log_interval_steps=100,
        post_jitter_metadata=DecoderPostJitterMetadata("unused", "unused"),
        structured_normalizer=_identity_normalizer(),
        fixed_t3_sigmas=[0.125],
    )
    decoder_dtypes = []
    projection = next(child for child in module.decoder.modules() if isinstance(child, torch.nn.Linear))
    projection.register_forward_hook(lambda child, args, output: decoder_dtypes.append(output.dtype))
    whitening_calls = []

    def spy_transform(method):
        def checked(values, mask):
            assert not torch.is_autocast_enabled("cpu")
            assert values.dtype == torch.float32
            result = method(values, mask)
            assert result.dtype == torch.float32
            whitening_calls.append(method.__name__)
            return result
        return checked

    monkeypatch.setattr(module.whitening, "whiten", spy_transform(module.whitening.whiten))
    monkeypatch.setattr(module.whitening, "unwhiten", spy_transform(module.whitening.unwhiten))
    assert torch.isfinite(trainer.train_step(batch).total)
    assert torch.isfinite(trainer.eval_step(batch).total)
    assert trainer._t3_robustness_metrics(batch)["T3_drift_at_0.125_cm"] >= 0.0
    assert decoder_dtypes == [torch.bfloat16] * 4
    assert whitening_calls.count("whiten") == 3
    assert whitening_calls.count("unwhiten") == 3


def test_decoder_post_loss_builder_uses_structured_normalizer():
    normalizer = _identity_normalizer()
    loss = build_decoder_post_jitter_loss_from_config(
        {"loss": {"stage1": {"decoder_post_jitter": {"lambda_world_marker": 1.0}}}},
        structured_normalizer=normalizer,
    )

    assert loss.reconstruction_loss.normalizer is normalizer


def test_decoder_post_preflight_reports_missing_whitening_artifact(tmp_path):
    ckpt_dir = tmp_path / "phase1" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    source = ckpt_dir / "final.pt"
    source.write_bytes(b"not-a-real-checkpoint")
    whitening = tmp_path / "runs" / "diagnostics" / "stage1_latent_whitening" / "run" / "whitening.pt"

    with pytest.raises(FileNotFoundError, match="generated under runs"):
        _validate_post_jitter_artifacts(
            source_path=source,
            whitening_path=whitening,
            data_root="/data/InterAct",
        )


def _make_post_jitter_module(*, sigma_target: float) -> tuple[Stage1DecoderPostJitterModule, dict]:
    batch = make_synthetic_stage1_structured_batch(batch_size=1, seq_len=5)
    batch["raw"]["human"] = {
        "pos": batch["model"]["human"]["pos_rootrel"] + batch["model"]["human"]["root"][:, :, None, :]
    }
    stage1_model = build_stage1_autoencoder_from_config(_autoencoder_config(), batch=batch)
    freeze_decoder_post_jitter_parameters(stage1_model)
    loss_fn = Stage1DecoderPostTrainingLoss(normalizer=_identity_normalizer())
    whitening = Stage1LatentWhiteningTransform(fit_stage1_latent_whitening_from_values(torch.randn(128, 8)))
    module = Stage1DecoderPostJitterModule(
        stage1_model,
        loss_fn,
        whitening,
        sigma_target=sigma_target,
        sigma_warmup_steps=0,
        sigma_ramp_end_step=1,
        clean_probability=1.0 if sigma_target == 0.0 else 0.25,
    )
    return module, batch


def _has_grad(param: torch.nn.Parameter) -> bool:
    return param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum().item() > 0.0


def _encode_zrec_mask(module: Stage1DecoderPostJitterModule, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    stage1 = module.stage1
    tok = stage1.tokenizer(batch, return_aux=False, return_concat=False, validate=False)
    latent = stage1.encoder(tok, batch=batch, return_diagnostics=False, validate=False)
    return latent.z_rec.float(), latent.mask.bool()


def _spy_decoder_z_override(module: Stage1DecoderPostJitterModule) -> dict:
    captured: dict = {}
    original = module.stage1.decoder.forward

    def _spy(latent, **kwargs):
        captured["z_override"] = kwargs.get("z_override")
        return original(latent, **kwargs)

    module.stage1.decoder.forward = _spy  # type: ignore[assignment]
    return captured


def test_decoder_post_module_sigma_zero_jitter_is_noop_roundtrip():
    """sigma=0 must feed an inverse-whitening roundtrip of z_rec (no-op) into the decoder."""
    module, batch = _make_post_jitter_module(sigma_target=0.0)
    captured = _spy_decoder_z_override(module)

    torch.manual_seed(0)
    module(batch, global_step=1)
    z_rec, mask = _encode_zrec_mask(module, batch)
    z_override = captured["z_override"]

    assert z_override is not None
    assert torch.isfinite(z_override).all()
    # invalid tokens stay exactly zero through whiten -> unwhiten -> zero_invalid
    invalid = z_override[~mask]
    if invalid.numel():
        assert invalid.abs().max().item() == 0.0
    # sigma=0: decoder input is the (lossless) roundtrip of the clean latent
    assert torch.allclose(z_override[mask], z_rec[mask], atol=1e-4)


def test_decoder_post_module_positive_sigma_actually_jitters_valid_tokens():
    """sigma>0 must perturb valid tokens (finite, non-zero change) while keeping invalid zero."""
    module, batch = _make_post_jitter_module(sigma_target=0.125)
    module.clean_probability = 0.0  # force the jitter branch for every sample
    captured = _spy_decoder_z_override(module)

    torch.manual_seed(0)
    out = module(batch, global_step=10**9)  # past ramp -> sigma_cap == sigma_target
    z_rec, mask = _encode_zrec_mask(module, batch)
    z_override = captured["z_override"]

    assert torch.isfinite(out.total)
    assert torch.isfinite(z_override).all()
    invalid = z_override[~mask]
    if invalid.numel():
        assert invalid.abs().max().item() == 0.0
    # the whitened jitter must reach the decoder input on valid tokens
    assert not torch.allclose(z_override[mask], z_rec[mask], atol=1e-6)


def test_decoder_post_t3_robustness_metrics_emit_best_checkpoint_key():
    """_t3_robustness_metrics must produce the finite dotted key that best-checkpoint selection reads."""
    module, batch = _make_post_jitter_module(sigma_target=0.125)
    cfg = {
        "training": {
            "stage1": {
                "run_name": "unit-t3",
                "output_root": "unused",
                "best_checkpoint_metric": "eval/T3_drift_at_0.125_cm",
                "best_checkpoint_mode": "min",
                "optimizer": {"type": "adamw", "lr": 2.0e-5, "weight_decay": 1.0e-4, "betas": [0.9, 0.95]},
            }
        },
        "loss": {"stage1": {"decoder_post_jitter": {}}},
    }
    optimizer = build_optimizer_from_config(cfg, module)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    prepared, _ = accelerator.prepare(module, optimizer)
    state = Stage1Phase1TrainState(
        accelerator=accelerator,
        model=prepared,
        loss_fn=accelerator.unwrap_model(prepared).loss_fn,
        optimizer=optimizer,
        scheduler=None,
        global_step=0,
        epoch=0,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        trainer = Stage1DecoderPostJitterTrainer(
            state=state,
            run_dir=tmp,
            resolved_config=cfg,
            grad_clip_norm=1.0,
            log_interval_steps=1,
            checkpoint_interval_steps=1,
            eval_interval_steps=1,
            post_jitter_metadata=DecoderPostJitterMetadata("source/final.pt", "whitening.pt"),
            structured_normalizer=_identity_normalizer(),
            fixed_t3_sigmas=[0.05, 0.125, 0.20],
        )
        metrics = trainer._t3_robustness_metrics(batch)

    assert "T3_drift_at_0.125_cm" in metrics
    assert all(isinstance(v, float) and v == v and v >= 0.0 for v in metrics.values())


def test_decoder_post_evaluate_averages_t3_over_configured_batches(tmp_path):
    module, batch = _make_post_jitter_module(sigma_target=0.0)
    whitening_path = tmp_path / "whitening.pt"
    whitening_path.write_bytes(b"whitening artifact")
    cfg = {
        "training": {
            "stage1": {
                "run_name": "unit-eval",
                "output_root": str(tmp_path),
                "best_checkpoint_metric": "eval/T3_drift_at_0.125_cm",
                "best_checkpoint_mode": "min",
                "optimizer": {"type": "adamw", "lr": 2.0e-5, "weight_decay": 1.0e-4, "betas": [0.9, 0.95]},
            }
        },
        "loss": {"stage1": {"decoder_post_jitter": {}}},
    }
    optimizer = build_optimizer_from_config(cfg, module)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    prepared, optimizer = accelerator.prepare(module, optimizer)
    state = Stage1Phase1TrainState(
        accelerator=accelerator,
        model=prepared,
        loss_fn=accelerator.unwrap_model(prepared).loss_fn,
        optimizer=optimizer,
        scheduler=None,
        global_step=0,
        epoch=0,
    )
    trainer = Stage1DecoderPostJitterTrainer(
        state=state,
        run_dir=tmp_path,
        resolved_config=cfg,
        grad_clip_norm=1.0,
        log_interval_steps=1,
        checkpoint_interval_steps=1,
        eval_interval_steps=1,
        post_jitter_metadata=DecoderPostJitterMetadata("source/final.pt", str(whitening_path)),
        structured_normalizer=_identity_normalizer(),
        fixed_t3_sigmas=[0.125],
        t3_eval_batches=2,
    )
    calls = {"count": 0}

    def _fake_t3(_batch):
        calls["count"] += 1
        return {"T3_drift_at_0.125_cm": float(calls["count"])}

    trainer._t3_robustness_metrics = _fake_t3  # type: ignore[assignment]
    out = trainer.evaluate([batch, batch, batch], max_batches=3)

    assert calls["count"] == 2
    assert out["eval/T3_drift_at_0.125_cm"] == pytest.approx(1.5)
