from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed

from hoi_latent.config.io import load_yaml
from hoi_latent.data.transforms import Stage1StructuredNormalizer
from hoi_latent.losses.stage1 import Stage1DecoderPostTrainingLoss, Stage1LossOutput
from hoi_latent.models.stage1.autoencoder.latent_whitening import (
    Stage1LatentWhiteningTransform,
    load_stage1_latent_whitening_transform,
)
from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.training.stage1.phase1_warmup import (
    Stage1Phase1TrainState,
    Stage1Phase1WarmupTrainer,
    _batch_size,
    _diagnostics_enabled,
    _optimizer_step_was_skipped,
    _run_dir_from_config,
    _set_current_cuda_device_for_distributed_barriers,
    _set_current_cuda_device_from_local_rank_env,
    _should_log,
    _stage1_train_cfg,
    _to_float,
    append_jsonl,
    build_optimizer_from_config,
    build_scheduler_from_config,
    build_stage1_phase1_dataloaders,
    current_lr,
    deep_merge,
    infer_frame_mask,
    load_or_compute_stage1_normalizers,
    move_to_device,
    prepare_run_directory,
    scheduler_metrics,
    validate_structured_batch,
    _write_run_metadata,
)
from hoi_latent.training.stage1.source_autoencoder import (
    build_stage1_source_autoencoder_from_config,
    load_stage1_source_config,
    load_stage1_source_component_state_dicts,
    merge_stage1_source_model_contract,
)


class Stage1DecoderPostJitterModule(torch.nn.Module):
    """DDP/Accelerate-safe decoder-only post-jitter wrapper."""

    def __init__(
        self,
        stage1: torch.nn.Module,
        loss_fn: Stage1DecoderPostTrainingLoss,
        whitening: Stage1LatentWhiteningTransform,
        *,
        sigma_target: float = 0.125,
        sigma_warmup_steps: int = 2000,
        sigma_ramp_end_step: int = 20000,
        clean_probability: float = 0.25,
    ) -> None:
        super().__init__()
        if sigma_target < 0.0:
            raise ValueError("sigma_target must be non-negative.")
        if sigma_warmup_steps < 0 or sigma_ramp_end_step <= sigma_warmup_steps:
            raise ValueError("sigma_ramp_end_step must be greater than sigma_warmup_steps.")
        if not 0.0 <= clean_probability <= 1.0:
            raise ValueError("clean_probability must be in [0,1].")
        # Only the decoder-side trainable modules are registered as children.
        # The encoder-side source model, loss object, and whitening transform are
        # frozen helpers used by forward(), so DDP should not broadcast/synchronize
        # their parameters or buffers during post-jitter training.
        object.__setattr__(self, "_stage1", stage1)
        object.__setattr__(self, "loss_fn", loss_fn)
        object.__setattr__(self, "whitening", whitening)
        self.decoder = _post_jitter_trainable_container(stage1)
        self.sigma_target = float(sigma_target)
        self.sigma_warmup_steps = int(sigma_warmup_steps)
        self.sigma_ramp_end_step = int(sigma_ramp_end_step)
        self.clean_probability = float(clean_probability)

    @property
    def stage1(self) -> torch.nn.Module:
        return self._stage1

    def component_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        if hasattr(self.stage1, "component_state_dicts"):
            return self.stage1.component_state_dicts()
        checkpoint_key = str(getattr(self.stage1, "checkpoint_key", "stage1"))
        return {checkpoint_key: self.stage1.state_dict()}

    def sigma_cap(self, global_step: int) -> float:
        return sigma_cap_schedule(
            int(global_step),
            warmup_steps=self.sigma_warmup_steps,
            ramp_end_step=self.sigma_ramp_end_step,
            sigma_target=self.sigma_target,
        )

    def sample_sigma(self, batch_size: int, *, device: torch.device, dtype: torch.dtype, global_step: int) -> torch.Tensor:
        return sample_per_sample_sigma(
            batch_size,
            sigma_cap=self.sigma_cap(global_step),
            clean_probability=self.clean_probability,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        batch: dict,
        *,
        global_step: int,
        return_diagnostics: bool = False,
        validate: bool = False,
    ) -> Stage1LossOutput:
        del validate
        frame_mask = infer_frame_mask(batch)
        target_num_frames = int(frame_mask.shape[1])
        for module in _post_jitter_frozen_modules(self.stage1).values():
            module.eval()
        self.whitening.eval()
        with torch.no_grad():
            latent = _post_jitter_encode(self.stage1, batch, return_diagnostics=False, validate=False)
            with torch.autocast(device_type=latent.z_rec.device.type, enabled=False):
                z_white = self.whitening.whiten(latent.z_rec.float(), latent.mask.bool())
        sigma_white = self.sample_sigma(
            int(z_white.shape[0]),
            device=z_white.device,
            dtype=z_white.dtype,
            global_step=global_step,
        )
        noise = torch.randn_like(z_white) * sigma_white[:, None, None]
        z_white_noisy = zero_invalid_tokens(z_white + noise, latent.mask.bool())
        with torch.autocast(device_type=z_white_noisy.device.type, enabled=False):
            z_rec_noisy = self.whitening.unwhiten(z_white_noisy.float(), latent.mask.bool())
        z_rec_noisy = zero_invalid_tokens(z_rec_noisy, latent.mask.bool())
        prediction = _post_jitter_decode(
            self.stage1,
            latent,
            batch,
            z_override=z_rec_noisy,
            target_num_frames=target_num_frames,
            target_frame_mask=frame_mask,
            return_diagnostics=False,
            validate=False,
        )
        loss_out = self.loss_fn(
            prediction,
            latent,
            batch,
            global_step=global_step,
            sigma_white=sigma_white,
            compute_group_diagnostics=return_diagnostics,
        )
        loss_out.diagnostics["sigma_cap"] = latent.z_rec.new_tensor(self.sigma_cap(global_step)).detach()
        return loss_out


@dataclass
class DecoderPostJitterMetadata:
    source_checkpoint: str
    whitening_artifact: str
    whitening_artifact_sha256: str | None = None


class Stage1DecoderPostJitterTrainer(Stage1Phase1WarmupTrainer):
    def __init__(
        self,
        *,
        post_jitter_metadata: DecoderPostJitterMetadata,
        structured_normalizer: Stage1StructuredNormalizer,
        fixed_t3_sigmas: list[float] | None = None,
        t3_eval_batches: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.post_jitter_metadata = post_jitter_metadata
        self.structured_normalizer = structured_normalizer
        self.fixed_t3_sigmas = fixed_t3_sigmas or [0.05, 0.10, 0.125, 0.15, 0.20]
        self.t3_eval_batches = max(int(t3_eval_batches), 0)
        self._cpu_gradient_sync_group = None

    def train_step(self, batch: dict) -> Stage1LossOutput:
        self.model.train()
        batch = move_to_device(batch, self.accelerator.device)
        next_step = self.state.global_step + 1
        return_diagnostics = _diagnostics_enabled(next_step, self.return_diagnostics_interval_steps)

        start = time.perf_counter()
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                loss_out = self.model(
                    batch,
                    global_step=self.state.global_step,
                    return_diagnostics=return_diagnostics,
                    validate=False,
                )
            self.accelerator.backward(loss_out.total)
            grad_norm = None
            if self.accelerator.sync_gradients:
                self._sync_trainable_gradients()
            if self.accelerator.sync_gradients and self.grad_clip_norm is not None and self.grad_clip_norm > 0.0:
                grad_norm = self.accelerator.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip_norm,
                )
            self.optimizer.step()
            if (
                self.accelerator.sync_gradients
                and self.scheduler is not None
                and not _optimizer_step_was_skipped(self.optimizer)
            ):
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.last_train_step_synced = bool(self.accelerator.sync_gradients)
        if self.last_train_step_synced:
            self.state.global_step += 1

        elapsed = max(time.perf_counter() - start, 1.0e-9)
        if self.last_train_step_synced:
            record = self._record_from_loss(
                loss_out,
                split="train",
                step=self.state.global_step,
                extra={
                    "lr": current_lr(self.optimizer),
                    **scheduler_metrics(self.scheduler, self.optimizer),
                    "grad_norm": _to_float(grad_norm) if grad_norm is not None else 0.0,
                    "examples_per_sec": _batch_size(batch) * self.accelerator.num_processes / elapsed,
                    "step_time_sec": elapsed,
                    "mixed_precision": self.accelerator.mixed_precision,
                    "manual_gradient_sync": float(self.accelerator.num_processes > 1),
                    "manual_gradient_sync_backend": "gloo_cpu" if self.accelerator.num_processes > 1 else "none",
                },
            )
            if _should_log(self.state.global_step, self.log_interval_steps):
                self.log_record(record, split="train")
        self.assert_frozen_modules_have_no_grad()
        return loss_out

    def _sync_trainable_gradients(self) -> None:
        if self.accelerator.num_processes <= 1:
            return
        group = self._get_cpu_gradient_sync_group()
        world_size = dist.get_world_size(group=group)
        for param in self.model.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad
            cpu_grad = grad.detach().float().cpu()
            dist.all_reduce(cpu_grad, op=dist.ReduceOp.SUM, group=group)
            cpu_grad.div_(float(world_size))
            grad.copy_(cpu_grad.to(device=grad.device, dtype=grad.dtype))

    def _get_cpu_gradient_sync_group(self):
        if self._cpu_gradient_sync_group is not None:
            return self._cpu_gradient_sync_group
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized for manual gradient sync.")
        if not dist.is_gloo_available():
            raise RuntimeError("Manual CPU gradient sync requires torch.distributed Gloo support.")
        self._cpu_gradient_sync_group = dist.new_group(backend="gloo")
        return self._cpu_gradient_sync_group

    def assert_frozen_modules_have_no_grad(self) -> None:
        module = self._unwrap_model()
        frozen = {**_post_jitter_frozen_modules(module.stage1), "whitening": module.whitening}
        for name, submodule in frozen.items():
            for param in submodule.parameters():
                if param.grad is not None:
                    raise AssertionError(f"Frozen {name} received gradients during decoder post-jitter training.")

    def save_checkpoint(self, tag: str) -> Path:
        if self.accelerator.is_main_process and self.post_jitter_metadata.whitening_artifact_sha256 is None:
            artifact = Path(self.post_jitter_metadata.whitening_artifact)
            self.post_jitter_metadata.whitening_artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        path = super().save_checkpoint(tag)
        if self.accelerator.is_main_process:
            payload = torch.load(path, map_location="cpu")
            payload["source_stage1_checkpoint"] = self.post_jitter_metadata.source_checkpoint
            payload["whitening_artifact"] = self.post_jitter_metadata.whitening_artifact
            payload["whitening_artifact_sha256"] = self.post_jitter_metadata.whitening_artifact_sha256
            payload["post_training_phase"] = "decoder_post_jitter"
            self.accelerator.save(payload, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        recorded_hash = checkpoint.get("whitening_artifact_sha256")
        if recorded_hash is not None:
            artifact = Path(self.post_jitter_metadata.whitening_artifact)
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != recorded_hash:
                raise ValueError("Post-jitter whitening artifact hash does not match resume checkpoint provenance.")
            self.post_jitter_metadata.whitening_artifact_sha256 = recorded_hash
        if "components" not in checkpoint:
            raise KeyError("Checkpoint missing 'components'.")
        load_stage1_source_component_state_dicts(self._unwrap_stage1_model(), checkpoint["components"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.state.global_step = int(checkpoint.get("global_step", 0))
        self.state.epoch = int(checkpoint.get("epoch", 0))
        best = checkpoint.get("best_eval_loss")
        self.state.best_eval_loss = None if best is None else float(best)

    @torch.no_grad()
    def evaluate(self, eval_loader: Iterable[dict], *, max_batches: int | None = None) -> dict[str, float]:
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        t3_sums: dict[str, float] = {}
        t3_counts: dict[str, int] = {}
        t3_batches = 0
        for batch_idx, batch in enumerate(eval_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch_on_device = move_to_device(batch, self.accelerator.device)
            loss_out = self.eval_step(batch_on_device)
            record = self._record_from_loss(loss_out, split="eval", step=self.state.global_step, extra={})
            self._accumulate_distributed_metrics(
                {key: float(value) for key, value in record.items() if key != "split" and isinstance(value, (int, float))},
                metric_sums,
                metric_counts,
            )
            if t3_batches < self.t3_eval_batches:
                robustness = self._t3_robustness_metrics(batch_on_device)
                self._accumulate_distributed_metrics(
                    {key: float(value) for key, value in robustness.items()},
                    t3_sums,
                    t3_counts,
                )
                t3_batches += 1
        if not metric_sums:
            raise ValueError("Evaluation produced no batches.")
        averaged = {key: metric_sums[key] / max(metric_counts[key], 1) for key in sorted(metric_sums)}
        for key in sorted(t3_sums):
            averaged[f"eval/{key}"] = t3_sums[key] / max(t3_counts[key], 1)
        averaged["step"] = int(self.state.global_step)
        averaged["epoch"] = int(self.state.epoch)
        averaged["split"] = "eval"
        self.log_record(averaged, split="eval")
        self._maybe_update_best(averaged)
        return averaged

    def _accumulate_distributed_metrics(
        self,
        local_metrics: dict[str, float],
        metric_sums: dict[str, float],
        metric_counts: dict[str, int],
    ) -> None:
        if self.accelerator.num_processes <= 1:
            for key, value in local_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                metric_counts[key] = metric_counts.get(key, 0) + 1
            return

        group = self._get_cpu_gradient_sync_group()
        world_size = dist.get_world_size(group=group)
        gathered_keys: list[list[str]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered_keys, sorted(local_metrics), group=group)
        all_keys = sorted({key for keys in gathered_keys for key in keys})
        for key in all_keys:
            present = key in local_metrics
            packed = torch.tensor(
                [float(local_metrics.get(key, 0.0)), 1.0 if present else 0.0],
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
            metric_sums[key] = metric_sums.get(key, 0.0) + float(packed[0].item())
            metric_counts[key] = metric_counts.get(key, 0) + int(packed[1].item())

    @torch.no_grad()
    def eval_step(self, batch: dict) -> Stage1LossOutput:
        self.model.eval()
        batch = move_to_device(batch, self.accelerator.device)
        with self.accelerator.autocast():
            return self.model(
                batch,
                global_step=self.state.global_step,
                return_diagnostics=True,
                validate=False,
            )

    @torch.no_grad()
    def _t3_robustness_metrics(self, batch: dict) -> dict[str, float]:
        module = self._unwrap_model()
        stage1 = module.stage1
        stage1.eval()
        with self.accelerator.autocast():
            latent = _post_jitter_encode(stage1, batch, return_diagnostics=False, validate=False)
        with torch.autocast(device_type=latent.z_rec.device.type, enabled=False):
            z_white = module.whitening.whiten(latent.z_rec.float(), latent.mask.bool())
        frame_mask = infer_frame_mask(batch)
        with self.accelerator.autocast():
            clean = _post_jitter_decode(
                stage1,
                latent,
                batch,
                target_frame_mask=frame_mask,
                return_diagnostics=False,
                validate=False,
            )
        world_clean = _prediction_world_markers(clean, self.structured_normalizer)
        hmask = batch["masks"]["human"].bool().cpu()
        out: dict[str, float] = {}
        for sigma in self.fixed_t3_sigmas:
            z_white_noisy = zero_invalid_tokens(
                z_white + torch.randn_like(z_white) * float(sigma),
                latent.mask.bool(),
            )
            with torch.autocast(device_type=z_white_noisy.device.type, enabled=False):
                z_rec_noisy = module.whitening.unwhiten(z_white_noisy.float(), latent.mask.bool())
            with self.accelerator.autocast():
                pred = _post_jitter_decode(
                    stage1,
                    latent,
                    batch,
                    z_override=z_rec_noisy,
                    target_frame_mask=frame_mask,
                    return_diagnostics=False,
                    validate=False,
                )
            world_noisy = _prediction_world_markers(pred, self.structured_normalizer)
            drift_cm = float((world_noisy - world_clean).norm(dim=-1)[hmask].mean() * 100.0)
            key = f"T3_drift_at_{sigma:g}_cm".replace(".", "_")
            out[key] = drift_cm
            if abs(float(sigma) - 0.125) < 1.0e-9:
                out["T3_drift_at_0.125_cm"] = drift_cm
        return out

    def _record_from_loss(self, loss_out: Stage1LossOutput, *, split: str, step: int, extra: dict) -> dict:
        record: dict[str, Any] = {"step": int(step), "epoch": int(self.state.epoch), "split": split}
        for key, value in loss_out.components.items():
            record[f"{split}/loss.{key}"] = _record_value(value)
        record[f"{split}/rec_loss_all"] = _record_value(loss_out.total)
        direct_diagnostic_keys = {
            "rec_loss_clean_samples": "rec_loss_clean_samples",
            "rec_loss_noisy_samples": "rec_loss_noisy_samples",
            "world_marker_clean_samples": "world_marker_clean_samples",
            "world_marker_noisy_samples": "world_marker_noisy_samples",
            "relation_dist_clean_samples": "relation_dist_clean_samples",
            "relation_dist_noisy_samples": "relation_dist_noisy_samples",
            "relation_delta_m2o_clean_samples": "relation_delta_m2o_clean_samples",
            "relation_delta_m2o_noisy_samples": "relation_delta_m2o_noisy_samples",
            "sigma_zero_fraction": "sigma_zero_fraction",
            "sigma_nonzero_fraction": "sigma_nonzero_fraction",
            "sigma_white_mean": "sigma_mean",
            "sigma_white_max": "sigma_max",
        }
        for key, value in loss_out.diagnostics.items():
            if key in direct_diagnostic_keys:
                record[f"{split}/{direct_diagnostic_keys[key]}"] = _record_value(value)
                continue
            prefix = "metric" if key == "world_marker_mpjpe_m" else "diag"
            record[f"{split}/{prefix}.{key}"] = _record_value(value)
        for key, value in extra.items():
            record[f"{split}/{key}"] = value if isinstance(value, str) else _record_value(value)
        return record

    def log_record(self, record: dict, *, split: str) -> None:
        if self.accelerator.is_main_process:
            log_path = self.train_log_path if split == "train" else self.eval_log_path
            append_jsonl(record, log_path)
        if self.accelerator.is_main_process and self.accelerator.trackers:
            self.accelerator.log(
                {key: value for key, value in record.items() if isinstance(value, (int, float))},
                step=record["step"],
            )

    def _unwrap_model(self) -> Stage1DecoderPostJitterModule:
        return self.accelerator.unwrap_model(self.model)

    def _unwrap_stage1_model(self) -> torch.nn.Module:
        return self._unwrap_model().stage1


def sigma_cap_schedule(
    step: int,
    *,
    warmup_steps: int = 2000,
    ramp_end_step: int = 20000,
    sigma_target: float = 0.125,
) -> float:
    if step < warmup_steps:
        return 0.0
    if step < ramp_end_step:
        progress = (step - warmup_steps) / max(ramp_end_step - warmup_steps, 1)
        return float(sigma_target) * float(progress)
    return float(sigma_target)


def sample_per_sample_sigma(
    batch_size: int,
    *,
    sigma_cap: float,
    clean_probability: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if sigma_cap < 0.0:
        raise ValueError("sigma_cap must be non-negative.")
    if not 0.0 <= clean_probability <= 1.0:
        raise ValueError("clean_probability must be in [0,1].")
    if sigma_cap == 0.0:
        return torch.zeros(batch_size, device=device, dtype=dtype)
    sigma = torch.rand(batch_size, device=device, dtype=dtype) * float(sigma_cap)
    clean = torch.rand(batch_size, device=device) < float(clean_probability)
    return sigma.masked_fill(clean, 0.0)


def run_stage1_decoder_post_jitter_training(
    *,
    data_config_path: str | Path,
    train_config_path: str | Path,
    source_run: str | Path,
    source_ckpt: str = "final.pt",
    whitening_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    data_config, train_config, resolved_config = load_decoder_post_jitter_training_config(
        data_config_path,
        train_config_path,
        overrides=overrides,
    )
    set_seed(int(resolved_config.get("seed", 42)), device_specific=False)
    source_model_config = load_stage1_source_config(source_run, fallback_config=resolved_config)
    resolved_config = merge_stage1_source_model_contract(resolved_config, source_model_config)
    train_cfg = _stage1_train_cfg(resolved_config)
    post_cfg = train_cfg.get("decoder_post_jitter", {})
    if source_run:
        train_cfg["source_run"] = str(source_run)
    train_cfg["source_ckpt"] = str(source_ckpt)
    train_cfg["whitening"] = str(whitening_path)
    source_path = Path(source_run) / "checkpoints" / str(source_ckpt)
    _validate_post_jitter_artifacts(
        source_path=source_path,
        whitening_path=Path(whitening_path),
        data_root=resolved_config.get("data", {}).get("root"),
    )
    wandb_cfg = train_cfg.get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", True)) and wandb_cfg.get("mode", "online") != "disabled"
    _set_current_cuda_device_from_local_rank_env()
    accelerator = Accelerator(
        mixed_precision=str(train_cfg.get("accelerate", {}).get("mixed_precision", "bf16")),
        gradient_accumulation_steps=int(train_cfg.get("accelerate", {}).get("gradient_accumulation_steps", 1)),
        log_with="wandb" if wandb_enabled else None,
        project_dir=str(_run_dir_from_config(train_cfg)),
    )
    _set_current_cuda_device_for_distributed_barriers(accelerator)
    run_dir = _run_dir_from_config(train_cfg)
    resume = train_cfg.get("resume")
    if accelerator.is_main_process:
        prepare_run_directory(run_dir, resume=resume)
        _write_run_metadata(run_dir, data_config_path, train_config_path, data_config, train_config, resolved_config)
    accelerator.wait_for_everyone()

    normalizer, structured_normalizer, normalizer_paths = load_or_compute_stage1_normalizers(
        resolved_config,
        accelerator=accelerator,
    )
    if structured_normalizer is None:
        raise ValueError("Decoder post-jitter training requires a structured normalizer.")
    train_loader, eval_loader = build_stage1_phase1_dataloaders(
        resolved_config,
        normalizer=normalizer,
        structured_normalizer=structured_normalizer,
    )
    stage1_model = build_stage1_source_autoencoder_from_config(source_model_config)
    source_payload = torch.load(source_path, map_location="cpu")
    load_stage1_source_component_state_dicts(stage1_model, source_payload["components"])
    freeze_decoder_post_jitter_parameters(stage1_model)
    whitening = load_stage1_latent_whitening_transform(whitening_path, map_location="cpu")
    _freeze_module(whitening)
    stage1_model.to(accelerator.device)
    whitening.to(accelerator.device)
    loss_fn = build_decoder_post_jitter_loss_from_config(resolved_config, structured_normalizer=structured_normalizer)
    model = Stage1DecoderPostJitterModule(
        stage1_model,
        loss_fn,
        whitening,
        sigma_target=float(post_cfg.get("sigma_white_target", 0.125)),
        sigma_warmup_steps=int(post_cfg.get("sigma_warmup_steps", 2000)),
        sigma_ramp_end_step=int(post_cfg.get("sigma_ramp_end_step", 20000)),
        clean_probability=float(post_cfg.get("clean_probability", 0.25)),
    )
    optimizer = build_optimizer_from_config(resolved_config, model)
    scheduler = build_scheduler_from_config(resolved_config, optimizer)
    # Do not wrap the post-jitter model in DDP.  This phase only trains the
    # decoder, and some server CUDA/NCCL stacks have failed during DDP's initial
    # parameter broadcast before forward().  We instead shard dataloaders with
    # Accelerate and manually all-reduce decoder gradients before optimizer.step.
    optimizer, train_loader, eval_loader = accelerator.prepare(optimizer, train_loader, eval_loader)
    state = Stage1Phase1TrainState(
        accelerator=accelerator,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=0,
        epoch=0,
    )
    trainer = Stage1DecoderPostJitterTrainer(
        state=state,
        run_dir=run_dir,
        resolved_config=resolved_config,
        normalizer_paths=normalizer_paths,
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        log_interval_steps=train_cfg.get("log_interval_steps", 50),
        checkpoint_interval_steps=train_cfg.get("checkpoint_interval_steps", 5000),
        save_periodic_step_checkpoints=train_cfg.get("save_periodic_step_checkpoints", False),
        eval_interval_steps=train_cfg.get("eval_interval_steps", 1000),
        return_diagnostics_interval_steps=post_cfg.get("return_diagnostics_interval_steps", 1000),
        best_checkpoint_metric=train_cfg.get("best_checkpoint_metric", "eval/T3_drift_at_0.125_cm"),
        best_checkpoint_mode=train_cfg.get("best_checkpoint_mode", "min"),
        post_jitter_metadata=DecoderPostJitterMetadata(str(source_path), str(whitening_path)),
        structured_normalizer=structured_normalizer,
        fixed_t3_sigmas=[float(x) for x in post_cfg.get("fixed_t3_sigmas", [0.05, 0.10, 0.125, 0.15, 0.20])],
        t3_eval_batches=int(post_cfg.get("t3_eval_batches", 4)),
    )
    if resume:
        trainer.load_checkpoint(resume)
    if wandb_enabled:
        accelerator.init_trackers(
            project_name=str(wandb_cfg.get("project", "combination-stage1")),
            config=resolved_config,
            init_kwargs={
                "wandb": {
                    "name": train_cfg.get("run_name"),
                    "entity": wandb_cfg.get("entity"),
                    "mode": wandb_cfg.get("mode", "online"),
                    "tags": wandb_cfg.get("tags", []),
                    **({"id": wandb_cfg.get("id"), "resume": wandb_cfg.get("resume", "allow")} if wandb_cfg.get("id") else {}),
                }
            },
        )
    if post_cfg.get("validate_first_batch", True):
        first_batch = next(iter(train_loader))
        validate_structured_batch(first_batch, window_size=resolved_config.get("data", {}).get("window_size"))

    max_steps = int(train_cfg.get("max_steps", 30000))
    max_eval_batches = train_cfg.get("max_eval_batches")
    max_eval_batches = None if max_eval_batches is None else int(max_eval_batches)
    while state.global_step < max_steps:
        for batch in train_loader:
            if state.global_step >= max_steps:
                break
            trainer.train_step(batch)
            if (
                trainer.last_train_step_synced
                and state.global_step > 0
                and state.global_step % trainer.eval_interval_steps == 0
            ):
                trainer.evaluate(eval_loader, max_batches=max_eval_batches)
            if (
                trainer.last_train_step_synced
                and state.global_step > 0
                and state.global_step % trainer.checkpoint_interval_steps == 0
            ):
                if trainer.save_periodic_step_checkpoints:
                    trainer.save_checkpoint(f"step_{state.global_step}")
                trainer.save_checkpoint("latest")
        state.epoch += 1
    trainer.evaluate(eval_loader, max_batches=max_eval_batches)
    trainer.save_checkpoint("latest")
    trainer.save_checkpoint("final")
    accelerator.wait_for_everyone()
    if dist.is_available() and dist.is_initialized():
        if trainer._cpu_gradient_sync_group is not None:
            dist.destroy_process_group(trainer._cpu_gradient_sync_group)
            trainer._cpu_gradient_sync_group = None
        dist.destroy_process_group()
    accelerator.end_training()
    return run_dir


def load_decoder_post_jitter_training_config(
    data_config_path: str | Path,
    train_config_path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> tuple[dict, dict, dict]:
    data_config = load_yaml(data_config_path)
    train_config = load_yaml(train_config_path)
    resolved = deep_merge(data_config, train_config)
    if overrides:
        resolved = apply_decoder_post_jitter_cli_overrides(resolved, overrides)
    return data_config, train_config, resolved


def apply_decoder_post_jitter_cli_overrides(config: dict, overrides: dict[str, Any]) -> dict:
    out = copy.deepcopy(config)
    mapping = {
        "run_name": ("training", "stage1", "run_name"),
        "output_root": ("training", "stage1", "output_root"),
        "seed": ("seed",),
        "resume": ("training", "stage1", "resume"),
        "max_steps": ("training", "stage1", "max_steps"),
        "batch_size": ("loader", "batch_size"),
        "num_workers": ("loader", "num_workers"),
        "lr": ("training", "stage1", "optimizer", "lr"),
        "weight_decay": ("training", "stage1", "optimizer", "weight_decay"),
        "eval_interval_steps": ("training", "stage1", "eval_interval_steps"),
        "checkpoint_interval_steps": ("training", "stage1", "checkpoint_interval_steps"),
        "log_interval_steps": ("training", "stage1", "log_interval_steps"),
        "debug_num_train_batches": ("training", "stage1", "debug_num_train_batches"),
        "debug_num_eval_batches": ("training", "stage1", "debug_num_eval_batches"),
        "mixed_precision": ("training", "stage1", "accelerate", "mixed_precision"),
        "gradient_accumulation_steps": ("training", "stage1", "accelerate", "gradient_accumulation_steps"),
        "wandb_project": ("training", "stage1", "wandb", "project"),
        "wandb_entity": ("training", "stage1", "wandb", "entity"),
        "wandb_mode": ("training", "stage1", "wandb", "mode"),
        "wandb_id": ("training", "stage1", "wandb", "id"),
        "wandb_resume": ("training", "stage1", "wandb", "resume"),
        "wandb_tags": ("training", "stage1", "wandb", "tags"),
        "validate_first_batch": ("training", "stage1", "decoder_post_jitter", "validate_first_batch"),
        "sigma_white_target": ("training", "stage1", "decoder_post_jitter", "sigma_white_target"),
        "clean_probability": ("training", "stage1", "decoder_post_jitter", "clean_probability"),
        "t3_eval_batches": ("training", "stage1", "decoder_post_jitter", "t3_eval_batches"),
    }
    for key, path in mapping.items():
        if key in overrides and overrides[key] is not None:
            _set_nested(out, path, overrides[key])
    return out


def build_decoder_post_jitter_loss_from_config(
    config: dict,
    *,
    structured_normalizer: Stage1StructuredNormalizer | None,
) -> Stage1DecoderPostTrainingLoss:
    loss_cfg = config.get("loss", {}).get("stage1", {}).get("decoder_post_jitter", {})
    kwargs = {key: value for key, value in loss_cfg.items() if value is not None}
    kwargs["normalizer"] = structured_normalizer
    return Stage1DecoderPostTrainingLoss(**kwargs)


def _validate_post_jitter_artifacts(
    *,
    source_path: Path,
    whitening_path: Path,
    data_root: str | None,
) -> None:
    if not source_path.exists():
        raise FileNotFoundError(
            "Stage1 decoder post-jitter source checkpoint is missing:\n"
            f"  {source_path}\n"
            "Pass --source-run and --source-ckpt pointing to an existing Phase1 checkpoint."
        )
    if whitening_path.exists():
        return
    data_root_text = str(data_root) if data_root else "<InterAct_root>"
    raise FileNotFoundError(
        "Stage1 latent whitening artifact is missing:\n"
        f"  {whitening_path}\n\n"
        "This file is generated under runs/ and is not tracked by git. Generate it on the server first, e.g.:\n"
        "  python scripts/diagnostics/stage1_latent/verify_latent_whitening_on_ckpt.py \\\n"
        f"    --run-dir {source_path.parent.parent} \\\n"
        f"    --ckpt {source_path.name} \\\n"
        f"    --data-root {data_root_text} \\\n"
        f"    --out {whitening_path}\n\n"
        "Alternatively, sync the existing local whitening.pt to that exact path."
    )


def _post_jitter_trainable_container(model: torch.nn.Module) -> torch.nn.Module:
    modules = _post_jitter_trainable_modules(model)
    if len(modules) == 1:
        return next(iter(modules.values()))
    return torch.nn.ModuleDict(modules)


def _post_jitter_trainable_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    provider = getattr(model, "post_jitter_trainable_modules", None)
    if callable(provider):
        modules = provider()
    else:
        if not hasattr(model, "decoder"):
            raise AttributeError(
                f"{type(model).__name__} must provide either .decoder or post_jitter_trainable_modules()."
            )
        modules = {"decoder": model.decoder}
    return _validate_post_jitter_module_mapping(modules, "post-jitter trainable")


def _post_jitter_frozen_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    provider = getattr(model, "post_jitter_frozen_modules", None)
    if callable(provider):
        modules = provider()
    else:
        missing = [name for name in ("tokenizer", "encoder") if not hasattr(model, name)]
        if missing:
            raise AttributeError(
                f"{type(model).__name__} must provide post_jitter_frozen_modules() "
                f"or production tokenizer/encoder modules; missing {missing}."
            )
        modules = {"tokenizer": model.tokenizer, "encoder": model.encoder}
    return _validate_post_jitter_module_mapping(modules, "post-jitter frozen")


def _validate_post_jitter_module_mapping(
    modules: dict[str, torch.nn.Module],
    label: str,
) -> dict[str, torch.nn.Module]:
    if not isinstance(modules, dict) or not modules:
        raise ValueError(f"{label} modules must be a non-empty dict[str, nn.Module].")
    out: dict[str, torch.nn.Module] = {}
    for name, module in modules.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} module names must be non-empty strings.")
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"{label} module {name!r} must be an nn.Module, got {type(module).__name__}.")
        out[name] = module
    return out


def _post_jitter_encode(
    model: torch.nn.Module,
    batch: dict,
    *,
    return_diagnostics: bool = False,
    validate: bool = False,
):
    encoder = getattr(model, "encode_for_post_jitter", None)
    if callable(encoder):
        return encoder(batch, return_diagnostics=return_diagnostics, validate=validate)
    tok = model.tokenizer(batch, return_aux=False, return_concat=False, validate=validate)
    return model.encoder(tok, batch=batch, return_diagnostics=return_diagnostics, validate=validate)


def _post_jitter_decode(
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
    decoder = getattr(model, "decode_for_post_jitter", None)
    if callable(decoder):
        return decoder(
            latent,
            batch,
            z_override=z_override,
            target_num_frames=target_num_frames,
            target_frame_mask=target_frame_mask,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
    return model.decoder(
        latent,
        z_override=z_override,
        target_num_frames=target_num_frames,
        target_frame_mask=target_frame_mask,
        return_diagnostics=return_diagnostics,
        validate=validate,
    )


def freeze_decoder_post_jitter_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for module in _post_jitter_frozen_modules(model).values():
        _freeze_module(module)
    for module in _post_jitter_trainable_modules(model).values():
        for param in module.parameters():
            param.requires_grad_(True)
        module.train()


def _freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()


def _prediction_world_markers(prediction, structured_normalizer: Stage1StructuredNormalizer) -> torch.Tensor:
    pos = structured_normalizer.denormalize_group_torch(
        "human",
        "pos_rootrel",
        prediction.human["pos_rootrel"].cpu(),
        require_stats=True,
    )
    root = structured_normalizer.denormalize_group_torch(
        "human",
        "root",
        prediction.human["root"].cpu(),
        require_stats=True,
    )
    return pos + root[:, :, None, :]


def _record_value(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return value
    return str(value)


def _set_nested(config: dict, path: tuple[str, ...], value: Any) -> None:
    node = config
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
