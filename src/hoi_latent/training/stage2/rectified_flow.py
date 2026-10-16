from __future__ import annotations

import copy
import hashlib
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from accelerate import Accelerator

from hoi_latent.config.io import load_yaml, save_yaml
from hoi_latent.config.stage1_anchor_overrides import apply_stage1_anchor_override
from hoi_latent.losses.stage2 import Stage2RFLossOutput, Stage2RectifiedFlowLoss
from hoi_latent.models.stage1.autoencoder.latent_whitening import (
    Stage1LatentWhiteningTransform,
    load_stage1_latent_whitening_transform,
)
from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.models.stage2 import (
    Stage2TriStreamMoTModel,
    build_stage2_tri_stream_mot_model_from_config,
    derive_stage2_latent_substream_ids,
)
from hoi_latent.training.stage2.flow_forcing_times import (
    build_stage2_training_flow_time_from_config,
    flow_time_log_diagnostics,
)
from hoi_latent.training.stage1.phase1_warmup import (
    _adamw_parameter_groups,
    _batch_size,
    _optimizer_step_was_skipped,
    _set_current_cuda_device_for_distributed_barriers,
    _set_current_cuda_device_from_local_rank_env,
    _should_log,
    _to_float,
    append_jsonl,
    build_stage1_phase1_dataloaders,
    deep_merge,
    load_or_compute_stage1_normalizers,
    move_to_device,
    prepare_run_directory,
    scheduler_metrics,
    validate_structured_batch,
)
from hoi_latent.training.stage1.source_autoencoder import (
    build_stage1_source_autoencoder_from_config,
    encode_stage1_source_latent,
    load_stage1_source_component_state_dicts,
    load_stage1_source_config,
)
from hoi_latent.utils.filesystem import ensure_dir
from hoi_latent.utils.json_io import save_json


@dataclass
class Stage2RectifiedFlowTrainState:
    accelerator: Accelerator
    stage2_model: torch.nn.Module
    stage1_model: torch.nn.Module
    whitening: Stage1LatentWhiteningTransform
    loss_fn: Stage2RectifiedFlowLoss
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    ema: "TrainableParameterEMA | None"
    global_step: int = 0
    epoch: int = 0
    best_eval_metric: float | None = None
    best_eval_step: int | None = None


@dataclass
class Stage2SourceArtifacts:
    stage1_checkpoint: str
    stage1_checkpoint_sha256: str
    whitening_artifact: str
    whitening_artifact_sha256: str
    whitening_payload: dict[str, Any]
    stage1_anchor_override: dict[str, Any] | None = None
    normalizers: dict[str, str | None] = field(default_factory=dict)
    normalizer_hashes: dict[str, str | None] = field(default_factory=dict)


class TrainableParameterEMA:
    """FP32 EMA over trainable parameters only."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        decay: float = 0.9999,
        warmup: bool = True,
        num_updates: int = 0,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0,1), got {decay}.")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = int(num_updates)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().float().clone()

    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        decay = self.decay_at_step()
        one_minus = 1.0 - decay
        params = dict(model.named_parameters())
        for name, shadow in self.shadow.items():
            param = params.get(name)
            if param is None:
                raise KeyError(f"EMA parameter {name!r} is missing from model.")
            if shadow.device != param.device:
                self.shadow[name] = shadow = shadow.to(device=param.device)
            shadow.mul_(decay).add_(param.detach().float(), alpha=one_minus)

    def decay_at_step(self) -> float:
        if not self.warmup:
            return self.decay
        # Common diffusion EMA warmup: avoid a stale randomly initialized EMA in
        # the first few thousand updates, then asymptote to target decay.
        warm = (1.0 + float(self.num_updates)) / (10.0 + float(self.num_updates))
        return min(self.decay, warm)

    def state_dict(self) -> dict[str, Any]:
        return {
            "shadow": {key: value.detach().cpu() for key, value in self.shadow.items()},
            "num_updates": int(self.num_updates),
            "target_decay": float(self.decay),
            "warmup": bool(self.warmup),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        shadow = state.get("shadow")
        if not isinstance(shadow, dict):
            raise KeyError("EMA state missing 'shadow'.")
        self.shadow = {str(key): value.detach().float().clone() for key, value in shadow.items()}
        self.num_updates = int(state.get("num_updates", 0))
        self.decay = float(state.get("target_decay", self.decay))
        self.warmup = bool(state.get("warmup", self.warmup))

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        params = dict(model.named_parameters())
        originals: dict[str, torch.Tensor] = {}
        try:
            for name, shadow in self.shadow.items():
                param = params.get(name)
                if param is None:
                    raise KeyError(f"EMA parameter {name!r} is missing from model.")
                originals[name] = param.detach().clone()
                param.data.copy_(shadow.to(device=param.device, dtype=param.dtype))
            yield
        finally:
            for name, original in originals.items():
                params[name].data.copy_(original.to(device=params[name].device, dtype=params[name].dtype))


class Stage2RectifiedFlowTrainer:
    def __init__(
        self,
        *,
        state: Stage2RectifiedFlowTrainState,
        run_dir: str | Path,
        resolved_config: dict,
        source_artifacts: Stage2SourceArtifacts,
        grad_clip_norm: float | None = 1.0,
        log_interval_steps: int = 50,
        checkpoint_interval_steps: int = 5000,
        velocity_eval_interval_steps: int = 1000,
        train_log_name: str = "train.jsonl",
        eval_log_name: str = "eval.jsonl",
        best_checkpoint_metric: str = "eval/rf_mse",
        best_checkpoint_mode: str = "min",
    ) -> None:
        self.state = state
        self.accelerator = state.accelerator
        self.run_dir = Path(run_dir)
        self.resolved_config = resolved_config
        self.source_artifacts = source_artifacts
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.log_interval_steps = int(log_interval_steps)
        self.checkpoint_interval_steps = int(checkpoint_interval_steps)
        self.velocity_eval_interval_steps = int(velocity_eval_interval_steps)
        self.best_checkpoint_metric = str(best_checkpoint_metric)
        self.best_checkpoint_mode = str(best_checkpoint_mode)
        self.last_train_step_synced = False
        self.train_log_path = self.run_dir / "logs" / train_log_name
        self.eval_log_path = self.run_dir / "logs" / eval_log_name
        self.checkpoint_dir = self.run_dir / "checkpoints"

    @property
    def stage2_model(self) -> torch.nn.Module:
        return self.state.stage2_model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.state.optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        return self.state.scheduler

    def train_step(self, batch: dict) -> Stage2RFLossOutput:
        self.stage2_model.train()
        _keep_frozen_text_encoder_eval(self._unwrap_stage2_model())
        self.state.stage1_model.eval()
        self.state.whitening.eval()
        batch = move_to_device(batch, self.accelerator.device)
        start = time.perf_counter()

        with self.accelerator.accumulate(self.stage2_model):
            latent, z_white_target = self._encode_target(batch)
            rf_state = self._sample_rf_state(batch, z_white_target, latent)
            force_text, force_geometry, dropout_diag = sample_categorical_condition_dropout(
                batch,
                self._condition_dropout_cfg(),
                device=z_white_target.device,
            )
            window_diag = _window_start_diagnostics(batch, device=z_white_target.device)
            with self.accelerator.autocast():
                out = self.stage2_model(
                    batch=batch,
                    z_white_t=rf_state["z_t"].to(dtype=z_white_target.dtype),
                    flow_time=rf_state["flow_time"],
                    latent_mask=latent.mask.bool(),
                    modality_ids=latent.modality_ids,
                    token_type_ids=latent.token_type_ids,
                    time_ids=latent.time_ids,
                    entity_ids=latent.entity_ids,
                    time_mask=latent.time_mask,
                    force_null_text_mask=force_text,
                    force_null_geometry_mask=force_geometry,
                    return_diagnostics=False,
                )
            substream_ids = derive_stage2_latent_substream_ids(
                latent.modality_ids,
                latent.token_type_ids,
                mask=latent.mask.bool(),
            )
            loss_out = self.state.loss_fn(
                out.velocity,
                rf_state["v_target"],
                latent.mask.bool(),
                latent_substream_ids=substream_ids,
            )
            _attach_rf_diagnostics(
                loss_out,
                rf_state,
                latent.mask.bool(),
                {**dropout_diag, **window_diag},
                compute_rms=_should_log(self.state.global_step + 1, self.log_interval_steps),
            )
            self.accelerator.backward(loss_out.total)
            grad_norm = None
            if self.accelerator.sync_gradients and self.grad_clip_norm is not None and self.grad_clip_norm > 0.0:
                grad_norm = self.accelerator.clip_grad_norm_(
                    [param for param in self.stage2_model.parameters() if param.requires_grad],
                    self.grad_clip_norm,
                )
            self.optimizer.step()
            stepped = self.accelerator.sync_gradients and not _optimizer_step_was_skipped(self.optimizer)
            if stepped and self.scheduler is not None:
                self.scheduler.step()
            if stepped and self.state.ema is not None:
                self.state.ema.update(self._unwrap_stage2_model())
            self.optimizer.zero_grad(set_to_none=True)

        self.last_train_step_synced = bool(self.accelerator.sync_gradients)
        if self.last_train_step_synced:
            self.state.global_step += 1
            elapsed = max(time.perf_counter() - start, 1.0e-9)
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
                    "ema_decay": self.state.ema.decay_at_step() if self.state.ema is not None else 0.0,
                    "ema_num_updates": self.state.ema.num_updates if self.state.ema is not None else 0.0,
                    "dynamic_noise_schedule": str(rf_state["dynamic_noise_schedule"]),
                    "static_time_policy": str(
                        _stage2_train_cfg(self.resolved_config)
                        .get("flow_forcing", {})
                        .get("static_time_policy", "independent_static")
                    ),
                },
            )
            if _should_log(self.state.global_step, self.log_interval_steps):
                self.log_record(record, split="train")
        self._assert_frozen_helpers_have_no_grad()
        return loss_out

    @torch.no_grad()
    def evaluate_velocity(self, eval_loader: Iterable[dict], *, max_batches: int | None = None) -> dict[str, float]:
        unwrapped = self._unwrap_stage2_model()
        context = self.state.ema.apply_to(unwrapped) if self.state.ema is not None else _null_context()
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        with context:
            self.stage2_model.eval()
            for batch_idx, batch in enumerate(eval_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                batch = move_to_device(batch, self.accelerator.device)
                latent, z_white_target = self._encode_target(batch)
                record = self._velocity_eval_record_for_batch(
                    batch,
                    latent,
                    z_white_target,
                    batch_seed=self._eval_seed() + batch_idx,
                )
                self._accumulate_metrics(record, metric_sums, metric_counts)
        if not metric_sums:
            raise ValueError("Velocity evaluation produced no batches.")
        averaged = {key: metric_sums[key] / max(metric_counts[key], 1) for key in sorted(metric_sums)}
        averaged.update({"step": int(self.state.global_step), "epoch": int(self.state.epoch), "split": "eval"})
        self.log_record(averaged, split="eval")
        self._maybe_update_best(averaged)
        return averaged

    @torch.no_grad()
    def _velocity_eval_record_for_batch(
        self,
        batch: dict,
        latent: Any,
        z_white_target: torch.Tensor,
        *,
        batch_seed: int,
    ) -> dict[str, Any]:
        mask = latent.mask.bool()
        rf_state = self._sample_rf_state(batch, z_white_target, latent, deterministic_seed=batch_seed)
        z1 = rf_state["z1"]
        v_target = rf_state["v_target"]
        substream_ids = derive_stage2_latent_substream_ids(
            latent.modality_ids,
            latent.token_type_ids,
            mask=mask,
        )
        target_rms = _valid_rms_float(v_target, mask)
        out = self.stage2_model(
            batch=batch,
            z_white_t=rf_state["z_t"].to(dtype=z_white_target.dtype),
            flow_time=rf_state["flow_time"],
            latent_mask=mask,
            modality_ids=latent.modality_ids,
            token_type_ids=latent.token_type_ids,
            time_ids=latent.time_ids,
            entity_ids=latent.entity_ids,
            time_mask=latent.time_mask,
            return_diagnostics=False,
        )
        loss_out = self.state.loss_fn(out.velocity, v_target, mask, latent_substream_ids=substream_ids)
        cosine = _velocity_cosine(out.velocity, v_target, mask)
        record = self._record_from_loss(loss_out, split="eval", step=self.state.global_step, extra={})
        pred_rms = _valid_rms_float(out.velocity, mask)
        merged: dict[str, Any] = {
            "step": int(self.state.global_step),
            "epoch": int(self.state.epoch),
            "split": "eval",
            "eval/rf_mse": _to_float(loss_out.total),
            f"eval/rf_mse_{rf_state['dynamic_noise_schedule']}": _to_float(loss_out.total),
            "eval/normalized_rf_mse": _to_float(loss_out.diagnostics["normalized_rf_mse"]),
            "eval/velocity_cosine": _to_float(cosine),
            "eval/velocity_pred_rms": pred_rms,
            "eval/velocity_rms_ratio": pred_rms / max(target_rms, 1.0e-8),
            "eval/diag.z_white_target_rms": _valid_rms_float(z1, mask),
            "eval/diag.velocity_target_rms": target_rms,
            "eval/diag.valid_latent_token_count": _to_float(mask.sum().float()),
        }
        for key, value in record.items():
            if key.startswith("eval/diag.velocity_mse_by_latent_substream."):
                merged[key] = value
        for key, values in list(merged.items()):
            if isinstance(values, list):
                merged[key] = sum(values) / max(len(values), 1)
        return merged

    def save_checkpoint(self, tag: str, *, include_resume_state: bool = False) -> Path:
        path = self.checkpoint_dir / f"{tag}.pt"
        if not self.accelerator.is_main_process:
            return path
        ensure_dir(self.checkpoint_dir)
        unwrapped = self._unwrap_stage2_model()
        payload: dict[str, Any] = {
            "format_version": 1,
            "stage2": _stage2_state_dict_without_frozen_text_encoder(unwrapped),
            "ema": self.state.ema.state_dict() if self.state.ema is not None else None,
            "global_step": int(self.state.global_step),
            "epoch": int(self.state.epoch),
            "best_metrics": {
                "eval": {"value": self.state.best_eval_metric, "step": self.state.best_eval_step},
            },
            "config": self.resolved_config,
            "source_artifacts": {
                "stage1_checkpoint": self.source_artifacts.stage1_checkpoint,
                "stage1_checkpoint_sha256": self.source_artifacts.stage1_checkpoint_sha256,
                "whitening_artifact": self.source_artifacts.whitening_artifact,
                "whitening_artifact_sha256": self.source_artifacts.whitening_artifact_sha256,
                "stage1_anchor_override": self.source_artifacts.stage1_anchor_override,
                "normalizers": self.source_artifacts.normalizers,
                "normalizer_hashes": self.source_artifacts.normalizer_hashes,
            },
            "whitening": self.source_artifacts.whitening_payload,
        }
        if include_resume_state:
            payload.update(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                    "accelerator": {
                        "mixed_precision": self.accelerator.mixed_precision,
                        "num_processes": self.accelerator.num_processes,
                        "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
                    },
                }
            )
        self.accelerator.save(payload, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "stage2" not in checkpoint:
            raise KeyError("Stage2 checkpoint missing 'stage2'.")
        self._validate_resume_source_artifacts(checkpoint)
        _load_stage2_state_without_frozen_text_encoder(self._unwrap_stage2_model(), checkpoint["stage2"])
        if checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.state.ema is not None and checkpoint.get("ema") is not None:
            self.state.ema.load_state_dict(checkpoint["ema"])
        self.state.global_step = int(checkpoint.get("global_step", 0))
        self.state.epoch = int(checkpoint.get("epoch", 0))
        best = checkpoint.get("best_metrics", {})
        eval_best = best.get("eval", {}) if isinstance(best, dict) else {}
        self.state.best_eval_metric = _optional_float(eval_best.get("value"))
        self.state.best_eval_step = _optional_int(eval_best.get("step"))

    def _validate_resume_source_artifacts(self, checkpoint: dict[str, Any]) -> None:
        recorded = checkpoint.get("source_artifacts", {})
        checks = {
            "stage1_checkpoint_sha256": self.source_artifacts.stage1_checkpoint_sha256,
            "whitening_artifact_sha256": self.source_artifacts.whitening_artifact_sha256,
        }
        for key, current in checks.items():
            previous = recorded.get(key)
            if previous is not None and str(previous) != str(current):
                raise ValueError(
                    f"Resume checkpoint {key} mismatch:\n"
                    f"  checkpoint: {previous}\n"
                    f"  current:    {current}\n"
                    "Refusing to resume with a different frozen Stage1/whitening target space."
                )
        previous_anchor = recorded.get("stage1_anchor_override")
        current_anchor = self.source_artifacts.stage1_anchor_override
        if previous_anchor != current_anchor:
            raise ValueError(
                "Resume checkpoint stage1_anchor_override mismatch:\n"
                f"  checkpoint: {previous_anchor}\n"
                f"  current:    {current_anchor}\n"
                "Refusing to resume with a different frozen Stage1 anchor target space."
            )
        previous_normalizers = recorded.get("normalizer_hashes", {})
        for key, current in self.source_artifacts.normalizer_hashes.items():
            previous = previous_normalizers.get(key) if isinstance(previous_normalizers, dict) else None
            if previous is not None and current is not None and str(previous) != str(current):
                raise ValueError(
                    f"Resume checkpoint normalizer hash mismatch for {key!r}:\n"
                    f"  checkpoint: {previous}\n"
                    f"  current:    {current}"
                )

    def log_record(self, record: dict, *, split: str) -> None:
        if self.accelerator.is_main_process:
            log_path = {
                "train": self.train_log_path,
                "eval": self.eval_log_path,
            }[split]
            append_jsonl(record, log_path)
        if self.accelerator.is_main_process and self.accelerator.trackers:
            self.accelerator.log(
                {key: value for key, value in record.items() if isinstance(value, (int, float))},
                step=record["step"],
            )

    def _encode_target(self, batch: dict) -> tuple[Any, torch.Tensor]:
        _stage2_window_starts(batch)
        with torch.no_grad():
            latent = encode_stage1_source_latent(
                self.state.stage1_model,
                batch,
                return_diagnostics=False,
                validate=False,
            )
            z_white = self.state.whitening.whiten(latent.z_rec.float(), latent.mask.bool()).float()
            z_white = zero_invalid_tokens(z_white, latent.mask.bool())
        return latent, z_white

    def _sample_rf_state(
        self,
        batch: dict,
        z_white_target: torch.Tensor,
        latent: Any,
        *,
        deterministic_seed: int | None = None,
    ) -> dict[str, Any]:
        mask = latent.mask.bool()
        generator = _torch_generator(z_white_target.device, deterministic_seed) if deterministic_seed is not None else None
        z1 = zero_invalid_tokens(z_white_target.float(), mask)
        z0 = torch.randn(z1.shape, device=z1.device, dtype=torch.float32, generator=generator)
        z0 = zero_invalid_tokens(z0, mask)
        flow_time = build_stage2_training_flow_time_from_config(
            self.resolved_config,
            latent_mask=mask,
            token_type_ids=latent.token_type_ids.to(device=z1.device),
            time_ids=latent.time_ids.to(device=z1.device),
            time_mask=latent.time_mask.to(device=z1.device),
            generator=generator,
            rollout_progress=_stage2_rollout_progress(batch, device=z1.device),
        )
        t_view = flow_time.t_latent_token[..., None].to(device=z1.device, dtype=torch.float32)
        z_t = (1.0 - t_view) * z0 + t_view * z1
        z_t = zero_invalid_tokens(z_t, mask)
        v_target = zero_invalid_tokens(z1 - z0, mask)
        block_size = int(
            _stage2_train_cfg(self.resolved_config).get("flow_forcing", {}).get("block_size_anchors", 8)
        )
        schedule = str(
            _stage2_train_cfg(self.resolved_config)
            .get("flow_forcing", {})
            .get("dynamic_noise_schedule", "random_all_blocks")
        )
        return {
            "z0": z0,
            "z1": z1,
            "z_t": z_t,
            "v_target": v_target,
            "flow_time": flow_time,
            "flow_block_size_anchors": block_size,
            "dynamic_noise_schedule": schedule,
        }

    def _record_from_loss(self, loss_out: Stage2RFLossOutput, *, split: str, step: int, extra: dict) -> dict[str, Any]:
        record: dict[str, Any] = {"step": int(step), "epoch": int(self.state.epoch), "split": split}
        record[f"{split}/rf_mse"] = _to_float(loss_out.total)
        if "normalized_rf_mse" in loss_out.diagnostics:
            record[f"{split}/normalized_rf_mse"] = _to_float(loss_out.diagnostics["normalized_rf_mse"])
        for key, value in loss_out.components.items():
            record[f"{split}/loss.{key}"] = _to_float(value)
        for key, value in loss_out.diagnostics.items():
            record[f"{split}/diag.{key}"] = _to_float(value)
        for key, value in extra.items():
            record[f"{split}/{key}"] = value if isinstance(value, str) else _to_float(value)
        return record

    def _accumulate_metrics(self, record: dict[str, Any], sums: dict[str, float], counts: dict[str, int]) -> None:
        for key, value in record.items():
            if key == "split" or not isinstance(value, (int, float)):
                continue
            gathered = self.accelerator.gather_for_metrics(
                torch.tensor([float(value)], device=self.accelerator.device)
            )
            sums[key] = sums.get(key, 0.0) + float(gathered.float().sum().item())
            counts[key] = counts.get(key, 0) + int(gathered.numel())

    def _maybe_update_best(self, record: dict[str, float]) -> None:
        value = record.get(self.best_checkpoint_metric)
        if value is None:
            return
        current = float(value)
        if _metric_improved(current, self.state.best_eval_metric, self.best_checkpoint_mode):
            self.state.best_eval_metric = current
            self.state.best_eval_step = int(self.state.global_step)
            self.save_checkpoint("best_val", include_resume_state=False)

    def _condition_dropout_cfg(self) -> dict[str, float]:
        return _stage2_train_cfg(self.resolved_config).get("condition_dropout", {})

    def _eval_seed(self) -> int:
        return int(_stage2_eval_cfg(self.resolved_config).get("velocity", {}).get("fixed_noise_seed", 12345))

    def _unwrap_stage2_model(self) -> Stage2TriStreamMoTModel:
        return self.accelerator.unwrap_model(self.stage2_model)

    def _assert_frozen_helpers_have_no_grad(self) -> None:
        for name, module in [("stage1", self.state.stage1_model), ("whitening", self.state.whitening)]:
            for param in module.parameters():
                if param.grad is not None:
                    raise AssertionError(f"Frozen {name} helper received gradients during Stage2 RF training.")


def run_stage2_rectified_flow_training(
    *,
    data_config_path: str | Path,
    train_config_path: str | Path,
    source_run: str | Path,
    source_ckpt: str = "best_val.pt",
    whitening_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    data_config, train_config, resolved_config = load_stage2_rectified_flow_training_config(
        data_config_path,
        train_config_path,
        overrides=overrides,
    )
    train_cfg = _stage2_train_cfg(resolved_config)
    stage1_anchor_override = apply_stage1_anchor_override(
        resolved_config,
        anchor_mode=train_cfg.get("stage1_anchor_mode"),
        anchor_stride=train_cfg.get("stage1_anchor_stride"),
        anchor_grid_origin=train_cfg.get("stage1_anchor_grid_origin"),
    )
    train_cfg["stage1_anchor_override"] = stage1_anchor_override or None
    _enforce_stage2_data_contract(resolved_config)
    train_cfg["source_run"] = str(source_run)
    train_cfg["source_ckpt"] = str(source_ckpt)
    train_cfg["whitening"] = str(whitening_path)
    _validate_resume_and_init_stage2_weights_are_exclusive(train_cfg)
    source_path = Path(source_run) / "checkpoints" / str(source_ckpt)
    _validate_source_files(source_path=source_path, whitening_path=Path(whitening_path))

    wandb_cfg = train_cfg.get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", True)) and wandb_cfg.get("mode", "online") != "disabled"
    _set_current_cuda_device_from_local_rank_env()
    accelerator = Accelerator(
        mixed_precision=str(train_cfg.get("accelerate", {}).get("mixed_precision", "bf16")),
        gradient_accumulation_steps=int(train_cfg.get("accelerate", {}).get("gradient_accumulation_steps", 2)),
        log_with="wandb" if wandb_enabled else None,
        project_dir=str(_stage2_run_dir_from_config(train_cfg)),
    )
    _set_current_cuda_device_for_distributed_barriers(accelerator)
    run_dir = _stage2_run_dir_from_config(train_cfg)
    resume = train_cfg.get("resume")
    if accelerator.is_main_process:
        prepare_run_directory(run_dir, resume=resume)
        _write_stage2_run_metadata(
            run_dir,
            data_config_path,
            train_config_path,
            data_config,
            train_config,
            resolved_config,
            source_path=source_path,
            whitening_path=Path(whitening_path),
            stage1_anchor_override=stage1_anchor_override or None,
            init_stage2_weights_path=train_cfg.get("init_stage2_weights"),
        )
    accelerator.wait_for_everyone()

    normalizer, structured_normalizer, normalizer_paths = load_or_compute_stage1_normalizers(
        resolved_config,
        accelerator=accelerator,
    )
    if structured_normalizer is None:
        raise ValueError("Stage2 RF training requires the structured normalizer shared with Stage1 data.")
    train_loader, eval_loader = build_stage1_phase1_dataloaders(
        _with_stage2_debug_limits_mirrored(resolved_config),
        normalizer=normalizer,
        structured_normalizer=structured_normalizer,
    )
    if accelerator.is_main_process:
        _write_stage2_data_filter_diagnostics(run_dir, train_loader, eval_loader)

    source_model_config = load_stage1_source_config(source_run, fallback_config=resolved_config)
    if stage1_anchor_override:
        apply_stage1_anchor_override(
            source_model_config,
            anchor_mode=train_cfg.get("stage1_anchor_mode"),
            anchor_stride=train_cfg.get("stage1_anchor_stride"),
            anchor_grid_origin=train_cfg.get("stage1_anchor_grid_origin"),
        )
    stage1_model = build_stage1_source_autoencoder_from_config(source_model_config)
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    load_stage1_source_component_state_dicts(stage1_model, source_payload["components"])
    _freeze_module(stage1_model)
    whitening = load_stage1_latent_whitening_transform(whitening_path, map_location="cpu")
    _freeze_module(whitening)
    _validate_whitening_matches_stage2(whitening, resolved_config)
    if stage1_anchor_override:
        _validate_whitening_anchor_override(Path(whitening_path), stage1_anchor_override)
    else:
        _validate_source_whitening_provenance(source_payload, whitening_path)

    stage2_model = build_stage2_tri_stream_mot_model_from_config(resolved_config)
    init_stage2_weights = train_cfg.get("init_stage2_weights")
    if init_stage2_weights:
        report = init_stage2_weights_from_checkpoint(stage2_model, init_stage2_weights)
        train_cfg["init_stage2_weights_report"] = report
        train_cfg["init_stage2_weights_sha256"] = _sha256_path(Path(init_stage2_weights))
        if accelerator.is_main_process:
            save_yaml(resolved_config, run_dir / "resolved_config.yaml")
    stage1_model.to(accelerator.device)
    whitening.to(accelerator.device)
    stage2_model.to(accelerator.device)
    loss_fn = Stage2RectifiedFlowLoss()
    optimizer = build_stage2_optimizer_from_config(resolved_config, stage2_model)
    scheduler = build_stage2_scheduler_from_config(resolved_config, optimizer)
    ema = build_ema_from_config(resolved_config, stage2_model)
    stage2_model, optimizer, train_loader, eval_loader = accelerator.prepare(
        stage2_model,
        optimizer,
        train_loader,
        eval_loader,
    )
    state = Stage2RectifiedFlowTrainState(
        accelerator=accelerator,
        stage2_model=stage2_model,
        stage1_model=stage1_model,
        whitening=whitening,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
    )
    source_artifacts = build_source_artifacts(
        source_path=source_path,
        whitening_path=Path(whitening_path),
        whitening=whitening,
        stage1_anchor_override=stage1_anchor_override or None,
        normalizer_paths=normalizer_paths,
    )
    trainer = Stage2RectifiedFlowTrainer(
        state=state,
        run_dir=run_dir,
        resolved_config=resolved_config,
        source_artifacts=source_artifacts,
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        log_interval_steps=train_cfg.get("log_interval_steps", 50),
        checkpoint_interval_steps=train_cfg.get("checkpoint_interval_steps", 5000),
        velocity_eval_interval_steps=_stage2_eval_cfg(resolved_config).get("velocity", {}).get("interval_steps", 1000),
        best_checkpoint_metric=train_cfg.get("best_checkpoint_metric", "eval/rf_mse"),
        best_checkpoint_mode=train_cfg.get("best_checkpoint_mode", "min"),
    )
    if resume:
        trainer.load_checkpoint(resume)
    if wandb_enabled:
        accelerator.init_trackers(
            project_name=str(wandb_cfg.get("project", "combination-stage2")),
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
    if train_cfg.get("validate_first_batch", True):
        first_batch = next(iter(train_loader))
        validate_stage2_first_batch(first_batch, window_size=resolved_config.get("data", {}).get("window_size"))

    max_steps = int(train_cfg.get("max_steps", 100000))
    eval_cfg = _stage2_eval_cfg(resolved_config)
    max_eval_batches = _optional_int(eval_cfg.get("velocity", {}).get("max_batches", 20))
    while state.global_step < max_steps:
        for batch in train_loader:
            if state.global_step >= max_steps:
                break
            trainer.train_step(batch)
            if trainer.last_train_step_synced and state.global_step > 0:
                if state.global_step % trainer.velocity_eval_interval_steps == 0:
                    trainer.evaluate_velocity(eval_loader, max_batches=max_eval_batches)
                if state.global_step % trainer.checkpoint_interval_steps == 0:
                    trainer.save_checkpoint("latest", include_resume_state=True)
        state.epoch += 1

    trainer.evaluate_velocity(eval_loader, max_batches=max_eval_batches)
    trainer.save_checkpoint("latest", include_resume_state=True)
    trainer.save_checkpoint("final", include_resume_state=False)
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return run_dir


def load_stage2_rectified_flow_training_config(
    data_config_path: str | Path,
    train_config_path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> tuple[dict, dict, dict]:
    data_config = load_yaml(data_config_path)
    train_config = load_yaml(train_config_path)
    resolved = deep_merge(data_config, train_config)
    if overrides:
        resolved = apply_stage2_rectified_flow_cli_overrides(resolved, overrides)
    return data_config, train_config, resolved


def apply_stage2_rectified_flow_cli_overrides(config: dict, overrides: dict[str, Any]) -> dict:
    out = copy.deepcopy(config)
    mapping = {
        "run_name": ("training", "stage2", "run_name"),
        "output_root": ("training", "stage2", "output_root"),
        "seed": ("seed",),
        "resume": ("training", "stage2", "resume"),
        "init_stage2_weights": ("training", "stage2", "init_stage2_weights"),
        "max_steps": ("training", "stage2", "max_steps"),
        "batch_size": ("loader", "batch_size"),
        "num_workers": ("loader", "num_workers"),
        "lr": ("training", "stage2", "optimizer", "lr"),
        "weight_decay": ("training", "stage2", "optimizer", "weight_decay"),
        "checkpoint_interval_steps": ("training", "stage2", "checkpoint_interval_steps"),
        "log_interval_steps": ("training", "stage2", "log_interval_steps"),
        "debug_num_train_batches": ("training", "stage2", "debug_num_train_batches"),
        "debug_num_eval_batches": ("training", "stage2", "debug_num_eval_batches"),
        "mixed_precision": ("training", "stage2", "accelerate", "mixed_precision"),
        "gradient_accumulation_steps": ("training", "stage2", "accelerate", "gradient_accumulation_steps"),
        "velocity_eval_interval_steps": ("evaluation", "stage2", "velocity", "interval_steps"),
        "wandb_project": ("training", "stage2", "wandb", "project"),
        "wandb_entity": ("training", "stage2", "wandb", "entity"),
        "wandb_mode": ("training", "stage2", "wandb", "mode"),
        "wandb_id": ("training", "stage2", "wandb", "id"),
        "wandb_resume": ("training", "stage2", "wandb", "resume"),
        "wandb_tags": ("training", "stage2", "wandb", "tags"),
        "validate_first_batch": ("training", "stage2", "validate_first_batch"),
        "stage1_anchor_mode": ("training", "stage2", "stage1_anchor_mode"),
        "stage1_anchor_stride": ("training", "stage2", "stage1_anchor_stride"),
        "stage1_anchor_grid_origin": ("training", "stage2", "stage1_anchor_grid_origin"),
    }
    for key, path in mapping.items():
        if key in overrides and overrides[key] is not None:
            _set_nested(out, path, overrides[key])
    return out


def _validate_resume_and_init_stage2_weights_are_exclusive(train_cfg: dict[str, Any]) -> None:
    if train_cfg.get("resume") and train_cfg.get("init_stage2_weights"):
        raise ValueError("--resume and --init-stage2-weights are mutually exclusive.")


def build_stage2_optimizer_from_config(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = _stage2_train_cfg(config).get("optimizer", {})
    opt_type = str(opt_cfg.get("type", "adamw")).lower()
    if opt_type != "adamw":
        raise ValueError("Stage2 RF v1 only supports AdamW.")
    groups = _adamw_parameter_groups(model, weight_decay=float(opt_cfg.get("weight_decay", 1.0e-2)))
    betas = opt_cfg.get("betas", [0.9, 0.95])
    return torch.optim.AdamW(
        groups,
        lr=float(opt_cfg.get("lr", 1.0e-4)),
        betas=(float(betas[0]), float(betas[1])),
    )


def build_stage2_scheduler_from_config(
    config: dict,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    train_cfg = _stage2_train_cfg(config)
    sched_cfg = train_cfg.get("scheduler")
    if not sched_cfg:
        return None
    sched_type = str(sched_cfg.get("type", "none")).lower()
    if sched_type in {"none", "off"}:
        return None
    supported = {"warmup_constant", "warmup_constant_cooldown", "constant"}
    if sched_type not in supported:
        raise ValueError(
            "Stage2 FF1 supports scheduler.type='warmup_constant', "
            "'warmup_constant_cooldown', or 'constant'."
        )
    warmup_steps = int(sched_cfg.get("warmup_steps", 2000))
    base_lr = float(train_cfg.get("optimizer", {}).get("lr", 1.0e-4))
    cooldown_start_step = int(sched_cfg.get("cooldown_start_step", train_cfg.get("max_steps", warmup_steps)))
    cooldown_steps = int(sched_cfg.get("cooldown_steps", 0))
    cooldown_type = str(sched_cfg.get("cooldown_type", "cosine")).lower()
    final_lr = float(sched_cfg.get("final_lr", sched_cfg.get("min_lr", base_lr)))
    if warmup_steps < 0:
        raise ValueError(f"scheduler.warmup_steps must be non-negative, got {warmup_steps}.")
    if cooldown_steps < 0:
        raise ValueError(f"scheduler.cooldown_steps must be non-negative, got {cooldown_steps}.")
    if final_lr < 0.0:
        raise ValueError(f"scheduler.final_lr must be non-negative, got {final_lr}.")
    if sched_type == "warmup_constant_cooldown" and cooldown_type != "cosine":
        raise ValueError(f"Stage2 cooldown_type must be 'cosine', got {cooldown_type!r}.")
    final_ratio = final_lr / base_lr if base_lr > 0.0 else 1.0

    def lr_lambda(step: int) -> float:
        if sched_type == "constant":
            return 1.0
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1.0e-8)
        if sched_type == "warmup_constant_cooldown" and cooldown_steps > 0 and step >= cooldown_start_step:
            progress = float(step - cooldown_start_step) / float(max(cooldown_steps - 1, 1))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_ratio + (1.0 - final_ratio) * cosine
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.phase1_metadata = {  # type: ignore[attr-defined]
        "type": sched_type,
        "base_lr": base_lr,
        "min_lr": final_lr if sched_type == "warmup_constant_cooldown" else base_lr,
        "max_steps": cooldown_start_step + cooldown_steps if sched_type == "warmup_constant_cooldown" else warmup_steps,
        "warmup_steps": warmup_steps,
        "cooldown_start_step": cooldown_start_step if sched_type == "warmup_constant_cooldown" else None,
        "cooldown_steps": cooldown_steps if sched_type == "warmup_constant_cooldown" else None,
        "cooldown_type": cooldown_type if sched_type == "warmup_constant_cooldown" else None,
    }
    return scheduler


def build_ema_from_config(config: dict, model: torch.nn.Module) -> TrainableParameterEMA | None:
    ema_cfg = _stage2_train_cfg(config).get("ema", {})
    if not bool(ema_cfg.get("enabled", True)):
        return None
    return TrainableParameterEMA(
        model,
        decay=float(ema_cfg.get("decay", 0.9999)),
        warmup=bool(ema_cfg.get("warmup", True)),
    )


def sample_categorical_condition_dropout(
    batch: dict,
    cfg: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    probs = {
        "both": float(cfg.get("both", 0.70)),
        "text_only": float(cfg.get("text_only", 0.10)),
        "geometry_only": float(cfg.get("geometry_only", 0.10)),
        "none": float(cfg.get("none", 0.10)),
    }
    bad_probs = {key: value for key, value in probs.items() if value < 0.0}
    if bad_probs:
        raise ValueError(f"condition_dropout probabilities must be non-negative, got {bad_probs}.")
    total = sum(probs.values())
    if total <= 0.0:
        raise ValueError("condition_dropout probabilities must have positive sum.")
    batch_size = _infer_batch_size_from_meta(batch)
    sample = torch.rand(batch_size, device=device)
    p_both = probs["both"] / total
    p_text_only = probs["text_only"] / total
    p_geometry_only = probs["geometry_only"] / total
    force_text = sample >= (p_both + p_text_only)
    force_geometry = (sample >= p_both) & (sample < p_both + p_text_only)
    force_geometry = force_geometry | (sample >= p_both + p_text_only + p_geometry_only)
    intrinsic_text = _intrinsic_null_text(batch, device)
    intrinsic_geometry = _intrinsic_null_geometry(batch, device)
    diagnostics = {
        "condition_dropout.forced_text_fraction": force_text.float().mean(),
        "condition_dropout.forced_geometry_fraction": force_geometry.float().mean(),
        "condition_dropout.effective_text_null_fraction": (force_text | intrinsic_text).float().mean(),
        "condition_dropout.effective_geometry_null_fraction": (force_geometry | intrinsic_geometry).float().mean(),
        "condition_dropout.intrinsic_text_null_fraction": intrinsic_text.float().mean(),
        "condition_dropout.intrinsic_geometry_null_fraction": intrinsic_geometry.float().mean(),
    }
    return force_text, force_geometry, diagnostics


def validate_stage2_first_batch(batch: dict, *, window_size: int | None = None) -> None:
    validate_structured_batch(batch, window_size=window_size)
    meta = batch.get("meta")
    if not isinstance(meta, dict) or "text" not in meta:
        raise KeyError("Stage2 batch requires batch['meta']['text'].")
    _stage2_window_starts(batch)
    _stage2_window_ends(batch)
    _stage2_sequence_lengths(batch)
    geometry = batch.get("geometry")
    if not isinstance(geometry, dict):
        raise KeyError("Stage2 batch requires batch['geometry'].")
    for key in ("object_bps", "object_bps_mask", "has_object_bps"):
        if key not in geometry:
            raise KeyError(f"Stage2 batch geometry missing {key!r}.")
    bps = geometry["object_bps"]
    if not isinstance(bps, torch.Tensor) or bps.ndim not in {3, 4} or bps.shape[-1] != 3:
        raise ValueError("batch['geometry']['object_bps'] must have shape [B,N,3] or [B,1,N,3].")
    has_bps = geometry["has_object_bps"]
    if not isinstance(has_bps, torch.Tensor):
        raise ValueError("batch['geometry']['has_object_bps'] must be a tensor.")
    if not has_bps.to(dtype=torch.bool).all():
        raise ValueError("Stage2 v1 requires object geometry for every training sample.")


def _stage2_window_starts(batch: dict, *, device: torch.device | None = None) -> torch.Tensor:
    meta = batch.get("meta")
    if not isinstance(meta, dict) or "start" not in meta:
        raise KeyError("Stage2 batch requires batch['meta']['start'] for Stage1 absolute-grid anchors.")
    starts = meta["start"]
    if not isinstance(starts, torch.Tensor):
        starts = torch.as_tensor(starts, dtype=torch.long)
    if starts.ndim != 1:
        raise ValueError(f"batch['meta']['start'] must have shape [B], got {tuple(starts.shape)}.")
    starts = starts.to(device=device, dtype=torch.long) if device is not None else starts.to(dtype=torch.long)
    return starts


def _stage2_window_ends(batch: dict, *, device: torch.device | None = None) -> torch.Tensor:
    meta = batch.get("meta")
    if not isinstance(meta, dict) or "end" not in meta:
        raise KeyError("Stage2 batch requires batch['meta']['end'] for rollout progress.")
    ends = meta["end"]
    if not isinstance(ends, torch.Tensor):
        ends = torch.as_tensor(ends, dtype=torch.long)
    if ends.ndim != 1:
        raise ValueError(f"batch['meta']['end'] must have shape [B], got {tuple(ends.shape)}.")
    ends = ends.to(device=device, dtype=torch.long) if device is not None else ends.to(dtype=torch.long)
    return ends


def _stage2_sequence_lengths(batch: dict, *, device: torch.device | None = None) -> torch.Tensor:
    meta = batch.get("meta")
    if not isinstance(meta, dict) or "sequence_length" not in meta:
        raise KeyError("Stage2 batch requires batch['meta']['sequence_length'] for rollout progress.")
    lengths = meta["sequence_length"]
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.as_tensor(lengths, dtype=torch.long)
    if lengths.ndim != 1:
        raise ValueError(f"batch['meta']['sequence_length'] must have shape [B], got {tuple(lengths.shape)}.")
    lengths = lengths.to(device=device, dtype=torch.long) if device is not None else lengths.to(dtype=torch.long)
    if (lengths <= 0).any():
        raise ValueError("batch['meta']['sequence_length'] must be positive.")
    return lengths


def _stage2_rollout_progress(batch: dict, *, device: torch.device) -> torch.Tensor:
    starts = _stage2_window_starts(batch, device=device).float()
    ends = _stage2_window_ends(batch, device=device).float()
    lengths = _stage2_sequence_lengths(batch, device=device).float()
    center = 0.5 * (starts + ends - 1.0)
    denom = (lengths - 1.0).clamp_min(1.0)
    return (center / denom).clamp(0.0, 1.0)


def _window_start_diagnostics(batch: dict, *, device: torch.device) -> dict[str, torch.Tensor]:
    starts = _stage2_window_starts(batch, device=device)
    progress = _stage2_rollout_progress(batch, device=device)
    starts_f = starts.float()
    return {
        "window_start_min": starts_f.min(),
        "window_start_max": starts_f.max(),
        "window_start_nonzero_fraction": (starts > 0).float().mean(),
        "rollout_progress_mean": progress.mean(),
        "rollout_progress_min": progress.min(),
        "rollout_progress_max": progress.max(),
    }


def build_source_artifacts(
    *,
    source_path: Path,
    whitening_path: Path,
    whitening: Stage1LatentWhiteningTransform,
    stage1_anchor_override: dict[str, Any] | None = None,
    normalizer_paths: dict[str, str | None],
) -> Stage2SourceArtifacts:
    normalizer_hashes = {
        key: (_sha256_path(Path(path)) if path is not None else None) for key, path in normalizer_paths.items()
    }
    return Stage2SourceArtifacts(
        stage1_checkpoint=str(source_path),
        stage1_checkpoint_sha256=_sha256_path(source_path),
        whitening_artifact=str(whitening_path),
        whitening_artifact_sha256=_sha256_path(whitening_path),
        whitening_payload=whitening.to_stats().to_payload(metadata={"source": str(whitening_path)}),
        stage1_anchor_override=copy.deepcopy(stage1_anchor_override),
        normalizers=normalizer_paths,
        normalizer_hashes=normalizer_hashes,
    )


def _attach_rf_diagnostics(
    loss_out: Stage2RFLossOutput,
    rf_state: dict[str, Any],
    latent_mask: torch.Tensor,
    extra: dict[str, torch.Tensor],
    *,
    compute_rms: bool = True,
) -> None:
    mask = latent_mask.bool()
    if compute_rms:
        loss_out.diagnostics["z_white_target_rms"] = torch.tensor(
            _valid_rms_float(rf_state["z1"], mask),
            device=rf_state["z1"].device,
        )
        loss_out.diagnostics["z_t_rms"] = torch.tensor(
            _valid_rms_float(rf_state["z_t"], mask),
            device=rf_state["z_t"].device,
        )
    flow_time = rf_state["flow_time"]
    for key, value in flow_time_log_diagnostics(
        flow_time,
        block_size_anchors=int(rf_state.get("flow_block_size_anchors", 8)),
        dynamic_noise_schedule=str(rf_state.get("dynamic_noise_schedule", "random_all_blocks")),
    ).items():
        if isinstance(value, torch.Tensor):
            loss_out.diagnostics[key] = value.to(device=rf_state["z_t"].device)
    for key, value in extra.items():
        loss_out.diagnostics[key] = value


def _valid_rms_float(value: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool().unsqueeze(-1).expand_as(value)
    if not valid.any():
        return 0.0
    return float(value.float()[valid].pow(2).mean().sqrt().detach().cpu().item())


def _velocity_cosine(velocity_pred: torch.Tensor, velocity_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if velocity_pred.shape != velocity_target.shape:
        raise ValueError("velocity_pred and velocity_target shapes must match.")
    valid = mask.to(device=velocity_pred.device, dtype=torch.bool).unsqueeze(-1).expand_as(velocity_pred)
    if not valid.any():
        return velocity_pred.new_zeros(())
    pred = velocity_pred.float()[valid]
    target = velocity_target.float()[valid]
    denom = pred.norm() * target.norm()
    if denom <= 1.0e-12:
        return velocity_pred.new_zeros(())
    return (pred * target).sum() / denom.clamp_min(1.0e-12)


def _stage2_state_dict_without_frozen_text_encoder(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    filter_text_encoder = _text_encoder_is_frozen(model)
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not (filter_text_encoder and _is_text_encoder_state_key(key))
    }


def _stage2_state_dict_without_frozen_clip(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return _stage2_state_dict_without_frozen_text_encoder(model)


def _load_stage2_state_without_frozen_text_encoder(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected Stage2 checkpoint keys: {unexpected[:20]}")
    allow_missing_text_encoder = _text_encoder_is_frozen(model)
    bad_missing = [
        key
        for key in result.missing_keys
        if not (allow_missing_text_encoder and _is_text_encoder_state_key(key))
    ]
    if bad_missing:
        raise RuntimeError(f"Stage2 checkpoint is missing non-text-encoder keys: {bad_missing[:20]}")


def _load_stage2_state_without_frozen_clip(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    _load_stage2_state_without_frozen_text_encoder(model, state)


def init_stage2_weights_from_checkpoint(model: torch.nn.Module, path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("stage2") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise KeyError(f"Stage2 init checkpoint {path} is missing a 'stage2' state dict.")
    current = model.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}
    unexpected: list[str] = []
    for key, value in state.items():
        if _is_text_encoder_state_key(key):
            continue
        if key not in current:
            unexpected.append(key)
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped[key] = f"shape {tuple(value.shape)} -> {tuple(current[key].shape)}"
            continue
        loadable[key] = value.detach().cpu()
    missing = [key for key in current if not _is_text_encoder_state_key(key) and key not in loadable and key not in skipped]
    loaded_count = len(loadable)
    eligible_count = loaded_count + len(skipped) + len(missing)
    loaded_ratio = float(loaded_count) / float(max(eligible_count, 1))
    if loaded_count == 0 or loaded_ratio < 0.10:
        raise RuntimeError(
            f"Catastrophic Stage2 init load from {path}: loaded_count={loaded_count}, "
            f"eligible_count={eligible_count}, loaded_ratio={loaded_ratio:.4f}."
        )
    updated = dict(current)
    updated.update(loadable)
    model.load_state_dict(updated, strict=True)
    return {
        "path": str(path),
        "sha256": _sha256_path(Path(path)),
        "loaded_count": loaded_count,
        "eligible_count": eligible_count,
        "loaded_ratio": loaded_ratio,
        "skipped_count": len(skipped),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "skipped": dict(list(skipped.items())[:50]),
        "missing": missing[:50],
        "unexpected": unexpected[:50],
    }


def _is_clip_state_key(key: str) -> bool:
    return _is_text_encoder_state_key(key)


def _is_text_encoder_state_key(key: str) -> bool:
    return (
        key.startswith("text_encoder.text_model.")
        or key.startswith("text_encoder.clip_model.")
        or ".text_model." in key
        or ".clip_model." in key
    )


def _clip_model_is_frozen(model: torch.nn.Module) -> bool:
    return _text_encoder_is_frozen(model)


def _text_encoder_is_frozen(model: torch.nn.Module) -> bool:
    text_encoder = getattr(model, "text_encoder", None)
    text_model = getattr(text_encoder, "text_model", None)
    if text_model is None:
        text_model = getattr(text_encoder, "clip_model", None)
    return text_model is not None and not any(param.requires_grad for param in text_model.parameters())


def _freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()


def _keep_frozen_clip_eval(model: Stage2TriStreamMoTModel) -> None:
    _keep_frozen_text_encoder_eval(model)


def _keep_frozen_text_encoder_eval(model: Stage2TriStreamMoTModel) -> None:
    text_encoder = getattr(model, "text_encoder", None)
    text_model = getattr(text_encoder, "text_model", None)
    if text_model is None:
        text_model = getattr(text_encoder, "clip_model", None)
    if text_model is not None and not any(param.requires_grad for param in text_model.parameters()):
        text_model.eval()


def _validate_whitening_matches_stage2(whitening: Stage1LatentWhiteningTransform, config: dict) -> None:
    latent_dim = int(config.get("model", {}).get("stage2", {}).get("latent_dim", 16))
    if int(whitening.z_dim) != latent_dim:
        raise ValueError(f"Whitening z_dim={whitening.z_dim} does not match Stage2 latent_dim={latent_dim}.")


def _validate_source_whitening_provenance(source_payload: dict[str, Any], whitening_path: str | Path) -> None:
    recorded = source_payload.get("whitening_artifact")
    if recorded is None:
        return
    if Path(str(recorded)).resolve() != Path(whitening_path).resolve():
        raise ValueError(
            "Stage2 whitening artifact does not match source checkpoint provenance:\n"
            f"  source checkpoint records: {recorded}\n"
            f"  requested whitening:       {whitening_path}"
        )


def _validate_whitening_anchor_override(whitening_path: str | Path, anchor_override: dict[str, Any]) -> None:
    payload = torch.load(whitening_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Whitening artifact payload must be a dict, got {type(payload).__name__}.")
    metadata = payload.get("metadata")
    recorded = None
    if isinstance(metadata, dict):
        raw_override = metadata.get("anchor_override")
        if isinstance(raw_override, dict):
            recorded = raw_override.get("after")
    requested = anchor_override.get("after") if isinstance(anchor_override, dict) else None
    if recorded != requested:
        raise ValueError(
            "Stage2 Stage1-anchor override does not match whitening artifact metadata:\n"
            f"  whitening records: {recorded}\n"
            f"  Stage2 requested:  {requested}"
        )


def _validate_source_files(*, source_path: Path, whitening_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Stage1 source checkpoint is missing: {source_path}")
    if not whitening_path.exists():
        raise FileNotFoundError(f"Whitening artifact is missing: {whitening_path}")


def _enforce_stage2_data_contract(config: dict) -> None:
    data_cfg = config.setdefault("data", {})
    data_cfg["load_text"] = True
    data_cfg["load_object_bps"] = True
    data_cfg["return_raw_motion"] = True
    data_cfg["return_structured"] = True
    structured = data_cfg.setdefault("structured", {})
    structured["enabled"] = True
    structured["require_object_geometry"] = True


def _with_stage2_debug_limits_mirrored(config: dict) -> dict:
    out = copy.deepcopy(config)
    stage2_cfg = _stage2_train_cfg(out)
    stage1_cfg = out.setdefault("training", {}).setdefault("stage1", {})
    for key in ("debug_num_train_batches", "debug_num_eval_batches"):
        if stage2_cfg.get(key) is not None:
            stage1_cfg[key] = stage2_cfg[key]
    return out


def _stage2_train_cfg(config: dict) -> dict:
    return config.setdefault("training", {}).setdefault("stage2", {})


def _stage2_eval_cfg(config: dict) -> dict:
    return config.setdefault("evaluation", {}).setdefault("stage2", {})


def _stage2_run_dir_from_config(train_cfg: dict) -> Path:
    output_root = Path(train_cfg.get("output_root", "runs/training/stage2"))
    run_name = train_cfg.get("run_name")
    if not run_name:
        raise ValueError("training.stage2.run_name is required.")
    return output_root / str(run_name)


def _set_nested(config: dict, path: tuple[str, ...], value: Any) -> None:
    node = config
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _write_stage2_run_metadata(
    run_dir: Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    data_config: dict,
    train_config: dict,
    resolved_config: dict,
    *,
    source_path: Path,
    whitening_path: Path,
    stage1_anchor_override: dict[str, Any] | None = None,
    init_stage2_weights_path: str | Path | None = None,
) -> None:
    save_yaml(data_config, run_dir / "data_config.yaml")
    save_yaml(train_config, run_dir / "train_config.yaml")
    save_yaml(resolved_config, run_dir / "resolved_config.yaml")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    save_json(
        {
            "data_config_path": str(data_config_path),
            "train_config_path": str(train_config_path),
            "stage1_source_checkpoint": str(source_path),
            "stage1_source_checkpoint_sha256": _sha256_path(source_path) if source_path.exists() else None,
            "whitening_artifact": str(whitening_path),
            "whitening_artifact_sha256": _sha256_path(whitening_path) if whitening_path.exists() else None,
            "stage1_anchor_override": stage1_anchor_override,
            "init_stage2_weights": str(init_stage2_weights_path) if init_stage2_weights_path else None,
            "init_stage2_weights_sha256": (
                _sha256_path(Path(init_stage2_weights_path))
                if init_stage2_weights_path and Path(init_stage2_weights_path).exists()
                else None
            ),
            "git_commit": _git_output(["rev-parse", "HEAD"]),
            "git_status_short": _git_output(["status", "--short"]),
        },
        run_dir / "git.json",
    )


def _write_stage2_data_filter_diagnostics(run_dir: Path, train_loader: Any, eval_loader: Any) -> None:
    save_json(
        {
            "train": _dataset_text_filter_diagnostics(getattr(train_loader, "dataset", None)),
            "eval": _dataset_text_filter_diagnostics(getattr(eval_loader, "dataset", None)),
        },
        run_dir / "data_filter_diagnostics.json",
    )


def _dataset_text_filter_diagnostics(dataset: Any) -> dict[str, Any] | None:
    if dataset is None:
        return None
    diagnostics = getattr(dataset, "text_filter_diagnostics", None)
    if diagnostics is not None:
        return dict(diagnostics)
    inner = getattr(dataset, "dataset", None)
    if inner is None:
        return None
    inner_diagnostics = _dataset_text_filter_diagnostics(inner)
    if inner_diagnostics is None:
        return None
    if hasattr(dataset, "indices"):
        inner_diagnostics["debug_subset_count"] = len(dataset)
    return inner_diagnostics


def _sha256_path(path: Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = child.relative_to(path).as_posix().encode("utf-8")
            h.update(rel)
            h.update(b"\0")
            _hash_file_into(child, h)
    else:
        _hash_file_into(path, h)
    return h.hexdigest()


def _hash_file_into(path: Path, digest: "hashlib._Hash") -> None:
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)


def _git_output(args: list[str]) -> str | None:
    import subprocess

    try:
        proc = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _torch_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(int(seed))
    return generator


@contextmanager
def _autocast_disabled(device: torch.device):
    if device.type in {"cuda", "cpu"}:
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    else:
        yield


def _infer_batch_size_from_meta(batch: dict) -> int:
    text = batch.get("meta", {}).get("text")
    if text is not None:
        return len(text)
    return _batch_size(batch)


def _intrinsic_null_text(batch: dict, device: torch.device) -> torch.Tensor:
    texts = list(batch.get("meta", {}).get("text", []))
    return torch.tensor([text is None for text in texts], dtype=torch.bool, device=device)


def _intrinsic_null_geometry(batch: dict, device: torch.device) -> torch.Tensor:
    geometry = batch.get("geometry", {})
    size = _infer_batch_size_from_meta(batch)
    value = geometry.get("has_object_bps") if isinstance(geometry, dict) else None
    if value is None:
        has = torch.ones(size, dtype=torch.bool, device=device)
    else:
        has = value.to(device=device, dtype=torch.bool).reshape(-1)
        if has.numel() != size:
            raise ValueError(f"has_object_bps must have {size} values, got {has.numel()}.")
    return ~has


def _metric_improved(current: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError("metric mode must be 'min' or 'max'.")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


@contextmanager
def _null_context():
    yield
