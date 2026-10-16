from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Subset

from hoi_latent.config.io import load_yaml, save_yaml
from hoi_latent.data.collate import collate_stage1_motion962
from hoi_latent.data.datasets import InterActStage1Dataset
from hoi_latent.data.datasets.interact_stage1 import build_interact_stage1_index
from hoi_latent.data.splits import InterActSplitAssigner
from hoi_latent.data.transforms import FeatureNormalizer, Stage1StructuredNormalizer
from hoi_latent.data.transforms.normalization import compute_streaming_normalizer
from hoi_latent.data.transforms.structured_normalization import compute_streaming_structured_normalizer
from hoi_latent.losses.stage1 import Stage1LossOutput, Stage1Phase1WarmupLoss
from hoi_latent.models.stage1 import Stage1AutoencoderModule, build_stage1_autoencoder_from_config
from hoi_latent.utils.filesystem import ensure_dir
from hoi_latent.utils.json_io import save_json


class Stage1Phase1WarmupModule(torch.nn.Module):
    """DDP/Accelerate-safe Stage1 reconstruction forward wrapper."""

    def __init__(self, stage1: Stage1AutoencoderModule, loss_fn: Stage1Phase1WarmupLoss) -> None:
        super().__init__()
        self.stage1 = stage1
        self.loss_fn = loss_fn

    def component_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        return self.stage1.component_state_dicts()

    def forward(
        self,
        batch: dict,
        *,
        global_step: int,
        return_diagnostics: bool = False,
        validate: bool = False,
    ) -> Stage1LossOutput:
        frame_mask = infer_frame_mask(batch)
        target_num_frames = int(frame_mask.shape[1])
        tokenizer_outputs = self.stage1.tokenizer(
            batch,
            return_aux=return_diagnostics,
            return_concat=False,
            validate=validate,
        )
        latent = self.stage1.encoder(
            tokenizer_outputs,
            batch=batch,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
        prediction = self.stage1.decoder(
            latent,
            target_num_frames=target_num_frames,
            target_frame_mask=frame_mask,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
        return self.loss_fn(prediction, latent, batch, global_step=global_step)


@dataclass
class Stage1Phase1TrainState:
    accelerator: Accelerator
    model: torch.nn.Module
    loss_fn: Stage1Phase1WarmupLoss
    optimizer: torch.optim.Optimizer
    scheduler: Any | None
    global_step: int
    epoch: int
    best_eval_loss: float | None = None


class Stage1Phase1WarmupTrainer:
    """Accelerate-backed trainer for Stage1 reconstruction warm-up."""

    def __init__(
        self,
        *,
        state: Stage1Phase1TrainState,
        run_dir: str | Path,
        resolved_config: dict,
        normalizer_paths: dict[str, str | None] | None = None,
        grad_clip_norm: float | None = 1.0,
        log_interval_steps: int = 50,
        checkpoint_interval_steps: int = 5000,
        save_periodic_step_checkpoints: bool = False,
        eval_interval_steps: int = 1000,
        return_diagnostics_interval_steps: int = 1000,
        best_checkpoint_metric: str = "eval/loss.total",
        best_checkpoint_mode: str = "min",
        train_log_name: str = "train.jsonl",
        eval_log_name: str = "eval.jsonl",
        manual_gradient_sync: bool = False,
    ) -> None:
        self.state = state
        self.accelerator = state.accelerator
        self.run_dir = Path(run_dir)
        self.resolved_config = resolved_config
        self.normalizer_paths = normalizer_paths or {}
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.log_interval_steps = int(log_interval_steps)
        self.checkpoint_interval_steps = int(checkpoint_interval_steps)
        self.save_periodic_step_checkpoints = bool(save_periodic_step_checkpoints)
        self.eval_interval_steps = int(eval_interval_steps)
        self.return_diagnostics_interval_steps = int(return_diagnostics_interval_steps)
        self.best_checkpoint_metric = str(best_checkpoint_metric)
        self.best_checkpoint_mode = str(best_checkpoint_mode)
        if self.best_checkpoint_mode not in {"min", "max"}:
            raise ValueError("best_checkpoint_mode must be 'min' or 'max'.")
        self.last_train_step_synced = False
        self.manual_gradient_sync = bool(manual_gradient_sync)
        self._cpu_gradient_sync_group = None
        self.train_log_path = self.run_dir / "logs" / train_log_name
        self.eval_log_path = self.run_dir / "logs" / eval_log_name
        self.checkpoint_dir = self.run_dir / "checkpoints"

    @property
    def model(self) -> torch.nn.Module:
        return self.state.model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.state.optimizer

    @property
    def scheduler(self) -> Any | None:
        return self.state.scheduler

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
            if self.manual_gradient_sync and self.accelerator.sync_gradients:
                self._sync_trainable_gradients()
            if self.accelerator.sync_gradients and self.grad_clip_norm is not None and self.grad_clip_norm > 0.0:
                grad_norm = self.accelerator.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip_norm,
                )
            self.optimizer.step()
            if self.accelerator.sync_gradients and self.scheduler is not None and not _optimizer_step_was_skipped(self.optimizer):
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
                    "manual_gradient_sync": float(self.manual_gradient_sync and self.accelerator.num_processes > 1),
                    "manual_gradient_sync_backend": (
                        "gloo_cpu" if self.manual_gradient_sync and self.accelerator.num_processes > 1 else "none"
                    ),
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

    @torch.no_grad()
    def eval_step(self, batch: dict) -> Stage1LossOutput:
        self.model.eval()
        batch = move_to_device(batch, self.accelerator.device)
        with self.accelerator.autocast():
            return self.model(
                batch,
                global_step=self.state.global_step,
                return_diagnostics=False,
                validate=False,
            )

    @torch.no_grad()
    def evaluate(self, eval_loader: Iterable[dict], *, max_batches: int | None = None) -> dict[str, float]:
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        for batch_idx, batch in enumerate(eval_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            loss_out = self.eval_step(batch)
            record = self._record_from_loss(loss_out, split="eval", step=self.state.global_step, extra={})
            for key, value in record.items():
                if key in {"split"}:
                    continue
                if isinstance(value, (int, float)):
                    gathered = self.accelerator.gather_for_metrics(
                        torch.tensor([float(value)], device=self.accelerator.device)
                    )
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(gathered.float().sum().item())
                    metric_counts[key] = metric_counts.get(key, 0) + int(gathered.numel())
        if not metric_sums:
            raise ValueError("Evaluation produced no batches.")
        averaged = {key: metric_sums[key] / max(metric_counts[key], 1) for key in sorted(metric_sums)}
        averaged["step"] = int(self.state.global_step)
        averaged["epoch"] = int(self.state.epoch)
        averaged["split"] = "eval"
        self.log_record(averaged, split="eval")
        self._maybe_update_best(averaged)
        return averaged

    def save_checkpoint(self, tag: str) -> Path:
        if not self.accelerator.is_main_process:
            return self.checkpoint_dir / f"{tag}.pt"
        ensure_dir(self.checkpoint_dir)
        path = self.checkpoint_dir / f"{tag}.pt"
        unwrapped = self._unwrap_model()
        payload = {
            "components": unwrapped.component_state_dicts(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "accelerator": {
                "mixed_precision": self.accelerator.mixed_precision,
                "num_processes": self.accelerator.num_processes,
                "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
            },
            "global_step": int(self.state.global_step),
            "epoch": int(self.state.epoch),
            "best_eval_loss": self.state.best_eval_loss,
            "config": self.resolved_config,
            "normalizers": self.normalizer_paths,
        }
        self.accelerator.save(payload, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        if "components" not in checkpoint:
            raise KeyError("Checkpoint missing 'components'.")
        load_component_state_dicts(self._unwrap_stage1_model(), checkpoint["components"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.state.global_step = int(checkpoint.get("global_step", 0))
        self.state.epoch = int(checkpoint.get("epoch", 0))
        best = checkpoint.get("best_eval_loss")
        self.state.best_eval_loss = None if best is None else float(best)

    def log_record(self, record: dict, *, split: str) -> None:
        if self.accelerator.is_main_process:
            log_path = self.train_log_path if split == "train" else self.eval_log_path
            append_jsonl(record, log_path)
        if self.accelerator.is_main_process and self.accelerator.trackers:
            self.accelerator.log({k: v for k, v in record.items() if isinstance(v, (int, float))}, step=record["step"])

    def assert_frozen_modules_have_no_grad(self) -> None:
        return None

    def _record_from_loss(self, loss_out: Stage1LossOutput, *, split: str, step: int, extra: dict) -> dict:
        record: dict[str, Any] = {
            "step": int(step),
            "epoch": int(self.state.epoch),
            "split": split,
        }
        for key, value in loss_out.components.items():
            record[f"{split}/loss.{key}"] = _to_float(value)
        for key, value in loss_out.diagnostics.items():
            prefix = "metric" if key == "world_marker_mpjpe_m" else "diag"
            record[f"{split}/{prefix}.{key}"] = _to_float(value)
        for key, value in extra.items():
            if isinstance(value, str):
                record[f"{split}/{key}"] = value
            else:
                record[f"{split}/{key}"] = _to_float(value)
        return record

    def _maybe_update_best(self, eval_record: dict[str, float]) -> None:
        metric_value = eval_record.get(self.best_checkpoint_metric)
        if metric_value is None:
            return
        current = float(metric_value)
        best = self.state.best_eval_loss
        improved = best is None
        if best is not None and self.best_checkpoint_mode == "min":
            improved = current < best
        elif best is not None and self.best_checkpoint_mode == "max":
            improved = current > best
        if improved:
            self.state.best_eval_loss = current
            self.save_checkpoint("best_val")

    def _unwrap_model(self) -> Stage1Phase1WarmupModule:
        return self.accelerator.unwrap_model(self.model)

    def _unwrap_stage1_model(self) -> Stage1AutoencoderModule:
        unwrapped = self._unwrap_model()
        return unwrapped.stage1 if isinstance(unwrapped, Stage1Phase1WarmupModule) else unwrapped


def run_stage1_phase1_warmup_training(
    *,
    data_config_path: str | Path,
    train_config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    data_config, train_config, resolved_config = load_phase1_training_config(
        data_config_path,
        train_config_path,
        overrides=overrides,
    )
    train_cfg = _stage1_train_cfg(resolved_config)
    wandb_cfg = train_cfg.get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", True)) and wandb_cfg.get("mode", "online") != "disabled"
    _set_current_cuda_device_from_local_rank_env()
    accelerate_cfg = train_cfg.get("accelerate", {})
    manual_gradient_sync = bool(accelerate_cfg.get("manual_gradient_sync", False))
    resume = train_cfg.get("resume")
    init_components = train_cfg.get("init_components")
    if resume and init_components:
        raise ValueError("training.stage1.resume and training.stage1.init_components are mutually exclusive.")
    if manual_gradient_sync and not (resume or init_components):
        raise ValueError(
            "training.stage1.accelerate.manual_gradient_sync currently requires resume or init_components so every "
            "rank loads identical model weights without DDP's initial parameter broadcast."
        )
    if accelerate_cfg.get("skip_initial_broadcast") and not (resume or init_components):
        raise ValueError(
            "training.stage1.accelerate.skip_initial_broadcast requires resume or init_components so every rank "
            "loads identical model weights after DDP wrapping."
        )
    if not manual_gradient_sync:
        _maybe_patch_ddp_initial_broadcast(accelerate_cfg)
    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=bool(accelerate_cfg.get("broadcast_buffers", True)),
        bucket_cap_mb=int(accelerate_cfg.get("bucket_cap_mb", 25)),
        find_unused_parameters=bool(accelerate_cfg.get("find_unused_parameters", False)),
    )
    accelerator = Accelerator(
        mixed_precision=str(accelerate_cfg.get("mixed_precision", "bf16")),
        gradient_accumulation_steps=int(accelerate_cfg.get("gradient_accumulation_steps", 1)),
        log_with="wandb" if wandb_enabled else None,
        project_dir=str(_run_dir_from_config(train_cfg)),
        kwargs_handlers=[ddp_kwargs],
    )
    _set_current_cuda_device_for_distributed_barriers(accelerator)
    run_dir = _run_dir_from_config(train_cfg)
    if accelerator.is_main_process:
        prepare_run_directory(run_dir, resume=resume)
        _write_run_metadata(run_dir, data_config_path, train_config_path, data_config, train_config, resolved_config)
    accelerator.wait_for_everyone()

    normalizer, structured_normalizer, normalizer_paths = load_or_compute_stage1_normalizers(
        resolved_config,
        accelerator=accelerator,
    )
    train_loader, eval_loader = build_stage1_phase1_dataloaders(
        resolved_config,
        normalizer=normalizer,
        structured_normalizer=structured_normalizer,
    )
    _assert_world_marker_normalizer_contract(resolved_config, structured_normalizer)

    stage1_model = build_stage1_autoencoder_from_config(resolved_config)
    freeze_inactive_phase1_parameters(stage1_model, resolved_config)
    loss_fn = build_phase1_loss_from_config(resolved_config, structured_normalizer=structured_normalizer)
    model = Stage1Phase1WarmupModule(stage1_model, loss_fn)
    if manual_gradient_sync:
        model.to(accelerator.device)
    optimizer = build_optimizer_from_config(resolved_config, model)
    scheduler = build_scheduler_from_config(resolved_config, optimizer)
    # Keep the scheduler unwrapped. AcceleratedScheduler advances num_processes
    # times per optimizer step when split_batches=False, while this schedule is
    # defined in global optimizer-update steps.
    if manual_gradient_sync:
        # Do not wrap the Stage1 model in DDP. Some CUDA/NCCL stacks fail in DDP's
        # initial parameter broadcast for this model; sharding dataloaders and
        # manually all-reducing gradients over a Gloo CPU group matches the
        # decoder post-jitter path already validated on the server.
        optimizer, train_loader, eval_loader = accelerator.prepare(optimizer, train_loader, eval_loader)
    else:
        model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)
    state = Stage1Phase1TrainState(
        accelerator=accelerator,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=0,
        epoch=0,
    )
    trainer = Stage1Phase1WarmupTrainer(
        state=state,
        run_dir=run_dir,
        resolved_config=resolved_config,
        normalizer_paths=normalizer_paths,
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        log_interval_steps=train_cfg.get("log_interval_steps", 50),
        checkpoint_interval_steps=train_cfg.get("checkpoint_interval_steps", 5000),
        save_periodic_step_checkpoints=train_cfg.get("save_periodic_step_checkpoints", False),
        eval_interval_steps=train_cfg.get("eval_interval_steps", 1000),
        return_diagnostics_interval_steps=train_cfg.get("phase1_warmup", {}).get(
            "return_diagnostics_interval_steps", 1000
        ),
        best_checkpoint_metric=train_cfg.get("best_checkpoint_metric", "eval/loss.total"),
        best_checkpoint_mode=train_cfg.get("best_checkpoint_mode", "min"),
        manual_gradient_sync=manual_gradient_sync,
    )
    if init_components:
        # Every rank reads the same checkpoint locally, avoiding NCCL's initial
        # parameter broadcast over the loaded 700MB Stage-1 weights.
        load_component_checkpoint(accelerator.unwrap_model(model), init_components)
        accelerator.wait_for_everyone()
    if resume:
        trainer.load_checkpoint(resume)

    if wandb_enabled:
        wandb_init = {
            "name": train_cfg.get("run_name"),
            "entity": wandb_cfg.get("entity"),
            "mode": wandb_cfg.get("mode", "online"),
            "tags": wandb_cfg.get("tags", []),
        }
        if wandb_cfg.get("id"):
            wandb_init["id"] = wandb_cfg.get("id")
            wandb_init["resume"] = wandb_cfg.get("resume", "allow")
        elif wandb_cfg.get("resume"):
            wandb_init["resume"] = wandb_cfg.get("resume")
        accelerator.init_trackers(
            project_name=str(wandb_cfg.get("project", "combination-stage1")),
            config=resolved_config,
            init_kwargs={"wandb": wandb_init},
        )

    if train_cfg.get("phase1_warmup", {}).get("validate_first_batch", True):
        first_batch = next(iter(train_loader))
        validate_structured_batch(first_batch, window_size=resolved_config.get("data", {}).get("window_size"))

    max_steps = int(train_cfg.get("max_steps", 100000))
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
    if manual_gradient_sync and dist.is_available() and dist.is_initialized():
        if trainer._cpu_gradient_sync_group is not None:
            dist.destroy_process_group(trainer._cpu_gradient_sync_group)
            trainer._cpu_gradient_sync_group = None
        dist.destroy_process_group()
    accelerator.end_training()
    return run_dir


def load_phase1_training_config(
    data_config_path: str | Path,
    train_config_path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> tuple[dict, dict, dict]:
    data_config = load_yaml(data_config_path)
    train_config = load_yaml(train_config_path)
    resolved = deep_merge(data_config, train_config)
    if overrides:
        resolved = apply_cli_overrides(resolved, overrides)
    return data_config, train_config, resolved


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_cli_overrides(config: dict, overrides: dict[str, Any]) -> dict:
    out = copy.deepcopy(config)
    mapping = {
        "run_name": ("training", "stage1", "run_name"),
        "output_root": ("training", "stage1", "output_root"),
        "seed": ("seed",),
        "resume": ("training", "stage1", "resume"),
        "init_components": ("training", "stage1", "init_components"),
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
        "validate_first_batch": ("training", "stage1", "phase1_warmup", "validate_first_batch"),
    }
    for key, path in mapping.items():
        if key not in overrides or overrides[key] is None:
            continue
        _set_nested(out, path, overrides[key])
    return out


def load_or_compute_stage1_normalizers(
    config: dict,
    *,
    accelerator: Accelerator | None = None,
) -> tuple[FeatureNormalizer | None, Stage1StructuredNormalizer | None, dict[str, str | None]]:
    accelerator = accelerator or Accelerator(cpu=True)
    data_cfg = config.get("data", {})
    norm_cfg = config.get("normalization", {})
    structured_cfg = data_cfg.get("structured", {})
    structured_norm_cfg = structured_cfg.get("normalization", {})
    records = None

    if accelerator.is_main_process:
        records = _assigned_records(config)
        _ensure_feature_normalizer_cache(config, records)
        _ensure_structured_normalizer_cache(config, records)
    accelerator.wait_for_everyone()

    normalizer = _load_feature_normalizer_from_config(norm_cfg)
    structured_normalizer = _load_structured_normalizer_from_config(structured_norm_cfg)
    return normalizer, structured_normalizer, {
        "legacy_motion962": str(norm_cfg.get("cache_path")) if norm_cfg.get("mode", "off") != "off" else None,
        "structured": (
            str(structured_norm_cfg.get("cache_path"))
            if structured_norm_cfg.get("enabled", structured_cfg.get("enabled", False))
            else None
        ),
    }


def build_stage1_phase1_dataloaders(
    config: dict,
    *,
    normalizer: FeatureNormalizer | None,
    structured_normalizer: Stage1StructuredNormalizer | None,
) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    split_cfg = config.get("split", {})
    loader_cfg = config.get("loader", {})
    train_cfg = _stage1_train_cfg(config)
    batch_size = int(loader_cfg.get("batch_size", 1))
    num_workers = int(loader_cfg.get("num_workers", 0))
    train_dataset = _make_dataset(
        config,
        split="train",
        window_mode=data_cfg.get("train_window_mode", "random"),
        normalizer=normalizer,
        structured_normalizer=structured_normalizer,
    )
    eval_dataset = _make_dataset(
        config,
        split="val",
        window_mode=data_cfg.get("eval_window_mode", "center"),
        normalizer=normalizer,
        structured_normalizer=structured_normalizer,
    )
    if train_cfg.get("debug_num_train_batches") is not None:
        train_dataset = _limit_dataset(train_dataset, int(train_cfg["debug_num_train_batches"]) * batch_size)
    if train_cfg.get("debug_num_eval_batches") is not None:
        eval_dataset = _limit_dataset(eval_dataset, int(train_cfg["debug_num_eval_batches"]) * batch_size)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", False)),
        "persistent_workers": bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0,
        "collate_fn": collate_stage1_motion962,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(eval_dataset, shuffle=False, **common),
    )


def build_phase1_loss_from_config(
    config: dict,
    *,
    structured_normalizer: Stage1StructuredNormalizer | None,
) -> Stage1Phase1WarmupLoss:
    loss_cfg = config.get("loss", {}).get("stage1", {}).get("phase1_warmup", {})
    kwargs = {key: value for key, value in loss_cfg.items() if value is not None}
    kwargs["normalizer"] = structured_normalizer
    return Stage1Phase1WarmupLoss(**kwargs)


def build_optimizer_from_config(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = _stage1_train_cfg(config).get("optimizer", {})
    opt_type = str(opt_cfg.get("type", "adamw")).lower()
    if opt_type != "adamw":
        raise ValueError("Phase1 warm-up v1 only supports AdamW.")
    weight_decay = float(opt_cfg.get("weight_decay", 1.0e-4))
    param_groups = _adamw_parameter_groups(model, weight_decay=weight_decay)
    betas = opt_cfg.get("betas", [0.9, 0.95])
    return torch.optim.AdamW(
        param_groups,
        lr=float(opt_cfg.get("lr", 1.0e-4)),
        betas=(float(betas[0]), float(betas[1])),
    )


def build_scheduler_from_config(
    config: dict,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    train_cfg = _stage1_train_cfg(config)
    sched_cfg = train_cfg.get("scheduler")
    if not sched_cfg:
        return None
    sched_type = str(sched_cfg.get("type", "none")).lower()
    if sched_type in {"none", "off"}:
        return None
    if sched_type not in {"warmup_cosine", "warmup_constant"}:
        raise ValueError("Phase1 warm-up v1 supports scheduler.type='warmup_cosine' or 'warmup_constant'.")
    max_steps = int(train_cfg.get("max_steps", 100000))
    warmup_steps = int(sched_cfg.get("warmup_steps", 2000))
    base_lr = float(train_cfg.get("optimizer", {}).get("lr", 1.0e-4))
    min_lr = base_lr if sched_type == "warmup_constant" else float(sched_cfg.get("min_lr", 1.0e-5))
    if max_steps <= 0:
        raise ValueError("scheduler requires training.stage1.max_steps > 0.")
    if warmup_steps < 0:
        raise ValueError("scheduler warmup_steps must be >= 0.")
    if min_lr < 0.0 or base_lr <= 0.0:
        raise ValueError("scheduler requires min_lr >= 0 and optimizer.lr > 0.")
    min_lr_ratio = min(min_lr / base_lr, 1.0)

    def lr_lambda(step: int) -> float:
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1.0e-8)
        if sched_type == "warmup_constant":
            return 1.0
        decay_steps = max(max_steps - warmup_steps, 1)
        progress = min(max(float(step - warmup_steps) / float(decay_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.phase1_metadata = {  # type: ignore[attr-defined]
        "type": sched_type,
        "base_lr": base_lr,
        "min_lr": min_lr,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
    }
    return scheduler


def _adamw_parameter_groups(model: torch.nn.Module, *, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for _, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            param_id = id(param)
            if param_id in seen or not param.requires_grad:
                continue
            seen.add(param_id)
            if _no_weight_decay_parameter(module, param_name, param):
                no_decay.append(param)
            else:
                decay.append(param)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _no_weight_decay_parameter(module: torch.nn.Module, param_name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim <= 1:
        return True
    if param_name == "bias":
        return True
    no_decay_modules = (
        torch.nn.Embedding,
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
    )
    return isinstance(module, no_decay_modules)


def freeze_inactive_phase1_parameters(model: Stage1AutoencoderModule, config: dict) -> None:
    """Freeze deterministic config-inactive branches before DDP/optimizer wrapping."""
    stage1_cfg = config.get("model", {}).get("stage1", {})
    tokenizer_cfg = stage1_cfg.get("tokenizers", {})
    object_cfg = tokenizer_cfg.get("object", {})
    object_tokenizer = model.tokenizer.object

    geometry_source = str(object_cfg.get("geometry_source", getattr(object_tokenizer, "geometry_source", "bps")))
    if geometry_source == "bps":
        _freeze_module(object_tokenizer.point_proj)
    elif geometry_source == "points":
        _freeze_module(object_tokenizer.bps_proj)
        _freeze_module(object_tokenizer.bps_index_embedding)
    elif geometry_source != "both":
        raise ValueError(f"Unsupported object geometry_source={geometry_source!r}.")

    num_geometry_slots = int(object_cfg.get("num_geometry_slots", getattr(object_tokenizer, "num_geometry_slots", 4)))
    num_frames = int(config.get("data", {}).get("window_size", stage1_cfg.get("max_seq_len", 128)))
    compressor = model.encoder.token_compressor
    budget = compressor._resolve_budget(num_frames=num_frames, num_geometry_tokens=num_geometry_slots)
    if num_geometry_slots <= budget.object_geometry_tokens:
        _freeze_module(compressor.geometry_compressor)


def _freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()


def validate_structured_batch(batch: dict, *, window_size: int | None = None) -> None:
    for key in ("raw", "model", "geometry", "ids", "masks"):
        if key not in batch:
            raise KeyError(f"Structured batch missing {key!r}.")
    ids = batch["ids"]
    for key in ("marker_ids", "relation_type_ids", "dataset_id", "object_id"):
        if key not in ids:
            raise KeyError(f"Structured batch ids missing {key!r}.")
    frame_mask = infer_frame_mask(batch)
    if frame_mask.ndim != 2:
        raise ValueError(f"frame mask must be [B,T], got {tuple(frame_mask.shape)}.")
    if window_size is not None and int(frame_mask.shape[1]) != int(window_size):
        raise ValueError(f"frame mask T={frame_mask.shape[1]} does not match window_size={window_size}.")
    human_mask = batch["masks"].get("human")
    relation_mask = batch["masks"].get("relation")
    if not isinstance(human_mask, torch.Tensor) or human_mask.ndim != 3 or human_mask.shape[-1] != 77:
        raise ValueError("batch['masks']['human'] must have shape [B,T,77].")
    if not isinstance(relation_mask, torch.Tensor) or relation_mask.ndim != 3 or relation_mask.shape[-1] != 78:
        raise ValueError("batch['masks']['relation'] must have shape [B,T,78].")
    _assert_finite_tree(batch["model"], "batch.model")
    _assert_finite_tree(batch["raw"], "batch.raw")


def infer_frame_mask(batch: dict) -> torch.Tensor:
    masks = batch.get("masks")
    if not isinstance(masks, dict):
        raise KeyError("batch missing masks dictionary.")
    for key in ("frame", "valid", "object"):
        value = masks.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value.to(dtype=torch.bool)
    raise KeyError("batch masks must include frame, valid, or object [B,T] mask.")


def move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    return obj


def append_jsonl(record: dict, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        json.dump(_jsonable(record), f, sort_keys=True)
        f.write("\n")


def load_component_state_dicts(model: torch.nn.Module, components: dict[str, dict[str, torch.Tensor]]) -> None:
    custom_loader = getattr(model, "load_component_state_dicts", None)
    if callable(custom_loader) and not isinstance(model, Stage1Phase1WarmupModule):
        custom_loader(components)
        return
    if isinstance(model, Stage1Phase1WarmupModule):
        model = model.stage1
    for key in ("tokenizer", "encoder", "decoder"):
        if key not in components:
            raise KeyError(f"Checkpoint components missing {key!r}.")
        getattr(model, key).load_state_dict(components[key])


def load_component_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    if "components" not in checkpoint:
        raise KeyError("Checkpoint missing 'components'.")
    load_component_state_dicts(model, checkpoint["components"])


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def scheduler_metrics(
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    if scheduler is None:
        lr = current_lr(optimizer)
        return {
            "lr_base": lr,
            "lr_min": lr,
            "scheduler_step": 0.0,
            "scheduler_warmup_progress": 1.0,
            "scheduler_decay_progress": 0.0,
        }
    metadata = getattr(scheduler, "phase1_metadata", {})
    base_lr = float(metadata.get("base_lr", optimizer.param_groups[0].get("initial_lr", current_lr(optimizer))))
    min_lr = float(metadata.get("min_lr", current_lr(optimizer)))
    max_steps = int(metadata.get("max_steps", 0))
    warmup_steps = int(metadata.get("warmup_steps", 0))
    sched_type = str(metadata.get("type", "none"))
    step = int(getattr(scheduler, "last_epoch", 0))
    warmup_progress = 1.0 if warmup_steps <= 0 else min(max(float(step) / float(warmup_steps), 0.0), 1.0)
    decay_steps = max(max_steps - warmup_steps, 1)
    decay_progress = (
        0.0
        if sched_type == "warmup_constant"
        else min(max(float(step - warmup_steps) / float(decay_steps), 0.0), 1.0)
    )
    return {
        "lr_base": base_lr,
        "lr_min": min_lr,
        "scheduler_step": float(step),
        "scheduler_warmup_progress": warmup_progress,
        "scheduler_decay_progress": decay_progress,
    }


def _optimizer_step_was_skipped(optimizer: torch.optim.Optimizer) -> bool:
    return bool(getattr(optimizer, "step_was_skipped", False))


def prepare_run_directory(run_dir: str | Path, *, resume: str | None = None) -> Path:
    out_dir = Path(run_dir)
    check_run_directory_available(out_dir, resume=resume)
    ensure_dir(out_dir / "logs")
    ensure_dir(out_dir / "checkpoints")
    return out_dir


def check_run_directory_available(run_dir: str | Path, *, resume: str | None = None) -> None:
    out_dir = Path(run_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not resume:
        raise FileExistsError(f"Run directory already exists and is non-empty: {out_dir}")


def _set_current_cuda_device_for_distributed_barriers(accelerator: Accelerator) -> None:
    if accelerator.device.type == "cuda" and torch.cuda.is_available():
        device_index = accelerator.device.index
        if device_index is None:
            device_index = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(device_index)


def _set_current_cuda_device_from_local_rank_env() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    device_index = int(local_rank)
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError(
            f"LOCAL_RANK={device_index} is outside available CUDA device count {torch.cuda.device_count()}."
    )
    torch.cuda.set_device(device_index)


def _maybe_patch_ddp_initial_broadcast(accelerate_cfg: dict) -> None:
    skip_initial_broadcast = bool(accelerate_cfg.get("skip_initial_broadcast", False))
    bucket_mb = accelerate_cfg.get("initial_broadcast_bucket_mb")
    if not skip_initial_broadcast and bucket_mb is None:
        _restore_ddp_initial_broadcast_if_patched()
        return
    bucket_bytes = None if bucket_mb is None else int(float(bucket_mb) * 1024 * 1024)
    if not skip_initial_broadcast and (bucket_bytes is None or bucket_bytes <= 0):
        return
    import torch.nn.parallel.distributed as ddp_distributed

    original = getattr(ddp_distributed, "_hoi_latent_original_sync_module_states", None)
    if original is None:
        original = ddp_distributed._sync_module_states
        ddp_distributed._hoi_latent_original_sync_module_states = original

    def capped_sync_module_states(*args, **kwargs):
        if skip_initial_broadcast:
            return None
        if "broadcast_bucket_size" in kwargs:
            kwargs["broadcast_bucket_size"] = min(int(kwargs["broadcast_bucket_size"]), int(bucket_bytes))
            return original(*args, **kwargs)
        if len(args) >= 3:
            args_list = list(args)
            args_list[2] = min(int(args_list[2]), int(bucket_bytes))
            return original(*args_list, **kwargs)
        return original(*args, **kwargs)

    ddp_distributed._sync_module_states = capped_sync_module_states


def _restore_ddp_initial_broadcast_if_patched() -> None:
    import torch.nn.parallel.distributed as ddp_distributed

    original = getattr(ddp_distributed, "_hoi_latent_original_sync_module_states", None)
    if original is not None:
        ddp_distributed._sync_module_states = original


def _write_run_metadata(
    run_dir: Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    data_config: dict,
    train_config: dict,
    resolved_config: dict,
) -> None:
    save_yaml(data_config, run_dir / "data_config.yaml")
    save_yaml(train_config, run_dir / "train_config.yaml")
    save_yaml(resolved_config, run_dir / "resolved_config.yaml")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    save_json(
        {
            "data_config_path": str(data_config_path),
            "train_config_path": str(train_config_path),
            "git_commit": _git_output(["rev-parse", "HEAD"]),
            "git_status_short": _git_output(["status", "--short"]),
        },
        run_dir / "git.json",
    )


def _ensure_feature_normalizer_cache(config: dict, records: list) -> None:
    norm_cfg = config.get("normalization", {})
    mode = norm_cfg.get("mode", "off")
    if mode == "off":
        return
    cache_path = norm_cfg.get("cache_path")
    if not cache_path:
        raise ValueError("normalization.cache_path is required when normalization is enabled.")
    cache_dir = Path(cache_path)
    if (cache_dir / "mean.npy").exists() and (cache_dir / "std.npy").exists():
        return
    if mode != "compute_if_missing":
        raise FileNotFoundError(f"Feature normalizer cache missing: {cache_dir}")
    normalizer, stats = compute_streaming_normalizer(
        records,
        eps=float(norm_cfg.get("eps", 1.0e-6)),
        datasets=config.get("data", {}).get("datasets") or [],
        split="train",
    )
    normalizer.save(cache_dir, stats=stats)


def _ensure_structured_normalizer_cache(config: dict, records: list) -> None:
    data_cfg = config.get("data", {})
    structured_cfg = data_cfg.get("structured", {})
    norm_cfg = structured_cfg.get("normalization", {})
    enabled = bool(norm_cfg.get("enabled", structured_cfg.get("enabled", False)))
    if not enabled or norm_cfg.get("mode", "off") == "off":
        return
    cache_path = norm_cfg.get("cache_path")
    if not cache_path:
        raise ValueError("data.structured.normalization.cache_path is required when structured normalization is enabled.")
    cache_dir = Path(cache_path)
    if (cache_dir / "stats.json").exists():
        return
    if norm_cfg.get("mode") != "compute_if_missing":
        raise FileNotFoundError(f"Structured normalizer cache missing: {cache_dir}")
    normalizer, metadata = compute_streaming_structured_normalizer(
        records,
        eps=float(norm_cfg.get("eps", config.get("normalization", {}).get("eps", 1.0e-6))),
        split="train",
        root_method=structured_cfg.get("root_method", "marker_mean"),
    )
    normalizer.save(cache_dir, metadata=metadata)


def _load_feature_normalizer_from_config(norm_cfg: dict) -> FeatureNormalizer | None:
    if norm_cfg.get("mode", "off") == "off":
        return None
    return FeatureNormalizer.load(norm_cfg["cache_path"])


def _load_structured_normalizer_from_config(norm_cfg: dict) -> Stage1StructuredNormalizer | None:
    if not norm_cfg or norm_cfg.get("mode", "off") == "off":
        return None
    return Stage1StructuredNormalizer.load(norm_cfg["cache_path"])


def _assigned_records(config: dict) -> list:
    data_cfg = config["data"]
    split_cfg = config.get("split", {})
    records, _ = build_interact_stage1_index(
        data_root=data_cfg["root"],
        datasets=data_cfg["datasets"],
        exclude_datasets=data_cfg.get("exclude_datasets") or [],
        require_motion=True,
        load_object_metadata=True,
        validate_motion_shape=True,
    )
    assigner = InterActSplitAssigner(
        test_seq_file=split_cfg.get("test_seq_file"),
        use_official_test_seq=split_cfg.get("use_official_test_seq", True),
        val_ratio_from_train=split_cfg.get("val_ratio_from_train", 0.05),
        val_hash_seed=split_cfg.get("val_hash_seed", 42),
    )
    assigned, _ = assigner.assign(records)
    return assigned


def _make_dataset(
    config: dict,
    *,
    split: str,
    window_mode: str,
    normalizer: FeatureNormalizer | None,
    structured_normalizer: Stage1StructuredNormalizer | None,
) -> InterActStage1Dataset:
    data_cfg = config["data"]
    structured_cfg = data_cfg.get("structured", {})
    return InterActStage1Dataset(
        data_root=data_cfg["root"],
        datasets=data_cfg["datasets"],
        split=split,
        split_config=config.get("split", {}),
        normalizer=normalizer,
        window_size=data_cfg.get("window_size"),
        window_mode=window_mode,
        representation=data_cfg.get("representation", "interact_motion_962"),
        load_text=data_cfg.get("load_text", True),
        load_object_points=True,
        load_object_bps=True,
        return_raw_motion=True,
        return_structured=True,
        structured_normalizer=structured_normalizer,
        require_object_geometry=True,
        text_filter=data_cfg.get("text_filter"),
        exclude_datasets=data_cfg.get("exclude_datasets") or [],
        object_geometry_cache_size=structured_cfg.get("object_geometry_cache_size", 256),
        marker_part_map_path=structured_cfg.get("marker_part_map_path"),
        root_method=structured_cfg.get("root_method", "marker_mean"),
        seed=config.get("seed", 42),
    )


def _limit_dataset(dataset, max_items: int):
    return Subset(dataset, list(range(min(len(dataset), max_items))))


def _stage1_train_cfg(config: dict) -> dict:
    return config.setdefault("training", {}).setdefault("stage1", {})


def _run_dir_from_config(train_cfg: dict) -> Path:
    output_root = Path(train_cfg.get("output_root", "runs/training/stage1"))
    run_name = train_cfg.get("run_name")
    if not run_name:
        raise ValueError("training.stage1.run_name is required.")
    return output_root / str(run_name)


def _set_nested(config: dict, path: tuple[str, ...], value: Any) -> None:
    node = config
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _assert_world_marker_normalizer_contract(config: dict, structured_normalizer: Stage1StructuredNormalizer | None) -> None:
    loss_cfg = config.get("loss", {}).get("stage1", {}).get("phase1_warmup", {})
    if float(loss_cfg.get("lambda_world_marker", 1.0)) > 0.0 and structured_normalizer is None:
        raise ValueError("lambda_world_marker > 0 requires a structured normalizer shared by dataset and loss.")


def _diagnostics_enabled(step: int, interval: int) -> bool:
    return interval > 0 and step > 0 and step % interval == 0


def _should_log(step: int, interval: int) -> bool:
    return interval <= 1 or (step > 0 and step % interval == 0)


def _batch_size(batch: dict) -> int:
    motion = batch.get("motion")
    if isinstance(motion, torch.Tensor):
        return int(motion.shape[0])
    frame = infer_frame_mask(batch)
    return int(frame.shape[0])


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to float.")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return _to_float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _assert_finite_tree(obj: Any, name: str) -> None:
    if isinstance(obj, torch.Tensor):
        if torch.is_floating_point(obj) and not torch.isfinite(obj).all():
            raise ValueError(f"{name} contains non-finite values.")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            _assert_finite_tree(value, f"{name}.{key}")


def _git_output(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()
