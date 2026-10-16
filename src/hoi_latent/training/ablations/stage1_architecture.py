from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from hoi_latent.ablations.stage1_architecture import (
    build_stage1_architecture_ablation_from_config,
    count_parameters,
)
from hoi_latent.losses.stage1 import Stage1LossOutput, Stage1Phase1WarmupLoss
from hoi_latent.training.stage1.phase1_warmup import (
    Stage1Phase1TrainState,
    Stage1Phase1WarmupTrainer,
    _assert_world_marker_normalizer_contract,
    _maybe_patch_ddp_initial_broadcast,
    _run_dir_from_config,
    _set_current_cuda_device_for_distributed_barriers,
    _set_current_cuda_device_from_local_rank_env,
    _stage1_train_cfg,
    build_optimizer_from_config,
    build_phase1_loss_from_config,
    build_scheduler_from_config,
    build_stage1_phase1_dataloaders,
    freeze_inactive_phase1_parameters,
    infer_frame_mask,
    load_component_checkpoint,
    load_or_compute_stage1_normalizers,
    load_phase1_training_config,
    prepare_run_directory,
    validate_structured_batch,
    _write_run_metadata,
)
from hoi_latent.utils.json_io import save_json


class Stage1ArchitectureAblationTrainingModule(torch.nn.Module):
    """DDP-safe training wrapper for Stage1 architecture ablation autoencoders."""

    def __init__(self, stage1: torch.nn.Module, loss_fn: Stage1Phase1WarmupLoss) -> None:
        super().__init__()
        self.stage1 = stage1
        self.loss_fn = loss_fn

    def component_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        if hasattr(self.stage1, "component_state_dicts"):
            return self.stage1.component_state_dicts()
        checkpoint_key = str(getattr(self.stage1, "checkpoint_key", "stage1_architecture_ablation"))
        return {checkpoint_key: self.stage1.state_dict()}

    def load_component_state_dicts(self, components: dict[str, dict[str, torch.Tensor]]) -> None:
        checkpoint_key = str(getattr(self.stage1, "checkpoint_key", "stage1_architecture_ablation"))
        if checkpoint_key in components:
            self.stage1.load_state_dict(components[checkpoint_key])
            return
        if "stage1_architecture_ablation" in components:
            self.stage1.load_state_dict(components["stage1_architecture_ablation"])
            return

        loaded_any = False
        for attr in ("tokenizer", "encoder", "decoder"):
            module = getattr(self.stage1, attr, None)
            if module is None:
                continue
            key = str(getattr(module, "checkpoint_key", attr))
            if key not in components:
                continue
            module.load_state_dict(components[key])
            loaded_any = True
        if not loaded_any:
            expected = [checkpoint_key, "stage1_architecture_ablation", "tokenizer", "encoder", "decoder"]
            raise KeyError(f"Checkpoint components do not match ablation wrapper. Expected one of {expected}.")

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
        out = self.stage1(
            batch,
            target_num_frames=target_num_frames,
            target_frame_mask=frame_mask,
            return_diagnostics=return_diagnostics,
            validate=validate,
        )
        return self.loss_fn(out.prediction, out.latent, batch, global_step=global_step)


def run_stage1_architecture_ablation_training(
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
    set_seed(int(resolved_config.get("seed", 42)), device_specific=False)
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
        write_stage1_architecture_ablation_wandb_document(
            run_dir,
            train_cfg=train_cfg,
            wandb_enabled=wandb_enabled,
            variant_name=_variant_name_from_config(resolved_config),
        )
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

    stage1_model = build_stage1_architecture_ablation_from_config(resolved_config)
    freeze_inactive_stage1_architecture_ablation_parameters(stage1_model, resolved_config)
    if accelerator.is_main_process:
        save_json(
            build_stage1_architecture_ablation_metadata(stage1_model, resolved_config),
            run_dir / "architecture_ablation_metadata.json",
        )

    loss_fn = build_phase1_loss_from_config(resolved_config, structured_normalizer=structured_normalizer)
    model = Stage1ArchitectureAblationTrainingModule(stage1_model, loss_fn)
    if manual_gradient_sync:
        model.to(accelerator.device)
    optimizer = build_optimizer_from_config(resolved_config, model)
    scheduler = build_scheduler_from_config(resolved_config, optimizer)
    if manual_gradient_sync:
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
            project_name=str(wandb_cfg.get("project", "combination-stage1-ablations")),
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


def freeze_inactive_stage1_architecture_ablation_parameters(model: torch.nn.Module, config: dict) -> None:
    if hasattr(model, "tokenizer") and hasattr(model, "encoder"):
        freeze_inactive_phase1_parameters(model, config)


def build_stage1_architecture_ablation_metadata(model: torch.nn.Module, config: dict) -> dict[str, Any]:
    resolved = getattr(model, "stage1_architecture_resolved_config", config)
    variant = str(getattr(model, "stage1_architecture_variant", _variant_name_from_config(resolved)))
    return {
        "variant_name": variant,
        "variant_params": count_parameters(model),
        "trainable_params": count_parameters(model, trainable_only=True),
        "model_class": type(model).__name__,
        "module_summary": _module_summary_for_metadata(model),
    }


def write_stage1_architecture_ablation_wandb_document(
    run_dir: str | Path,
    *,
    train_cfg: dict,
    wandb_enabled: bool,
    variant_name: str,
) -> None:
    wandb_cfg = train_cfg.get("wandb", {})
    save_json(
        {
            "enabled": bool(wandb_enabled),
            "project": wandb_cfg.get("project", "combination-stage1-ablations"),
            "entity": wandb_cfg.get("entity"),
            "mode": wandb_cfg.get("mode", "online"),
            "run_name": train_cfg.get("run_name"),
            "id": wandb_cfg.get("id"),
            "resume": wandb_cfg.get("resume"),
            "tags": wandb_cfg.get("tags", []),
            "variant": variant_name,
            "local_jsonl_logs": {
                "train": "logs/train.jsonl",
                "eval": "logs/eval.jsonl",
            },
            "metadata_files": {
                "resolved_config": "resolved_config.yaml",
                "architecture": "architecture_ablation_metadata.json",
                "wandb": "wandb_run.json",
            },
            "plot_ready_metric_keys": _wandb_metric_key_document(),
            "note": "The trainer logs the same scalar keys to W&B and local JSONL logs.",
        },
        Path(run_dir) / "wandb_run.json",
    )


def _variant_name_from_config(config: dict) -> str:
    ablation_cfg = config.get("ablation", {})
    variant = str(ablation_cfg.get("variant", ablation_cfg.get("name", "")))
    if variant == "stage1_architecture":
        variant = str(ablation_cfg.get("stage1_architecture", {}).get("variant", ""))
    return variant


def _module_summary_for_metadata(model: torch.nn.Module) -> dict[str, Any]:
    encoder_joint = getattr(getattr(model, "encoder", None), "joint_encoder", None)
    decoder_joint = getattr(getattr(model, "decoder", None), "joint_stack", None)
    return {
        "encoder_joint_depth": getattr(encoder_joint, "depth", None),
        "encoder_ffn_hidden_dim": getattr(encoder_joint, "ffn_hidden_dim", None),
        "decoder_joint_depth": getattr(decoder_joint, "depth", None),
        "decoder_ffn_hidden_dim": getattr(decoder_joint, "ffn_hidden_dim", None),
    }


def _wandb_metric_key_document() -> dict[str, list[str]]:
    return {
        "loss": [
            "train/loss.total",
            "eval/loss.total",
            "train/loss.human",
            "eval/loss.human",
            "train/loss.object",
            "eval/loss.object",
            "train/loss.relation",
            "eval/loss.relation",
            "train/loss.world_marker",
            "eval/loss.world_marker",
            "train/loss.latent_cov",
            "eval/loss.latent_cov",
        ],
        "relation_detail": [
            "train/loss.relation.delta_m2o",
            "eval/loss.relation.delta_m2o",
            "train/loss.relation.dist",
            "eval/loss.relation.dist",
        ],
        "diagnostics": [
            "train/metric.world_marker_mpjpe_m",
            "eval/metric.world_marker_mpjpe_m",
            "train/diag.lambda_cov_floor",
            "eval/diag.lambda_cov_floor",
            "train/examples_per_sec",
            "train/step_time_sec",
            "train/grad_norm",
            "train/lr",
        ],
    }
