from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "src"))
    return root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage2 tri-stream MoT rectified-flow model with Accelerate.")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-ckpt", default="best_val.pt")
    parser.add_argument("--whitening", required=True)
    parser.add_argument("--stage1-anchor-mode", choices=["hybrid", "absolute_grid"], default=None)
    parser.add_argument("--stage1-anchor-stride", type=int, default=None)
    parser.add_argument("--stage1-anchor-grid-origin", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-stage2-weights", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=None)
    parser.add_argument("--log-interval-steps", type=int, default=None)
    parser.add_argument("--velocity-eval-interval-steps", type=int, default=None)
    parser.add_argument("--debug-num-train-batches", type=int, default=None)
    parser.add_argument("--debug-num-eval-batches", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=["no", "bf16", "fp16"], default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--wandb-id", default=None)
    parser.add_argument("--wandb-resume", choices=["allow", "must", "never", "auto"], default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    validate_group = parser.add_mutually_exclusive_group()
    validate_group.add_argument("--validate-first-batch", dest="validate_first_batch", action="store_true")
    validate_group.add_argument("--no-validate-first-batch", dest="validate_first_batch", action="store_false")
    parser.set_defaults(validate_first_batch=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _add_src_to_path()
    from hoi_latent.training.stage2 import run_stage2_rectified_flow_training

    overrides = {
        "run_name": args.run_name,
        "output_root": args.output_root,
        "seed": args.seed,
        "resume": args.resume,
        "init_stage2_weights": args.init_stage2_weights,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "checkpoint_interval_steps": args.checkpoint_interval_steps,
        "log_interval_steps": args.log_interval_steps,
        "velocity_eval_interval_steps": args.velocity_eval_interval_steps,
        "debug_num_train_batches": args.debug_num_train_batches,
        "debug_num_eval_batches": args.debug_num_eval_batches,
        "mixed_precision": args.mixed_precision,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
        "wandb_id": args.wandb_id,
        "wandb_resume": args.wandb_resume,
        "wandb_tags": args.wandb_tags,
        "validate_first_batch": args.validate_first_batch,
        "stage1_anchor_mode": args.stage1_anchor_mode,
        "stage1_anchor_stride": args.stage1_anchor_stride,
        "stage1_anchor_grid_origin": args.stage1_anchor_grid_origin,
    }
    run_stage2_rectified_flow_training(
        data_config_path=args.data_config,
        train_config_path=args.train_config,
        source_run=args.source_run,
        source_ckpt=args.source_ckpt,
        whitening_path=args.whitening,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
