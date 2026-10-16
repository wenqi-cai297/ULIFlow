from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoi_latent.data.collate import collate_stage1_motion962  # noqa: E402
from hoi_latent.config.stage1_anchor_overrides import apply_stage1_anchor_override  # noqa: E402
from hoi_latent.inference.stage2 import (  # noqa: E402
    Stage2FlowForcingRolloutConfig,
    Stage2GuidanceConfig,
    Stage2InferenceLatentTemplateSpec,
    batch_gt_to_world_markers_and_object_points,
    compute_overlap_candidate_disagreement,
    decode_stage2_z_white_windows,
    prediction_to_world_markers_and_object_points,
    render_stage2_marker_comparison_mp4,
    sample_stage2_flow_forcing_rollout,
    stitch_stage1_predictions_preferred_slot,
    write_stage2_render_summary,
)
from hoi_latent.models.stage1.autoencoder.latent_whitening import (  # noqa: E402
    load_stage1_latent_whitening_transform,
)
from hoi_latent.models.stage2 import build_stage2_tri_stream_mot_model_from_config  # noqa: E402
from hoi_latent.training.stage1.phase1_warmup import (  # noqa: E402
    build_stage1_phase1_dataloaders,
    load_or_compute_stage1_normalizers,
    move_to_device,
    validate_structured_batch,
)
from hoi_latent.training.stage1.source_autoencoder import (  # noqa: E402
    build_stage1_source_autoencoder_from_config,
    load_stage1_source_component_state_dicts,
)
from hoi_latent.training.stage2.rectified_flow import (  # noqa: E402
    _load_stage2_state_without_frozen_clip,
    load_stage2_rectified_flow_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample Stage2 Flow-Forcing latents, decode with frozen Stage1 decoder, and render GT comparison MP4s."
    )
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--stage2-run", type=Path, required=True)
    parser.add_argument("--stage2-ckpt", default="latest.pt")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-ckpt", default="best_val.pt")
    parser.add_argument("--whitening", type=Path, required=True)
    parser.add_argument("--template-spec", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("runs/diagnostics/stage2_flow_forcing_renders"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--requested-frames", type=int, default=128)
    parser.add_argument("--requested-frames-source", default="fixed", choices=["fixed", "sequence_length"])
    parser.add_argument("--samples-per-dataset", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    ema_group.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--steps-per-block", type=int, default=32)
    parser.add_argument("--window-blocks", type=int, default=4)
    parser.add_argument("--clean-history-blocks", type=int, default=1)
    parser.add_argument("--solver", default="euler", choices=["euler", "heun"])
    parser.add_argument("--guidance-mode", default="none", choices=["none", "joint", "separated"])
    parser.add_argument("--joint-scale", type=float, default=1.0)
    parser.add_argument("--text-scale", type=float, default=1.0)
    parser.add_argument("--geometry-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument("--render-mode", default="generated_only", choices=["generated_only", "comparison"])
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--stage1-anchor-mode", default="absolute_grid", choices=["hybrid", "absolute_grid"])
    parser.add_argument("--stage1-anchor-stride", type=int, default=4)
    parser.add_argument("--stage1-anchor-grid-origin", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.requested_frames <= 0:
        raise ValueError("--requested-frames must be positive.")
    if args.requested_frames_source == "sequence_length" and args.batch_size not in (None, 1):
        raise ValueError("--requested-frames-source sequence_length requires --batch-size 1.")
    if args.samples_per_dataset is not None and args.samples_per_dataset <= 0:
        raise ValueError("--samples-per-dataset must be positive when provided.")

    run_name = args.run_name or f"{time.strftime('%Y%m%d_%H%M%S')}_{args.stage2_run.name}_{args.stage2_ckpt}"
    out_dir = args.out_root / run_name.replace(".pt", "")
    out_dir.mkdir(parents=True, exist_ok=False)

    device = _resolve_device(args.device)
    template_spec = _load_template_spec(args.template_spec)
    data_config, train_config, resolved_config = load_stage2_rectified_flow_training_config(
        args.data_config,
        args.train_config,
        overrides={
            "batch_size": int(args.batch_size or args.num_samples),
            "num_workers": int(args.num_workers),
            "stage1_anchor_mode": args.stage1_anchor_mode,
            "stage1_anchor_stride": args.stage1_anchor_stride,
            "stage1_anchor_grid_origin": args.stage1_anchor_grid_origin,
        },
    )
    stage1_anchor_override = apply_stage1_anchor_override(
        resolved_config,
        anchor_mode=args.stage1_anchor_mode,
        anchor_stride=args.stage1_anchor_stride,
        anchor_grid_origin=args.stage1_anchor_grid_origin,
    )
    if stage1_anchor_override.get("after") != template_spec.stage1_anchor_override.get("after", template_spec.stage1_anchor_override):
        # Older saved specs may store only the after policy; current training provenance stores before/after.
        expected = stage1_anchor_override.get("after")
        recorded = template_spec.stage1_anchor_override.get("after", template_spec.stage1_anchor_override)
        if expected != recorded:
            raise ValueError(f"Template anchor override mismatch: expected {expected}, recorded {recorded}.")
    _force_render_loader_contract(
        resolved_config,
        batch_size=_effective_batch_size(args),
        num_workers=int(args.num_workers),
        num_samples=int(args.num_samples),
        stratified=bool(args.samples_per_dataset),
    )

    normalizer, structured_normalizer, _ = load_or_compute_stage1_normalizers(resolved_config)
    if structured_normalizer is None:
        raise ValueError("Stage2 render requires the structured normalizer.")
    train_loader, eval_loader = build_stage1_phase1_dataloaders(
        resolved_config,
        normalizer=normalizer,
        structured_normalizer=structured_normalizer,
    )
    loader = train_loader if args.split == "train" else eval_loader
    if args.samples_per_dataset is not None:
        loader = _stratified_dataset_loader(
            loader,
            samples_per_dataset=int(args.samples_per_dataset),
            datasets=args.datasets,
            batch_size=_effective_batch_size(args),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
        )

    stage1_model = _load_stage1_model(resolved_config, args.source_run, args.source_ckpt, device=device)
    stage2_model = _load_stage2_model(resolved_config, args.stage2_run, args.stage2_ckpt, device=device, use_ema=bool(args.use_ema))
    whitening = load_stage1_latent_whitening_transform(args.whitening, map_location="cpu").to(device)
    whitening.eval()

    guidance = Stage2GuidanceConfig(
        enabled=args.guidance_mode != "none",
        mode=args.guidance_mode,
        joint_scale=float(args.joint_scale),
        text_scale=float(args.text_scale),
        geometry_scale=float(args.geometry_scale),
    )
    rollout_cfg = Stage2FlowForcingRolloutConfig(
        steps_per_block=int(args.steps_per_block),
        window_blocks=int(args.window_blocks),
        clean_history_blocks=int(args.clean_history_blocks),
        solver=args.solver,
    )

    summary: dict[str, Any] = {
        "data_config": str(args.data_config),
        "train_config": str(args.train_config),
        "stage2_checkpoint": str(_checkpoint_path(args.stage2_run, args.stage2_ckpt)),
        "source_checkpoint": str(_checkpoint_path(args.source_run, args.source_ckpt)),
        "whitening": str(args.whitening),
        "template_spec": str(args.template_spec),
        "template_spec_hash": template_spec.spec_hash,
        "requested_frames": int(args.requested_frames),
        "requested_frames_source": args.requested_frames_source,
        "seed": int(args.seed),
        "solver": args.solver,
        "steps_per_block": int(args.steps_per_block),
        "render_mode": args.render_mode,
        "guidance": guidance.__dict__,
        "samples": [],
    }

    sample_index = 0
    for batch_index, batch in enumerate(loader):
        validate_structured_batch(batch, window_size=resolved_config.get("data", {}).get("window_size"))
        batch = move_to_device(batch, device)
        batch_size = _batch_size_from_batch(batch)
        requested_frames = _requested_frames_for_batch(args, batch)
        rollout = sample_stage2_flow_forcing_rollout(
            stage2_model,
            batch=batch,
            template_spec=template_spec,
            requested_frames=int(requested_frames),
            guidance=guidance,
            solver=args.solver,
            seed=int(args.seed) + batch_index,
            config=rollout_cfg,
        )
        decoded = decode_stage2_z_white_windows(
            stage1_decoder=stage1_model.decoder,
            whitening=whitening,
            rollout=rollout,
            template_spec=template_spec,
            requested_frames=int(requested_frames),
        )
        stitched = stitch_stage1_predictions_preferred_slot(
            window_predictions=decoded.predictions,
            decode_window_starts=decoded.decode_window_starts,
            preferred_decode_window_start=rollout.preferred_decode_window_start,
            preferred_local_slot=rollout.preferred_local_slot,
            requested_frames=int(requested_frames),
            block_size_frames=int(template_spec.block_size_anchors * template_spec.anchor_stride_frames),
        )
        generated_world = prediction_to_world_markers_and_object_points(
            stitched,
            structured_normalizer=structured_normalizer,
            object_points_local=batch["geometry"]["object_points_local"],
        )
        gt_world = None
        if args.render_mode == "comparison":
            gt_world = batch_gt_to_world_markers_and_object_points(batch)
            _require_gt_frame_coverage(gt_world, requested_frames=int(requested_frames))
        overlap = compute_overlap_candidate_disagreement(
            window_predictions=decoded.predictions,
            decode_window_starts=decoded.decode_window_starts,
            preferred_decode_window_start=rollout.preferred_decode_window_start,
            preferred_local_slot=rollout.preferred_local_slot,
            block_size_frames=int(template_spec.block_size_anchors * template_spec.anchor_stride_frames),
        )

        for local_idx in range(batch_size):
            if sample_index >= args.num_samples:
                break
            sample_dir = out_dir / f"sample_{sample_index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=False)
            condition_text = _condition_text(batch, local_idx)
            gen_item = _slice_world(generated_world, local_idx, requested_frames=int(requested_frames))
            gt_item = None if gt_world is None else _slice_world(gt_world, local_idx, requested_frames=int(requested_frames))
            video_name = "generated.mp4" if gt_item is None else "comparison.mp4"
            render_stage2_marker_comparison_mp4(
                generated=gen_item,
                gt=gt_item,
                condition_text=condition_text,
                video_path=sample_dir / video_name,
                fps=int(args.fps),
                stride=int(args.render_stride),
                dpi=int(args.dpi),
                metadata={
                    "seed": int(args.seed) + batch_index,
                    "solver": args.solver,
                    "steps_per_block": int(args.steps_per_block),
                    "guidance_mode": args.guidance_mode,
                    "requested_frames": int(requested_frames),
                    "sequence_name": _meta_at(batch.get("meta", {}).get("sequence_name"), local_idx),
                },
            )
            arrays = {
                "generated_human_markers": gen_item.human_markers,
                "generated_object_points": gen_item.object_points,
                "frame_mask": gen_item.frame_mask,
            }
            if gt_item is not None:
                arrays["gt_human_markers"] = gt_item.human_markers
                arrays["gt_object_points"] = gt_item.object_points
            np.savez_compressed(sample_dir / "render_arrays.npz", **arrays)
            record = {
                "sample_index": sample_index,
                "batch_index": batch_index,
                "local_index": local_idx,
                "dataset": _meta_at(batch.get("meta", {}).get("dataset"), local_idx),
                "sequence_name": _meta_at(batch.get("meta", {}).get("sequence_name"), local_idx),
                "object_name": _meta_at(batch.get("meta", {}).get("object_name"), local_idx),
                "condition_text": condition_text,
                "requested_frames": int(requested_frames),
                "video_path": str(sample_dir / video_name),
                "first_frame_path": str((sample_dir / video_name).with_suffix(".frame000.png")),
            }
            write_stage2_render_summary(sample_dir / "summary.json", record)
            summary["samples"].append(record)
            sample_index += 1
        summary["rollout_diagnostics"] = _jsonable(rollout.diagnostics)
        summary["decode_diagnostics"] = _jsonable(decoded.diagnostics)
        summary["boundary_diagnostics"] = _jsonable(stitched.aux.get("stage2_decode_stitch", {}).get("boundary_diagnostics", {}))
        summary["overlap_candidate_disagreement"] = _jsonable(overlap)
        if sample_index >= args.num_samples:
            break

    write_stage2_render_summary(out_dir / "summary.json", summary)
    print(f"Wrote {out_dir / 'summary.json'}")
    for item in summary["samples"]:
        print(f"Wrote {item['video_path']}")


def _force_render_loader_contract(
    config: dict[str, Any],
    *,
    batch_size: int,
    num_workers: int,
    num_samples: int,
    stratified: bool = False,
) -> None:
    config.setdefault("loader", {})["batch_size"] = int(batch_size)
    config.setdefault("loader", {})["num_workers"] = int(num_workers)
    if not stratified:
        stage2_train = config.setdefault("training", {}).setdefault("stage2", {})
        max_batches = max(1, int(np.ceil(float(num_samples) / float(max(batch_size, 1)))))
        stage2_train["debug_num_eval_batches"] = max_batches
        stage2_train["debug_num_train_batches"] = max_batches
        stage1_train = config.setdefault("training", {}).setdefault("stage1", {})
        stage1_train["debug_num_eval_batches"] = max_batches
        stage1_train["debug_num_train_batches"] = max_batches
    config.setdefault("data", {})["window_size"] = 128
    config["data"]["eval_window_mode"] = "center"


def _load_stage1_model(config: dict[str, Any], run_dir: Path, ckpt_name: str, *, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(_checkpoint_path(run_dir, ckpt_name), map_location="cpu", weights_only=False)
    model = build_stage1_source_autoencoder_from_config(config)
    load_stage1_source_component_state_dicts(model, checkpoint["components"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _load_stage2_model(
    config: dict[str, Any],
    run_dir: Path,
    ckpt_name: str,
    *,
    device: torch.device,
    use_ema: bool,
) -> torch.nn.Module:
    checkpoint = torch.load(_checkpoint_path(run_dir, ckpt_name), map_location="cpu", weights_only=False)
    model = build_stage2_tri_stream_mot_model_from_config(config)
    state = checkpoint.get("stage2")
    if not isinstance(state, dict):
        raise KeyError(f"Stage2 checkpoint {_checkpoint_path(run_dir, ckpt_name)} is missing 'stage2'.")
    _load_stage2_state_without_frozen_clip(model, state)
    if use_ema and isinstance(checkpoint.get("ema"), dict):
        shadow = checkpoint["ema"].get("shadow")
        if isinstance(shadow, dict):
            _copy_ema_shadow_to_model(model, shadow)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _copy_ema_shadow_to_model(model: torch.nn.Module, shadow: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    copied = 0
    for name, value in shadow.items():
        if name not in params:
            continue
        params[name].data.copy_(value.to(device=params[name].device, dtype=params[name].dtype))
        copied += 1
    if copied == 0:
        raise RuntimeError("Requested --use-ema, but no EMA shadow parameters matched the Stage2 model.")


def _load_template_spec(path: Path) -> Stage2InferenceLatentTemplateSpec:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Stage2InferenceLatentTemplateSpec):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("template_spec"), Stage2InferenceLatentTemplateSpec):
        return payload["template_spec"]
    if isinstance(payload, dict) and {"metadata_1d", "spec_hash", "window_frames"}.issubset(payload):
        return Stage2InferenceLatentTemplateSpec(**payload)
    raise TypeError(f"Unsupported Stage2 template spec payload in {path}.")


def _checkpoint_path(run_dir: Path, ckpt_name: str | Path) -> Path:
    ckpt = Path(ckpt_name)
    return ckpt if ckpt.exists() else run_dir / "checkpoints" / ckpt


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _batch_size_from_batch(batch: dict[str, Any]) -> int:
    model = batch.get("model", {})
    human = model.get("human", {}) if isinstance(model, dict) else {}
    root = human.get("root") if isinstance(human, dict) else None
    if not isinstance(root, torch.Tensor):
        raise KeyError("Cannot infer batch size; expected batch['model']['human']['root'].")
    return int(root.shape[0])


def _condition_text(batch: dict[str, Any], index: int) -> str:
    meta = batch.get("meta", {})
    text = meta.get("text") if isinstance(meta, dict) else None
    if isinstance(text, (list, tuple)):
        value = text[index]
        return "" if value is None else str(value)
    return "" if text is None else str(text)


def _effective_batch_size(args: argparse.Namespace) -> int:
    if args.requested_frames_source == "sequence_length":
        return 1
    return int(args.batch_size or args.num_samples)


def _requested_frames_for_batch(args: argparse.Namespace, batch: dict[str, Any]) -> int:
    if args.requested_frames_source == "fixed":
        return int(args.requested_frames)
    meta = batch.get("meta", {})
    seq_len = meta.get("sequence_length") if isinstance(meta, dict) else None
    if not isinstance(seq_len, torch.Tensor):
        raise KeyError("--requested-frames-source sequence_length requires batch['meta']['sequence_length'].")
    values = seq_len.detach().cpu().reshape(-1).tolist()
    if not values:
        raise ValueError("sequence_length metadata is empty.")
    if len(set(int(v) for v in values)) != 1:
        raise ValueError("--requested-frames-source sequence_length requires all samples in a batch to have the same length.")
    return int(values[0])


def _stratified_dataset_loader(
    loader,
    *,
    samples_per_dataset: int,
    datasets: list[str] | None,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    dataset = _unwrap_subset(loader.dataset)
    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("Stratified rendering requires a dataset with a 'records' attribute.")
    wanted = set(datasets or sorted({record.dataset for record in records}))
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for dataset_name in sorted(wanted):
        candidates = [idx for idx, record in enumerate(records) if record.dataset == dataset_name]
        if not candidates:
            continue
        rng.shuffle(candidates)
        indices.extend(candidates[: int(samples_per_dataset)])
    if not indices:
        raise ValueError(f"No records found for requested datasets {sorted(wanted)}.")
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        persistent_workers=False,
        collate_fn=collate_stage1_motion962,
    )


def _unwrap_subset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _meta_at(value: Any, index: int, default: Any = None) -> Any:
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        return flat[index].item() if index < flat.numel() else default
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else default
    return default if value is None else value


def _require_gt_frame_coverage(arrays, *, requested_frames: int) -> None:
    gt_frames = int(arrays.human_markers.shape[1])
    if gt_frames < int(requested_frames):
        raise ValueError(
            "This diagnostic renders generated output side-by-side with dataset GT, but the current "
            f"structured dataloader only provides {gt_frames} GT frames while requested_frames={requested_frames}. "
            "Use requested_frames <= the dataloader window length for GT comparison, or add a full-sequence GT loader."
        )


def _slice_world(arrays, index: int, *, requested_frames: int):
    from hoi_latent.inference.stage2 import Stage2WorldRenderArrays

    return Stage2WorldRenderArrays(
        human_markers=arrays.human_markers[index : index + 1, :requested_frames],
        object_points=arrays.object_points[index : index + 1, :requested_frames],
        frame_mask=arrays.frame_mask[index : index + 1, :requested_frames],
        object_trans=None if arrays.object_trans is None else arrays.object_trans[index : index + 1, :requested_frames],
        object_rotmat=None if arrays.object_rotmat is None else arrays.object_rotmat[index : index + 1, :requested_frames],
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
