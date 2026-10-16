from __future__ import annotations

import argparse
import copy
import json
import math
import numpy as np
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoi_latent.config.io import load_yaml, save_yaml  # noqa: E402
from hoi_latent.config.stage1_anchor_overrides import apply_stage1_anchor_override  # noqa: E402
from hoi_latent.data.collate import collate_stage1_motion962  # noqa: E402
from hoi_latent.evaluation.stage2 import (  # noqa: E402
    InterActFeatureArtifactPaths,
    InterActFeatureEvaluatorUnavailable,
    Stage2DecodedSampleRecord,
    build_local_stage1_window_batch,
    build_interact_feature_evaluator_from_artifacts,
    audit_interact_official_split_no_leak,
    compute_stage1_reconstruction_ceiling,
    compute_windowed_stage1_reconstruction_ceiling,
    default_interact_feature_artifact_paths,
    evaluate_stage2_decoded_records,
    load_interact_feature_artifacts,
    read_interact_split_audit_report,
    read_marker_bps_alignment_report,
    read_stage2_decoded_records_from_dir,
    run_interact_feature_parity_check,
    sha256_file,
    split_batched_world_arrays,
    write_interact_split_audit_report,
    write_marker_bps_alignment_report,
    write_stage2_decoded_record,
    write_stage2_quantitative_outputs,
)
from hoi_latent.inference.stage2 import (  # noqa: E402
    Stage2FlowForcingRolloutConfig,
    Stage2GuidanceConfig,
    Stage2InferenceLatentTemplateSpec,
    batch_gt_to_world_markers_and_object_points,
    decode_stage2_z_white_windows,
    extract_stage2_inference_latent_template_spec,
    prediction_to_world_markers_and_object_points,
    sample_stage2_flow_forcing_rollout,
    stitch_stage1_predictions_preferred_slot,
)
from hoi_latent.models.stage1.autoencoder.latent_whitening import load_stage1_latent_whitening_transform  # noqa: E402
from hoi_latent.models.stage2 import build_stage2_tri_stream_mot_model_from_config  # noqa: E402
from hoi_latent.training.stage1.phase1_warmup import (  # noqa: E402
    build_stage1_phase1_dataloaders,
    infer_frame_mask,
    load_or_compute_stage1_normalizers,
    move_to_device,
    validate_structured_batch,
    _limit_dataset,
    _make_dataset,
)
from hoi_latent.training.stage1.source_autoencoder import (  # noqa: E402
    build_stage1_source_autoencoder_from_config,
    encode_stage1_source_latent,
    load_stage1_source_component_state_dicts,
    load_stage1_source_config,
    merge_stage1_source_model_contract,
)
from hoi_latent.training.stage2.rectified_flow import (  # noqa: E402
    _load_stage2_state_without_frozen_clip,
    load_stage2_rectified_flow_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Stage2 Flow-Forcing decoded records into quantitative evaluation metrics."
    )
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--stage2-run", type=Path, required=True)
    parser.add_argument("--stage2-ckpt", default="latest.pt")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-ckpt", default="best_val.pt")
    parser.add_argument("--whitening", type=Path, required=True)
    parser.add_argument(
        "--template-spec",
        type=Path,
        default=None,
        help="Existing latent template spec. Omit to extract one from the frozen Stage1 source.",
    )
    parser.add_argument(
        "--eval-protocol",
        choices=["eval_128_dev", "eval_128_report", "eval_long_v1", "interact_official_v1"],
        required=True,
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("runs/evaluation/stage2"))
    parser.add_argument("--decoded-records-dir", type=Path, default=None)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Write decoded records and exit before loading them for metric aggregation.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-conditions", type=int, default=None)
    parser.add_argument(
        "--min-sequence-frames",
        type=int,
        default=None,
        help=(
            "Restrict evaluation to source sequences at least this long. For eval_long_v1, "
            "the dataloader then returns the requested-length prefix so generated and GT "
            "motions remain paired."
        ),
    )
    parser.add_argument(
        "--sequence-name",
        action="append",
        default=[],
        help=(
            "Restrict generation to an exact dataset sequence name. Repeat for multiple "
            "conditions; selected sequences retain the CLI order."
        ),
    )
    parser.add_argument(
        "--disable-text-filter",
        action="store_true",
        help=(
            "Keep the fixed evaluation condition set even when the training config filtered "
            "captions by tokenizer length. The text encoder still applies its configured "
            "maximum sequence length."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--requested-frames", type=int, default=128)
    parser.add_argument("--requested-frames-list", type=int, nargs="*", default=[128, 196, 256, 384])
    parser.add_argument("--samples-per-condition", type=int, default=1)
    parser.add_argument("--multimodality-samples", type=int, default=10)
    parser.add_argument(
        "--multimodality-pairs-per-condition",
        type=int,
        default=None,
        help=(
            "If set, compute InterAct/T2M-style MModality by sampling this many "
            "pairs inside every condition. Leave unset to use the legacy total-pair protocol."
        ),
    )
    parser.add_argument("--best-of-k", type=int, default=10)
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    ema_group.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--steps-per-block", type=int, default=32)
    parser.add_argument("--window-blocks", type=int, default=4)
    parser.add_argument("--clean-history-blocks", type=int, default=1)
    parser.add_argument("--guidance-mode", choices=["none", "joint", "separated"], default="none")
    parser.add_argument("--joint-scale", type=float, default=1.0)
    parser.add_argument("--text-scale", type=float, default=1.0)
    parser.add_argument("--geometry-scale", type=float, default=1.0)
    parser.add_argument("--decode", choices=["decoder", "coarse"], default="decoder")
    parser.add_argument(
        "--stage1-anchor-mode",
        choices=["source", "hybrid", "absolute_grid"],
        default="absolute_grid",
        help="Use 'source' for Stage1 architecture ablations with their native latent layout.",
    )
    parser.add_argument("--stage1-anchor-stride", type=int, default=4)
    parser.add_argument("--stage1-anchor-grid-origin", type=int, default=0)
    parser.add_argument("--feature-evaluator-mode", choices=["official_artifact", "repo_trained", "disabled"], default="official_artifact")
    parser.add_argument("--feature-artifact-root", type=Path, default=Path("assets/evaluation/interact_markersbps"))
    parser.add_argument("--feature-evaluator-ckpt", type=Path, default=None)
    parser.add_argument("--feature-state-dict", type=Path, default=None)
    parser.add_argument("--feature-mean", type=Path, default=None)
    parser.add_argument("--feature-std", type=Path, default=None)
    parser.add_argument("--feature-bps-basis", type=Path, default=None)
    parser.add_argument("--feature-parity-dir", type=Path, default=None)
    parser.add_argument("--split-audit-json", type=Path, default=None)
    parser.add_argument("--marker-bps-audit-json", type=Path, default=None)
    parser.add_argument("--require-feature-parity", action="store_true")
    parser.add_argument("--allow-missing-feature-evaluator", action="store_true")
    parser.add_argument("--r-precision-pool-size", type=int, default=32)
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=64,
        help="Batch size for GPU feature preprocessing/encoding; <=0 keeps the old all-at-once path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    resolved = _resolved_eval_config(args)
    save_yaml(resolved, run_dir / "resolved_config.yaml")
    _write_json(run_dir / "git.json", _git_metadata(args))

    records_dir = _resolve_or_generate_records_dir(args, run_dir)
    if args.generate_only:
        record_count = sum(1 for _ in records_dir.glob("records_*.npz"))
        if record_count == 0:
            raise RuntimeError(f"No decoded records found under {records_dir}.")
        print(f"Decoded records ready: {record_count} under {records_dir}; skipping metric aggregation.")
        return
    records = read_stage2_decoded_records_from_dir(records_dir)
    if records_dir.resolve() != (run_dir / "predictions").resolve():
        _write_records(run_dir / "predictions", records)
    split_audit_report = _load_or_run_split_audit(args, records)
    marker_bps_report = _load_marker_bps_audit(args)

    feature_evaluator = None
    feature_artifacts: dict[str, Any] = {"mode": args.feature_evaluator_mode}
    feature_required = _feature_required(args)
    feature_enabled = args.feature_evaluator_mode != "disabled" and args.eval_protocol != "eval_long_v1"
    if feature_enabled:
        try:
            feature_evaluator, feature_artifacts = _build_feature_evaluator(args)
        except InterActFeatureEvaluatorUnavailable as exc:
            feature_artifacts["unavailable_reason"] = str(exc)
            if feature_required or not args.allow_missing_feature_evaluator:
                raise

    paired_enabled = args.eval_protocol != "eval_long_v1"
    result = evaluate_stage2_decoded_records(
        records,
        protocol=args.eval_protocol,
        feature_evaluator=feature_evaluator,
        feature_enabled=feature_enabled,
        feature_required=feature_required,
        paired_enabled=paired_enabled,
        interaction_enabled=True,
        stage1_ceiling_enabled=paired_enabled,
        r_precision_pool_size=int(args.r_precision_pool_size),
        r_precision_seed=int(args.seed),
        block_size_frames=32,
        feature_metrics_mode=args.feature_evaluator_mode,
        feature_parity_passed=_feature_parity_passed(feature_artifacts),
        feature_parity_golden_source=_feature_parity_golden_source(feature_artifacts),
        feature_artifact_manifest_valid=_feature_manifest_valid(feature_artifacts),
        split_audit_passed=None if split_audit_report is None else bool(split_audit_report.passed),
        marker_bps_alignment_passed=None if marker_bps_report is None else bool(marker_bps_report.passed),
        multimodality_min_samples=int(args.multimodality_samples),
        multimodality_pairs_per_condition=args.multimodality_pairs_per_condition,
        feature_batch_size=int(args.feature_batch_size),
    )
    if args.multimodality_pairs_per_condition is not None and "feature/multimodality" not in result.feature_metrics:
        samples_per_condition = _samples_per_condition_from_records(records)
        raise RuntimeError(
            "Requested standard per-condition MModality, but feature/multimodality was not produced. "
            "This usually means at least one condition has too few or uneven generated samples. "
            f"samples_per_condition={samples_per_condition}, required={int(args.multimodality_samples)}."
        )
    artifacts = _artifact_manifest(
        args,
        records_dir=records_dir,
        feature_artifacts=feature_artifacts,
        split_audit=split_audit_report,
        marker_bps_alignment=marker_bps_report,
    )
    write_stage2_quantitative_outputs(result, run_dir, artifacts=artifacts)
    if split_audit_report is not None:
        write_interact_split_audit_report(split_audit_report, run_dir / "metrics" / "split_audit.json")
    if marker_bps_report is not None:
        write_marker_bps_alignment_report(marker_bps_report, run_dir / "metrics" / "marker_bps_alignment.json")
    print(f"Wrote {run_dir / 'metrics' / 'summary.json'}")


def _resolve_or_generate_records_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.decoded_records_dir is not None:
        return args.decoded_records_dir
    candidate = args.stage2_run / "predictions"
    if candidate.exists():
        return candidate
    return _generate_decoded_records(args, run_dir / "predictions")


@torch.no_grad()
def _generate_decoded_records(args: argparse.Namespace, records_dir: Path) -> Path:
    records_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    _, _, resolved_config = load_stage2_rectified_flow_training_config(
        args.data_config,
        args.train_config,
        overrides={
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "stage1_anchor_mode": args.stage1_anchor_mode,
            "stage1_anchor_stride": int(args.stage1_anchor_stride),
            "stage1_anchor_grid_origin": int(args.stage1_anchor_grid_origin),
        },
    )
    source_config = load_stage1_source_config(args.source_run, fallback_config=resolved_config)
    model_config = merge_stage1_source_model_contract(resolved_config, source_config)
    if args.stage1_anchor_mode == "source":
        anchor_override: dict[str, Any] = {}
    else:
        anchor_override = apply_stage1_anchor_override(
            model_config,
            anchor_mode=args.stage1_anchor_mode,
            anchor_stride=int(args.stage1_anchor_stride),
            anchor_grid_origin=int(args.stage1_anchor_grid_origin),
        )
    loader_config = copy.deepcopy(resolved_config)
    _force_evaluation_loader_contract(loader_config, args)

    normalizer, structured_normalizer, _ = load_or_compute_stage1_normalizers(loader_config)
    if structured_normalizer is None:
        raise ValueError("Stage2 quantitative evaluation requires the structured normalizer.")
    loader = _build_stage2_evaluation_loader(
        loader_config,
        args,
        normalizer=normalizer,
        structured_normalizer=structured_normalizer,
    )

    stage1_model = _load_stage1_model(model_config, args.source_run, args.source_ckpt, device=device)
    stage2_model = _load_stage2_model(model_config, args.stage2_run, args.stage2_ckpt, device=device, use_ema=bool(args.use_ema))
    whitening = load_stage1_latent_whitening_transform(args.whitening, map_location="cpu").to(device)
    whitening.eval()
    template_spec = _load_or_extract_template_spec(
        args,
        loader=loader,
        stage1_model=stage1_model,
        model_config=model_config,
        structured_normalizer=structured_normalizer,
        anchor_override=anchor_override,
        records_dir=records_dir,
        device=device,
    )
    _validate_template_anchor(template_spec, anchor_override)

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

    max_conditions = int(args.num_conditions) if args.num_conditions is not None else None
    condition_count = 0
    record_count = 0
    generated_frames = 0
    generation_seconds = 0.0
    for batch_index, batch in enumerate(loader):
        validate_structured_batch(batch, window_size=loader_config.get("data", {}).get("window_size"))
        batch = move_to_device(batch, device)
        batch_size = _batch_size_from_batch(batch)
        condition_indices: list[int] = []
        for _local_idx in range(batch_size):
            if max_conditions is not None and condition_count >= max_conditions:
                break
            condition_indices.append(condition_count)
            condition_count += 1
        if not condition_indices:
            break

        requested_frames = _requested_frames_for_batch(args, batch)
        gt_world = batch_gt_to_world_markers_and_object_points(batch)
        gt_has_requested = int(gt_world.human_markers.shape[1]) >= int(requested_frames)
        _clip_world_frame_mask(gt_world, requested_frames)
        ceiling_world = None
        ceiling_decode_metadata: dict[str, Any] = {}
        if gt_has_requested:
            if int(requested_frames) > int(template_spec.window_frames) or args.eval_protocol == "interact_official_v1":
                ceiling = compute_windowed_stage1_reconstruction_ceiling(
                    stage1_model=stage1_model,
                    whitening=whitening,
                    batch=batch,
                    structured_normalizer=structured_normalizer,
                    template_spec=template_spec,
                    requested_frames=int(requested_frames),
                    clean_history_blocks=int(args.clean_history_blocks),
                    root_method=loader_config.get("data", {}).get("structured", {}).get("root_method", "marker_mean"),
                )
                ceiling_decode_metadata = dict(ceiling.decode_metadata)
            else:
                ceiling = compute_stage1_reconstruction_ceiling(
                    stage1_model=stage1_model,
                    whitening=whitening,
                    batch=batch,
                    structured_normalizer=structured_normalizer,
                )
            ceiling_world = ceiling.world
            _clip_world_frame_mask(ceiling_world, requested_frames)
        elif args.eval_protocol != "eval_long_v1":
            raise ValueError(
                f"{args.eval_protocol} requires GT coverage for requested_frames={requested_frames}, "
                f"but the dataloader batch contains {gt_world.human_markers.shape[1]} frames."
            )

        for sample_index in range(int(args.samples_per_condition)):
            _synchronize_if_cuda(device)
            generation_started = time.perf_counter()
            rollout = sample_stage2_flow_forcing_rollout(
                stage2_model,
                batch=batch,
                template_spec=template_spec,
                requested_frames=int(requested_frames),
                guidance=guidance,
                solver=args.solver,
                seed=int(args.seed) + batch_index * 1009 + sample_index * 100003,
                config=rollout_cfg,
            )
            decoded = decode_stage2_z_white_windows(
                stage1_model=stage1_model,
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
            _synchronize_if_cuda(device)
            generation_seconds += time.perf_counter() - generation_started
            generated_frames += int(requested_frames) * len(condition_indices)
            for local_idx, condition_index in enumerate(condition_indices):
                record = _build_decoded_record(
                    generated_world=generated_world,
                    gt_world=gt_world if gt_has_requested else None,
                    ceiling_world=ceiling_world,
                    batch=batch,
                    batch_index=local_idx,
                    split=args.split,
                    requested_frames=int(requested_frames),
                    fps=30.0,
                    seed=int(args.seed) + batch_index * 1009 + sample_index * 100003,
                    condition_index=condition_index,
                    sample_index=sample_index,
                    rollout_metadata={
                        **_jsonable(rollout.diagnostics),
                        "update_counts": rollout.update_counts.detach().cpu().tolist(),
                        "static_update_count": rollout.static_update_count.detach().cpu().tolist(),
                    },
                    decode_metadata={
                        **_jsonable(decoded.diagnostics),
                        "decode_window_starts": decoded.decode_window_starts.detach().cpu().tolist(),
                        "preferred_decode_window_start": rollout.preferred_decode_window_start.detach().cpu().tolist(),
                        "preferred_local_slot": rollout.preferred_local_slot.detach().cpu().tolist(),
                        "stage1_ceiling": ceiling_decode_metadata,
                    },
                    provenance={
                        "stage2_checkpoint": str(_checkpoint_path(args.stage2_run, args.stage2_ckpt)),
                        "stage1_checkpoint": str(_checkpoint_path(args.source_run, args.source_ckpt)),
                        "whitening": str(args.whitening),
                        "template_spec": str(args.template_spec),
                        "stage1_anchor_override": anchor_override,
                    },
                )
                write_stage2_decoded_record(record, records_dir / f"records_{record_count:06d}.npz")
                record_count += 1
        if max_conditions is not None and condition_count >= max_conditions:
            break

    if record_count == 0:
        raise RuntimeError("Stage2 quantitative evaluation produced zero decoded records.")
    _write_json(
        records_dir.parent / "generation_timing.json",
        {
            "record_count": int(record_count),
            "generated_frames": int(generated_frames),
            "generation_seconds": float(generation_seconds),
            "seconds_per_100_frames": (
                float(generation_seconds) * 100.0 / float(generated_frames)
                if generated_frames > 0
                else None
            ),
        },
    )
    return records_dir


def _write_records(target_dir: Path, records) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        write_stage2_decoded_record(record, target_dir / f"records_{index:06d}.npz")


def _samples_per_condition_from_records(records) -> int | None:
    counts: dict[int, int] = {}
    for record in records:
        counts[int(record.condition_index)] = counts.get(int(record.condition_index), 0) + 1
    values = set(counts.values())
    return next(iter(values)) if len(values) == 1 else None


def _build_decoded_record(
    *,
    generated_world,
    gt_world,
    ceiling_world,
    batch: Mapping[str, Any],
    batch_index: int,
    split: str,
    requested_frames: int,
    fps: float,
    seed: int,
    condition_index: int,
    sample_index: int,
    rollout_metadata: Mapping[str, Any],
    decode_metadata: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Stage2DecodedSampleRecord:
    meta = batch.get("meta", {}) if isinstance(batch, Mapping) else {}
    dataset = _meta_value(meta, "dataset", batch_index)
    sequence = _meta_value(meta, "sequence_name", batch_index) or _meta_value(meta, "seq_name", batch_index)
    text = _meta_value(meta, "text", batch_index)
    object_name = _meta_value(meta, "object_name", batch_index) or _meta_value(meta, "object", batch_index)
    return Stage2DecodedSampleRecord(
        sample_id=_make_sample_id(dataset, sequence, condition_index, sample_index),
        split=str(split),
        dataset=_none_or_str(dataset),
        sequence_name=_none_or_str(sequence),
        text=_none_or_str(text),
        object_name=_none_or_str(object_name),
        requested_frames=int(requested_frames),
        fps=float(fps),
        seed=int(seed),
        condition_index=int(condition_index),
        sample_index=int(sample_index),
        generated=split_batched_world_arrays(generated_world, batch_index),
        gt=split_batched_world_arrays(gt_world, batch_index) if gt_world is not None else None,
        stage1_ceiling=split_batched_world_arrays(ceiling_world, batch_index) if ceiling_world is not None else None,
        rollout_metadata=dict(rollout_metadata),
        decode_metadata=dict(decode_metadata),
        provenance=dict(provenance),
    )


def _force_evaluation_loader_contract(config: dict[str, Any], args: argparse.Namespace) -> None:
    min_sequence_frames = getattr(args, "min_sequence_frames", None)
    if args.eval_protocol == "interact_official_v1":
        if args.split != "test":
            raise ValueError("interact_official_v1 must be run with --split test.")
        if int(args.batch_size) != 1:
            raise ValueError("interact_official_v1 currently requires --batch-size 1 for per-sample requested_frames.")
    config.setdefault("loader", {})["batch_size"] = int(args.batch_size)
    config.setdefault("loader", {})["num_workers"] = int(args.num_workers)
    max_batches = None
    if args.num_conditions is not None:
        max_batches = max(1, int(math.ceil(float(args.num_conditions) / float(max(int(args.batch_size), 1)))))
    stage2_train = config.setdefault("training", {}).setdefault("stage2", {})
    stage1_train = config.setdefault("training", {}).setdefault("stage1", {})
    if max_batches is not None:
        stage2_train["debug_num_eval_batches"] = max_batches
        stage2_train["debug_num_train_batches"] = max_batches
        stage1_train["debug_num_eval_batches"] = max_batches
        stage1_train["debug_num_train_batches"] = max_batches
    data_cfg = config.setdefault("data", {})
    if bool(getattr(args, "disable_text_filter", False)):
        data_cfg.setdefault("text_filter", {})["enabled"] = False
    if args.eval_protocol == "interact_official_v1":
        data_cfg["window_size"] = 300
        data_cfg["eval_window_mode"] = "first"
        data_cfg["train_window_mode"] = "first"
    elif min_sequence_frames is not None:
        if int(min_sequence_frames) < int(args.requested_frames):
            raise ValueError("--min-sequence-frames must be at least --requested-frames.")
        data_cfg["window_size"] = int(args.requested_frames)
        data_cfg["eval_window_mode"] = "first"
        data_cfg["train_window_mode"] = "first"
    else:
        data_cfg["window_size"] = 128
        data_cfg["eval_window_mode"] = "center"
        data_cfg["train_window_mode"] = "random"


def _build_stage2_evaluation_loader(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    normalizer,
    structured_normalizer,
):
    min_sequence_frames = getattr(args, "min_sequence_frames", None)
    if args.eval_protocol != "interact_official_v1":
        if args.sequence_name or min_sequence_frames is not None:
            data_cfg = config["data"]
            window_mode = (
                data_cfg.get("train_window_mode", "random")
                if args.split == "train"
                else data_cfg.get("eval_window_mode", "center")
            )
            dataset = _make_dataset(
                config,
                split=args.split,
                window_mode=window_mode,
                normalizer=normalizer,
                structured_normalizer=structured_normalizer,
            )
            if args.sequence_name:
                dataset = _select_sequence_subset(dataset, args.sequence_name)
            if min_sequence_frames is not None:
                dataset = _select_minimum_sequence_length_subset(
                    dataset,
                    int(min_sequence_frames),
                )
            if getattr(args, "num_conditions", None) is not None:
                dataset = _limit_dataset(dataset, int(args.num_conditions))
            return _build_selected_sequence_loader(dataset, config)
        train_loader, eval_loader = build_stage1_phase1_dataloaders(
            config,
            normalizer=normalizer,
            structured_normalizer=structured_normalizer,
        )
        return train_loader if args.split == "train" else eval_loader

    data_cfg = config["data"]
    loader_cfg = config.get("loader", {})
    stage1_train = config.setdefault("training", {}).setdefault("stage1", {})
    dataset = _make_dataset(
        config,
        split="test",
        window_mode=data_cfg.get("eval_window_mode", "first"),
        normalizer=normalizer,
        structured_normalizer=structured_normalizer,
    )
    if args.sequence_name:
        dataset = _select_sequence_subset(dataset, args.sequence_name)
    if stage1_train.get("debug_num_eval_batches") is not None:
        dataset = _limit_dataset(dataset, int(stage1_train["debug_num_eval_batches"]) * int(loader_cfg.get("batch_size", 1)))
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", False)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)) and int(loader_cfg.get("num_workers", 0)) > 0,
        collate_fn=collate_stage1_motion962,
    )


def _select_sequence_subset(dataset, sequence_names: list[str]) -> Subset:
    requested = [str(name) for name in sequence_names]
    if len(set(requested)) != len(requested):
        raise ValueError("--sequence-name values must be unique.")

    base_dataset = dataset
    candidate_indices: list[int]
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        candidate_indices = [int(index) for index in dataset.indices]
    else:
        candidate_indices = list(range(len(dataset)))

    records = getattr(base_dataset, "records", None)
    if records is None:
        raise TypeError("Sequence selection requires a dataset with a records attribute.")

    matches: dict[str, list[int]] = {name: [] for name in requested}
    for index in candidate_indices:
        sequence_name = str(getattr(records[index], "sequence_name", ""))
        if sequence_name in matches:
            matches[sequence_name].append(index)

    invalid = {name: indices for name, indices in matches.items() if len(indices) != 1}
    if invalid:
        details = ", ".join(f"{name!r}: {len(indices)} matches" for name, indices in invalid.items())
        raise ValueError(f"Each selected sequence must match exactly one dataset record; {details}.")
    return Subset(base_dataset, [matches[name][0] for name in requested])


def _select_minimum_sequence_length_subset(dataset, min_frames: int) -> Subset:
    if min_frames <= 0:
        raise ValueError(f"min_frames must be positive, got {min_frames}.")
    base_dataset = dataset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        candidate_indices = [int(index) for index in dataset.indices]
    else:
        candidate_indices = list(range(len(dataset)))
    records = getattr(base_dataset, "records", None)
    if records is None:
        raise TypeError("Length selection requires a dataset with a records attribute.")
    selected = [index for index in candidate_indices if int(records[index].length) >= min_frames]
    if not selected:
        raise ValueError(f"No evaluation sequences have at least {min_frames} frames.")
    return Subset(base_dataset, selected)


def _build_selected_sequence_loader(dataset: Subset, config: Mapping[str, Any]) -> DataLoader:
    loader_cfg = config.get("loader", {})
    num_workers = int(loader_cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", False)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0,
        collate_fn=collate_stage1_motion962,
    )


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


@torch.no_grad()
def _load_or_extract_template_spec(
    args: argparse.Namespace,
    *,
    loader,
    stage1_model: torch.nn.Module,
    model_config: dict[str, Any],
    structured_normalizer: Any,
    anchor_override: dict[str, Any],
    records_dir: Path,
    device: torch.device,
) -> Stage2InferenceLatentTemplateSpec:
    if args.template_spec is not None:
        return _load_template_spec(args.template_spec)

    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("Cannot extract a Stage2 latent template from an empty evaluation loader.") from exc
    validate_structured_batch(batch, window_size=None)
    batch = move_to_device(batch, device)
    window_frames = int(model_config.get("model", {}).get("stage1", {}).get("max_seq_len", 128))
    local_batch = build_local_stage1_window_batch(
        batch,
        structured_normalizer=structured_normalizer,
        valid_mask=infer_frame_mask(batch).to(device=device, dtype=torch.bool),
        start_frame=0,
        window_frames=window_frames,
        root_method=model_config.get("data", {}).get("structured", {}).get("root_method", "marker_mean"),
    )
    latent = encode_stage1_source_latent(
        stage1_model,
        local_batch,
        return_diagnostics=False,
        validate=False,
    )
    block_size_anchors = int(
        model_config.get("training", {})
        .get("stage2", {})
        .get("flow_forcing", {})
        .get("block_size_anchors", 8)
    )
    spec = extract_stage2_inference_latent_template_spec(
        latent,
        window_frames=window_frames,
        anchor_stride_frames=int(args.stage1_anchor_stride),
        block_size_anchors=block_size_anchors,
        source_stage1_checkpoint_sha256=sha256_file(_checkpoint_path(args.source_run, args.source_ckpt)),
        stage1_anchor_override=anchor_override,
    )
    path = records_dir.parent / "template_spec.pt"
    torch.save({"template_spec": spec}, path)
    args.template_spec = path
    return spec


def _validate_template_anchor(template_spec: Stage2InferenceLatentTemplateSpec, active_override: dict[str, Any] | None) -> None:
    if not active_override:
        return
    expected = active_override.get("after", active_override)
    recorded = template_spec.stage1_anchor_override.get("after", template_spec.stage1_anchor_override)
    if expected != recorded:
        raise ValueError(f"Template anchor override mismatch: expected {expected}, recorded {recorded}.")


def _batch_size_from_batch(batch: Mapping[str, Any]) -> int:
    model = batch.get("model", {})
    human = model.get("human", {}) if isinstance(model, Mapping) else {}
    root = human.get("root") if isinstance(human, Mapping) else None
    if not isinstance(root, torch.Tensor):
        raise KeyError("Cannot infer batch size; expected batch['model']['human']['root'].")
    return int(root.shape[0])


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_feature_evaluator(args: argparse.Namespace):
    if args.feature_evaluator_mode == "repo_trained":
        raise InterActFeatureEvaluatorUnavailable("repo_trained feature evaluator mode is not implemented yet.")
    defaults = default_interact_feature_artifact_paths(args.feature_artifact_root)
    paths = InterActFeatureArtifactPaths(
        checkpoint=args.feature_evaluator_ckpt or defaults.checkpoint,
        mean=args.feature_mean or defaults.mean,
        std=args.feature_std or defaults.std,
        bps_basis=args.feature_bps_basis or defaults.bps_basis,
        state_dict=args.feature_state_dict or defaults.state_dict,
        manifest=defaults.manifest,
    )
    bundle = load_interact_feature_artifacts(paths)
    evaluator = build_interact_feature_evaluator_from_artifacts(bundle)
    artifacts: dict[str, Any] = {
        "mode": args.feature_evaluator_mode,
        "checkpoint": str(paths.checkpoint),
        "mean": str(paths.mean),
        "std": str(paths.std),
        "bps_basis": str(paths.bps_basis),
        "state_dict": str(paths.state_dict) if paths.state_dict else None,
        "sha256": bundle.sha256,
        "manifest": bundle.manifest,
        "manifest_valid": bool(bundle.manifest_valid),
    }
    parity_required = _feature_required(args) or bool(args.require_feature_parity)
    parity_dir = args.feature_parity_dir or (args.feature_artifact_root / "parity")
    if parity_required:
        report = run_interact_feature_parity_check(evaluator, parity_dir=parity_dir)
        artifacts["parity"] = report.to_dict()
        artifacts["parity_dir"] = str(parity_dir)
        if not report.passed:
            raise InterActFeatureEvaluatorUnavailable(f"feature parity check failed: {report.to_dict()}")
        if report.golden_source != "external":
            raise InterActFeatureEvaluatorUnavailable(
                "feature parity fixture must be generated by the external InterAct evaluator; "
                f"got golden_source={report.golden_source!r}."
            )
    elif parity_dir.exists():
        report = run_interact_feature_parity_check(evaluator, parity_dir=parity_dir)
        artifacts["parity"] = report.to_dict()
        artifacts["parity_dir"] = str(parity_dir)
    return evaluator, artifacts


def _feature_required(args: argparse.Namespace) -> bool:
    return bool(args.require_feature_parity or args.eval_protocol in {"eval_128_report", "interact_official_v1"})


def _clip_world_frame_mask(world: Any, length: int) -> None:
    """Mask out padding past the true ``length`` so GT/ceiling features ignore it.

    interact_official_v1 loads each sequence into a fixed 300-frame window with the
    padding frames marked valid, while generated motion is produced at the true
    sequence length. Without this clip the GT/ceiling features would be encoded over
    the static padding tail (window minus true length), inflating FID and skewing
    R-precision / MM-Dist. No-op when the mask is already <= length.
    """
    if world is None:
        return
    fm = np.asarray(world.frame_mask, dtype=bool).copy()
    n = max(0, int(length))
    if fm.ndim == 2:
        fm[:, n:] = False
    else:
        fm[n:] = False
    world.frame_mask = fm


def _requested_frames_for_batch(args: argparse.Namespace, batch: Mapping[str, Any]) -> int:
    if args.eval_protocol != "interact_official_v1":
        return int(args.requested_frames)
    batch_size = _batch_size_from_batch(batch)
    if batch_size != 1:
        raise ValueError("interact_official_v1 requires one sequence per batch so requested_frames is unambiguous.")
    meta = batch.get("meta", {}) if isinstance(batch, Mapping) else {}
    value = _meta_value(meta, "sequence_length", 0)
    if value is None:
        lengths = batch.get("lengths")
        if isinstance(lengths, torch.Tensor) and lengths.numel() > 0:
            value = int(lengths[0].item())
    if value is None:
        raise KeyError("interact_official_v1 requires batch['meta']['sequence_length'] or batch['lengths'].")
    return max(1, min(int(value), 300))


def _feature_parity_passed(feature_artifacts: Mapping[str, Any]) -> bool | None:
    parity = feature_artifacts.get("parity")
    if not isinstance(parity, Mapping) or "passed" not in parity:
        return None
    return bool(parity["passed"])


def _feature_parity_golden_source(feature_artifacts: Mapping[str, Any]) -> str | None:
    parity = feature_artifacts.get("parity")
    if not isinstance(parity, Mapping):
        return None
    value = parity.get("golden_source")
    return str(value) if value is not None else None


def _feature_manifest_valid(feature_artifacts: Mapping[str, Any]) -> bool | None:
    if "manifest_valid" not in feature_artifacts:
        return None
    return bool(feature_artifacts["manifest_valid"])


def _load_or_run_split_audit(args: argparse.Namespace, records: list[Stage2DecodedSampleRecord]):
    if args.split_audit_json is not None:
        return read_interact_split_audit_report(args.split_audit_json)
    if args.eval_protocol != "interact_official_v1":
        return None
    stage1_config = _safe_load_yaml(Path(args.source_run) / "resolved_config.yaml")
    stage2_config = _safe_load_yaml(Path(args.stage2_run) / "resolved_config.yaml")
    if not isinstance(stage1_config, Mapping):
        raise FileNotFoundError(f"Stage1 source run is missing resolved_config.yaml: {args.source_run}")
    if not isinstance(stage2_config, Mapping):
        raise FileNotFoundError(f"Stage2 run is missing resolved_config.yaml: {args.stage2_run}")
    return audit_interact_official_split_no_leak(
        eval_records=records,
        eval_split=args.split,
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        stage1_source_run=args.source_run,
        stage2_run=args.stage2_run,
    )


def _load_marker_bps_audit(args: argparse.Namespace):
    if args.marker_bps_audit_json is None:
        return None
    return read_marker_bps_alignment_report(args.marker_bps_audit_json)


def _resolved_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data_config": _safe_load_yaml(args.data_config),
        "train_config": _safe_load_yaml(args.train_config),
        "evaluation": {
            "stage2": {
                "protocol": args.eval_protocol,
                "output_root": str(args.out_root),
                "run_name": args.run_name,
                "seed": int(args.seed),
                "device": args.device,
                "data": {
                    "split": args.split,
                    "num_conditions": args.num_conditions,
                    "min_sequence_frames": args.min_sequence_frames,
                    "disable_text_filter": bool(getattr(args, "disable_text_filter", False)),
                    "batch_size": int(args.batch_size),
                    "num_workers": int(args.num_workers),
                },
                "sampler": {
                    "requested_frames": int(args.requested_frames),
                    "official_requested_frames": "min(sequence_length,300)" if args.eval_protocol == "interact_official_v1" else None,
                    "requested_frames_list": [int(v) for v in args.requested_frames_list],
                    "samples_per_condition": int(args.samples_per_condition),
                    "multimodality_samples": int(args.multimodality_samples),
                    "multimodality_pairs_per_condition": args.multimodality_pairs_per_condition,
                    "best_of_k": int(args.best_of_k),
                    "use_ema": bool(args.use_ema),
                    "solver": args.solver,
                    "steps_per_block": int(args.steps_per_block),
                    "window_blocks": int(args.window_blocks),
                    "clean_history_blocks": int(args.clean_history_blocks),
                },
                "guidance": {
                    "mode": args.guidance_mode,
                    "joint_scale": float(args.joint_scale),
                    "text_scale": float(args.text_scale),
                    "geometry_scale": float(args.geometry_scale),
                },
                "decode": {"mode": args.decode},
                "feature_evaluator": {
                    "mode": args.feature_evaluator_mode,
                    "artifact_root": str(args.feature_artifact_root),
                    "require_parity": _feature_required(args),
                    "batch_size": int(args.feature_batch_size),
                },
            }
        },
    }


def _artifact_manifest(
    args: argparse.Namespace,
    *,
    records_dir: Path,
    feature_artifacts: dict[str, Any],
    split_audit,
    marker_bps_alignment,
) -> dict[str, Any]:
    stage2_ckpt = _checkpoint_path(args.stage2_run, args.stage2_ckpt)
    stage1_ckpt = _checkpoint_path(args.source_run, args.source_ckpt)
    return {
        "decoded_records_dir": str(records_dir),
        "decoded_records_dir_sha256": _sha256_optional(records_dir),
        "stage2_checkpoint": str(stage2_ckpt),
        "stage2_checkpoint_sha256": _sha256_optional(stage2_ckpt),
        "stage1_checkpoint": str(stage1_ckpt),
        "stage1_checkpoint_sha256": _sha256_optional(stage1_ckpt),
        "whitening": str(args.whitening),
        "whitening_sha256": _sha256_optional(args.whitening),
        "template_spec": None if args.template_spec is None else str(args.template_spec),
        "template_spec_hash": _sha256_optional(args.template_spec),
        "feature_evaluator": feature_artifacts,
        "split_audit": None if split_audit is None else split_audit.to_dict(),
        "marker_bps_alignment": None if marker_bps_alignment is None else marker_bps_alignment.to_dict(),
    }


def _checkpoint_path(run_dir: Path, ckpt_name: str | Path) -> Path:
    ckpt = Path(ckpt_name)
    return ckpt if ckpt.exists() else Path(run_dir) / "checkpoints" / ckpt


def _sha256_optional(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.is_dir():
        import hashlib

        digest = hashlib.sha256()
        for child in sorted(item for item in p.rglob("*") if item.is_file()):
            digest.update(child.relative_to(p).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_file(child).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
    return sha256_file(p)


def _safe_load_yaml(path: Path) -> Any:
    return load_yaml(path) if Path(path).exists() else None


def _git_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "data_config_path": str(args.data_config),
        "train_config_path": str(args.train_config),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
    }


def _git_output(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *cmd], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _meta_value(meta: Any, key: str, index: int) -> Any:
    if not isinstance(meta, Mapping) or key not in meta:
        return None
    value = meta[key]
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    if hasattr(value, "detach"):
        arr = value.detach().cpu()
        return arr[index].item() if arr.ndim > 0 else arr.item()
    if hasattr(value, "ndim"):
        return value[index].item() if value.ndim > 0 else value.item()
    return value


def _none_or_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _make_sample_id(dataset: Any, sequence: Any, condition_index: int, sample_index: int) -> str:
    parts = [str(value) for value in (dataset, sequence) if value is not None]
    parts.extend([f"cond{int(condition_index):06d}", f"sample{int(sample_index):02d}"])
    return "/".join(parts)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().numpy())
    return value


if __name__ == "__main__":
    main()
