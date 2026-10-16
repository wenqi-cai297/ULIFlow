from __future__ import annotations

from collections.abc import Iterator
import json
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hoi_latent.metrics.dynamics import (
    boundary_jump_cm,
    foot_skate_stance_cm_per_frame,
    foot_sliding_interact_mm,
    mean_acceleration_cm_per_frame2,
    mean_jerk_cm_per_frame3,
)
from hoi_latent.metrics.interaction import contact_metrics
from hoi_latent.metrics.reconstruction import (
    marker_mpjpe_cm,
    marker_velocity_error_cm_per_frame,
    object_rotation_geodesic_deg,
    object_translation_error_cm,
)
from hoi_latent.metrics.relation import marker_object_distance_error_cm
from hoi_latent.data.text_utils import clean_interact_caption

from .decoded_records import Stage2DecodedSampleRecord, read_stage2_decoded_record
from .feature_metrics import compute_stage2_feature_metrics


@dataclass
class Stage2QuantitativeEvaluationResult:
    protocol: str
    records: list[Stage2DecodedSampleRecord]
    paired_metrics: dict[str, float] = field(default_factory=dict)
    interaction_metrics: dict[str, float] = field(default_factory=dict)
    stage1_ceiling_metrics: dict[str, float] = field(default_factory=dict)
    feature_metrics: dict[str, float] = field(default_factory=dict)
    rollout_metrics: dict[str, float] = field(default_factory=dict)
    per_sample_metrics: list[dict[str, Any]] = field(default_factory=list)
    availability: dict[str, str] = field(default_factory=dict)
    comparability: dict[str, Any] = field(default_factory=dict)

    @property
    def summary_metrics(self) -> dict[str, float]:
        merged: dict[str, float] = {}
        for group in (
            self.paired_metrics,
            self.interaction_metrics,
            self.stage1_ceiling_metrics,
            self.feature_metrics,
            self.rollout_metrics,
        ):
            merged.update(group)
        return merged


def read_stage2_decoded_records_from_dir(records_dir: str | Path) -> list[Stage2DecodedSampleRecord]:
    return list(iter_stage2_decoded_records_from_dir(records_dir))


def iter_stage2_decoded_records_from_dir(records_dir: str | Path) -> Iterator[Stage2DecodedSampleRecord]:
    root = Path(records_dir)
    if not root.exists():
        raise FileNotFoundError(f"decoded records directory does not exist: {root}")
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise ValueError(f"decoded records directory contains no .npz files: {root}")
    for path in paths:
        yield read_stage2_decoded_record(path)


def evaluate_stage2_decoded_records(
    records: Sequence[Stage2DecodedSampleRecord],
    *,
    protocol: str,
    feature_evaluator: Any | None = None,
    feature_enabled: bool = False,
    feature_required: bool = False,
    paired_enabled: bool = True,
    interaction_enabled: bool = True,
    stage1_ceiling_enabled: bool = True,
    r_precision_pool_size: int = 32,
    r_precision_seed: int = 0,
    block_size_frames: int = 32,
    feature_metrics_mode: str | None = None,
    feature_parity_passed: bool | None = None,
    feature_parity_golden_source: str | None = None,
    feature_artifact_manifest_valid: bool | None = None,
    split_audit_passed: bool | None = None,
    marker_bps_alignment_passed: bool | None = None,
    multimodality_min_samples: int = 2,
    multimodality_pairs_per_condition: int | None = None,
    feature_batch_size: int | None = None,
) -> Stage2QuantitativeEvaluationResult:
    records_list = list(records)
    if not records_list:
        raise ValueError("records must not be empty.")

    per_sample: list[dict[str, Any]] = []
    paired_rows = []
    interaction_rows = []
    ceiling_rows = []
    rollout_rows = []
    for record in records_list:
        row: dict[str, Any] = {"sample_id": record.sample_id}
        paired = compute_stage2_paired_metrics(record) if paired_enabled else {}
        interaction = compute_stage2_interaction_metrics(record, block_size_frames=block_size_frames) if interaction_enabled else {}
        ceiling = compute_stage2_stage1_ceiling_metrics(record) if stage1_ceiling_enabled else {}
        rollout = compute_stage2_rollout_metrics(record)
        row.update(paired)
        row.update(interaction)
        row.update(ceiling)
        row.update(rollout)
        per_sample.append(row)
        if paired:
            paired_rows.append(paired)
        if interaction:
            interaction_rows.append(interaction)
        if ceiling:
            ceiling_rows.append(ceiling)
        if rollout:
            rollout_rows.append(rollout)

    availability: dict[str, str] = {
        "paired_metrics": "available" if paired_rows else ("disabled" if not paired_enabled else "unavailable_missing_gt"),
        "interaction_metrics": "available" if interaction_rows else ("disabled" if not interaction_enabled else "unavailable_missing_gt"),
        "stage1_ceiling_metrics": "available" if ceiling_rows else (
            "disabled" if not stage1_ceiling_enabled else "unavailable_missing_stage1_ceiling"
        ),
        "penetration": "unavailable_no_sdf",
    }
    feature_values: dict[str, float] = {}
    if not feature_enabled:
        availability["feature_metrics"] = "disabled"
    elif feature_evaluator is None:
        availability["feature_metrics"] = "unavailable_missing_evaluator"
        if feature_required:
            raise RuntimeError("feature metrics are required, but no feature evaluator was provided.")
    else:
        feature_values = compute_stage2_feature_metrics_for_records(
            records_list,
            feature_evaluator=feature_evaluator,
            r_precision_pool_size=r_precision_pool_size,
            r_precision_seed=r_precision_seed,
            multimodality_min_samples=int(multimodality_min_samples),
            multimodality_pairs_per_condition=multimodality_pairs_per_condition,
            feature_batch_size=feature_batch_size,
        )
        availability["feature_metrics"] = "available" if feature_values else "unavailable_missing_gt"
        if feature_required and not feature_values:
            raise RuntimeError("feature metrics are required, but no GT-paired records were available.")
    comparability = _feature_comparability_status(
        protocol=str(protocol),
        feature_metrics_mode=feature_metrics_mode,
        feature_enabled=feature_enabled,
        feature_parity_passed=feature_parity_passed,
        feature_parity_golden_source=feature_parity_golden_source,
        feature_artifact_manifest_valid=feature_artifact_manifest_valid,
        split_audit_passed=split_audit_passed,
        marker_bps_alignment_passed=marker_bps_alignment_passed,
        feature_values=feature_values,
    )

    return Stage2QuantitativeEvaluationResult(
        protocol=str(protocol),
        records=records_list,
        paired_metrics=_mean_dicts(paired_rows),
        interaction_metrics=_mean_dicts(interaction_rows),
        stage1_ceiling_metrics=_mean_dicts(ceiling_rows),
        feature_metrics=feature_values,
        rollout_metrics=_mean_dicts(rollout_rows),
        per_sample_metrics=per_sample,
        availability=availability,
        comparability=comparability,
    )


def compute_stage2_paired_metrics(record: Stage2DecodedSampleRecord) -> dict[str, float]:
    if record.gt is None:
        return {}
    gen = record.generated
    gt = record.gt
    limit = int(record.requested_frames)
    frame_mask = _joint_frame_mask(gen.frame_mask, gt.frame_mask, limit)[:limit]
    gen_markers = gen.human_markers[:limit]
    gt_markers = gt.human_markers[:limit]
    gt_object_points = gt.object_points[:limit]
    out = {
        "paired/marker_mpjpe_cm": marker_mpjpe_cm(gen_markers, gt_markers, frame_mask),
        "paired/marker_velocity_error_cm_per_frame": marker_velocity_error_cm_per_frame(
            gen_markers,
            gt_markers,
            frame_mask,
        ),
        "paired/marker_object_distance_error_cm": marker_object_distance_error_cm(
            gen_markers,
            gt_markers,
            gt_object_points,
            frame_mask,
        ),
    }
    if gen.object_trans is not None and gt.object_trans is not None:
        out["paired/object_translation_error_cm"] = object_translation_error_cm(gen.object_trans[:limit], gt.object_trans[:limit], frame_mask)
    if gen.object_rotmat is not None and gt.object_rotmat is not None:
        out["paired/object_rotation_geodesic_deg"] = object_rotation_geodesic_deg(gen.object_rotmat[:limit], gt.object_rotmat[:limit], frame_mask)
    return out


def compute_stage2_stage1_ceiling_metrics(record: Stage2DecodedSampleRecord) -> dict[str, float]:
    if record.gt is None or record.stage1_ceiling is None:
        return {}
    ceiling = record.stage1_ceiling
    gt = record.gt
    limit = int(record.requested_frames)
    frame_mask = _joint_frame_mask(ceiling.frame_mask, gt.frame_mask, limit)[:limit]
    ceiling_markers = ceiling.human_markers[:limit]
    gt_markers = gt.human_markers[:limit]
    gt_object_points = gt.object_points[:limit]
    out = {
        "stage1_ceiling/marker_mpjpe_cm": marker_mpjpe_cm(ceiling_markers, gt_markers, frame_mask),
        "stage1_ceiling/marker_velocity_error_cm_per_frame": marker_velocity_error_cm_per_frame(
            ceiling_markers,
            gt_markers,
            frame_mask,
        ),
        "stage1_ceiling/marker_object_distance_error_cm": marker_object_distance_error_cm(
            ceiling_markers,
            gt_markers,
            gt_object_points,
            frame_mask,
        ),
    }
    if ceiling.object_trans is not None and gt.object_trans is not None:
        out["stage1_ceiling/object_translation_error_cm"] = object_translation_error_cm(
            ceiling.object_trans[:limit],
            gt.object_trans[:limit],
            frame_mask,
        )
    if ceiling.object_rotmat is not None and gt.object_rotmat is not None:
        out["stage1_ceiling/object_rotation_geodesic_deg"] = object_rotation_geodesic_deg(
            ceiling.object_rotmat[:limit],
            gt.object_rotmat[:limit],
            frame_mask,
        )
    return out


def compute_stage2_interaction_metrics(
    record: Stage2DecodedSampleRecord,
    *,
    block_size_frames: int = 32,
) -> dict[str, float]:
    gen = record.generated
    limit = int(record.requested_frames)
    frame_mask = _prefix_frame_mask(gen.frame_mask, limit)[:limit]
    gen_markers = gen.human_markers[:limit]
    out = {
        "interaction/foot_sliding_interact_mm": foot_sliding_interact_mm(
            gen_markers,
            frame_mask=frame_mask,
            fps=record.fps,
        )["foot_sliding_interact_mm"],
        "interaction/mean_acceleration_cm_per_frame2": mean_acceleration_cm_per_frame2(gen_markers, frame_mask),
        "interaction/mean_jerk_cm_per_frame3": mean_jerk_cm_per_frame3(gen_markers, frame_mask),
    }
    out.update({f"interaction/{key}": value for key, value in boundary_jump_cm(gen_markers, block_size_frames=block_size_frames, frame_mask=frame_mask).items()})
    if record.gt is not None:
        gt = record.gt
        paired_mask = _joint_frame_mask(gen.frame_mask, gt.frame_mask, limit)[:limit]
        gt_markers = gt.human_markers[:limit]
        gt_object_points = gt.object_points[:limit]
        # Use GT object points for paired contact to isolate human contact behavior.
        out.update(
            {
                f"interaction/{key}": value
                for key, value in contact_metrics(
                    gen_markers,
                    gt_markers,
                    gt_object_points,
                    frame_mask=paired_mask,
                ).items()
            }
        )
        out.update(
            {
                f"interaction/{key}": value
                for key, value in foot_skate_stance_cm_per_frame(
                    gen_markers,
                    gt_markers,
                    frame_mask=paired_mask,
                ).items()
            }
        )
    return out


def compute_stage2_rollout_metrics(record: Stage2DecodedSampleRecord) -> dict[str, float]:
    out: dict[str, float] = {}
    update_counts = _numeric_array(record.rollout_metadata.get("update_counts"))
    if update_counts is not None and update_counts.size:
        out["rollout/update_count_min"] = float(update_counts.min())
        out["rollout/update_count_max"] = float(update_counts.max())
        out["rollout/update_count_mean"] = float(update_counts.mean())
    static_count = _numeric_array(record.rollout_metadata.get("static_update_count"))
    if static_count is not None and static_count.size:
        out["rollout/static_update_count_mean"] = float(static_count.mean())
    for key in ("steps_per_block", "num_blocks", "num_requested_frames"):
        value = _numeric_array(record.rollout_metadata.get(key))
        if value is not None and value.size:
            out[f"rollout/{key}"] = float(value.mean())
    return out


def compute_stage2_feature_metrics_for_records(
    records: Sequence[Stage2DecodedSampleRecord],
    *,
    feature_evaluator: Any,
    r_precision_pool_size: int = 32,
    r_precision_seed: int = 0,
    multimodality_min_samples: int = 2,
    multimodality_pairs_per_condition: int | None = None,
    feature_batch_size: int | None = None,
) -> dict[str, float]:
    paired = [record for record in records if record.gt is not None]
    if not paired:
        return {}
    gt_samples = [record.gt for record in paired if record.gt is not None]
    gen_samples = [record.generated for record in paired]
    real = _encode_motion_features_for_samples_batched(
        gt_samples,
        feature_evaluator=feature_evaluator,
        batch_size=feature_batch_size,
    )
    generated = _encode_motion_features_for_samples_batched(
        gen_samples,
        feature_evaluator=feature_evaluator,
        batch_size=feature_batch_size,
    )
    texts = [clean_interact_caption(record.text) or "" for record in paired]
    text_features = _encode_text_features_batched(
        texts,
        feature_evaluator=feature_evaluator,
        batch_size=feature_batch_size,
    ) if texts else None
    multimodal = _multimodal_feature_array(paired, generated, min_samples=int(multimodality_min_samples))
    values = compute_stage2_feature_metrics(
        real_motion_features=real,
        generated_motion_features=generated,
        text_features=text_features,
        multimodal_generated_features=multimodal,
        r_precision_pool_size=r_precision_pool_size,
        r_precision_seed=r_precision_seed,
        multimodality_pairs_per_condition=multimodality_pairs_per_condition,
    ).values
    if text_features is not None:
        # Real/GT retrieval ceiling (InterAct "Real" row): run R-precision and
        # matching-score on the GT motion features themselves, reusing the exact
        # generated pooling path by treating `real` as the prediction.
        real_self = compute_stage2_feature_metrics(
            real_motion_features=real,
            generated_motion_features=real,
            text_features=text_features,
            r_precision_pool_size=r_precision_pool_size,
            r_precision_seed=r_precision_seed,
        ).values
        for _key in ("r_precision_top1", "r_precision_top2", "r_precision_top3", "mm_dist"):
            _source = f"feature/{_key}"
            if _source in real_self:
                values[f"feature/real_{_key}"] = real_self[_source]
    ceiling_samples = [record.stage1_ceiling for record in paired if record.stage1_ceiling is not None]
    if len(ceiling_samples) == len(paired):
        ceiling_features = _encode_motion_features_for_samples_batched(
            ceiling_samples,
            feature_evaluator=feature_evaluator,
            batch_size=feature_batch_size,
        )
        ceiling_values = compute_stage2_feature_metrics(
            real_motion_features=real,
            generated_motion_features=ceiling_features,
            text_features=text_features,
            r_precision_pool_size=r_precision_pool_size,
            r_precision_seed=r_precision_seed,
        ).values
        values.update(_stage1_ceiling_feature_metric_keys(ceiling_values))
        if "feature/fid" in values and "feature/stage1_ceiling_fid" in values:
            values["feature/fid_generation_gap"] = values["feature/fid"] - values["feature/stage1_ceiling_fid"]
    return values


def _encode_motion_features_for_samples_batched(
    samples: Sequence[Any],
    *,
    feature_evaluator: Any,
    batch_size: int | None,
) -> np.ndarray:
    if not samples:
        return np.empty((0, 0), dtype=np.float32)
    features: list[np.ndarray] = []
    for chunk in _chunk_sequence(samples, batch_size):
        preprocess = feature_evaluator.preprocess(chunk)
        encoded = feature_evaluator.encode_motion(preprocess.normalized_features, preprocess.lengths)
        features.append(encoded.detach().cpu().numpy())
        del encoded, preprocess
        _release_torch_cache()
    return np.concatenate(features, axis=0)


def _encode_text_features_batched(
    texts: Sequence[str],
    *,
    feature_evaluator: Any,
    batch_size: int | None,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    features: list[np.ndarray] = []
    for chunk in _chunk_sequence(texts, batch_size):
        encoded = feature_evaluator.encode_text(chunk)
        features.append(encoded.detach().cpu().numpy())
        del encoded
        _release_torch_cache()
    return np.concatenate(features, axis=0)


def _chunk_sequence(values: Sequence[Any], batch_size: int | None) -> list[Sequence[Any]]:
    if batch_size is None or int(batch_size) <= 0 or int(batch_size) >= len(values):
        return [values]
    size = int(batch_size)
    return [values[start : start + size] for start in range(0, len(values), size)]


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def write_stage2_quantitative_outputs(
    result: Stage2QuantitativeEvaluationResult,
    run_dir: str | Path,
    *,
    artifacts: Mapping[str, Any] | None = None,
    resolved_config: Mapping[str, Any] | None = None,
) -> None:
    root = Path(run_dir)
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if resolved_config is not None:
        _write_json(root / "resolved_config.json", resolved_config)
    if artifacts is not None:
        artifact_payload = dict(artifacts)
        artifact_payload["comparability"] = result.comparability
        _write_json(root / "artifacts.json", artifact_payload)
    _write_json(metrics_dir / "paired_metrics.json", result.paired_metrics)
    _write_json(metrics_dir / "interaction_metrics.json", result.interaction_metrics)
    _write_json(metrics_dir / "stage1_ceiling_metrics.json", result.stage1_ceiling_metrics)
    _write_json(metrics_dir / "feature_metrics.json", result.feature_metrics)
    _write_json(metrics_dir / "rollout_metrics.json", result.rollout_metrics)
    with (metrics_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in result.per_sample_metrics:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    _write_json(
        metrics_dir / "summary.json",
        {
            "protocol": result.protocol,
            "num_conditions": len({record.condition_index for record in result.records}),
            "num_records": len(result.records),
            "samples_per_condition": _samples_per_condition(result.records),
            "requested_frames": sorted({int(record.requested_frames) for record in result.records}),
            "metrics": result.summary_metrics,
            "availability": result.availability,
            "comparability": result.comparability,
        },
    )


def _stage1_ceiling_feature_metric_keys(values: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    mapping = {
        "feature/fid": "feature/stage1_ceiling_fid",
        "feature/mm_dist": "feature/stage1_ceiling_mm_dist",
        "feature/r_precision_top1": "feature/stage1_ceiling_r_precision_top1",
        "feature/r_precision_top2": "feature/stage1_ceiling_r_precision_top2",
        "feature/r_precision_top3": "feature/stage1_ceiling_r_precision_top3",
        "feature/generated_diversity": "feature/stage1_ceiling_diversity",
        "feature/generated_pair_count": "feature/stage1_ceiling_pair_count",
    }
    for source, target in mapping.items():
        if source in values:
            out[target] = float(values[source])
    return out


def _feature_comparability_status(
    *,
    protocol: str,
    feature_metrics_mode: str | None,
    feature_enabled: bool,
    feature_parity_passed: bool | None,
    feature_parity_golden_source: str | None,
    feature_artifact_manifest_valid: bool | None,
    split_audit_passed: bool | None,
    marker_bps_alignment_passed: bool | None,
    feature_values: Mapping[str, float],
) -> dict[str, Any]:
    mode = str(feature_metrics_mode or ("disabled" if not feature_enabled else "unknown"))
    blocking: list[str] = []
    if not feature_enabled:
        blocking.append("feature_metrics_disabled")
    elif mode != "official_artifact":
        blocking.append("feature_evaluator_not_official_artifact")
    if protocol == "eval_128_dev":
        blocking.append("fixed_128_internal_protocol")
    elif protocol == "eval_long_v1":
        blocking.append("eval_long_feature_protocol_not_official_comparable")
    elif protocol == "interact_official_v1":
        if split_audit_passed is not True:
            blocking.append("split_audit_missing" if split_audit_passed is None else "split_audit_failed")
        if marker_bps_alignment_passed is not True:
            blocking.append(
                "marker_bps_alignment_missing"
                if marker_bps_alignment_passed is None
                else "marker_bps_alignment_failed"
            )
    else:
        blocking.append(f"unsupported_official_protocol:{protocol}")
    if feature_enabled and mode == "official_artifact" and feature_parity_passed is not True:
        blocking.append("missing_external_golden_parity" if feature_parity_passed is None else "failed_external_golden_parity")
    if (
        feature_enabled
        and mode == "official_artifact"
        and feature_parity_passed is True
        and protocol != "eval_128_dev"
        and feature_parity_golden_source != "external"
    ):
        if feature_parity_golden_source is None:
            blocking.append("missing_external_golden_source")
        else:
            blocking.append(f"non_external_golden_source:{feature_parity_golden_source}")
    if (
        feature_enabled
        and mode == "official_artifact"
        and protocol != "eval_128_dev"
        and feature_artifact_manifest_valid is not True
    ):
        blocking.append("missing_or_invalid_feature_artifact_manifest")
    if feature_enabled and "feature/stage1_ceiling_fid" not in feature_values:
        blocking.append("missing_stage1_ceiling_feature_fid")
    return {
        "feature_metrics_mode": mode,
        "protocol": str(protocol),
        "feature_parity_golden_source": feature_parity_golden_source,
        "feature_artifact_manifest_valid": feature_artifact_manifest_valid,
        "split_audit_passed": split_audit_passed,
        "marker_bps_alignment_passed": marker_bps_alignment_passed,
        "interact_comparable": len(blocking) == 0,
        "blocking_reasons": sorted(set(blocking)),
    }


def _mean_dicts(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                totals.setdefault(str(key), []).append(float(value))
    return {key: float(np.mean(values)) for key, values in sorted(totals.items()) if values}


def _joint_frame_mask(a: np.ndarray, b: np.ndarray, requested_frames: int) -> np.ndarray:
    limit = int(requested_frames)
    return _prefix_frame_mask(a, limit)[:limit] & _prefix_frame_mask(b, limit)[:limit]


def _prefix_frame_mask(mask: np.ndarray, requested_frames: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 1:
        raise ValueError(f"frame_mask must have shape [T], got {arr.shape}")
    limit = min(int(requested_frames), arr.shape[0])
    out = np.zeros(arr.shape[0], dtype=bool)
    out[:limit] = arr[:limit]
    return out


def _numeric_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def _multimodal_feature_array(
    records: Sequence[Stage2DecodedSampleRecord],
    features: np.ndarray,
    *,
    min_samples: int,
) -> np.ndarray | None:
    by_condition: dict[int, list[int]] = {}
    for idx, record in enumerate(records):
        by_condition.setdefault(int(record.condition_index), []).append(idx)
    sizes = {len(indices) for indices in by_condition.values()}
    required = max(2, int(min_samples))
    if len(sizes) != 1 or next(iter(sizes)) < required:
        return None
    grouped = [features[indices] for _, indices in sorted(by_condition.items())]
    return np.stack(grouped, axis=0)


def _samples_per_condition(records: Sequence[Stage2DecodedSampleRecord]) -> int | None:
    counts: dict[int, int] = {}
    for record in records:
        counts[int(record.condition_index)] = counts.get(int(record.condition_index), 0) + 1
    if not counts:
        return None
    values = set(counts.values())
    return next(iter(values)) if len(values) == 1 else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().numpy())
    return value
