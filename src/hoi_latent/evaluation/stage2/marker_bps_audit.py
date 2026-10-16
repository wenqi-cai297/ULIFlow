from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import permutations, product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .feature_preprocess import preprocess_interact_marker_bps_samples
from .interact_feature_evaluator import _samples_from_fixture_inputs


@dataclass(frozen=True)
class MarkerBPSAlignmentReport:
    fixture_inputs: str
    external_features: str
    marker_world_max_abs: float | None
    object_points_world_max_abs: float | None
    bps_raw_feature_max_abs: float
    normalized_feature_max_abs: float
    tolerance: float
    unit_scale_guess: float | None
    axis_flip_detected: bool
    passed: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_marker_bps_alignment_report(report: MarkerBPSAlignmentReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_marker_bps_alignment_report(path: str | Path) -> MarkerBPSAlignmentReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MarkerBPSAlignmentReport(
        fixture_inputs=str(payload.get("fixture_inputs", "")),
        external_features=str(payload.get("external_features", "")),
        marker_world_max_abs=_optional_float(payload.get("marker_world_max_abs")),
        object_points_world_max_abs=_optional_float(payload.get("object_points_world_max_abs")),
        bps_raw_feature_max_abs=float(payload.get("bps_raw_feature_max_abs", float("inf"))),
        normalized_feature_max_abs=float(payload.get("normalized_feature_max_abs", float("inf"))),
        tolerance=float(payload.get("tolerance", 0.0)),
        unit_scale_guess=_optional_float(payload.get("unit_scale_guess")),
        axis_flip_detected=bool(payload.get("axis_flip_detected", False)),
        passed=bool(payload.get("passed", False)),
        notes=list(payload.get("notes", [])),
    )


def audit_marker_bps_representation_from_external_features(
    *,
    fixture_inputs: str | Path,
    external_features: str | Path,
    mean: np.ndarray,
    std: np.ndarray,
    bps_basis: np.ndarray,
    tolerance: float = 1.0e-5,
) -> MarkerBPSAlignmentReport:
    arrays = compute_marker_bps_alignment_arrays(
        fixture_inputs=fixture_inputs,
        external_features=external_features,
        mean=mean,
        std=std,
        bps_basis=bps_basis,
    )
    fixture_path = Path(fixture_inputs)
    external_path = Path(external_features)
    raw = arrays["repo_raw_feature_615"]
    normalized = arrays["repo_normalized_feature_615"]
    raw_ref = arrays["external_raw_feature_615"]
    norm_ref = arrays["external_normalized_feature_615"]
    bps_raw_feature_max_abs = _max_abs(raw - raw_ref)
    normalized_feature_max_abs = _max_abs(normalized - norm_ref)

    marker_world_max_abs = None
    if "fixture_human_markers_world" in arrays and "external_human_markers_world" in arrays:
        marker_world_max_abs = _max_abs(arrays["fixture_human_markers_world"] - arrays["external_human_markers_world"])
    object_points_world_max_abs = None
    if "repo_object_points_world" in arrays and "external_object_points_world" in arrays:
        object_points_world_max_abs = _max_abs(arrays["repo_object_points_world"] - arrays["external_object_points_world"])

    passed = bps_raw_feature_max_abs <= tolerance and normalized_feature_max_abs <= tolerance
    notes = []
    if marker_world_max_abs is None:
        notes.append("external_features did not include human_markers_world; marker_world_max_abs unavailable")
    if object_points_world_max_abs is None:
        notes.append("external_features did not include object_points_world; object_points_world_max_abs unavailable")
    axis_flip_detected = _axis_permutation_or_flip_detected(
        raw,
        raw_ref,
        baseline_max_abs=bps_raw_feature_max_abs,
        tolerance=float(tolerance),
    )
    if axis_flip_detected:
        notes.append("raw 615D feature mismatch is strongly explained by an xyz axis permutation/sign flip candidate")
    return MarkerBPSAlignmentReport(
        fixture_inputs=str(fixture_path),
        external_features=str(external_path),
        marker_world_max_abs=marker_world_max_abs,
        object_points_world_max_abs=object_points_world_max_abs,
        bps_raw_feature_max_abs=bps_raw_feature_max_abs,
        normalized_feature_max_abs=normalized_feature_max_abs,
        tolerance=float(tolerance),
        unit_scale_guess=_unit_scale_guess(raw, raw_ref),
        axis_flip_detected=axis_flip_detected,
        passed=bool(passed),
        notes=notes,
    )


def compute_marker_bps_alignment_arrays(
    *,
    fixture_inputs: str | Path,
    external_features: str | Path,
    mean: np.ndarray,
    std: np.ndarray,
    bps_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    fixture_path = Path(fixture_inputs)
    external_path = Path(external_features)
    with np.load(fixture_path) as fixture_payload:
        fixture = {key: fixture_payload[key] for key in fixture_payload.files}
    with np.load(external_path) as external_payload:
        external = {key: external_payload[key] for key in external_payload.files}

    samples = _samples_from_fixture_inputs(fixture)
    pre = preprocess_interact_marker_bps_samples(
        samples,
        mean=mean,
        std=std,
        bps_basis=bps_basis,
    )
    arrays: dict[str, np.ndarray] = {
        "repo_raw_feature_615": pre.raw_features.detach().cpu().numpy(),
        "repo_normalized_feature_615": pre.normalized_features.detach().cpu().numpy(),
        "external_raw_feature_615": _require_external_feature(external, "raw_feature_615"),
        "external_normalized_feature_615": _require_external_feature(external, "normalized_feature_615"),
    }
    if "human_markers_world" in fixture:
        arrays["fixture_human_markers_world"] = np.asarray(fixture["human_markers_world"], dtype=np.float32)
    if "human_markers_world" in external:
        arrays["external_human_markers_world"] = np.asarray(external["human_markers_world"], dtype=np.float32)
    if "object_points_world" in external:
        arrays["repo_object_points_world"] = np.stack(
            [np.asarray(sample.object_points, dtype=np.float32) for sample in samples],
            axis=0,
        )
        arrays["external_object_points_world"] = np.asarray(external["object_points_world"], dtype=np.float32)
    return arrays


def _require_external_feature(payload: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"external feature file is missing {key!r}.")
    value = np.asarray(payload[key], dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 615:
        raise ValueError(f"{key} must have shape [B,T,615], got {value.shape}.")
    return value


def _max_abs(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _unit_scale_guess(raw: np.ndarray, reference: np.ndarray) -> float | None:
    lhs = np.asarray(raw, dtype=np.float32).reshape(-1)
    rhs = np.asarray(reference, dtype=np.float32).reshape(-1)
    mask = np.abs(lhs) > 1.0e-8
    if not np.any(mask):
        return None
    ratios = rhs[mask] / lhs[mask]
    ratios = ratios[np.isfinite(ratios)]
    return None if ratios.size == 0 else float(np.median(ratios))


def _axis_permutation_or_flip_detected(
    raw: np.ndarray,
    reference: np.ndarray,
    *,
    baseline_max_abs: float,
    tolerance: float,
) -> bool:
    lhs = np.asarray(raw, dtype=np.float32)
    rhs = np.asarray(reference, dtype=np.float32)
    if lhs.shape != rhs.shape or lhs.ndim < 1 or lhs.shape[-1] % 3 != 0:
        return False
    if baseline_max_abs <= tolerance:
        return False

    triples = lhs.reshape(*lhs.shape[:-1], lhs.shape[-1] // 3, 3)
    identity_perm = (0, 1, 2)
    identity_signs = (1.0, 1.0, 1.0)
    best = float("inf")
    for perm in permutations(identity_perm):
        for signs in product((-1.0, 1.0), repeat=3):
            if perm == identity_perm and signs == identity_signs:
                continue
            candidate = triples[..., list(perm)] * np.asarray(signs, dtype=np.float32)
            best = min(best, _max_abs(candidate.reshape(lhs.shape) - rhs))
    return bool(best <= tolerance or best <= baseline_max_abs * 0.1)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
