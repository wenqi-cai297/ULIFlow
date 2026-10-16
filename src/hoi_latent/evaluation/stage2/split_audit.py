from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from hoi_latent.data.datasets.interact_stage1 import build_interact_stage1_index
from hoi_latent.data.schemas import InterActSequenceRecord
from hoi_latent.data.splits.interact_splits import InterActSplitAssigner, match_official_test_name

from .decoded_records import Stage2DecodedSampleRecord


@dataclass(frozen=True)
class SplitSourceAudit:
    run: str
    config_path: str | None
    audit_source: str
    use_official_test_seq: bool
    official_test_count: int
    matched_test_count: int
    unmatched_official_names: list[str]
    training_overlap_count: int
    overlap_sequence_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InterActSplitAuditReport:
    test_seq_file: str | None
    official_test_count: int
    eval_split: str
    eval_sequence_count: int
    eval_non_test_count: int
    eval_non_official_count: int
    stage1_source: SplitSourceAudit
    stage2_source: SplitSourceAudit
    passed: bool
    notes: list[str] = field(default_factory=list)

    @property
    def stage1_training_overlap_count(self) -> int:
        return int(self.stage1_source.training_overlap_count)

    @property
    def stage2_training_overlap_count(self) -> int:
        return int(self.stage2_source.training_overlap_count)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage1_training_overlap_count"] = self.stage1_training_overlap_count
        payload["stage2_training_overlap_count"] = self.stage2_training_overlap_count
        return payload


def write_interact_split_audit_report(report: InterActSplitAuditReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_interact_split_audit_report(path: str | Path) -> InterActSplitAuditReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return InterActSplitAuditReport(
        test_seq_file=payload.get("test_seq_file"),
        official_test_count=int(payload.get("official_test_count", 0)),
        eval_split=str(payload.get("eval_split", "")),
        eval_sequence_count=int(payload.get("eval_sequence_count", 0)),
        eval_non_test_count=int(payload.get("eval_non_test_count", 0)),
        eval_non_official_count=int(payload.get("eval_non_official_count", 0)),
        stage1_source=_source_from_dict(payload["stage1_source"]),
        stage2_source=_source_from_dict(payload["stage2_source"]),
        passed=bool(payload.get("passed", False)),
        notes=list(payload.get("notes", [])),
    )


def audit_interact_official_split_no_leak(
    *,
    eval_records: Sequence[Stage2DecodedSampleRecord],
    eval_split: str,
    stage1_config: Mapping[str, Any],
    stage2_config: Mapping[str, Any],
    stage1_source_run: str | Path,
    stage2_run: str | Path,
) -> InterActSplitAuditReport:
    """Reconstruct InterAct split assignment and check official-test leakage.

    Current training runs do not persist exact sequence manifests, so this audit
    reconstructs assignments from each run's resolved config and the dataset
    metadata. The report records that provenance explicitly.
    """
    stage1_source = audit_interact_split_source(
        config=stage1_config,
        run=stage1_source_run,
        config_path=_config_path(stage1_source_run),
    )
    stage2_source = audit_interact_split_source(
        config=stage2_config,
        run=stage2_run,
        config_path=_config_path(stage2_run),
    )
    official_names = _official_names_from_config(stage2_config) or _official_names_from_config(stage1_config)
    eval_sequence_names = {str(record.sequence_name) for record in eval_records if record.sequence_name}
    eval_non_test = [record.sample_id for record in eval_records if str(record.split) != "test"]
    eval_non_official = sorted(
        name for name in eval_sequence_names if official_names and not _matches_official_name(name, official_names)
    )
    notes = [
        "training manifests were not available; split assignment was reconstructed from resolved configs",
        "reconstruction uses CURRENT split code; checkpoints trained before a split fix are NOT re-audited here",
    ]
    full_coverage = (
        stage1_source.matched_test_count == stage1_source.official_test_count
        and stage2_source.matched_test_count == stage2_source.official_test_count
    )
    if not full_coverage:
        notes.append(
            "official test coverage is incomplete (matched_test_count < official_test_count); "
            "evaluation does not span the full official test split"
        )
    passed = (
        str(eval_split) == "test"
        and len(eval_records) > 0
        and len(eval_sequence_names) > 0
        and not eval_non_test
        and not eval_non_official
        and stage1_source.training_overlap_count == 0
        and stage2_source.training_overlap_count == 0
        and stage1_source.official_test_count > 0
        and stage2_source.official_test_count > 0
        and full_coverage
    )
    return InterActSplitAuditReport(
        test_seq_file=_split_config(stage2_config).get("test_seq_file") or _split_config(stage1_config).get("test_seq_file"),
        official_test_count=max(stage1_source.official_test_count, stage2_source.official_test_count),
        eval_split=str(eval_split),
        eval_sequence_count=len(eval_sequence_names),
        eval_non_test_count=len(eval_non_test),
        eval_non_official_count=len(eval_non_official),
        stage1_source=stage1_source,
        stage2_source=stage2_source,
        passed=passed,
        notes=notes,
    )


def audit_interact_split_source(
    *,
    config: Mapping[str, Any],
    run: str | Path,
    config_path: str | Path | None = None,
) -> SplitSourceAudit:
    records, _ = build_interact_stage1_index(
        _data_config(config)["root"],
        list(_data_config(config).get("datasets", [])),
        exclude_datasets=_data_config(config).get("exclude_datasets") or [],
        require_motion=True,
        load_object_metadata=True,
        validate_motion_shape=False,
    )
    return audit_interact_split_records(
        records=records,
        split_config=_split_config(config),
        run=run,
        config_path=config_path,
    )


def audit_interact_split_records(
    *,
    records: Sequence[InterActSequenceRecord],
    split_config: Mapping[str, Any],
    run: str | Path,
    config_path: str | Path | None = None,
) -> SplitSourceAudit:
    assigner = InterActSplitAssigner(
        test_seq_file=split_config.get("test_seq_file"),
        use_official_test_seq=bool(split_config.get("use_official_test_seq", True)),
        val_ratio_from_train=float(split_config.get("val_ratio_from_train", 0.05)),
        val_hash_seed=int(split_config.get("val_hash_seed", 42)),
    )
    assigned, diagnostics = assigner.assign(list(records))
    official_names = _load_official_names(split_config.get("test_seq_file"))
    overlaps = [
        f"{record.dataset}/{record.sequence_name}"
        for record in assigned
        if record.split != "test" and _matches_official_name(record.sequence_name, official_names)
    ]
    return SplitSourceAudit(
        run=str(run),
        config_path=str(config_path) if config_path is not None else None,
        audit_source="reconstructed_from_resolved_config",
        use_official_test_seq=bool(split_config.get("use_official_test_seq", True)),
        official_test_count=len(official_names),
        matched_test_count=int(diagnostics.get("matched_test_count", 0)),
        unmatched_official_names=list(diagnostics.get("unmatched_official_names", [])),
        training_overlap_count=len(overlaps),
        overlap_sequence_names=sorted(overlaps),
    )


def _source_from_dict(payload: Mapping[str, Any]) -> SplitSourceAudit:
    return SplitSourceAudit(
        run=str(payload.get("run", "")),
        config_path=payload.get("config_path"),
        audit_source=str(payload.get("audit_source", "")),
        use_official_test_seq=bool(payload.get("use_official_test_seq", False)),
        official_test_count=int(payload.get("official_test_count", 0)),
        matched_test_count=int(payload.get("matched_test_count", 0)),
        unmatched_official_names=list(payload.get("unmatched_official_names", [])),
        training_overlap_count=int(payload.get("training_overlap_count", 0)),
        overlap_sequence_names=list(payload.get("overlap_sequence_names", [])),
    )


def _matches_official_name(sequence_name: str, official_names: set[str]) -> bool:
    if not official_names:
        return False
    official_basenames = {Path(item).name for item in official_names}
    return (
        match_official_test_name(str(sequence_name), official_names, official_basenames)
        is not None
    )


def _official_names_from_config(config: Mapping[str, Any]) -> set[str]:
    return _load_official_names(_split_config(config).get("test_seq_file"))


def _load_official_names(path: Any) -> set[str]:
    if not path:
        return set()
    source = Path(path)
    if not source.exists():
        return set()
    return {line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()}


def _data_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise KeyError("split audit requires config['data'].")
    return data


def _split_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("split", {})
    return value if isinstance(value, Mapping) else {}


def _config_path(run: str | Path) -> str:
    return str(Path(run) / "resolved_config.yaml")
