from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hoi_latent.evaluation.stage2 import (
    Stage2DecodedSampleRecord,
    Stage2WorldMotionArrays,
    evaluate_stage2_decoded_records,
    read_stage2_decoded_records_from_dir,
    summarize_stage2_repeat_runs,
    write_stage2_decoded_record,
    write_stage2_quantitative_outputs,
    write_stage2_repeat_report,
)


def test_quantitative_aggregation_computes_paired_interaction_and_ceiling(tmp_path: Path) -> None:
    record = _record(generated_offset=0.01, ceiling_offset=0.005)

    result = evaluate_stage2_decoded_records([record], protocol="eval_128_dev")
    write_stage2_quantitative_outputs(result, tmp_path / "eval", artifacts={"stage2_checkpoint": "dummy.pt"})

    assert result.paired_metrics["paired/marker_mpjpe_cm"] == pytest.approx(1.0)
    assert result.stage1_ceiling_metrics["stage1_ceiling/marker_mpjpe_cm"] == pytest.approx(0.5)
    assert "interaction/foot_sliding_interact_mm" in result.interaction_metrics
    summary = json.loads((tmp_path / "eval" / "metrics" / "summary.json").read_text(encoding="utf-8"))
    artifacts = json.loads((tmp_path / "eval" / "artifacts.json").read_text(encoding="utf-8"))
    assert summary["availability"]["paired_metrics"] == "available"
    assert summary["metrics"]["paired/marker_mpjpe_cm"] == pytest.approx(1.0)
    assert summary["comparability"]["interact_comparable"] is False
    assert "fixed_128_internal_protocol" in summary["comparability"]["blocking_reasons"]
    assert artifacts["comparability"] == summary["comparability"]


def test_eval_long_can_disable_paired_metrics_but_keep_generated_diagnostics() -> None:
    record = _record(generated_offset=0.01, ceiling_offset=0.005)

    result = evaluate_stage2_decoded_records(
        [record],
        protocol="eval_long_v1",
        paired_enabled=False,
        stage1_ceiling_enabled=False,
        feature_enabled=False,
    )

    assert result.availability["paired_metrics"] == "disabled"
    assert result.availability["stage1_ceiling_metrics"] == "disabled"
    assert not result.paired_metrics
    assert "interaction/mean_acceleration_cm_per_frame2" in result.interaction_metrics


def test_metrics_crop_arrays_to_requested_frames_before_shape_sensitive_formulas() -> None:
    record = Stage2DecodedSampleRecord(
        sample_id="behave/seq/cond000001/sample00",
        split="val",
        dataset="behave",
        sequence_name="seq",
        text="lift box",
        object_name="box",
        requested_frames=4,
        fps=30.0,
        seed=11,
        condition_index=1,
        sample_index=0,
        generated=_world(frames=6, marker_offset=0.01),
        gt=_world(frames=4, marker_offset=0.0),
        stage1_ceiling=_world(frames=5, marker_offset=0.005),
    )

    result = evaluate_stage2_decoded_records([record], protocol="eval_128_dev")

    assert result.paired_metrics["paired/marker_mpjpe_cm"] == pytest.approx(1.0)
    assert result.stage1_ceiling_metrics["stage1_ceiling/marker_mpjpe_cm"] == pytest.approx(0.5)


def test_feature_required_raises_without_evaluator() -> None:
    with pytest.raises(RuntimeError, match="feature metrics are required"):
        evaluate_stage2_decoded_records(
            [_record(generated_offset=0.0, ceiling_offset=0.0)],
            protocol="eval_128_report",
            feature_enabled=True,
            feature_required=True,
            feature_evaluator=None,
        )


def test_feature_metrics_use_fake_evaluator() -> None:
    record = _record(generated_offset=0.0, ceiling_offset=0.0)

    result = evaluate_stage2_decoded_records(
        [record],
        protocol="eval_128_dev",
        feature_enabled=True,
        feature_evaluator=FakeFeatureEvaluator(),
        r_precision_pool_size=1,
        feature_metrics_mode="official_artifact",
        feature_parity_passed=True,
    )

    assert result.availability["feature_metrics"] == "available"
    assert result.feature_metrics["feature/fid"] == pytest.approx(0.0)
    assert result.feature_metrics["feature/r_precision_top1"] == pytest.approx(1.0)
    assert result.feature_metrics["feature/stage1_ceiling_fid"] == pytest.approx(0.0)
    assert result.feature_metrics["feature/stage1_ceiling_mm_dist"] == pytest.approx(0.0)
    assert result.feature_metrics["feature/stage1_ceiling_r_precision_top1"] == pytest.approx(1.0)
    assert result.feature_metrics["feature/fid_generation_gap"] == pytest.approx(0.0)
    assert result.comparability["interact_comparable"] is False
    assert result.comparability["blocking_reasons"] == ["fixed_128_internal_protocol"]


def test_feature_metrics_batching_matches_all_at_once() -> None:
    records = []
    for condition_index in range(2):
        for sample_index, offset in enumerate((0.0, 0.1)):
            record = _record(generated_offset=offset + condition_index, ceiling_offset=0.0)
            record.condition_index = condition_index
            record.sample_index = sample_index
            records.append(record)

    all_at_once = evaluate_stage2_decoded_records(
        records,
        protocol="eval_128_dev",
        feature_enabled=True,
        feature_evaluator=FakeFeatureEvaluator(),
        r_precision_pool_size=1,
        multimodality_min_samples=2,
        multimodality_pairs_per_condition=1,
    )
    batched = evaluate_stage2_decoded_records(
        records,
        protocol="eval_128_dev",
        feature_enabled=True,
        feature_evaluator=FakeFeatureEvaluator(),
        r_precision_pool_size=1,
        multimodality_min_samples=2,
        multimodality_pairs_per_condition=1,
        feature_batch_size=1,
    )

    assert batched.feature_metrics.keys() == all_at_once.feature_metrics.keys()
    for key, value in all_at_once.feature_metrics.items():
        assert batched.feature_metrics[key] == pytest.approx(value)


def test_multimodality_requires_configured_samples_per_condition() -> None:
    records = []
    for condition_index in range(2):
        for sample_index, offset in enumerate((0.0, 0.1, 0.2)):
            record = _record(generated_offset=offset + condition_index, ceiling_offset=0.0)
            record.condition_index = condition_index
            record.sample_index = sample_index
            records.append(record)

    disabled = evaluate_stage2_decoded_records(
        records,
        protocol="eval_128_dev",
        feature_enabled=True,
        feature_evaluator=FakeFeatureEvaluator(),
        r_precision_pool_size=1,
        multimodality_min_samples=4,
    )
    enabled = evaluate_stage2_decoded_records(
        records,
        protocol="eval_128_dev",
        feature_enabled=True,
        feature_evaluator=FakeFeatureEvaluator(),
        r_precision_pool_size=1,
        multimodality_min_samples=3,
        multimodality_pairs_per_condition=1,
    )

    assert "feature/multimodality" not in disabled.feature_metrics
    assert enabled.feature_metrics["feature/multimodality"] > 0.0
    assert enabled.feature_metrics["feature/pair_count"] == 2.0


def test_feature_metrics_without_ceiling_are_marked_not_comparable() -> None:
    record = _record(generated_offset=0.0, ceiling_offset=0.0)
    record.stage1_ceiling = None

    result = evaluate_stage2_decoded_records(
        [record],
        protocol="eval_128_report",
        feature_enabled=True,
        feature_evaluator=FakeFeatureEvaluator(),
        r_precision_pool_size=1,
        feature_metrics_mode="official_artifact",
        feature_parity_passed=True,
        feature_parity_golden_source="external",
        feature_artifact_manifest_valid=True,
    )

    assert "feature/fid" in result.feature_metrics
    assert "feature/stage1_ceiling_fid" not in result.feature_metrics
    assert result.comparability["interact_comparable"] is False
    assert "missing_stage1_ceiling_feature_fid" in result.comparability["blocking_reasons"]


def test_report_protocol_requires_external_golden_source() -> None:
    record = _record(generated_offset=0.0, ceiling_offset=0.0)

    result = evaluate_stage2_decoded_records(
        [record],
        protocol="eval_128_report",
        feature_enabled=True,
        feature_evaluator=FakeFeatureEvaluator(),
        r_precision_pool_size=1,
        feature_metrics_mode="official_artifact",
        feature_parity_passed=True,
        feature_parity_golden_source="self",
        feature_artifact_manifest_valid=True,
    )

    assert result.comparability["interact_comparable"] is False
    assert "non_external_golden_source:self" in result.comparability["blocking_reasons"]


def test_report_protocol_requires_feature_artifact_manifest() -> None:
    record = _record(generated_offset=0.0, ceiling_offset=0.0)

    result = evaluate_stage2_decoded_records(
        [record],
        protocol="eval_128_report",
        feature_enabled=True,
        feature_evaluator=FakeFeatureEvaluator(),
        r_precision_pool_size=1,
        feature_metrics_mode="official_artifact",
        feature_parity_passed=True,
        feature_parity_golden_source="external",
        feature_artifact_manifest_valid=False,
    )

    assert result.comparability["interact_comparable"] is False
    assert "missing_or_invalid_feature_artifact_manifest" in result.comparability["blocking_reasons"]


def test_interact_official_protocol_fails_closed_until_audits_exist() -> None:
    record = _record(generated_offset=0.0, ceiling_offset=0.0)

    result = evaluate_stage2_decoded_records(
        [record],
        protocol="interact_official_v1",
        feature_enabled=True,
        feature_evaluator=FakeFeatureEvaluator(),
        r_precision_pool_size=1,
        feature_metrics_mode="official_artifact",
        feature_parity_passed=True,
        feature_parity_golden_source="external",
        feature_artifact_manifest_valid=True,
    )

    assert result.comparability["interact_comparable"] is False
    assert "split_audit_missing" in result.comparability["blocking_reasons"]
    assert "marker_bps_alignment_missing" in result.comparability["blocking_reasons"]


def test_interact_official_protocol_uses_split_and_marker_audit_gates() -> None:
    record = _record(generated_offset=0.0, ceiling_offset=0.0)

    failed = evaluate_stage2_decoded_records(
        [record],
        protocol="interact_official_v1",
        feature_enabled=True,
        feature_evaluator=FakeFeatureEvaluator(),
        r_precision_pool_size=1,
        feature_metrics_mode="official_artifact",
        feature_parity_passed=True,
        feature_parity_golden_source="external",
        feature_artifact_manifest_valid=True,
        split_audit_passed=False,
        marker_bps_alignment_passed=False,
    )
    assert "split_audit_failed" in failed.comparability["blocking_reasons"]
    assert "marker_bps_alignment_failed" in failed.comparability["blocking_reasons"]

    passed = evaluate_stage2_decoded_records(
        [record],
        protocol="interact_official_v1",
        feature_enabled=True,
        feature_evaluator=FakeFeatureEvaluator(),
        r_precision_pool_size=1,
        feature_metrics_mode="official_artifact",
        feature_parity_passed=True,
        feature_parity_golden_source="external",
        feature_artifact_manifest_valid=True,
        split_audit_passed=True,
        marker_bps_alignment_passed=True,
    )
    assert passed.comparability["interact_comparable"] is True
    assert passed.comparability["blocking_reasons"] == []


def test_cli_aggregates_records_dir_and_writes_batch4_layout(tmp_path: Path) -> None:
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    write_stage2_decoded_record(_record(generated_offset=0.01, ceiling_offset=0.005), records_dir / "records_000000.npz")
    data_cfg = tmp_path / "data.yaml"
    train_cfg = tmp_path / "train.yaml"
    data_cfg.write_text("data: {}\n", encoding="utf-8")
    train_cfg.write_text("training: {}\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/evaluation/stage2/evaluate_flow_forcing_quantitative.py",
        "--data-config",
        str(data_cfg),
        "--train-config",
        str(train_cfg),
        "--stage2-run",
        str(tmp_path / "stage2_run"),
        "--source-run",
        str(tmp_path / "stage1_run"),
        "--whitening",
        str(tmp_path / "whitening.pt"),
        "--template-spec",
        str(tmp_path / "template.pt"),
        "--eval-protocol",
        "eval_128_dev",
        "--run-name",
        "unit_eval",
        "--out-root",
        str(tmp_path / "out"),
        "--decoded-records-dir",
        str(records_dir),
        "--feature-evaluator-mode",
        "disabled",
    ]
    completed = subprocess.run(cmd, cwd=Path.cwd(), check=True, text=True, capture_output=True)

    run_dir = tmp_path / "out" / "unit_eval"
    assert "summary.json" in completed.stdout
    assert (run_dir / "predictions" / "records_000000.npz").exists()
    assert (run_dir / "artifacts.json").exists()
    summary = json.loads((run_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["protocol"] == "eval_128_dev"
    assert summary["metrics"]["paired/marker_mpjpe_cm"] == pytest.approx(1.0)
    loaded = read_stage2_decoded_records_from_dir(run_dir / "predictions")
    assert len(loaded) == 1


def test_cli_generate_only_skips_record_aggregation(tmp_path: Path) -> None:
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    write_stage2_decoded_record(
        _record(generated_offset=0.01, ceiling_offset=0.005),
        records_dir / "records_000000.npz",
    )
    data_cfg = tmp_path / "data.yaml"
    train_cfg = tmp_path / "train.yaml"
    data_cfg.write_text("data: {}\n", encoding="utf-8")
    train_cfg.write_text("training: {}\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/evaluation/stage2/evaluate_flow_forcing_quantitative.py",
        "--data-config",
        str(data_cfg),
        "--train-config",
        str(train_cfg),
        "--stage2-run",
        str(tmp_path / "stage2_run"),
        "--source-run",
        str(tmp_path / "stage1_run"),
        "--whitening",
        str(tmp_path / "whitening.pt"),
        "--eval-protocol",
        "eval_128_dev",
        "--run-name",
        "generate_only",
        "--out-root",
        str(tmp_path / "out"),
        "--decoded-records-dir",
        str(records_dir),
        "--generate-only",
    ]
    completed = subprocess.run(cmd, cwd=Path.cwd(), check=True, text=True, capture_output=True)

    run_dir = tmp_path / "out" / "generate_only"
    assert "Decoded records ready: 1" in completed.stdout
    assert (run_dir / "command.txt").exists()
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "git.json").exists()
    assert not (run_dir / "predictions").exists()
    assert not (run_dir / "metrics" / "summary.json").exists()


def test_repeat_reporting_writes_mean_ci_tables(tmp_path: Path) -> None:
    repeat0 = tmp_path / "repeat_00"
    repeat1 = tmp_path / "repeat_01"
    _write_repeat_summary(repeat0, seed=101, fid=10.0, comparable=True)
    _write_repeat_summary(repeat1, seed=202, fid=14.0, comparable=True)

    report = summarize_stage2_repeat_runs([repeat0, repeat1])
    out_dir = tmp_path / "summary"
    write_stage2_repeat_report(report, out_dir)

    item = report.metrics["feature/fid"]
    assert item.value_mean == pytest.approx(12.0)
    assert item.value_std == pytest.approx(2.0)
    assert item.value_ci95 == pytest.approx(1.96 * item.value_std / (2.0**0.5))
    assert report.seeds == [101, 202]
    assert report.all_interact_comparable
    assert (out_dir / "metrics" / "report_table.json").exists()
    assert "`feature/fid`" in (out_dir / "metrics" / "report_table.md").read_text(encoding="utf-8")


def test_cli_records_resolution_generates_when_no_records_dir(monkeypatch, tmp_path: Path) -> None:
    module = _load_eval_cli_module()
    generated_dir = tmp_path / "generated_records"
    args = SimpleNamespace(decoded_records_dir=None, stage2_run=tmp_path / "stage2_run")
    calls = []

    def fake_generate(decoded_args, run_predictions):
        calls.append((decoded_args, run_predictions))
        return generated_dir

    monkeypatch.setattr(module, "_generate_decoded_records", fake_generate)

    resolved = module._resolve_or_generate_records_dir(args, tmp_path / "run")

    assert resolved == generated_dir
    assert calls and calls[0][1] == tmp_path / "run" / "predictions"


def test_cli_records_resolution_prefers_explicit_records_dir(monkeypatch, tmp_path: Path) -> None:
    module = _load_eval_cli_module()
    explicit = tmp_path / "records"
    args = SimpleNamespace(decoded_records_dir=explicit, stage2_run=tmp_path / "stage2_run")

    def fail_generate(decoded_args, run_predictions):  # pragma: no cover - should never be called
        raise AssertionError("explicit records dir should bypass generation")

    monkeypatch.setattr(module, "_generate_decoded_records", fail_generate)

    assert module._resolve_or_generate_records_dir(args, tmp_path / "run") == explicit


def test_official_protocol_requested_frames_uses_sequence_length_cap() -> None:
    module = _load_eval_cli_module()
    args = SimpleNamespace(eval_protocol="interact_official_v1", requested_frames=128)
    batch = {
        "model": {"human": {"root": torch.zeros(1, 300, 3)}},
        "meta": {"sequence_length": torch.tensor([412])},
    }

    assert module._requested_frames_for_batch(args, batch) == 300

    batch["meta"]["sequence_length"] = torch.tensor([196])
    assert module._requested_frames_for_batch(args, batch) == 196


def test_official_protocol_loader_contract_uses_test_prefix_windows() -> None:
    module = _load_eval_cli_module()
    args = SimpleNamespace(
        eval_protocol="interact_official_v1",
        split="test",
        batch_size=1,
        num_workers=0,
        num_conditions=3,
    )
    config = {"data": {}, "loader": {}, "training": {"stage1": {}, "stage2": {}}}

    module._force_evaluation_loader_contract(config, args)

    assert config["data"]["window_size"] == 300
    assert config["data"]["eval_window_mode"] == "first"
    assert config["training"]["stage1"]["debug_num_eval_batches"] == 3


def test_evaluation_loader_contract_can_preserve_a_fixed_condition_set() -> None:
    module = _load_eval_cli_module()
    args = SimpleNamespace(
        eval_protocol="interact_official_v1",
        split="test",
        batch_size=1,
        num_workers=0,
        num_conditions=240,
        disable_text_filter=True,
    )
    config = {
        "data": {
            "text_filter": {
                "enabled": True,
                "tokenizer": "google/flan-t5-xl",
                "max_tokens": 77,
            }
        },
        "loader": {},
        "training": {"stage1": {}, "stage2": {}},
    }

    module._force_evaluation_loader_contract(config, args)

    assert config["data"]["text_filter"] == {
        "enabled": False,
        "tokenizer": "google/flan-t5-xl",
        "max_tokens": 77,
    }


def test_sequence_subset_selects_exact_records_in_cli_order() -> None:
    module = _load_eval_cli_module()

    class Record:
        def __init__(self, sequence_name: str):
            self.sequence_name = sequence_name

    class Dataset:
        records = [Record("seq-a"), Record("seq-b"), Record("seq-c")]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return index

    subset = module._select_sequence_subset(Dataset(), ["seq-c", "seq-a"])

    assert subset.indices == [2, 0]


def test_sequence_subset_rejects_missing_and_ambiguous_names() -> None:
    module = _load_eval_cli_module()

    class Record:
        def __init__(self, sequence_name: str):
            self.sequence_name = sequence_name

    class Dataset:
        records = [Record("same"), Record("same")]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return index

    with pytest.raises(ValueError, match="same.*2 matches"):
        module._select_sequence_subset(Dataset(), ["same"])
    with pytest.raises(ValueError, match="missing.*0 matches"):
        module._select_sequence_subset(Dataset(), ["missing"])


def test_minimum_length_subset_preserves_dataset_order() -> None:
    module = _load_eval_cli_module()

    class Record:
        def __init__(self, length: int):
            self.length = length

    class Dataset:
        records = [Record(512), Record(127), Record(384), Record(383)]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return index

    subset = module._select_minimum_sequence_length_subset(Dataset(), 384)

    assert subset.indices == [0, 2]


def test_long_paired_loader_contract_uses_requested_prefix() -> None:
    module = _load_eval_cli_module()
    args = SimpleNamespace(
        eval_protocol="eval_long_v1",
        split="test",
        batch_size=2,
        num_workers=0,
        num_conditions=3,
        requested_frames=384,
        min_sequence_frames=384,
    )
    config = {"data": {}, "loader": {}, "training": {"stage1": {}, "stage2": {}}}

    module._force_evaluation_loader_contract(config, args)

    assert config["data"]["window_size"] == 384
    assert config["data"]["eval_window_mode"] == "first"
    assert config["data"]["train_window_mode"] == "first"
    assert config["training"]["stage1"]["debug_num_eval_batches"] == 2

    args.min_sequence_frames = 256
    with pytest.raises(ValueError, match="must be at least"):
        module._force_evaluation_loader_contract(config, args)


def test_selected_sequence_loader_uses_requested_split(monkeypatch) -> None:
    module = _load_eval_cli_module()
    observed: dict[str, object] = {}

    class Record:
        def __init__(self, sequence_name: str):
            self.sequence_name = sequence_name

    class Dataset:
        records = [Record("seq-a"), Record("seq-b")]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return index

    def fake_make_dataset(_config, *, split, window_mode, **_kwargs):
        observed.update(split=split, window_mode=window_mode)
        return Dataset()

    monkeypatch.setattr(module, "_make_dataset", fake_make_dataset)
    monkeypatch.setattr(
        module,
        "build_stage1_phase1_dataloaders",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selected sequences must bypass the implicit val loader")
        ),
    )
    args = SimpleNamespace(
        eval_protocol="eval_long_v1",
        split="test",
        sequence_name=["seq-b"],
    )
    config = {
        "data": {"train_window_mode": "random", "eval_window_mode": "center"},
        "loader": {"batch_size": 1, "num_workers": 0},
    }

    loader = module._build_stage2_evaluation_loader(
        config,
        args,
        normalizer=None,
        structured_normalizer=None,
    )

    assert observed == {"split": "test", "window_mode": "center"}
    assert loader.dataset.indices == [1]


def test_long_loader_filters_before_applying_condition_limit(monkeypatch) -> None:
    module = _load_eval_cli_module()

    class Record:
        def __init__(self, sequence_name: str, length: int):
            self.sequence_name = sequence_name
            self.length = length

    class Dataset:
        records = [
            Record("long-a", 512),
            Record("short", 127),
            Record("long-b", 384),
            Record("long-c", 420),
        ]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return index

    monkeypatch.setattr(module, "_make_dataset", lambda *_args, **_kwargs: Dataset())
    args = SimpleNamespace(
        eval_protocol="eval_long_v1",
        split="test",
        sequence_name=[],
        min_sequence_frames=384,
        num_conditions=2,
    )
    config = {
        "data": {"train_window_mode": "first", "eval_window_mode": "first"},
        "loader": {"batch_size": 1, "num_workers": 0},
    }

    loader = module._build_stage2_evaluation_loader(
        config,
        args,
        normalizer=None,
        structured_normalizer=None,
    )

    assert [loader.dataset[index] for index in range(len(loader.dataset))] == [0, 2]


def test_official_protocol_requires_test_split_and_batch_size_one() -> None:
    module = _load_eval_cli_module()
    config = {"data": {}, "loader": {}, "training": {"stage1": {}, "stage2": {}}}
    args = SimpleNamespace(
        eval_protocol="interact_official_v1",
        split="val",
        batch_size=1,
        num_workers=0,
        num_conditions=None,
    )

    with pytest.raises(ValueError, match="--split test"):
        module._force_evaluation_loader_contract(config, args)

    args.split = "test"
    args.batch_size = 2
    with pytest.raises(ValueError, match="--batch-size 1"):
        module._force_evaluation_loader_contract(config, args)


def test_cli_required_parity_rejects_non_external_golden(monkeypatch, tmp_path: Path) -> None:
    module = _load_eval_cli_module()
    args = SimpleNamespace(
        feature_evaluator_mode="official_artifact",
        feature_artifact_root=tmp_path,
        feature_evaluator_ckpt=None,
        feature_mean=None,
        feature_std=None,
        feature_bps_basis=None,
        feature_state_dict=None,
        feature_parity_dir=tmp_path / "parity",
        require_feature_parity=True,
        eval_protocol="eval_128_dev",
    )

    monkeypatch.setattr(
        module,
        "load_interact_feature_artifacts",
        lambda paths: SimpleNamespace(sha256={}, manifest={}, manifest_valid=True),
    )
    monkeypatch.setattr(module, "build_interact_feature_evaluator_from_artifacts", lambda bundle: object())
    monkeypatch.setattr(
        module,
        "run_interact_feature_parity_check",
        lambda evaluator, parity_dir: SimpleNamespace(
            passed=True,
            golden_source="self",
            to_dict=lambda: {"passed": True, "golden_source": "self"},
        ),
    )

    with pytest.raises(module.InterActFeatureEvaluatorUnavailable, match="external InterAct evaluator"):
        module._build_feature_evaluator(args)


def _record(*, generated_offset: float, ceiling_offset: float) -> Stage2DecodedSampleRecord:
    return Stage2DecodedSampleRecord(
        sample_id="behave/seq/cond000001/sample00",
        split="val",
        dataset="behave",
        sequence_name="seq",
        text="lift box",
        object_name="box",
        requested_frames=4,
        fps=30.0,
        seed=11,
        condition_index=1,
        sample_index=0,
        generated=_world(frames=4, marker_offset=generated_offset),
        gt=_world(frames=4, marker_offset=0.0),
        stage1_ceiling=_world(frames=4, marker_offset=ceiling_offset),
        rollout_metadata={"update_counts": [32, 32, 32, 32], "steps_per_block": 32, "num_blocks": 4},
    )


def _write_repeat_summary(root: Path, *, seed: int, fid: float, comparable: bool) -> None:
    (root / "metrics").mkdir(parents=True)
    (root / "metrics" / "summary.json").write_text(
        json.dumps(
            {
                "metrics": {"feature/fid": fid, "feature/mm_dist": fid + 1.0},
                "comparability": {"interact_comparable": comparable, "blocking_reasons": [] if comparable else ["x"]},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "resolved_config.yaml").write_text(
        f"evaluation:\n  stage2:\n    seed: {seed}\n",
        encoding="utf-8",
    )


def _world(frames: int, *, marker_offset: float) -> Stage2WorldMotionArrays:
    human = np.zeros((frames, 77, 3), dtype=np.float32)
    human[..., 0] = np.float32(marker_offset)
    object_points = np.zeros((frames, 2, 3), dtype=np.float32)
    object_trans = np.zeros((frames, 3), dtype=np.float32)
    object_rotmat = np.eye(3, dtype=np.float32)[None].repeat(frames, axis=0)
    return Stage2WorldMotionArrays(
        human_markers=human,
        object_points=object_points,
        frame_mask=np.ones(frames, dtype=bool),
        object_trans=object_trans,
        object_rotmat=object_rotmat,
    )


class FakeFeatureEvaluator:
    def preprocess(self, samples):
        values = []
        for sample in samples:
            values.append(float(sample.human_markers.mean()))
        features = torch.tensor(values, dtype=torch.float32).view(len(values), 1, 1)
        return SimpleNamespace(normalized_features=features, lengths=[1] * len(values))

    def encode_motion(self, normalized_features, lengths):
        return normalized_features[:, 0, :]

    def encode_text(self, texts):
        return torch.zeros(len(texts), 1)


def _load_eval_cli_module():
    path = Path("scripts/evaluation/stage2/evaluate_flow_forcing_quantitative.py")
    spec = importlib.util.spec_from_file_location("stage2_eval_cli_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
