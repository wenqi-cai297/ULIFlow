from __future__ import annotations

from pathlib import Path

from hoi_latent.data.datasets import interact_stage1
from hoi_latent.data.schemas import InterActSequenceRecord


class _WhitespaceTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, truncation: bool) -> dict:
        assert truncation is False
        count = len(text.split()) + int(add_special_tokens)
        return {"input_ids": list(range(count))}


def test_text_token_filter_drops_overflow_and_missing_text(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        interact_stage1,
        "_load_text_filter_tokenizer",
        lambda tokenizer_name: _WhitespaceTokenizer(),
    )
    records = [
        _record(tmp_path, "short", "pick cup"),
        _record(tmp_path, "long", "this caption is too long"),
        _record(tmp_path, "missing", None),
    ]

    kept, diagnostics = interact_stage1._apply_text_token_filter(
        records,
        {
            "enabled": True,
            "tokenizer": "fake-t5",
            "max_tokens": 3,
            "add_special_tokens": True,
            "require_text": True,
        },
    )

    assert [record.sequence_name for record in kept] == ["short"]
    assert diagnostics["before_count"] == 3
    assert diagnostics["after_count"] == 1
    assert diagnostics["dropped_count"] == 2
    assert diagnostics["dropped_too_long_count"] == 1
    assert diagnostics["dropped_missing_text_count"] == 1
    assert diagnostics["max_kept_tokens"] == 3
    assert diagnostics["max_seen_tokens"] == 6


def test_text_token_filter_disabled_keeps_records(tmp_path) -> None:
    records = [_record(tmp_path, "short", "pick cup")]

    kept, diagnostics = interact_stage1._apply_text_token_filter(records, None)

    assert kept == records
    assert diagnostics["enabled"] is False
    assert diagnostics["dropped_count"] == 0


def _record(tmp_path: Path, name: str, text: str | None) -> InterActSequenceRecord:
    text_path = None
    if text is not None:
        text_file = tmp_path / f"{name}.txt"
        text_file.write_text(text, encoding="utf-8")
        text_path = str(text_file)
    return InterActSequenceRecord(
        dataset="toy",
        sequence_name=name,
        sequence_dir=str(tmp_path / name),
        motion_path=str(tmp_path / name / "motion.npy"),
        human_path=None,
        object_path=None,
        text_path=text_path,
        object_name=None,
        object_dir=None,
        sample_points_path=None,
        bps_path=None,
        length=8,
        motion_dim=962,
        dtype="float32",
        split="train",
    )
