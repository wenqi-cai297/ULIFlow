from __future__ import annotations

from hoi_latent.data.schemas import InterActSequenceRecord
from hoi_latent.data.splits.interact_splits import (
    InterActSplitAssigner,
    match_official_test_name,
)


def _official(names: list[str]) -> tuple[set[str], set[str]]:
    from pathlib import Path

    official = set(names)
    return official, {Path(n).name for n in official}


def test_exact_and_basename_official_match() -> None:
    official, basenames = _official(["Date07_Sub08_backpack_hand_812", "behave/s1_box_lift_0"])
    # behave-style official names already carry the segment frame and match exactly.
    assert match_official_test_name("Date07_Sub08_backpack_hand_812", official, basenames) == "Date07_Sub08_backpack_hand_812"
    # basename match when the record drops the dataset prefix.
    assert match_official_test_name("s1_box_lift_0", official, basenames) == "s1_box_lift_0"


def test_canonical_segment_suffix_matches_official_base() -> None:
    # GRAB regression: official lists base names, canonical records append _<segment>.
    official, basenames = _official(["s10_hammer_lift", "s7_cup_pour_1"])
    assert match_official_test_name("s10_hammer_lift_0", official, basenames) == "s10_hammer_lift"
    assert match_official_test_name("s10_hammer_lift_142", official, basenames) == "s10_hammer_lift"
    # action name ending in a digit must not be over-stripped.
    assert match_official_test_name("s7_cup_pour_1_0", official, basenames) == "s7_cup_pour_1"


def test_no_false_positive_for_non_official_segments() -> None:
    official, basenames = _official(["s10_hammer_lift"])
    # a different action whose stripped base is not official must not match.
    assert match_official_test_name("s10_hammer_pass_0", official, basenames) is None
    # similarly-prefixed but distinct sequence must not match.
    assert match_official_test_name("s10_hammer_lift2_0", official, basenames) is None


def test_assigner_holds_out_canonical_segment_records_from_train() -> None:
    records = [
        _rec("grab", "s10_hammer_lift_0"),      # official test (segment of s10_hammer_lift)
        _rec("grab", "s10_hammer_lift_142"),    # same official test, second segment
        _rec("grab", "s10_apple_pass_1_0"),     # NOT official -> train/val
    ]
    assigner = InterActSplitAssigner(
        test_seq_file=None, use_official_test_seq=True, val_ratio_from_train=0.0,
    )
    # inject official names directly (avoid file IO)
    assigner._load_official_names = lambda: {"s10_hammer_lift"}  # type: ignore[assignment]
    assigned, diag = assigner.assign(records)
    by_name = {r.sequence_name: r.split for r in assigned}
    assert by_name["s10_hammer_lift_0"] == "test"
    assert by_name["s10_hammer_lift_142"] == "test"
    assert by_name["s10_apple_pass_1_0"] == "train"
    assert diag["matched_test_count"] == 1  # one official base, two canonical segments


def _rec(dataset: str, sequence_name: str) -> InterActSequenceRecord:
    return InterActSequenceRecord(
        dataset=dataset,
        sequence_name=sequence_name,
        sequence_dir=f"/tmp/{dataset}/{sequence_name}",
        motion_path=f"/tmp/{dataset}/{sequence_name}/motion.npy",
        human_path=None,
        object_path=None,
        text_path=None,
        object_name=None,
        object_dir=None,
        sample_points_path=None,
        bps_path=None,
        length=128,
        motion_dim=962,
        dtype="float32",
    )
