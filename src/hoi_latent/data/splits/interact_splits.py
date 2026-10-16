from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Optional

from hoi_latent.data.schemas import InterActSequenceRecord
from hoi_latent.utils.hash_utils import stable_float01

_SEGMENT_SUFFIX = re.compile(r"_\d+$")


def match_official_test_name(
    sequence_name: str,
    official_names: set[str],
    official_basenames: set[str],
) -> Optional[str]:
    """Return the official test name a record matches, or None.

    InterAct's official ``test_seq.txt`` lists whole-sequence base names (e.g.
    ``s10_hammer_lift``), but our ``sequences_canonical`` records append a
    ``_<segment_start_frame>`` suffix (e.g. ``s10_hammer_lift_0``). An exact /
    basename match alone therefore silently FAILS to recognise those records as
    test, leaking them into train. We additionally match a record whose trailing
    ``_<digits>`` segment, once stripped, equals an official base name. Any such
    canonical segment is, by construction, part of an official test sequence.
    """

    if sequence_name in official_names:
        return sequence_name
    base = Path(sequence_name).name
    if base in official_basenames:
        return base
    stripped = _SEGMENT_SUFFIX.sub("", sequence_name)
    if stripped != sequence_name:
        if stripped in official_names:
            return stripped
        stripped_base = Path(stripped).name
        if stripped_base in official_basenames:
            return stripped_base
    return None


class InterActSplitAssigner:
    def __init__(
        self,
        test_seq_file: Optional[str],
        use_official_test_seq: bool = True,
        val_ratio_from_train: float = 0.05,
        val_hash_seed: int = 42,
    ):
        self.test_seq_file = test_seq_file
        self.use_official_test_seq = use_official_test_seq
        self.val_ratio_from_train = val_ratio_from_train
        self.val_hash_seed = val_hash_seed

    def assign(
        self, records: list[InterActSequenceRecord]
    ) -> tuple[list[InterActSequenceRecord], dict]:
        official_names = self._load_official_names()
        official_basenames = {Path(name).name for name in official_names}
        matched_official = set()
        assigned = []
        per_dataset_split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for record in records:
            split = "train"
            if self.use_official_test_seq:
                matched = match_official_test_name(
                    record.sequence_name, official_names, official_basenames
                )
                if matched is not None:
                    split = "test"
                    matched_official.add(matched)

            if split == "train":
                key = f"{record.dataset}/{record.sequence_name}"
                if stable_float01(key, seed=self.val_hash_seed) < self.val_ratio_from_train:
                    split = "val"

            assigned_record = replace(record, split=split)
            assigned.append(assigned_record)
            per_dataset_split_counts[record.dataset][split] += 1

        diagnostics = {
            "test_seq_file": self.test_seq_file,
            "use_official_test_seq": self.use_official_test_seq,
            "official_test_count": len(official_names),
            "matched_test_count": len(matched_official),
            "unmatched_official_names": sorted(list(official_names - matched_official)),
            "val_ratio_from_train": self.val_ratio_from_train,
            "val_hash_seed": self.val_hash_seed,
            "per_dataset_split_counts": {
                dataset: dict(counts) for dataset, counts in per_dataset_split_counts.items()
            },
        }
        return assigned, diagnostics

    def _load_official_names(self) -> set[str]:
        if not self.use_official_test_seq or not self.test_seq_file:
            return set()
        path = Path(self.test_seq_file)
        if not path.exists():
            return set()
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
