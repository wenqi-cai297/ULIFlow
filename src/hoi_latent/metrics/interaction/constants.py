from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

LEFT_HAND_PARTS: frozenset[int] = frozenset({3, 9})
RIGHT_HAND_PARTS: frozenset[int] = frozenset({4, 10})
HAND_PARTS: frozenset[int] = LEFT_HAND_PARTS | RIGHT_HAND_PARTS
FOOT_PARTS: frozenset[int] = frozenset({5, 6, 7, 8})


def _default_marker_part_map_path() -> Path:
    return Path(__file__).resolve().parents[4] / "assets" / "interact" / "marker_part_map_v1.json"


@lru_cache(maxsize=4)
def load_marker_part_ids(marker_part_map_path: str | Path | None = None) -> tuple[int, ...]:
    path = Path(marker_part_map_path) if marker_part_map_path is not None else _default_marker_part_map_path()
    if not path.exists():
        raise FileNotFoundError(f"marker part map not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    part_ids = tuple(int(v) for v in payload["marker_part_ids"])
    if len(part_ids) != 77:
        raise ValueError(f"expected 77 marker part ids, got {len(part_ids)} from {path}")
    return part_ids


def marker_indices_for_parts(parts: set[int] | frozenset[int], marker_part_map_path: str | Path | None = None) -> tuple[int, ...]:
    part_set = set(int(v) for v in parts)
    return tuple(i for i, part_id in enumerate(load_marker_part_ids(marker_part_map_path)) if part_id in part_set)


LEFT_HAND_MARKERS: tuple[int, ...] = marker_indices_for_parts(LEFT_HAND_PARTS)
RIGHT_HAND_MARKERS: tuple[int, ...] = marker_indices_for_parts(RIGHT_HAND_PARTS)
HAND_MARKERS: tuple[int, ...] = marker_indices_for_parts(HAND_PARTS)
FOOT_MARKERS: tuple[int, ...] = marker_indices_for_parts(FOOT_PARTS)
FULL_MARKERS: tuple[int, ...] = tuple(range(77))
NON_HAND_MARKERS: tuple[int, ...] = tuple(i for i in FULL_MARKERS if i not in set(HAND_MARKERS))

MARKER_SUBSETS: dict[str, tuple[int, ...]] = {
    "left_hand": LEFT_HAND_MARKERS,
    "right_hand": RIGHT_HAND_MARKERS,
    "hand": HAND_MARKERS,
    "foot": FOOT_MARKERS,
    "non_hand": NON_HAND_MARKERS,
    "full": FULL_MARKERS,
}
