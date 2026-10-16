from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from hoi_latent.utils.json_io import load_json

NUM_MARKERS = 77
NUM_RELATION_TOKENS = 78
REL_TYPE_HUMAN_OBJECT = 0
REL_TYPE_OBJECT_GROUND = 1
PART_UNKNOWN = 0
PART_GROUND = 99


def get_marker_ids() -> np.ndarray:
    return np.arange(NUM_MARKERS, dtype=np.int64)


def get_marker_part_ids(map_path: Optional[str] = None) -> np.ndarray:
    if map_path and Path(map_path).exists():
        data = load_json(map_path)
        marker_part_ids = np.asarray(data["marker_part_ids"], dtype=np.int64)
        if marker_part_ids.shape != (NUM_MARKERS,):
            raise ValueError(
                f"marker_part_ids must have shape ({NUM_MARKERS},), got {marker_part_ids.shape}."
            )
        return marker_part_ids
    return np.full((NUM_MARKERS,), PART_UNKNOWN, dtype=np.int64)


def get_relation_ids() -> np.ndarray:
    return np.arange(NUM_RELATION_TOKENS, dtype=np.int64)


def get_relation_marker_ids() -> np.ndarray:
    marker_ids = np.full((NUM_RELATION_TOKENS,), -1, dtype=np.int64)
    marker_ids[:NUM_MARKERS] = np.arange(NUM_MARKERS, dtype=np.int64)
    return marker_ids


def get_relation_part_ids(marker_part_ids: np.ndarray) -> np.ndarray:
    marker_part_ids = np.asarray(marker_part_ids, dtype=np.int64)
    if marker_part_ids.shape != (NUM_MARKERS,):
        raise ValueError(f"marker_part_ids must have shape ({NUM_MARKERS},), got {marker_part_ids.shape}.")
    relation_part_ids = np.full((NUM_RELATION_TOKENS,), PART_GROUND, dtype=np.int64)
    relation_part_ids[:NUM_MARKERS] = marker_part_ids
    return relation_part_ids


def get_relation_type_ids() -> np.ndarray:
    type_ids = np.full((NUM_RELATION_TOKENS,), REL_TYPE_HUMAN_OBJECT, dtype=np.int64)
    type_ids[-1] = REL_TYPE_OBJECT_GROUND
    return type_ids


def build_stage1_ids(map_path: Optional[str] = None) -> dict:
    marker_part_ids = get_marker_part_ids(map_path)
    return {
        "marker_ids": get_marker_ids(),
        "marker_part_ids": marker_part_ids,
        "relation_ids": get_relation_ids(),
        "relation_marker_ids": get_relation_marker_ids(),
        "relation_part_ids": get_relation_part_ids(marker_part_ids),
        "relation_type_ids": get_relation_type_ids(),
    }


def build_marker_part_diagnostics(map_path: Optional[str], marker_part_ids: np.ndarray) -> dict:
    marker_part_ids = np.asarray(marker_part_ids, dtype=np.int64)
    map_exists = bool(map_path and Path(map_path).exists())
    all_unknown = bool(np.all(marker_part_ids == PART_UNKNOWN))
    return {
        "marker_part_map_path": str(map_path) if map_path else None,
        "marker_part_map_exists": map_exists,
        "num_markers": NUM_MARKERS,
        "unique_marker_part_ids": np.unique(marker_part_ids).astype(np.int64).tolist(),
        "all_marker_part_ids_unknown": all_unknown,
        "part_pooling_available": not all_unknown,
        "recommended_tokenization": "marker_tokens_only" if all_unknown else "marker_and_part_tokens",
        "warning": (
            "All marker_part_ids are PART_UNKNOWN. v0 encoders must use marker-level tokens only "
            "and must not rely on body-part pooling."
            if all_unknown
            else None
        ),
    }
