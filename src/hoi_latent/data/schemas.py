from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InterActSequenceRecord:
    dataset: str
    sequence_name: str
    sequence_dir: str
    motion_path: str
    human_path: Optional[str]
    object_path: Optional[str]
    text_path: Optional[str]
    object_name: Optional[str]
    object_dir: Optional[str]
    sample_points_path: Optional[str]
    bps_path: Optional[str]
    length: int
    motion_dim: int
    dtype: str
    split: Optional[str] = None
