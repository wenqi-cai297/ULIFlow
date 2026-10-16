from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for parser-only checks.
    torch = None


@dataclass
class InterActMotion962Groups:
    human_pos: Any
    human_vel: Any
    left_heights: Any
    right_heights: Any
    object_rot6d: Any
    object_trans: Any
    object_vel: Any
    relation: Any

    @property
    def relation_delta(self) -> Any:
        return self.relation[..., 0:3]

    @property
    def relation_point(self) -> Any:
        return self.relation[..., 3:6]


class InterActMotion962:
    DIM = 962
    NUM_MARKERS = 77
    NUM_RELATION_ENTRIES = 78

    HUMAN_POS = slice(0, 231)
    HUMAN_VEL = slice(231, 462)
    LEFT_HEIGHTS = slice(462, 469)
    RIGHT_HEIGHTS = slice(469, 476)
    OBJECT_POSE = slice(476, 485)
    OBJECT_VEL = slice(485, 494)
    RELATION = slice(494, 962)

    @classmethod
    def validate_shape(cls, x: Any, *, name: str = "motion") -> None:
        shape = getattr(x, "shape", None)
        if shape is None:
            raise TypeError(f"{name} must have a shape attribute.")
        if len(shape) < 2:
            raise ValueError(f"{name} must have at least 2 dimensions, got shape {tuple(shape)}.")
        if int(shape[-1]) != cls.DIM:
            raise ValueError(f"{name} last dimension must be {cls.DIM}, got shape {tuple(shape)}.")

    @classmethod
    def parse(cls, x: Any) -> InterActMotion962Groups:
        cls.validate_shape(x)
        prefix = tuple(x.shape[:-1])
        human_pos = x[..., cls.HUMAN_POS].reshape(prefix + (cls.NUM_MARKERS, 3))
        human_vel = x[..., cls.HUMAN_VEL].reshape(prefix + (cls.NUM_MARKERS, 3))
        object_pose = x[..., cls.OBJECT_POSE]
        relation = x[..., cls.RELATION].reshape(prefix + (cls.NUM_RELATION_ENTRIES, 6))
        # TODO: Confirm relation_delta/relation_point naming against
        # external/InterAct/process/motion_representation.py before treating them as semantics.
        return InterActMotion962Groups(
            human_pos=human_pos,
            human_vel=human_vel,
            left_heights=x[..., cls.LEFT_HEIGHTS],
            right_heights=x[..., cls.RIGHT_HEIGHTS],
            object_rot6d=object_pose[..., 0:6],
            object_trans=object_pose[..., 6:9],
            object_vel=x[..., cls.OBJECT_VEL],
            relation=relation,
        )

    @classmethod
    def compose(cls, groups: InterActMotion962Groups) -> Any:
        pieces = [
            cls._flatten_last(groups.human_pos, cls.NUM_MARKERS * 3),
            cls._flatten_last(groups.human_vel, cls.NUM_MARKERS * 3),
            groups.left_heights,
            groups.right_heights,
            cls._concat([groups.object_rot6d, groups.object_trans], axis=-1),
            groups.object_vel,
            cls._flatten_last(groups.relation, cls.NUM_RELATION_ENTRIES * 6),
        ]
        out = cls._concat(pieces, axis=-1)
        cls.validate_shape(out, name="composed_motion")
        return out

    @classmethod
    def group_slices(cls) -> dict[str, slice]:
        return {
            "human_pos": cls.HUMAN_POS,
            "human_vel": cls.HUMAN_VEL,
            "left_heights": cls.LEFT_HEIGHTS,
            "right_heights": cls.RIGHT_HEIGHTS,
            "object_pose": cls.OBJECT_POSE,
            "object_vel": cls.OBJECT_VEL,
            "relation": cls.RELATION,
        }

    @classmethod
    def group_dims(cls) -> dict[str, int]:
        return {
            name: sl.stop - sl.start for name, sl in cls.group_slices().items()
        }

    @staticmethod
    def _is_torch(x: Any) -> bool:
        return torch is not None and isinstance(x, torch.Tensor)

    @classmethod
    def _concat(cls, pieces: list[Any], axis: int = -1) -> Any:
        if cls._is_torch(pieces[0]):
            return torch.cat(pieces, dim=axis)
        return np.concatenate(pieces, axis=axis)

    @staticmethod
    def _flatten_last(x: Any, dim: int) -> Any:
        return x.reshape(tuple(x.shape[:-2]) + (dim,))
