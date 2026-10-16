from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hoi_latent.data.representations.interact_motion_962 import InterActMotion962


@dataclass
class Stage1HumanRaw:
    pos: Any
    vel: Any
    acc: Any
    vel_official: Any
    root: Any
    root_vel: Any
    root_acc: Any
    pos_rootrel: Any
    vel_rootrel: Any
    acc_rootrel: Any
    left_heights: Any
    right_heights: Any


@dataclass
class Stage1ObjectRaw:
    rot6d: Any
    rotmat: Any
    trans: Any
    trans_vel: Any
    trans_acc: Any
    vel9_official: Any
    rot6d_vel_official: Any
    trans_vel_official: Any


@dataclass
class Stage1RelationRaw:
    delta_m2o: Any
    delta_o2m: Any
    point_canon: Any
    dist: Any
    delta_vel: Any
    delta_acc: Any
    nearest_point_world: Any
    nearest_point_vel: Any
    nearest_point_acc: Any
    ground_marker: Any
    root_object_delta: Any
    root_object_dist: Any
    root_object_delta_vel: Any
    root_object_delta_acc: Any


def diff_pad_last(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    if x.shape[0] > 1:
        out[:-1] = x[1:] - x[:-1]
    return out


def diff_pad_last_valid(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    if x.shape[0] <= 1:
        return out
    valid = np.asarray(valid_mask, dtype=bool)
    pair_valid = valid[:-1] & valid[1:]
    expand_shape = (pair_valid.shape[0],) + (1,) * (x.ndim - 1)
    out[:-1] = np.where(pair_valid.reshape(expand_shape), x[1:] - x[:-1], 0.0)
    return out


def derivative_mask(valid_mask: np.ndarray, order: int) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool)
    out = np.zeros_like(valid, dtype=bool)
    if order <= 0:
        return valid.copy()
    if len(valid) <= order:
        return out
    stacked = [valid[offset : len(valid) - order + offset] for offset in range(order + 1)]
    out[: len(valid) - order] = np.logical_and.reduce(stacked)
    return out


def safe_norm(
    x: np.ndarray,
    axis: int = -1,
    keepdims: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    sq = np.sum(np.square(x, dtype=np.float32), axis=axis, keepdims=keepdims)
    return np.sqrt(np.maximum(sq, eps * eps), dtype=np.float32)


def rot6d_to_matrix_np(rot6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d last dimension must be 6, got {rot6d.shape}.")
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = _normalize(a1, eps=eps)
    b2 = _normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1, eps=eps)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-2).astype(np.float32, copy=False)


def compute_root_proxy(human_pos: np.ndarray, method: str = "marker_mean") -> np.ndarray:
    if method != "marker_mean":
        raise ValueError(f"Unsupported root proxy method {method!r}.")
    return np.mean(human_pos, axis=-2, dtype=np.float32)


def build_stage1_raw_groups_from_motion_window(
    raw_motion_window: np.ndarray,
    valid_mask: np.ndarray,
    root_method: str = "marker_mean",
) -> dict:
    raw_motion = np.asarray(raw_motion_window, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    InterActMotion962.validate_shape(raw_motion, name="raw_motion_window")
    if raw_motion.ndim != 2:
        raise ValueError(f"raw_motion_window must be [T, 962], got {raw_motion.shape}.")
    if valid.shape != (raw_motion.shape[0],):
        raise ValueError(f"valid_mask must be [{raw_motion.shape[0]}], got {valid.shape}.")

    groups = InterActMotion962.parse(raw_motion)
    human_pos = _zero_invalid(groups.human_pos.astype(np.float32, copy=False), valid)
    human_vel_official = _zero_invalid(groups.human_vel.astype(np.float32, copy=False), valid)
    left_heights = _zero_invalid(groups.left_heights.astype(np.float32, copy=False), valid)
    right_heights = _zero_invalid(groups.right_heights.astype(np.float32, copy=False), valid)

    vel_mask = derivative_mask(valid, order=1)
    acc_mask = derivative_mask(valid, order=2)
    human_vel = _zero_by_mask(diff_pad_last_valid(human_pos, valid), vel_mask)
    human_acc = _zero_by_mask(diff_pad_last_valid(human_vel, vel_mask), acc_mask)

    root = _zero_invalid(compute_root_proxy(human_pos, method=root_method), valid)
    root_vel = _zero_by_mask(diff_pad_last_valid(root, valid), vel_mask)
    root_acc = _zero_by_mask(diff_pad_last_valid(root_vel, vel_mask), acc_mask)

    pos_rootrel = _zero_invalid(human_pos - root[:, None, :], valid)
    vel_rootrel = _zero_by_mask(human_vel - root_vel[:, None, :], vel_mask)
    acc_rootrel = _zero_by_mask(human_acc - root_acc[:, None, :], acc_mask)

    object_rot6d = _zero_invalid(groups.object_rot6d.astype(np.float32, copy=False), valid)
    object_trans = _zero_invalid(groups.object_trans.astype(np.float32, copy=False), valid)
    object_vel9 = _zero_invalid(groups.object_vel.astype(np.float32, copy=False), valid)
    object_trans_vel = _zero_by_mask(diff_pad_last_valid(object_trans, valid), vel_mask)
    object_trans_acc = _zero_by_mask(diff_pad_last_valid(object_trans_vel, vel_mask), acc_mask)
    object_raw = Stage1ObjectRaw(
        rot6d=object_rot6d,
        rotmat=_zero_invalid(rot6d_to_matrix_np(object_rot6d), valid),
        trans=object_trans,
        trans_vel=object_trans_vel,
        trans_acc=object_trans_acc,
        vel9_official=object_vel9,
        rot6d_vel_official=object_vel9[..., 0:6],
        trans_vel_official=object_vel9[..., 6:9],
    )

    relation = _zero_invalid(groups.relation.astype(np.float32, copy=False), valid)
    delta_m2o = relation[..., 0:3]
    delta_o2m = -delta_m2o
    point_canon = relation[..., 3:6]
    delta_vel = _zero_by_mask(diff_pad_last_valid(delta_m2o, valid), vel_mask)
    delta_acc = _zero_by_mask(diff_pad_last_valid(delta_vel, vel_mask), acc_mask)
    dist = _zero_invalid(safe_norm(delta_m2o, axis=-1, keepdims=True), valid)

    ground_marker = object_trans.copy()
    ground_marker[..., 1] = 0.0
    ground_marker = _zero_invalid(ground_marker, valid)
    nearest_point_world = np.zeros_like(delta_m2o, dtype=np.float32)
    nearest_point_world[..., :77, :] = human_pos + delta_m2o[..., :77, :]
    nearest_point_world[..., 77, :] = ground_marker + delta_m2o[..., 77, :]
    nearest_point_world = _zero_invalid(nearest_point_world, valid)
    nearest_point_vel = _zero_by_mask(diff_pad_last_valid(nearest_point_world, valid), vel_mask)
    nearest_point_acc = _zero_by_mask(diff_pad_last_valid(nearest_point_vel, vel_mask), acc_mask)

    root_object_delta = _zero_invalid(object_trans - root, valid)
    root_object_dist = _zero_invalid(safe_norm(root_object_delta, axis=-1, keepdims=True), valid)
    root_object_delta_vel = _zero_by_mask(diff_pad_last_valid(root_object_delta, valid), vel_mask)
    root_object_delta_acc = _zero_by_mask(diff_pad_last_valid(root_object_delta_vel, vel_mask), acc_mask)

    human_raw = Stage1HumanRaw(
        pos=human_pos,
        vel=human_vel,
        acc=human_acc,
        vel_official=human_vel_official,
        root=root,
        root_vel=root_vel,
        root_acc=root_acc,
        pos_rootrel=pos_rootrel,
        vel_rootrel=vel_rootrel,
        acc_rootrel=acc_rootrel,
        left_heights=left_heights,
        right_heights=right_heights,
    )
    relation_raw = Stage1RelationRaw(
        delta_m2o=delta_m2o,
        delta_o2m=delta_o2m,
        point_canon=point_canon,
        dist=dist,
        delta_vel=delta_vel,
        delta_acc=delta_acc,
        nearest_point_world=nearest_point_world,
        nearest_point_vel=nearest_point_vel,
        nearest_point_acc=nearest_point_acc,
        ground_marker=ground_marker,
        root_object_delta=root_object_delta,
        root_object_dist=root_object_dist,
        root_object_delta_vel=root_object_delta_vel,
        root_object_delta_acc=root_object_delta_acc,
    )
    return {
        "human": _dataclass_to_dict(human_raw),
        "object": _dataclass_to_dict(object_raw),
        "relation": _dataclass_to_dict(relation_raw),
    }


def _normalize(x: np.ndarray, eps: float) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, eps)


def _zero_invalid(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return _zero_by_mask(x, np.asarray(valid_mask, dtype=bool))


def _zero_by_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    expand_shape = tuple(mask.shape) + (1,) * (out.ndim - mask.ndim)
    out *= mask.reshape(expand_shape).astype(np.float32)
    return out


def _dataclass_to_dict(obj: Any) -> dict:
    return {name: getattr(obj, name) for name in obj.__dataclass_fields__}
