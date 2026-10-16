from __future__ import annotations

import numpy as np


def pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lhs = _as_matrix(a, name="a")
    rhs = _as_matrix(b, name="b")
    if lhs.shape[1] != rhs.shape[1]:
        raise ValueError(f"feature dims differ: {lhs.shape[1]} vs {rhs.shape[1]}")
    diff = lhs[:, None, :] - rhs[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def r_precision(
    text_features: np.ndarray,
    motion_features: np.ndarray,
    *,
    top_k: tuple[int, ...] = (1, 2, 3),
) -> dict[str, float]:
    dist = pairwise_l2(text_features, motion_features)
    if dist.shape[0] != dist.shape[1]:
        raise ValueError("R-Precision expects matching [N, D] text and motion features")
    ranking = np.argsort(dist, axis=1)
    target = np.arange(dist.shape[0])[:, None]
    out: dict[str, float] = {}
    for k in top_k:
        if k <= 0:
            raise ValueError(f"top-k values must be positive, got {k}")
        capped = min(k, dist.shape[1])
        out[f"r_precision_top{int(k)}"] = float((ranking[:, :capped] == target).any(axis=1).mean())
    return out


def r_precision_pooled(
    text_features: np.ndarray,
    motion_features: np.ndarray,
    *,
    pool_size: int = 32,
    top_k: tuple[int, ...] = (1, 2, 3),
    seed: int | None = None,
    drop_remainder: bool = True,
) -> dict[str, float]:
    """R-Precision averaged over InterAct-style retrieval pools.

    The reference evaluator computes retrieval within each evaluation batch
    rather than against the full dataset. ``pool_size=32`` mirrors the common
    InterAct/T2M protocol. Inputs may be deterministically shuffled first.
    """

    text = _as_matrix(text_features, name="text_features")
    motion = _as_matrix(motion_features, name="motion_features")
    if text.shape != motion.shape:
        raise ValueError(f"text and motion feature shapes must match, got {text.shape} and {motion.shape}")
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")

    order = np.arange(text.shape[0])
    if seed is not None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(order)
    text = text[order]
    motion = motion[order]

    totals = {f"r_precision_top{int(k)}": 0.0 for k in top_k}
    total_count = 0
    for start in range(0, text.shape[0], pool_size):
        end = min(start + pool_size, text.shape[0])
        if end - start < pool_size and drop_remainder:
            continue
        if end == start:
            continue
        scores = r_precision(text[start:end], motion[start:end], top_k=top_k)
        count = end - start
        for key, value in scores.items():
            totals[key] += float(value) * count
        total_count += count
    if total_count == 0:
        raise ValueError("no retrieval pools were evaluated; reduce pool_size or set drop_remainder=False")
    return {key: value / total_count for key, value in totals.items()}


def mm_dist(text_features: np.ndarray, motion_features: np.ndarray) -> float:
    dist = pairwise_l2(text_features, motion_features)
    if dist.shape[0] != dist.shape[1]:
        raise ValueError("MM-Dist expects matching [N, D] text and motion features")
    return float(np.diag(dist).mean())


def matching_score(text_features: np.ndarray, motion_features: np.ndarray) -> float:
    return mm_dist(text_features, motion_features)


def _as_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr
