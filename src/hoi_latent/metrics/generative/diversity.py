from __future__ import annotations

import numpy as np


def feature_diversity(
    features: np.ndarray,
    *,
    num_pairs: int = 300,
    seed: int = 0,
    avoid_self_pairs: bool = False,
) -> dict[str, float]:
    matrix = _as_matrix(features, name="features")
    if matrix.shape[0] < 2:
        return {"diversity": 0.0, "pair_count": 0.0}
    i, j = _sample_pairs(matrix.shape[0], num_pairs=num_pairs, seed=seed, avoid_self_pairs=avoid_self_pairs)
    dist = np.linalg.norm(matrix[i] - matrix[j], axis=-1)
    return {"diversity": float(dist.mean()), "pair_count": float(dist.shape[0])}


def feature_multimodality(
    features_by_condition: np.ndarray,
    *,
    num_pairs: int = 300,
    seed: int = 0,
    avoid_self_pairs: bool = False,
) -> dict[str, float]:
    """Mean feature distance between samples generated for the same condition.

    ``features_by_condition`` is shaped ``[C, K, D]``.
    """

    arr = np.asarray(features_by_condition, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"features_by_condition must have shape [C, K, D], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("features_by_condition contains non-finite values")
    if arr.shape[0] == 0 or arr.shape[1] < 2:
        return {"multimodality": 0.0, "pair_count": 0.0}

    rng = np.random.default_rng(seed)
    per_condition = max(1, int(np.ceil(num_pairs / arr.shape[0])))
    values: list[np.ndarray] = []
    for condition_features in arr:
        i, j = _sample_pairs(
            condition_features.shape[0],
            num_pairs=per_condition,
            rng=rng,
            avoid_self_pairs=avoid_self_pairs,
        )
        values.append(np.linalg.norm(condition_features[i] - condition_features[j], axis=-1))
    distances = np.concatenate(values, axis=0)[:num_pairs]
    return {"multimodality": float(distances.mean()), "pair_count": float(distances.shape[0])}


def feature_multimodality_per_condition(
    features_by_condition: np.ndarray,
    *,
    pairs_per_condition: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """InterAct/T2M-style multimodality over repeated samples per condition.

    ``features_by_condition`` is shaped ``[C, K, D]``. For each condition, sample
    ``pairs_per_condition`` first indices and ``pairs_per_condition`` second
    indices from the K generated samples without replacement, then average the
    per-condition feature distances. This mirrors the common
    ``calculate_multimodality`` evaluator convention.
    """

    arr = np.asarray(features_by_condition, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"features_by_condition must have shape [C, K, D], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("features_by_condition contains non-finite values")
    if arr.shape[0] == 0 or arr.shape[1] < 2:
        return {"multimodality": 0.0, "pair_count": 0.0}
    count = int(pairs_per_condition)
    if count <= 0:
        return {"multimodality": 0.0, "pair_count": 0.0}
    if arr.shape[1] <= count:
        raise ValueError(
            "features_by_condition must contain more samples per condition than "
            f"pairs_per_condition, got K={arr.shape[1]}, pairs_per_condition={count}."
        )

    rng = np.random.default_rng(seed)
    first = rng.choice(arr.shape[1], size=count, replace=False)
    second = rng.choice(arr.shape[1], size=count, replace=False)
    dist = np.linalg.norm(arr[:, first] - arr[:, second], axis=-1)
    return {"multimodality": float(dist.mean()), "pair_count": float(dist.size)}


def _sample_pairs(
    count: int,
    *,
    num_pairs: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    avoid_self_pairs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if count < 2 or num_pairs <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    generator = rng if rng is not None else np.random.default_rng(seed)
    if avoid_self_pairs:
        i = generator.integers(0, count, size=num_pairs, endpoint=False)
        j = generator.integers(0, count - 1, size=num_pairs, endpoint=False)
        j = j + (j >= i)
    else:
        i = generator.choice(count, size=num_pairs, replace=count < num_pairs)
        j = generator.choice(count, size=num_pairs, replace=count < num_pairs)
    return i.astype(np.int64), j.astype(np.int64)


def _as_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr
