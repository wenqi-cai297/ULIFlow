from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hoi_latent.metrics.generative import (
    feature_diversity,
    feature_multimodality,
    feature_multimodality_per_condition,
    frechet_distance,
    matching_score,
    r_precision_pooled,
)


@dataclass
class Stage2FeatureMetrics:
    values: dict[str, float] = field(default_factory=dict)


def compute_stage2_feature_metrics(
    *,
    real_motion_features: np.ndarray,
    generated_motion_features: np.ndarray,
    text_features: np.ndarray | None = None,
    multimodal_generated_features: np.ndarray | None = None,
    r_precision_pool_size: int = 32,
    r_precision_seed: int = 0,
    diversity_pairs: int = 300,
    multimodality_pairs_per_condition: int | None = None,
) -> Stage2FeatureMetrics:
    real = _as_matrix(real_motion_features, "real_motion_features")
    generated = _as_matrix(generated_motion_features, "generated_motion_features")
    if real.shape[1] != generated.shape[1]:
        raise ValueError(f"feature dims differ: {real.shape[1]} vs {generated.shape[1]}.")
    values: dict[str, float] = {
        "feature/fid": frechet_distance(real, generated),
        **{f"feature/real_{k}": v for k, v in feature_diversity(real, num_pairs=diversity_pairs).items()},
        **{f"feature/generated_{k}": v for k, v in feature_diversity(generated, num_pairs=diversity_pairs).items()},
    }
    if text_features is not None:
        text = _as_matrix(text_features, "text_features")
        if text.shape != generated.shape:
            raise ValueError(f"text and generated feature shapes must match, got {text.shape} and {generated.shape}.")
        pooled = r_precision_pooled(
            text,
            generated,
            pool_size=r_precision_pool_size,
            top_k=(1, 2, 3),
            seed=r_precision_seed,
            drop_remainder=text.shape[0] >= r_precision_pool_size,
        )
        values.update({f"feature/{key}": value for key, value in pooled.items()})
        values["feature/mm_dist"] = matching_score(text, generated)
    if multimodal_generated_features is not None:
        if multimodality_pairs_per_condition is not None:
            multimodality_values = feature_multimodality_per_condition(
                multimodal_generated_features,
                pairs_per_condition=int(multimodality_pairs_per_condition),
                seed=r_precision_seed,
            )
        else:
            multimodality_values = feature_multimodality(
                multimodal_generated_features,
                num_pairs=diversity_pairs,
            )
        values.update(
            {
                f"feature/{key}": value
                for key, value in multimodality_values.items()
            }
        )
    return Stage2FeatureMetrics(values=values)


def _as_matrix(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [N,D], got {arr.shape}.")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr
