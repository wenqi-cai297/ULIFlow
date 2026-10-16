from __future__ import annotations

import numpy as np


def frechet_distance(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Compute FID-style Frechet distance between two feature sets.

    Inputs are shaped ``[N, D]``. This implementation intentionally uses only
    numpy so the formula layer does not add a SciPy dependency.
    """

    real = _as_feature_matrix(real_features, name="real_features")
    generated = _as_feature_matrix(generated_features, name="generated_features")
    if real.shape[1] != generated.shape[1]:
        raise ValueError(f"feature dims differ: {real.shape[1]} vs {generated.shape[1]}")

    mu_real = real.mean(axis=0)
    mu_generated = generated.mean(axis=0)
    cov_real = _covariance(real)
    cov_generated = _covariance(generated)

    diff = mu_real - mu_generated
    covmean_trace = _trace_sqrt_product(cov_real, cov_generated, eps=eps)
    fid = float(diff.dot(diff) + np.trace(cov_real) + np.trace(cov_generated) - 2.0 * covmean_trace)
    return max(fid, 0.0)


def _as_feature_matrix(features: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _covariance(features: np.ndarray) -> np.ndarray:
    if features.shape[0] == 1:
        return np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    return np.atleast_2d(np.cov(features, rowvar=False)).astype(np.float64, copy=False)


def _trace_sqrt_product(cov_a: np.ndarray, cov_b: np.ndarray, *, eps: float) -> float:
    if cov_a.shape != cov_b.shape:
        raise ValueError(f"covariance shapes differ: {cov_a.shape} vs {cov_b.shape}")
    if cov_a.size == 0:
        return 0.0

    product = cov_a @ cov_b
    eigvals = np.linalg.eigvals(product)
    if np.iscomplexobj(eigvals):
        if np.max(np.abs(eigvals.imag)) > 1e-5:
            # Numerical products of SPD matrices may have tiny imaginary parts;
            # larger imaginary residue means the input is not behaving like SPD.
            product = (cov_a + np.eye(cov_a.shape[0]) * eps) @ (cov_b + np.eye(cov_b.shape[0]) * eps)
            eigvals = np.linalg.eigvals(product)
        eigvals = eigvals.real
    eigvals = np.clip(eigvals, 0.0, None)
    return float(np.sqrt(eigvals).sum())
