from __future__ import annotations

import numpy as np


def marker_object_penetration_metrics(
    markers_world: np.ndarray,
    mesh_vertices_local: np.ndarray,
    mesh_faces: np.ndarray,
    object_rotmat: np.ndarray,
    object_trans: np.ndarray,
    *,
    frame_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Marker-vs-object penetration via a signed-distance query against the posed mesh.

    Markers are transformed into the object's local frame (the exact inverse of
    ``object_points_world``: ``local = (world - trans) @ R^T``) and queried against the
    static object mesh, so the mesh is built once. ``trimesh.signed_distance`` returns
    POSITIVE inside the mesh, matching :func:`sdf_penetration_metrics`. Best-effort:
    a non-watertight mesh or a missing ``trimesh`` returns ``skipped_count``.
    """

    markers = np.asarray(markers_world, dtype=np.float64)
    R = np.asarray(object_rotmat, dtype=np.float64)
    t = np.asarray(object_trans, dtype=np.float64)
    if markers.ndim != 3 or markers.shape[-1] != 3:
        raise ValueError(f"markers_world must be [T,M,3], got {markers.shape}")
    if R.shape != (markers.shape[0], 3, 3) or t.shape != (markers.shape[0], 3):
        raise ValueError("object_rotmat/object_trans must be [T,3,3]/[T,3] matching markers")
    try:
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh_vertices_local, dtype=np.float64),
            faces=np.asarray(mesh_faces),
            process=False,
        )
        local = np.einsum("tmj,tcj->tmc", markers - t[:, None, :], R)  # inverse pose
        signed = trimesh.proximity.signed_distance(mesh, local.reshape(-1, 3)).reshape(markers.shape[:2])
    except Exception:
        return sdf_penetration_metrics(np.zeros(markers.shape[:2]), mask=np.zeros(markers.shape[:2], dtype=bool), skipped_count=int(markers.shape[0]))
    if frame_mask is None:
        mask = np.ones(markers.shape[:2], dtype=bool)
    else:
        fm = np.asarray(frame_mask, dtype=bool).reshape(-1)
        mask = np.broadcast_to(fm[:, None], markers.shape[:2]).copy()
    signed = np.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
    return sdf_penetration_metrics(signed, mask=mask)


def sdf_penetration_metrics(
    signed_depth_m: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    skipped_count: int = 0,
) -> dict[str, float]:
    """Summarize SDF-style penetration depths.

    Positive signed depth means the query point is inside the object.
    """

    depth = np.asarray(signed_depth_m, dtype=np.float64)
    if not np.isfinite(depth).all():
        raise ValueError("signed_depth_m contains non-finite values")
    if mask is None:
        valid = np.ones(depth.shape, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != depth.shape:
            raise ValueError(f"mask shape {valid.shape} does not match signed_depth_m shape {depth.shape}")
    if not valid.any():
        return {
            "sdf_penetration_ratio": 0.0,
            "sdf_penetration_mean_cm": 0.0,
            "sdf_penetration_p95_cm": 0.0,
            "sdf_penetration_max_cm": 0.0,
            "sdf_penetration_valid_count": 0.0,
            "sdf_penetration_skipped_count": float(skipped_count),
        }
    positive = np.clip(depth[valid], 0.0, None)
    penetrating = positive > 0.0
    if not penetrating.any():
        mean_cm = p95_cm = max_cm = 0.0
    else:
        values_cm = positive[penetrating] * 100.0
        mean_cm = float(values_cm.mean())
        p95_cm = float(np.percentile(values_cm, 95.0))
        max_cm = float(values_cm.max())
    return {
        "sdf_penetration_ratio": float(penetrating.mean()),
        "sdf_penetration_mean_cm": mean_cm,
        "sdf_penetration_p95_cm": p95_cm,
        "sdf_penetration_max_cm": max_cm,
        "sdf_penetration_valid_count": float(valid.sum()),
        "sdf_penetration_skipped_count": float(skipped_count),
    }
