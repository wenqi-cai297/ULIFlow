from __future__ import annotations

from typing import Any


_VALID_ANCHOR_MODES = {"hybrid", "absolute_grid"}


def apply_stage1_anchor_override(
    config: dict[str, Any],
    *,
    anchor_mode: str | None = None,
    anchor_stride: int | None = None,
    anchor_grid_origin: int | None = None,
) -> dict[str, Any]:
    """Mutate a Stage1 config with an explicit temporal-anchor override.

    This is used by diagnostics that need to evaluate an existing checkpoint
    under a different non-learned anchor selection policy without editing the
    checkpoint or run directory.
    """
    if anchor_mode is None and anchor_stride is None and anchor_grid_origin is None:
        return {}

    mode = None if anchor_mode is None else str(anchor_mode).lower()
    if mode is not None and mode not in _VALID_ANCHOR_MODES:
        raise ValueError(f"anchor_mode must be one of {sorted(_VALID_ANCHOR_MODES)}, got {anchor_mode!r}.")
    if anchor_stride is not None and int(anchor_stride) <= 0:
        raise ValueError(f"anchor_stride must be positive, got {anchor_stride}.")

    temporal = (
        config.setdefault("model", {})
        .setdefault("stage1", {})
        .setdefault("token_compressor", {})
        .setdefault("temporal", {})
    )
    before = {
        "anchor_mode": temporal.get("anchor_mode"),
        "anchor_stride": temporal.get("anchor_stride"),
        "anchor_grid_origin": temporal.get("anchor_grid_origin"),
    }

    if mode is not None:
        temporal["anchor_mode"] = mode
    if anchor_stride is not None:
        temporal["anchor_stride"] = int(anchor_stride)
    if anchor_grid_origin is not None:
        temporal["anchor_grid_origin"] = int(anchor_grid_origin)

    after = {
        "anchor_mode": temporal.get("anchor_mode"),
        "anchor_stride": temporal.get("anchor_stride"),
        "anchor_grid_origin": temporal.get("anchor_grid_origin"),
    }
    return {"before": before, "after": after}
