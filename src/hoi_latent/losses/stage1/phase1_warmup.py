from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hoi_latent.data.transforms.structured_normalization import Stage1StructuredNormalizer
from hoi_latent.models.stage1.autoencoder.latent_bottleneck import (
    Stage1LatentFieldOutput,
    aggregate_mean_covariance_regularization,
    compute_aggregate_latent_stats,
)
from hoi_latent.models.stage1.decoder.readout_heads import Stage1CoarsePrediction


@dataclass
class Stage1LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


DEFAULT_HUMAN_FIELD_WEIGHTS = {
    "pos_rootrel": 1.0,
    "root": 1.0,
    "vel_rootrel": 0.25,
    "root_vel": 0.25,
    "acc_rootrel": 0.10,
    "root_acc": 0.10,
    "left_heights": 0.50,
    "right_heights": 0.50,
}

DEFAULT_OBJECT_FIELD_WEIGHTS = {
    "rot6d": 1.0,
    "trans": 1.0,
    "trans_vel": 0.25,
    "trans_acc": 0.10,
}

DEFAULT_RELATION_FIELD_WEIGHTS = {
    "delta_m2o": 1.0,
    "dist": 1.0,
    "point_canon": 0.50,
    "delta_vel": 0.25,
    "delta_acc": 0.10,
    "root_object_delta": 1.0,
    "root_object_dist": 1.0,
    "root_object_delta_vel": 0.25,
    "root_object_delta_acc": 0.10,
}


class Stage1Phase1WarmupLoss(nn.Module):
    """Phase 1 coarse reconstruction warm-up objective."""

    def __init__(
        self,
        *,
        normalizer: Stage1StructuredNormalizer | None = None,
        lambda_human: float = 1.0,
        lambda_object: float = 1.0,
        lambda_relation: float = 2.0,
        lambda_world_marker: float = 1.0,
        lambda_latent_cov_floor: float = 1.0e-4,
        smooth_l1_beta: float = 1.0,
        world_marker_error_scale_m: float = 0.05,
        world_marker_smooth_l1_beta: float = 1.0,
        latent_cov_start_step: int = 10000,
        latent_cov_end_step: int = 30000,
        human_field_weights: dict[str, float] | None = None,
        object_field_weights: dict[str, float] | None = None,
        relation_field_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if smooth_l1_beta <= 0.0:
            raise ValueError("smooth_l1_beta must be positive.")
        if world_marker_error_scale_m <= 0.0:
            raise ValueError("world_marker_error_scale_m must be positive.")
        if world_marker_smooth_l1_beta <= 0.0:
            raise ValueError("world_marker_smooth_l1_beta must be positive.")
        if latent_cov_start_step < 0 or latent_cov_end_step <= latent_cov_start_step:
            raise ValueError("latent_cov_end_step must be greater than latent_cov_start_step.")
        self.normalizer = normalizer
        self.lambda_human = float(lambda_human)
        self.lambda_object = float(lambda_object)
        self.lambda_relation = float(lambda_relation)
        self.lambda_world_marker = float(lambda_world_marker)
        self.lambda_latent_cov_floor = float(lambda_latent_cov_floor)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.world_marker_error_scale_m = float(world_marker_error_scale_m)
        self.world_marker_smooth_l1_beta = float(world_marker_smooth_l1_beta)
        self.latent_cov_start_step = int(latent_cov_start_step)
        self.latent_cov_end_step = int(latent_cov_end_step)
        self.human_field_weights = _merged_weights(DEFAULT_HUMAN_FIELD_WEIGHTS, human_field_weights)
        self.object_field_weights = _merged_weights(DEFAULT_OBJECT_FIELD_WEIGHTS, object_field_weights)
        self.relation_field_weights = _merged_weights(DEFAULT_RELATION_FIELD_WEIGHTS, relation_field_weights)

    def latent_cov_floor_weight(self, global_step: int) -> float:
        if global_step < self.latent_cov_start_step:
            return 0.0
        if global_step < self.latent_cov_end_step:
            progress = (global_step - self.latent_cov_start_step) / (
                self.latent_cov_end_step - self.latent_cov_start_step
            )
            return progress * self.lambda_latent_cov_floor
        return self.lambda_latent_cov_floor

    def forward(
        self,
        coarse: Stage1CoarsePrediction,
        latent: Stage1LatentFieldOutput,
        batch: dict,
        *,
        global_step: int,
    ) -> Stage1LossOutput:
        frame_mask = _infer_frame_mask(batch)
        vel_mask = _derivative_mask(frame_mask, order=1)
        acc_mask = _derivative_mask(frame_mask, order=2)
        model = _require_mapping(batch, "model")

        components: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, torch.Tensor] = {}

        human_loss = self._group_loss(
            group_name="human",
            prediction=coarse.human,
            target=_require_mapping(model, "human"),
            field_weights=self.human_field_weights,
            base_frame_mask=frame_mask,
            marker_mask=_human_marker_mask(batch, frame_mask),
            vel_mask=vel_mask,
            acc_mask=acc_mask,
            components=components,
        )
        object_loss = self._group_loss(
            group_name="object",
            prediction=coarse.object,
            target=_require_mapping(model, "object"),
            field_weights=self.object_field_weights,
            base_frame_mask=frame_mask,
            marker_mask=None,
            vel_mask=vel_mask,
            acc_mask=acc_mask,
            components=components,
        )
        relation_loss = self._group_loss(
            group_name="relation",
            prediction=coarse.relation,
            target=_require_mapping(model, "relation"),
            field_weights=self.relation_field_weights,
            base_frame_mask=frame_mask,
            marker_mask=_relation_entry_mask(batch, frame_mask),
            vel_mask=vel_mask,
            acc_mask=acc_mask,
            components=components,
        )

        weighted_human = human_loss * self.lambda_human
        weighted_object = object_loss * self.lambda_object
        weighted_relation = relation_loss * self.lambda_relation
        components["human"] = weighted_human
        components["object"] = weighted_object
        components["relation"] = weighted_relation

        world_marker_scaled = self._world_marker_loss(coarse, batch, frame_mask, components, diagnostics)
        weighted_world_marker = world_marker_scaled * self.lambda_world_marker
        components["world_marker"] = weighted_world_marker

        cov_losses = aggregate_mean_covariance_regularization(latent.z_rec, latent.mask)
        latent_stats = compute_aggregate_latent_stats(
            latent.z_rec.detach(),
            latent.mask,
            modality_ids=latent.modality_ids,
            token_type_ids=latent.token_type_ids,
            time_ids=latent.time_ids,
        )
        cov_weight = self.latent_cov_floor_weight(int(global_step))
        latent_cov = cov_losses["total"] * cov_weight
        components["latent_cov"] = latent_cov
        components["latent_cov_mean"] = cov_losses["mean"]
        components["latent_cov_offdiag"] = cov_losses["cov_offdiag"]
        components["latent_cov_diag"] = cov_losses["cov_diag"]

        total = weighted_human + weighted_object + weighted_relation + weighted_world_marker + latent_cov
        components["total"] = total
        diagnostics.update(
            {
                "valid_frame_count": frame_mask.to(dtype=torch.long).sum().detach(),
                "valid_human_marker_count": _human_marker_mask(batch, frame_mask).to(dtype=torch.long).sum().detach(),
                "valid_relation_entry_count": _relation_entry_mask(batch, frame_mask).to(dtype=torch.long).sum().detach(),
                "valid_velocity_frame_count": vel_mask.to(dtype=torch.long).sum().detach(),
                "valid_acceleration_frame_count": acc_mask.to(dtype=torch.long).sum().detach(),
                "lambda_cov_floor": latent.z_rec.new_tensor(cov_weight).detach(),
                "latent_z_effective_rank": latent_stats["effective_rank"].detach(),
                "relation_dist_loss": components["relation.dist"].detach(),
                "relation_delta_m2o_loss": components["relation.delta_m2o"].detach(),
            }
        )
        return Stage1LossOutput(total=total, components=components, diagnostics=diagnostics)

    def _group_loss(
        self,
        *,
        group_name: str,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        field_weights: dict[str, float],
        base_frame_mask: torch.Tensor,
        marker_mask: torch.Tensor | None,
        vel_mask: torch.Tensor,
        acc_mask: torch.Tensor,
        components: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = _zero_like_reference(prediction, target)
        for field_name, weight in field_weights.items():
            pred = _require_tensor(prediction, field_name, f"coarse.{group_name}")
            tgt = _require_tensor(target, field_name, f"batch.model.{group_name}").to(device=pred.device, dtype=pred.dtype)
            if tuple(pred.shape) != tuple(tgt.shape):
                raise ValueError(
                    f"{group_name}.{field_name} prediction/target shapes differ: "
                    f"{tuple(pred.shape)} vs {tuple(tgt.shape)}."
                )
            mask = _field_mask(field_name, pred, base_frame_mask, marker_mask, vel_mask, acc_mask)
            field_loss = _masked_smooth_l1(pred, tgt, mask, beta=self.smooth_l1_beta)
            components[f"{group_name}.{field_name}"] = field_loss
            total = total + field_loss * float(weight)
        return total

    def _world_marker_loss(
        self,
        coarse: Stage1CoarsePrediction,
        batch: dict,
        frame_mask: torch.Tensor,
        components: dict[str, torch.Tensor],
        diagnostics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        reference = coarse.human["pos_rootrel"]
        zero = reference.sum() * 0.0
        if self.lambda_world_marker == 0.0:
            components["world_marker_scaled_loss"] = zero
            diagnostics["world_marker_target_source"] = zero.new_tensor(0)
            diagnostics["world_marker_error_scale_m"] = zero.new_tensor(self.world_marker_error_scale_m)
            diagnostics["world_marker_mpjpe_m"] = zero.detach()
            return zero
        if self.normalizer is None:
            raise ValueError(
                "Stage1Phase1WarmupLoss requires structured normalizer stats for world-space marker loss."
            )

        pred_pos = self.normalizer.denormalize_group_torch(
            "human", "pos_rootrel", coarse.human["pos_rootrel"], require_stats=True
        )
        pred_root = self.normalizer.denormalize_group_torch("human", "root", coarse.human["root"], require_stats=True)
        pred_marker = pred_pos + pred_root[:, :, None, :]

        target_marker, source_id = self._world_marker_target(batch, pred_marker)
        mask = _human_marker_mask(batch, frame_mask).to(device=pred_marker.device)
        marker_error_m = pred_marker - target_marker
        marker_error_unit = marker_error_m / self.world_marker_error_scale_m
        scaled_loss = _masked_smooth_l1(
            marker_error_unit,
            torch.zeros_like(marker_error_unit),
            mask,
            beta=self.world_marker_smooth_l1_beta,
        )
        components["world_marker_scaled_loss"] = scaled_loss
        mpjpe = _masked_marker_mpjpe(marker_error_m, mask)
        diagnostics["world_marker_target_source"] = pred_marker.new_tensor(source_id)
        diagnostics["world_marker_error_scale_m"] = pred_marker.new_tensor(self.world_marker_error_scale_m)
        diagnostics["world_marker_mpjpe_m"] = mpjpe.detach()
        return scaled_loss

    def _world_marker_target(self, batch: dict, reference: torch.Tensor) -> tuple[torch.Tensor, int]:
        raw = batch.get("raw")
        if isinstance(raw, dict):
            raw_human = raw.get("human")
            if isinstance(raw_human, dict) and isinstance(raw_human.get("pos"), torch.Tensor):
                return raw_human["pos"].to(device=reference.device, dtype=reference.dtype), 1
        model_human = _require_mapping(_require_mapping(batch, "model"), "human")
        target_pos = self.normalizer.denormalize_group_torch(
            "human",
            "pos_rootrel",
            _require_tensor(model_human, "pos_rootrel", "batch.model.human").to(
                device=reference.device, dtype=reference.dtype
            ),
            require_stats=True,
        )
        target_root = self.normalizer.denormalize_group_torch(
            "human",
            "root",
            _require_tensor(model_human, "root", "batch.model.human").to(device=reference.device, dtype=reference.dtype),
            require_stats=True,
        )
        return target_pos + target_root[:, :, None, :], 0


def _merged_weights(defaults: dict[str, float], override: dict[str, float] | None) -> dict[str, float]:
    merged = dict(defaults)
    if override:
        merged.update({key: float(value) for key, value in override.items()})
    return merged


def _infer_frame_mask(batch: dict) -> torch.Tensor:
    masks = batch.get("masks")
    if not isinstance(masks, dict):
        raise KeyError("Stage1Phase1WarmupLoss requires batch['masks'].")
    for key in ("frame", "valid", "object"):
        value = masks.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value.to(dtype=torch.bool)
    raise KeyError("Stage1Phase1WarmupLoss requires batch['masks']['frame'], ['valid'], or ['object'].")


def _derivative_mask(valid_mask: torch.Tensor, order: int) -> torch.Tensor:
    valid = valid_mask.to(dtype=torch.bool)
    out = torch.zeros_like(valid, dtype=torch.bool)
    if order <= 0:
        return valid.clone()
    if valid.shape[1] <= order:
        return out
    segment = valid[:, : valid.shape[1] - order].clone()
    for offset in range(1, order + 1):
        segment = segment & valid[:, offset : valid.shape[1] - order + offset]
    out[:, : valid.shape[1] - order] = segment
    return out


def _field_mask(
    field_name: str,
    value: torch.Tensor,
    frame_mask: torch.Tensor,
    marker_mask: torch.Tensor | None,
    vel_mask: torch.Tensor,
    acc_mask: torch.Tensor,
) -> torch.Tensor:
    lower = field_name.lower()
    time_mask = acc_mask if "acc" in lower else vel_mask if "vel" in lower else frame_mask
    time_mask = time_mask.to(device=value.device, dtype=torch.bool)
    if value.ndim >= 4:
        if marker_mask is None:
            raise ValueError(f"{field_name} needs a per-entry mask for shape {tuple(value.shape)}.")
        mask = marker_mask.to(device=value.device, dtype=torch.bool) & time_mask[:, :, None]
        if tuple(mask.shape) != tuple(value.shape[:-1]):
            raise ValueError(f"{field_name} mask shape {tuple(mask.shape)} does not match {tuple(value.shape[:-1])}.")
        return mask
    if tuple(time_mask.shape) != tuple(value.shape[:2]):
        raise ValueError(f"{field_name} mask shape {tuple(time_mask.shape)} does not match {tuple(value.shape[:2])}.")
    return time_mask


def _masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    mask = mask.to(device=loss.device, dtype=torch.bool)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(loss)
    count = expanded.to(dtype=loss.dtype).sum()
    return (loss * expanded.to(dtype=loss.dtype)).sum() / count.clamp_min(1.0)


def _masked_marker_mpjpe(marker_error_m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dist = marker_error_m.norm(dim=-1)
    mask = mask.to(device=dist.device, dtype=torch.bool)
    count = mask.to(dtype=dist.dtype).sum()
    return (dist * mask.to(dtype=dist.dtype)).sum() / count.clamp_min(1.0)


def _human_marker_mask(batch: dict, frame_mask: torch.Tensor) -> torch.Tensor:
    masks = batch.get("masks", {})
    human = masks.get("human") if isinstance(masks, dict) else None
    if isinstance(human, torch.Tensor) and human.ndim == 3:
        return human.to(device=frame_mask.device, dtype=torch.bool) & frame_mask[:, :, None]
    return frame_mask[:, :, None].expand(*frame_mask.shape, 77)


def _relation_entry_mask(batch: dict, frame_mask: torch.Tensor) -> torch.Tensor:
    masks = batch.get("masks", {})
    relation = masks.get("relation") if isinstance(masks, dict) else None
    if isinstance(relation, torch.Tensor) and relation.ndim == 3:
        return relation.to(device=frame_mask.device, dtype=torch.bool) & frame_mask[:, :, None]
    return frame_mask[:, :, None].expand(*frame_mask.shape, 78)


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Expected mapping at key {key!r}.")
    return value


def _require_tensor(mapping: dict[str, Any], key: str, context: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"{context} missing tensor field {key!r}.")
    return value


def _zero_like_reference(*mappings: dict[str, torch.Tensor]) -> torch.Tensor:
    for mapping in mappings:
        for value in mapping.values():
            if isinstance(value, torch.Tensor):
                return value.sum() * 0.0
    raise ValueError("Cannot build zero loss without any tensor reference.")
