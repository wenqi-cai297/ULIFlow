from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hoi_latent.data.transforms.structured_normalization import Stage1StructuredNormalizer
from hoi_latent.losses.stage1.human_jitter import human_world_marker_finite_difference_loss
from hoi_latent.losses.stage1.object_jitter import (
    object_translation_finite_difference_loss,
    object_world_points_loss,
)
from hoi_latent.losses.stage1.phase1_warmup import Stage1LossOutput, Stage1Phase1WarmupLoss
from hoi_latent.models.stage1.autoencoder.latent_bottleneck import Stage1LatentFieldOutput
from hoi_latent.models.stage1.decoder.readout_heads import Stage1CoarsePrediction


class Stage1DecoderPostTrainingLoss(nn.Module):
    """Decoder-only post-training objective for whitened latent jitter robustness."""

    def __init__(
        self,
        *,
        normalizer: Stage1StructuredNormalizer | None = None,
        clean_sigma_epsilon: float = 0.0,
        lambda_object_world_points: float = 0.0,
        object_world_point_error_scale_m: float = 0.05,
        object_world_point_smooth_l1_beta: float = 1.0,
        lambda_object_fd_velocity: float = 0.0,
        lambda_object_fd_acceleration: float = 0.0,
        object_fd_velocity_error_scale_m: float = 0.05,
        object_fd_acceleration_error_scale_m: float = 0.02,
        object_fd_smooth_l1_beta: float = 1.0,
        lambda_human_fd_velocity: float = 0.0,
        lambda_human_fd_acceleration: float = 0.0,
        human_fd_velocity_error_scale_m: float = 0.05,
        human_fd_acceleration_error_scale_m: float = 0.02,
        human_fd_smooth_l1_beta: float = 1.0,
        **phase1_loss_kwargs: Any,
    ) -> None:
        super().__init__()
        for name, value in {
            "lambda_object_world_points": lambda_object_world_points,
            "lambda_object_fd_velocity": lambda_object_fd_velocity,
            "lambda_object_fd_acceleration": lambda_object_fd_acceleration,
            "lambda_human_fd_velocity": lambda_human_fd_velocity,
            "lambda_human_fd_acceleration": lambda_human_fd_acceleration,
        }.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        for name, value in {
            "object_world_point_error_scale_m": object_world_point_error_scale_m,
            "object_world_point_smooth_l1_beta": object_world_point_smooth_l1_beta,
            "object_fd_velocity_error_scale_m": object_fd_velocity_error_scale_m,
            "object_fd_acceleration_error_scale_m": object_fd_acceleration_error_scale_m,
            "object_fd_smooth_l1_beta": object_fd_smooth_l1_beta,
            "human_fd_velocity_error_scale_m": human_fd_velocity_error_scale_m,
            "human_fd_acceleration_error_scale_m": human_fd_acceleration_error_scale_m,
            "human_fd_smooth_l1_beta": human_fd_smooth_l1_beta,
        }.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        phase1_loss_kwargs = dict(phase1_loss_kwargs)
        phase1_loss_kwargs["lambda_latent_cov_floor"] = 0.0
        phase1_loss_kwargs["normalizer"] = normalizer
        self.reconstruction_loss = Stage1Phase1WarmupLoss(**phase1_loss_kwargs)
        if clean_sigma_epsilon < 0.0:
            raise ValueError("clean_sigma_epsilon must be non-negative.")
        self.normalizer = normalizer
        self.clean_sigma_epsilon = float(clean_sigma_epsilon)
        self.lambda_object_world_points = float(lambda_object_world_points)
        self.object_world_point_error_scale_m = float(object_world_point_error_scale_m)
        self.object_world_point_smooth_l1_beta = float(object_world_point_smooth_l1_beta)
        self.lambda_object_fd_velocity = float(lambda_object_fd_velocity)
        self.lambda_object_fd_acceleration = float(lambda_object_fd_acceleration)
        self.object_fd_velocity_error_scale_m = float(object_fd_velocity_error_scale_m)
        self.object_fd_acceleration_error_scale_m = float(object_fd_acceleration_error_scale_m)
        self.object_fd_smooth_l1_beta = float(object_fd_smooth_l1_beta)
        self.lambda_human_fd_velocity = float(lambda_human_fd_velocity)
        self.lambda_human_fd_acceleration = float(lambda_human_fd_acceleration)
        self.human_fd_velocity_error_scale_m = float(human_fd_velocity_error_scale_m)
        self.human_fd_acceleration_error_scale_m = float(human_fd_acceleration_error_scale_m)
        self.human_fd_smooth_l1_beta = float(human_fd_smooth_l1_beta)

    def forward(
        self,
        prediction: Stage1CoarsePrediction,
        latent: Stage1LatentFieldOutput,
        batch: dict,
        *,
        global_step: int,
        sigma_white: torch.Tensor,
        compute_group_diagnostics: bool = True,
    ) -> Stage1LossOutput:
        if sigma_white.ndim != 1:
            raise ValueError(f"sigma_white must be [B], got {tuple(sigma_white.shape)}.")
        batch_size = _batch_size_from_prediction(prediction)
        if int(sigma_white.shape[0]) != batch_size:
            raise ValueError(f"sigma_white B={sigma_white.shape[0]} does not match prediction B={batch_size}.")

        pooled = self.reconstruction_loss(prediction, latent, batch, global_step=global_step)
        components = dict(pooled.components)
        diagnostics = dict(pooled.diagnostics)
        zero = pooled.total.detach() * 0.0
        components["latent_cov"] = zero
        components["latent_cov_mean"] = zero
        components["latent_cov_offdiag"] = zero
        components["latent_cov_diag"] = zero
        total = pooled.total
        total = self._add_object_jitter_terms(total, prediction, batch, components, diagnostics)
        total = self._add_human_jitter_terms(total, prediction, batch, components, diagnostics)
        components["total"] = total

        clean_mask = sigma_white.detach() <= self.clean_sigma_epsilon
        noisy_mask = ~clean_mask
        diagnostics.update(
            {
                "sigma_white_mean": sigma_white.detach().float().mean(),
                "sigma_white_max": sigma_white.detach().float().max(),
                "sigma_zero_fraction": clean_mask.to(dtype=torch.float32).mean().to(device=pooled.total.device),
                "sigma_nonzero_fraction": noisy_mask.to(dtype=torch.float32).mean().to(device=pooled.total.device),
            }
        )
        if compute_group_diagnostics:
            self._add_group_diagnostics(
                diagnostics,
                group_name="clean_samples",
                prediction=prediction,
                latent=latent,
                batch=batch,
                sample_mask=clean_mask,
                global_step=global_step,
            )
            self._add_group_diagnostics(
                diagnostics,
                group_name="noisy_samples",
                prediction=prediction,
                latent=latent,
                batch=batch,
                sample_mask=noisy_mask,
                global_step=global_step,
            )
        else:
            self._add_skipped_group_diagnostics(diagnostics, "clean_samples", latent, clean_mask)
            self._add_skipped_group_diagnostics(diagnostics, "noisy_samples", latent, noisy_mask)
        return Stage1LossOutput(total=total, components=components, diagnostics=diagnostics)

    def _add_object_jitter_terms(
        self,
        total: torch.Tensor,
        prediction: Stage1CoarsePrediction,
        batch: dict,
        components: dict[str, torch.Tensor],
        diagnostics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if (
            self.lambda_object_world_points == 0.0
            and self.lambda_object_fd_velocity == 0.0
            and self.lambda_object_fd_acceleration == 0.0
        ):
            return total
        if self.normalizer is None:
            raise ValueError("Object jitter losses require structured normalizer stats.")

        if self.lambda_object_world_points > 0.0:
            world = object_world_points_loss(
                prediction,
                batch,
                structured_normalizer=self.normalizer,
                error_scale_m=self.object_world_point_error_scale_m,
                beta=self.object_world_point_smooth_l1_beta,
            )
            weighted = world.loss * self.lambda_object_world_points
            components["object_world_points_scaled_loss"] = world.loss
            components["object_world_points"] = weighted
            diagnostics.update(world.diagnostics)
            total = total + weighted

        if self.lambda_object_fd_velocity > 0.0 or self.lambda_object_fd_acceleration > 0.0:
            fd = object_translation_finite_difference_loss(
                prediction,
                batch,
                structured_normalizer=self.normalizer,
                velocity_error_scale_m=self.object_fd_velocity_error_scale_m,
                acceleration_error_scale_m=self.object_fd_acceleration_error_scale_m,
                beta=self.object_fd_smooth_l1_beta,
            )
            weighted_vel = fd.velocity_loss * self.lambda_object_fd_velocity
            weighted_acc = fd.acceleration_loss * self.lambda_object_fd_acceleration
            components["object_fd_velocity_scaled_loss"] = fd.velocity_loss
            components["object_fd_acceleration_scaled_loss"] = fd.acceleration_loss
            components["object_fd_velocity"] = weighted_vel
            components["object_fd_acceleration"] = weighted_acc
            diagnostics.update(fd.diagnostics)
            total = total + weighted_vel + weighted_acc
        return total

    def _add_human_jitter_terms(
        self,
        total: torch.Tensor,
        prediction: Stage1CoarsePrediction,
        batch: dict,
        components: dict[str, torch.Tensor],
        diagnostics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.lambda_human_fd_velocity == 0.0 and self.lambda_human_fd_acceleration == 0.0:
            return total
        if self.normalizer is None:
            raise ValueError("Human jitter losses require structured normalizer stats.")

        fd = human_world_marker_finite_difference_loss(
            prediction,
            batch,
            structured_normalizer=self.normalizer,
            velocity_error_scale_m=self.human_fd_velocity_error_scale_m,
            acceleration_error_scale_m=self.human_fd_acceleration_error_scale_m,
            beta=self.human_fd_smooth_l1_beta,
        )
        weighted_vel = fd.velocity_loss * self.lambda_human_fd_velocity
        weighted_acc = fd.acceleration_loss * self.lambda_human_fd_acceleration
        components["human_fd_velocity_scaled_loss"] = fd.velocity_loss
        components["human_fd_acceleration_scaled_loss"] = fd.acceleration_loss
        components["human_fd_velocity"] = weighted_vel
        components["human_fd_acceleration"] = weighted_acc
        diagnostics.update(fd.diagnostics)
        return total + weighted_vel + weighted_acc

    def _add_skipped_group_diagnostics(
        self,
        diagnostics: dict[str, Any],
        group_name: str,
        latent: Stage1LatentFieldOutput,
        sample_mask: torch.Tensor,
    ) -> None:
        count = int(sample_mask.to(torch.bool).sum().item())
        diagnostics[f"{group_name}.sample_count"] = latent.z_rec.new_tensor(count, dtype=torch.long)
        diagnostics[f"rec_loss_{group_name}"] = None
        diagnostics[f"world_marker_{group_name}"] = None
        diagnostics[f"relation_dist_{group_name}"] = None
        diagnostics[f"relation_delta_m2o_{group_name}"] = None

    @torch.no_grad()
    def _add_group_diagnostics(
        self,
        diagnostics: dict[str, Any],
        *,
        group_name: str,
        prediction: Stage1CoarsePrediction,
        latent: Stage1LatentFieldOutput,
        batch: dict,
        sample_mask: torch.Tensor,
        global_step: int,
    ) -> None:
        count = int(sample_mask.to(torch.bool).sum().item())
        diagnostics[f"{group_name}.sample_count"] = latent.z_rec.new_tensor(count, dtype=torch.long)
        if count == 0:
            diagnostics[f"rec_loss_{group_name}"] = None
            diagnostics[f"world_marker_{group_name}"] = None
            diagnostics[f"relation_dist_{group_name}"] = None
            diagnostics[f"relation_delta_m2o_{group_name}"] = None
            return
        masked_batch = _batch_with_sample_masked_loss_masks(batch, sample_mask)
        group_out = self.reconstruction_loss(prediction, latent, masked_batch, global_step=global_step)
        diagnostics[f"rec_loss_{group_name}"] = group_out.total.detach()
        diagnostics[f"world_marker_{group_name}"] = group_out.components["world_marker"].detach()
        diagnostics[f"relation_dist_{group_name}"] = group_out.components["relation.dist"].detach()
        diagnostics[f"relation_delta_m2o_{group_name}"] = group_out.components["relation.delta_m2o"].detach()


def _batch_size_from_prediction(prediction: Stage1CoarsePrediction) -> int:
    for group in (prediction.human, prediction.object, prediction.relation):
        for value in group.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
    raise ValueError("Cannot infer batch size from Stage1CoarsePrediction.")


def _batch_with_sample_masked_loss_masks(batch: dict, sample_mask: torch.Tensor) -> dict:
    sample_mask = sample_mask.to(dtype=torch.bool)
    out = dict(batch)
    masks = dict(batch.get("masks", {}))
    for key, value in list(masks.items()):
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != sample_mask.shape[0]:
            continue
        shape = [sample_mask.shape[0]] + [1] * (value.ndim - 1)
        keep = sample_mask.to(device=value.device, dtype=torch.bool).reshape(shape)
        masks[key] = value.to(dtype=torch.bool) & keep
    out["masks"] = masks
    return out
