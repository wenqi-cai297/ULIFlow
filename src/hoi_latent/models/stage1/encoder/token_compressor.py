from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from hoi_latent.models.stage1.mask_utils import safe_masked_softmax, zero_invalid_tokens
from hoi_latent.models.stage1.time_bias import RelativeTimeBias
from hoi_latent.models.stage1.tokenizer_types import (
    ModalityId,
    TokenizerOutput,
    TokenTypeId,
    concat_tokenizer_outputs,
    validate_tokenizer_output,
)


@dataclass
class Stage1ParsedStemOutputs:
    human: torch.Tensor
    human_mask: torch.Tensor
    human_meta: dict[str, torch.Tensor]
    object_pose: torch.Tensor
    object_pose_mask: torch.Tensor
    object_pose_meta: dict[str, torch.Tensor]
    object_geometry: torch.Tensor
    object_geometry_mask: torch.Tensor
    object_geometry_meta: dict[str, torch.Tensor]
    relation: torch.Tensor
    relation_mask: torch.Tensor
    relation_meta: dict[str, torch.Tensor]
    valid_frame_mask: torch.Tensor


@dataclass
class Stage1TemporalAnchors:
    time_ids: torch.Tensor
    valid_mask: torch.Tensor
    sources: torch.Tensor
    scores: torch.Tensor


@dataclass
class Stage1CompressionBudget:
    temporal_anchors: int
    object_geometry_tokens: int


@dataclass
class Stage1SaliencyScores:
    relation: torch.Tensor
    motion: torch.Tensor
    source: str


class Stage1TokenCompressorAndBudgeter(nn.Module):
    checkpoint_key = "token_compressor"
    output_keys = ("human", "object", "relation")
    concat_key = "concat"

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        human_slots_per_frame: int = 11,
        relation_slots_per_frame: int = 10,
        compression_factor: int = 4,
        max_temporal_anchors: int = 32,
        min_temporal_anchors: int = 1,
        geometry_target_tokens: int = 16,
        geometry_min_tokens: int = 1,
        global_cap: int = 768,
        require_raw_relation_dist: bool = True,
        use_relation_distance: bool = True,
        use_relation_velocity: bool = True,
        use_relation_acceleration: bool = True,
        use_motion_energy: bool = True,
        use_stem_learned_saliency: bool = True,
        normalize_per_source: bool = True,
        anchor_mode: str = "hybrid",
        anchor_uniform_quota: float = 0.50,
        anchor_relation_quota: float = 0.30,
        anchor_motion_quota: float = 0.20,
        anchor_stride: int | None = None,
        anchor_grid_origin: int = 0,
        relative_time_alpha_init: float = 0.1,
        gate_bias_init: float = 0.0,
        residual_gamma_init: float = 1e-3,
        adapter_output_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.human_slots_per_frame = int(human_slots_per_frame)
        self.relation_slots_per_frame = int(relation_slots_per_frame)
        self.compression_factor = int(compression_factor)
        self.max_temporal_anchors = int(max_temporal_anchors)
        self.min_temporal_anchors = int(min_temporal_anchors)
        self.geometry_target_tokens = int(geometry_target_tokens)
        self.geometry_min_tokens = int(geometry_min_tokens)
        self.global_cap = int(global_cap)
        self.require_raw_relation_dist = bool(require_raw_relation_dist)
        self.use_relation_distance = bool(use_relation_distance)
        self.use_relation_velocity = bool(use_relation_velocity)
        self.use_relation_acceleration = bool(use_relation_acceleration)
        self.use_motion_energy = bool(use_motion_energy)
        self.use_stem_learned_saliency = bool(use_stem_learned_saliency)
        self.normalize_per_source = bool(normalize_per_source)
        self.anchor_mode = str(anchor_mode).lower()
        self.anchor_uniform_quota = float(anchor_uniform_quota)
        self.anchor_relation_quota = float(anchor_relation_quota)
        self.anchor_motion_quota = float(anchor_motion_quota)
        self.anchor_stride = None if anchor_stride is None else int(anchor_stride)
        self.anchor_grid_origin = int(anchor_grid_origin)

        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if self.compression_factor <= 0:
            raise ValueError(f"compression_factor must be positive, got {compression_factor}.")
        if self.max_temporal_anchors <= 0:
            raise ValueError(f"max_temporal_anchors must be positive, got {max_temporal_anchors}.")
        if self.geometry_target_tokens <= 0:
            raise ValueError(f"geometry_target_tokens must be positive, got {geometry_target_tokens}.")
        if self.global_cap <= 0:
            raise ValueError(f"global_cap must be positive, got {global_cap}.")

        self.parser = Stage1StemOutputParser(self.human_slots_per_frame, self.relation_slots_per_frame)
        self.saliency_scorer = Stage1CompressionSaliencyScorer(
            require_raw_relation_dist=self.require_raw_relation_dist,
            use_relation_distance=self.use_relation_distance,
            use_relation_velocity=self.use_relation_velocity,
            use_relation_acceleration=self.use_relation_acceleration,
            use_motion_energy=self.use_motion_energy,
            use_stem_learned_saliency=self.use_stem_learned_saliency,
            normalize_per_source=self.normalize_per_source,
        )
        if self.anchor_mode == "hybrid":
            self.anchor_selector = HybridTemporalAnchorSelector(
                compression_factor=self.compression_factor,
                max_temporal_anchors=self.max_temporal_anchors,
                min_temporal_anchors=self.min_temporal_anchors,
                uniform_quota=self.anchor_uniform_quota,
                relation_quota=self.anchor_relation_quota,
                motion_quota=self.anchor_motion_quota,
            )
        elif self.anchor_mode == "absolute_grid":
            self.anchor_selector = AbsoluteGridTemporalAnchorSelector(
                compression_factor=self.compression_factor,
                max_temporal_anchors=self.max_temporal_anchors,
                min_temporal_anchors=self.min_temporal_anchors,
                anchor_stride=self.anchor_stride,
                grid_origin=self.anchor_grid_origin,
            )
        else:
            raise ValueError("anchor_mode must be 'hybrid' or 'absolute_grid'.")
        self.human_pool = HybridKeepPoolCompressor(
            self.d_model,
            num_heads=self.num_heads,
            relative_time_alpha_init=relative_time_alpha_init,
            gate_bias_init=gate_bias_init,
            dropout=dropout,
        )
        self.object_pose_pool = HybridKeepPoolCompressor(
            self.d_model,
            num_heads=self.num_heads,
            relative_time_alpha_init=relative_time_alpha_init,
            gate_bias_init=gate_bias_init,
            dropout=dropout,
        )
        self.relation_pool = HybridKeepPoolCompressor(
            self.d_model,
            num_heads=self.num_heads,
            relative_time_alpha_init=relative_time_alpha_init,
            gate_bias_init=gate_bias_init,
            dropout=dropout,
        )
        self.geometry_compressor = GeometryTokenCompressor(
            self.d_model,
            num_heads=self.num_heads,
            target_tokens=self.geometry_target_tokens,
            min_tokens=self.geometry_min_tokens,
            dropout=dropout,
        )
        self.shared_space_adapter = SharedSpaceOutputAdapter(
            self.d_model,
            adapter_keys=("human_dynamic", "object_pose", "object_geometry", "relation_dynamic"),
            residual_gamma_init=residual_gamma_init,
            output_norm=adapter_output_norm,
        )

    def forward(
        self,
        stem_outputs: dict[str, TokenizerOutput],
        batch: Optional[dict] = None,
        return_concat: bool = False,
        validate: bool = False,
    ) -> dict[str, TokenizerOutput]:
        missing = [key for key in self.output_keys if key not in stem_outputs]
        if missing:
            raise KeyError(f"TokenCompressor input is missing keys: {missing}.")

        groups = self.parser(stem_outputs)
        budget = self._resolve_budget(
            num_frames=groups.valid_frame_mask.shape[1],
            num_geometry_tokens=groups.object_geometry.shape[1],
        )
        window_start = _window_start_from_batch(
            batch,
            batch_size=groups.valid_frame_mask.shape[0],
            device=groups.valid_frame_mask.device,
        )
        saliency = self.saliency_scorer(groups, stem_outputs, batch=batch)
        anchors = self.anchor_selector(
            valid_mask=groups.valid_frame_mask,
            relation_saliency=saliency.relation,
            motion_saliency=saliency.motion,
            max_temporal_anchors=budget.temporal_anchors,
            window_start=window_start,
        )
        near_contact_recall = _near_contact_recall(batch, anchors)

        human_tokens, human_mask = self.human_pool(
            groups.human,
            groups.human_mask,
            anchors,
            saliency_bias=saliency.motion,
        )
        object_pose_tokens, object_pose_mask = self.object_pose_pool(
            groups.object_pose[:, :, None, :],
            groups.object_pose_mask[:, :, None],
            anchors,
            saliency_bias=saliency.motion,
        )
        object_pose_tokens = object_pose_tokens[:, :, 0, :]
        object_pose_mask = object_pose_mask[:, :, 0]
        relation_tokens, relation_mask = self.relation_pool(
            groups.relation,
            groups.relation_mask,
            anchors,
            saliency_bias=saliency.relation,
        )
        geometry_tokens, geometry_mask, geometry_entity_ids = self.geometry_compressor(
            groups.object_geometry,
            groups.object_geometry_mask,
            groups.object_geometry_meta["entity_ids"],
            target_tokens=budget.object_geometry_tokens,
        )

        human_out = self._build_dynamic_output(
            source=stem_outputs["human"],
            tokens=human_tokens,
            mask=human_mask,
            anchors=anchors,
            meta=groups.human_meta,
            adapter_key="human_dynamic",
            saliency_source=saliency.source,
            original_num_tokens=stem_outputs["human"].tokens.shape[1],
            budget=budget,
            near_contact_recall=near_contact_recall,
        )
        relation_out = self._build_dynamic_output(
            source=stem_outputs["relation"],
            tokens=relation_tokens,
            mask=relation_mask,
            anchors=anchors,
            meta=groups.relation_meta,
            adapter_key="relation_dynamic",
            saliency_source=saliency.source,
            original_num_tokens=stem_outputs["relation"].tokens.shape[1],
            budget=budget,
            near_contact_recall=near_contact_recall,
        )
        object_out = self._build_object_output(
            source=stem_outputs["object"],
            pose_tokens=object_pose_tokens,
            pose_mask=object_pose_mask,
            pose_meta=groups.object_pose_meta,
            geometry_tokens=geometry_tokens,
            geometry_mask=geometry_mask,
            geometry_entity_ids=geometry_entity_ids,
            anchors=anchors,
            saliency_source=saliency.source,
            budget=budget,
            near_contact_recall=near_contact_recall,
        )

        outputs = {"human": human_out, "object": object_out, "relation": relation_out}
        if validate:
            self.validate_outputs(outputs)
        if return_concat:
            outputs[self.concat_key] = concat_tokenizer_outputs(outputs[key] for key in self.output_keys)
            if validate:
                validate_tokenizer_output(outputs[self.concat_key], f"{self.__class__.__name__}.{self.concat_key}")
        return outputs

    def validate_outputs(self, outputs: dict[str, TokenizerOutput]) -> None:
        total = 0
        for key in self.output_keys:
            if key not in outputs:
                raise KeyError(f"TokenCompressor output is missing key {key!r}.")
            validate_tokenizer_output(outputs[key], f"{self.__class__.__name__}.{key}")
            total += outputs[key].tokens.shape[1]
        if total > self.global_cap:
            raise ValueError(f"Compressed token count {total} exceeds global_cap={self.global_cap}.")

    def _resolve_budget(self, num_frames: int, num_geometry_tokens: int) -> Stage1CompressionBudget:
        temporal_anchors = self.anchor_selector.resolve_num_anchors(num_frames)
        geometry_tokens = num_geometry_tokens if num_geometry_tokens <= self.geometry_target_tokens else self.geometry_target_tokens
        geometry_tokens = max(self.geometry_min_tokens, geometry_tokens)

        def total(k: int, g: int) -> int:
            return k * (self.human_slots_per_frame + self.relation_slots_per_frame + 1) + g

        while geometry_tokens > self.geometry_min_tokens and total(temporal_anchors, geometry_tokens) > self.global_cap:
            geometry_tokens -= 1

        min_temporal = self.anchor_selector.resolve_min_anchors(num_frames)
        while temporal_anchors > min_temporal and total(temporal_anchors, geometry_tokens) > self.global_cap:
            temporal_anchors -= 1

        if total(temporal_anchors, geometry_tokens) > self.global_cap:
            raise ValueError(
                "Stage1TokenCompressorAndBudgeter cannot satisfy global_cap="
                f"{self.global_cap} with min_temporal_anchors={min_temporal}, "
                f"geometry_min_tokens={self.geometry_min_tokens}, "
                f"human_slots={self.human_slots_per_frame}, relation_slots={self.relation_slots_per_frame}."
            )
        return Stage1CompressionBudget(
            temporal_anchors=int(temporal_anchors),
            object_geometry_tokens=int(geometry_tokens),
        )

    def _build_dynamic_output(
        self,
        source: TokenizerOutput,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        anchors: Stage1TemporalAnchors,
        meta: dict[str, torch.Tensor],
        adapter_key: str,
        saliency_source: str,
        original_num_tokens: int,
        budget: Stage1CompressionBudget,
        near_contact_recall: Optional[torch.Tensor],
    ) -> TokenizerOutput:
        b, k, s, d = tokens.shape
        flat_tokens = tokens.reshape(b, k * s, d)
        flat_mask = mask.reshape(b, k * s)
        out = TokenizerOutput(
            tokens=flat_tokens,
            mask=flat_mask,
            modality_ids=_gather_by_anchor(meta["modality_ids"], anchors.time_ids).reshape(b, k * s),
            token_type_ids=_gather_by_anchor(meta["token_type_ids"], anchors.time_ids).reshape(b, k * s),
            time_ids=_gather_by_anchor(meta["time_ids"], anchors.time_ids).reshape(b, k * s),
            entity_ids=_gather_by_anchor(meta["entity_ids"], anchors.time_ids).reshape(b, k * s),
            time_mask=flat_mask.clone(),
            aux=self._compression_aux(
                source.aux,
                anchors,
                saliency_source,
                original_num_tokens=original_num_tokens,
                compressed_num_tokens=k * s,
                substream_tokens={"dynamic": int(k * s)},
                budget=budget,
                near_contact_recall=near_contact_recall,
            ),
        )
        out = self.shared_space_adapter(out, adapter_key)
        out.aux["compression_diagnostics"]["adapter"] = {
            adapter_key: out.aux.pop("shared_space_adapter_diagnostics")
        }
        return out

    def _build_object_output(
        self,
        source: TokenizerOutput,
        pose_tokens: torch.Tensor,
        pose_mask: torch.Tensor,
        pose_meta: dict[str, torch.Tensor],
        geometry_tokens: torch.Tensor,
        geometry_mask: torch.Tensor,
        geometry_entity_ids: torch.Tensor,
        anchors: Stage1TemporalAnchors,
        saliency_source: str,
        budget: Stage1CompressionBudget,
        near_contact_recall: Optional[torch.Tensor],
    ) -> TokenizerOutput:
        b, k, d = pose_tokens.shape
        g = geometry_tokens.shape[1]
        pose_out = TokenizerOutput(
            tokens=pose_tokens,
            mask=pose_mask,
            modality_ids=_gather_by_anchor(pose_meta["modality_ids"][:, :, None], anchors.time_ids).reshape(b, k),
            token_type_ids=_gather_by_anchor(pose_meta["token_type_ids"][:, :, None], anchors.time_ids).reshape(b, k),
            time_ids=_gather_by_anchor(pose_meta["time_ids"][:, :, None], anchors.time_ids).reshape(b, k),
            entity_ids=_gather_by_anchor(pose_meta["entity_ids"][:, :, None], anchors.time_ids).reshape(b, k),
            time_mask=pose_mask.clone(),
            aux={},
        )
        geometry_out = TokenizerOutput(
            tokens=geometry_tokens,
            mask=geometry_mask,
            modality_ids=torch.full((b, g), int(ModalityId.OBJECT), dtype=torch.long, device=geometry_tokens.device),
            token_type_ids=torch.full(
                (b, g), int(TokenTypeId.OBJECT_GEOMETRY), dtype=torch.long, device=geometry_tokens.device
            ),
            time_ids=torch.zeros(b, g, dtype=torch.long, device=geometry_tokens.device),
            entity_ids=geometry_entity_ids,
            time_mask=torch.zeros(b, g, dtype=torch.bool, device=geometry_tokens.device),
            aux={},
        )
        pose_out = self.shared_space_adapter(pose_out, "object_pose")
        geometry_out = self.shared_space_adapter(geometry_out, "object_geometry")
        adapter_diagnostics = {
            "object_pose": pose_out.aux.pop("shared_space_adapter_diagnostics"),
            "object_geometry": geometry_out.aux.pop("shared_space_adapter_diagnostics"),
        }
        out = concat_tokenizer_outputs([pose_out, geometry_out])
        out.aux = self._compression_aux(
            source.aux,
            anchors,
            saliency_source,
            original_num_tokens=source.tokens.shape[1],
            compressed_num_tokens=k + g,
            substream_tokens={"object_pose": int(k), "object_geometry": int(g)},
            budget=budget,
            near_contact_recall=near_contact_recall,
        )
        out.aux["compression_diagnostics"]["adapter"] = adapter_diagnostics
        return out

    def _compression_aux(
        self,
        base_aux: dict,
        anchors: Stage1TemporalAnchors,
        saliency_source: str,
        original_num_tokens: int,
        compressed_num_tokens: int,
        substream_tokens: dict[str, int],
        budget: Stage1CompressionBudget,
        near_contact_recall: Optional[torch.Tensor],
    ) -> dict:
        anchor_sources = anchors.sources
        valid = anchors.valid_mask
        denom = valid.sum().clamp_min(1).to(dtype=torch.float32)
        source_ratio = {
            name: ((anchor_sources == value) & valid).sum().to(dtype=torch.float32).div(denom)
            for value, name in [(0, "uniform"), (1, "relation"), (2, "motion"), (3, "fill")]
        }
        return {
            **base_aux,
            "compression": {
                "method": "hybrid_keep_global_attention_pooling",
                "anchor_time_ids": anchors.time_ids,
                "anchor_valid_mask": anchors.valid_mask,
                "anchor_sources": anchors.sources,
                "anchor_scores": anchors.scores,
                "attention_scope": "global",
                "pooling_mode": "hybrid_keep_pool",
                "shared_space_adapter": "residual_per_substream_projection",
                "saliency_source": saliency_source,
                "source_time_spans": None,
                "source_attention_weights": None,
                "original_num_tokens": int(original_num_tokens),
                "compressed_num_tokens": int(compressed_num_tokens),
                "temporal_compression_factor": self.compression_factor,
                "temporal_anchor_policy": (
                    "absolute_grid" if self.anchor_mode == "absolute_grid" else "ceil_T_over_factor_clamped_by_max_and_budget"
                ),
                "actual_temporal_anchors": int(anchors.time_ids.shape[1]),
                "budgeted_temporal_anchors": int(budget.temporal_anchors),
                "budgeted_object_geometry_tokens": int(budget.object_geometry_tokens),
                "global_cap": int(self.global_cap),
            },
            "compression_diagnostics": {
                **substream_tokens,
                "total_tokens": int(compressed_num_tokens),
                "anchor_source_ratio": source_ratio,
                "coverage_gap_max": _coverage_gap_max(anchors.time_ids, anchors.valid_mask),
                "near_contact_recall": near_contact_recall,
                "saliency_source": saliency_source,
                "attention_scope": "global",
                "pooling_mode": "hybrid_keep_pool",
                "shared_space_adapter": "residual_per_substream_projection",
                "anchor_gradient_boundary": "hard_selection_no_gradient_pooling_projection_differentiable",
                "shared_anchor_tradeoff": "temporal_alignment_over_modality_private_event_coverage",
                "adapter_gamma": {
                    key: param.detach() for key, param in self.shared_space_adapter.gamma.items()
                },
            },
        }


class Stage1StemOutputParser(nn.Module):
    def __init__(self, human_slots_per_frame: int, relation_slots_per_frame: int) -> None:
        super().__init__()
        self.human_slots_per_frame = int(human_slots_per_frame)
        self.relation_slots_per_frame = int(relation_slots_per_frame)

    def forward(self, outputs: dict[str, TokenizerOutput]) -> Stage1ParsedStemOutputs:
        human, human_mask, human_meta = _view_time_slots(
            outputs["human"], self.human_slots_per_frame, "TokenCompressor.human"
        )
        relation, relation_mask, relation_meta = _view_time_slots(
            outputs["relation"], self.relation_slots_per_frame, "TokenCompressor.relation"
        )
        object_out = outputs["object"]
        pose_index = _static_type_index(object_out, TokenTypeId.OBJECT_POSE, "TokenCompressor.object_pose")
        geometry_index = _static_type_index(object_out, TokenTypeId.OBJECT_GEOMETRY, "TokenCompressor.object_geometry")
        if not pose_index.any() or not geometry_index.any():
            raise ValueError("TokenCompressor requires both OBJECT_POSE and OBJECT_GEOMETRY tokens.")
        object_pose = object_out.tokens[:, pose_index, :]
        object_pose_mask = object_out.mask[:, pose_index]
        object_geometry = object_out.tokens[:, geometry_index, :]
        object_geometry_mask = object_out.mask[:, geometry_index]
        object_pose_meta = _select_meta(object_out, pose_index)
        object_geometry_meta = _select_meta(object_out, geometry_index)
        t = human.shape[1]
        if relation.shape[1] != t or object_pose.shape[1] != t:
            raise ValueError(
                "TokenCompressor expects human, object pose, and relation to share the same T; "
                f"got human={t}, object_pose={object_pose.shape[1]}, relation={relation.shape[1]}."
            )
        valid_frame_mask = human_mask.any(dim=2) | relation_mask.any(dim=2) | object_pose_mask
        return Stage1ParsedStemOutputs(
            human=human,
            human_mask=human_mask,
            human_meta=human_meta,
            object_pose=object_pose,
            object_pose_mask=object_pose_mask,
            object_pose_meta=object_pose_meta,
            object_geometry=object_geometry,
            object_geometry_mask=object_geometry_mask,
            object_geometry_meta=object_geometry_meta,
            relation=relation,
            relation_mask=relation_mask,
            relation_meta=relation_meta,
            valid_frame_mask=valid_frame_mask,
        )


class Stage1CompressionSaliencyScorer(nn.Module):
    def __init__(
        self,
        require_raw_relation_dist: bool,
        use_relation_distance: bool,
        use_relation_velocity: bool,
        use_relation_acceleration: bool,
        use_motion_energy: bool,
        use_stem_learned_saliency: bool,
        normalize_per_source: bool,
    ) -> None:
        super().__init__()
        self.require_raw_relation_dist = bool(require_raw_relation_dist)
        self.use_relation_distance = bool(use_relation_distance)
        self.use_relation_velocity = bool(use_relation_velocity)
        self.use_relation_acceleration = bool(use_relation_acceleration)
        self.use_motion_energy = bool(use_motion_energy)
        self.use_stem_learned_saliency = bool(use_stem_learned_saliency)
        self.normalize_per_source = bool(normalize_per_source)

    def forward(
        self,
        groups: Stage1ParsedStemOutputs,
        stem_outputs: dict[str, TokenizerOutput],
        batch: Optional[dict],
    ) -> Stage1SaliencyScores:
        device = groups.human.device
        dtype = groups.human.dtype
        b, t = groups.valid_frame_mask.shape
        relation_terms = []
        sources: list[str] = []

        if self.use_relation_distance:
            dist = _get_nested(batch, ("raw", "relation", "dist")) if batch is not None else None
            if dist is None and self.require_raw_relation_dist:
                raise KeyError(
                    "Stage1TokenCompressorAndBudgeter requires batch['raw']['relation']['dist'] "
                    "when require_raw_relation_dist=True."
                )
            if dist is not None:
                dist = dist.to(device=device, dtype=dtype)
                relation_terms.append(_normalize_frame_score(-dist.squeeze(-1).amin(dim=-1), groups.valid_frame_mask))
                sources.append("raw_relation_distance")
            elif not self.require_raw_relation_dist:
                model_dist = _get_nested(batch, ("model", "relation", "dist")) if batch is not None else None
                if model_dist is not None:
                    model_dist = model_dist.to(device=device, dtype=dtype)
                    relation_terms.append(
                        _normalize_frame_score(-model_dist.squeeze(-1).amin(dim=-1), groups.valid_frame_mask)
                    )
                    sources.append("model_relation_distance")

        if self.use_relation_velocity:
            vel = _get_nested(batch, ("model", "relation", "delta_vel")) if batch is not None else None
            if vel is not None:
                vel_score = vel.to(device=device, dtype=dtype).norm(dim=-1).amax(dim=-1)
                relation_terms.append(_normalize_frame_score(vel_score, groups.valid_frame_mask))
                sources.append("relation_velocity")

        if self.use_relation_acceleration:
            acc = _get_nested(batch, ("model", "relation", "delta_acc")) if batch is not None else None
            if acc is not None:
                acc_score = acc.to(device=device, dtype=dtype).norm(dim=-1).amax(dim=-1)
                relation_terms.append(_normalize_frame_score(acc_score, groups.valid_frame_mask))
                sources.append("relation_acceleration")

        if self.use_stem_learned_saliency and "temporal_saliency_logits" in stem_outputs["relation"].aux:
            learned = stem_outputs["relation"].aux["temporal_saliency_logits"].to(device=device, dtype=dtype)
            learned_score = learned.amax(dim=-1)
            relation_terms.append(_normalize_frame_score(learned_score, groups.valid_frame_mask))
            sources.append("stem_learned_saliency")

        if relation_terms:
            relation = torch.stack(relation_terms, dim=0).mean(dim=0)
        else:
            relation = _token_delta_score(groups.relation, groups.relation_mask)
            sources.append("token_delta_fallback")

        if self.use_motion_energy:
            motion_terms = []
            human_vel = _get_nested(batch, ("model", "human", "root_vel")) if batch is not None else None
            human_acc = _get_nested(batch, ("model", "human", "root_acc")) if batch is not None else None
            object_vel = _get_nested(batch, ("model", "object", "trans_vel")) if batch is not None else None
            object_acc = _get_nested(batch, ("model", "object", "trans_acc")) if batch is not None else None

            for name, value in [
                ("human_root_velocity", human_vel),
                ("human_root_acceleration", human_acc),
                ("object_translation_velocity", object_vel),
                ("object_translation_acceleration", object_acc),
            ]:
                score = _frame_vector_norm_score(value, groups.valid_frame_mask, device=device, dtype=dtype)
                if score is not None:
                    motion_terms.append(score)
                    sources.append(name)

            if not motion_terms:
                human_motion = _token_delta_score(groups.human, groups.human_mask)
                object_motion = _token_delta_score(groups.object_pose[:, :, None, :], groups.object_pose_mask[:, :, None])
                motion_terms.append(0.5 * (human_motion + object_motion))
                sources.append("token_motion_delta")

            motion = torch.stack(motion_terms, dim=0).mean(dim=0)
        else:
            motion = torch.zeros(b, t, dtype=dtype, device=device)

        if self.normalize_per_source:
            relation = _normalize_frame_score(relation, groups.valid_frame_mask)
            motion = _normalize_frame_score(motion, groups.valid_frame_mask)
        return Stage1SaliencyScores(
            relation=relation * groups.valid_frame_mask.to(dtype=dtype),
            motion=motion * groups.valid_frame_mask.to(dtype=dtype),
            source="+".join(dict.fromkeys(sources)),
        )


class HybridTemporalAnchorSelector(nn.Module):
    def __init__(
        self,
        compression_factor: int = 4,
        max_temporal_anchors: int = 32,
        min_temporal_anchors: int = 1,
        uniform_quota: float = 0.50,
        relation_quota: float = 0.30,
        motion_quota: float = 0.20,
    ) -> None:
        super().__init__()
        self.compression_factor = int(compression_factor)
        self.max_temporal_anchors = int(max_temporal_anchors)
        self.min_temporal_anchors = int(min_temporal_anchors)
        self.uniform_quota = float(uniform_quota)
        self.relation_quota = float(relation_quota)
        self.motion_quota = float(motion_quota)

    def forward(
        self,
        valid_mask: torch.Tensor,
        relation_saliency: torch.Tensor,
        motion_saliency: torch.Tensor,
        max_temporal_anchors: Optional[int] = None,
        window_start: Optional[torch.Tensor] = None,
    ) -> Stage1TemporalAnchors:
        if valid_mask.ndim != 2:
            raise ValueError(f"valid_mask must be [B,T], got {tuple(valid_mask.shape)}.")
        b, t = valid_mask.shape
        k = self.resolve_num_anchors(t, max_temporal_anchors=max_temporal_anchors)

        device = valid_mask.device
        valid_cpu = valid_mask.detach().to(device="cpu", dtype=torch.bool)
        relation_cpu = relation_saliency.detach().to(device="cpu")
        motion_cpu = motion_saliency.detach().to(device="cpu")
        time_ids_cpu = torch.zeros(b, k, dtype=torch.long)
        anchor_valid_cpu = torch.zeros(b, k, dtype=torch.bool)
        sources_cpu = torch.full((b, k), 3, dtype=torch.long)
        scores_cpu = torch.zeros(b, k, dtype=relation_cpu.dtype)

        uniform_count = min(k, max(1, int(round(k * self.uniform_quota))))
        relation_count = min(k - uniform_count, max(0, int(round(k * self.relation_quota))))
        motion_count = max(0, k - uniform_count - relation_count)
        min_gap = max(1, t // max(k, 1) // 2)

        with torch.no_grad():
            for batch_idx in range(b):
                valid_times = torch.nonzero(valid_cpu[batch_idx], as_tuple=False).flatten()
                if valid_times.numel() == 0:
                    continue
                selected: list[int] = []
                selected_sources: list[int] = []
                selected_scores: list[float] = []

                for time in _uniform_pick(valid_times, min(uniform_count, valid_times.numel())):
                    selected.append(time)
                    selected_sources.append(0)
                    selected_scores.append(0.0)

                rel = _topk_with_gap(
                    relation_cpu[batch_idx],
                    valid_cpu[batch_idx],
                    relation_count,
                    min_gap,
                    blocked=set(selected),
                )
                for time in rel:
                    selected.append(time)
                    selected_sources.append(1)
                    selected_scores.append(float(relation_cpu[batch_idx, time].item()))

                mot = _topk_with_gap(
                    motion_cpu[batch_idx],
                    valid_cpu[batch_idx],
                    motion_count,
                    min_gap,
                    blocked=set(selected),
                )
                for time in mot:
                    selected.append(time)
                    selected_sources.append(2)
                    selected_scores.append(float(motion_cpu[batch_idx, time].item()))

                for time in _uniform_pick(valid_times, valid_times.numel()):
                    if len(selected) >= k:
                        break
                    if time in selected:
                        continue
                    selected.append(time)
                    selected_sources.append(3)
                    selected_scores.append(0.0)

                order = sorted(range(len(selected)), key=lambda idx: selected[idx])
                ordered = [(selected[idx], selected_sources[idx], selected_scores[idx]) for idx in order[:k]]
                for out_idx, (time, source, score) in enumerate(ordered):
                    time_ids_cpu[batch_idx, out_idx] = int(time)
                    anchor_valid_cpu[batch_idx, out_idx] = True
                    sources_cpu[batch_idx, out_idx] = int(source)
                    scores_cpu[batch_idx, out_idx] = float(score)

        return Stage1TemporalAnchors(
            time_ids=time_ids_cpu.to(device=device),
            valid_mask=anchor_valid_cpu.to(device=device),
            sources=sources_cpu.to(device=device),
            scores=scores_cpu.to(device=device, dtype=relation_saliency.dtype),
        )

    def resolve_num_anchors(self, num_frames: int, max_temporal_anchors: Optional[int] = None) -> int:
        max_anchors = self.max_temporal_anchors
        if max_temporal_anchors is not None:
            max_anchors = min(max_anchors, int(max_temporal_anchors))
        k = max(1, math.ceil(int(num_frames) / self.compression_factor))
        k = min(max_anchors, k)
        if num_frames >= self.min_temporal_anchors:
            k = max(k, self.min_temporal_anchors)
        return min(k, int(num_frames))

    def resolve_min_anchors(self, num_frames: int) -> int:
        if num_frames <= 0:
            return 0
        if num_frames >= self.min_temporal_anchors:
            return min(self.min_temporal_anchors, num_frames)
        return 1


class AbsoluteGridTemporalAnchorSelector(nn.Module):
    def __init__(
        self,
        compression_factor: int = 4,
        max_temporal_anchors: int = 32,
        min_temporal_anchors: int = 1,
        anchor_stride: int | None = None,
        grid_origin: int = 0,
    ) -> None:
        super().__init__()
        self.compression_factor = int(compression_factor)
        self.max_temporal_anchors = int(max_temporal_anchors)
        self.min_temporal_anchors = int(min_temporal_anchors)
        self.anchor_stride = int(anchor_stride) if anchor_stride is not None else self.compression_factor
        self.grid_origin = int(grid_origin)
        if self.compression_factor <= 0:
            raise ValueError(f"compression_factor must be positive, got {compression_factor}.")
        if self.max_temporal_anchors <= 0:
            raise ValueError(f"max_temporal_anchors must be positive, got {max_temporal_anchors}.")
        if self.min_temporal_anchors <= 0:
            raise ValueError(f"min_temporal_anchors must be positive, got {min_temporal_anchors}.")
        if self.anchor_stride <= 0:
            raise ValueError(f"anchor_stride must be positive, got {anchor_stride}.")

    def forward(
        self,
        valid_mask: torch.Tensor,
        relation_saliency: torch.Tensor,
        motion_saliency: torch.Tensor,
        max_temporal_anchors: Optional[int] = None,
        window_start: Optional[torch.Tensor] = None,
    ) -> Stage1TemporalAnchors:
        if valid_mask.ndim != 2:
            raise ValueError(f"valid_mask must be [B,T], got {tuple(valid_mask.shape)}.")
        b, t = valid_mask.shape
        k = self.resolve_num_anchors(t, max_temporal_anchors=max_temporal_anchors)

        device = valid_mask.device
        valid_cpu = valid_mask.detach().to(device="cpu", dtype=torch.bool)
        start_cpu = _window_start_or_zeros(window_start, batch_size=b).to(device="cpu")
        time_ids_cpu = torch.zeros(b, k, dtype=torch.long)
        anchor_valid_cpu = torch.zeros(b, k, dtype=torch.bool)
        sources_cpu = torch.full((b, k), 3, dtype=torch.long)
        scores_cpu = torch.zeros(b, k, dtype=relation_saliency.dtype)

        with torch.no_grad():
            for batch_idx in range(b):
                valid_times = torch.nonzero(valid_cpu[batch_idx], as_tuple=False).flatten()
                if valid_times.numel() == 0:
                    continue
                physical = start_cpu[batch_idx] + valid_times
                grid_mask = torch.remainder(physical - self.grid_origin, self.anchor_stride) == 0
                grid_picks = valid_times[grid_mask]

                if grid_picks.numel() > k:
                    selected = _uniform_pick(grid_picks, k)
                    selected_sources = [0] * len(selected)
                else:
                    selected = [int(item) for item in grid_picks.tolist()]
                    selected_sources = [0] * len(selected)

                target_valid = min(k, int(valid_times.numel()))
                if len(selected) < target_valid:
                    blocked = set(selected)
                    for time in _uniform_pick(valid_times, valid_times.numel()):
                        if len(selected) >= target_valid:
                            break
                        if time in blocked:
                            continue
                        selected.append(int(time))
                        selected_sources.append(3)
                        blocked.add(int(time))

                order = sorted(range(len(selected)), key=lambda idx: selected[idx])
                ordered = [(selected[idx], selected_sources[idx]) for idx in order[:target_valid]]
                for out_idx, (time, source) in enumerate(ordered):
                    time_ids_cpu[batch_idx, out_idx] = int(time)
                    anchor_valid_cpu[batch_idx, out_idx] = True
                    sources_cpu[batch_idx, out_idx] = int(source)

        return Stage1TemporalAnchors(
            time_ids=time_ids_cpu.to(device=device),
            valid_mask=anchor_valid_cpu.to(device=device),
            sources=sources_cpu.to(device=device),
            scores=scores_cpu.to(device=device, dtype=relation_saliency.dtype),
        )

    def resolve_num_anchors(self, num_frames: int, max_temporal_anchors: Optional[int] = None) -> int:
        max_anchors = self.max_temporal_anchors
        if max_temporal_anchors is not None:
            max_anchors = min(max_anchors, int(max_temporal_anchors))
        k = max(1, math.ceil(int(num_frames) / self.compression_factor))
        k = min(max_anchors, k)
        if num_frames >= self.min_temporal_anchors:
            k = max(k, self.min_temporal_anchors)
        return min(k, int(num_frames))

    def resolve_min_anchors(self, num_frames: int) -> int:
        if num_frames <= 0:
            return 0
        if num_frames >= self.min_temporal_anchors:
            return min(self.min_temporal_anchors, num_frames)
        return 1


class HybridKeepPoolCompressor(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        relative_time_alpha_init: float = 0.1,
        gate_bias_init: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}.")
        self.head_dim = self.d_model // self.num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.pool_proj = nn.Linear(d_model, d_model)
        self.gate_mlp = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relative_time_bias = RelativeTimeBias(alpha_init=relative_time_alpha_init)
        nn.init.constant_(self.gate_mlp.bias, float(gate_bias_init))

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        anchors: Stage1TemporalAnchors,
        saliency_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 4:
            raise ValueError(f"tokens must have shape [B,T,S,D], got {tuple(tokens.shape)}.")
        if mask.shape != tokens.shape[:3]:
            raise ValueError(f"mask must have shape {tuple(tokens.shape[:3])}, got {tuple(mask.shape)}.")
        b, t, s, d = tokens.shape
        if d != self.d_model:
            raise ValueError(f"tokens last dim must be d_model={self.d_model}, got {d}.")
        k = anchors.time_ids.shape[1]
        x_anchor = _gather_by_anchor(tokens, anchors.time_ids)
        anchor_mask = _gather_by_anchor(mask, anchors.time_ids) & anchors.valid_mask[:, :, None]

        flat_anchor = x_anchor.permute(0, 2, 1, 3).reshape(b * s, k, d)
        flat_source = tokens.permute(0, 2, 1, 3).reshape(b * s, t, d)
        flat_key_mask = mask.permute(0, 2, 1).reshape(b * s, t)
        flat_query_mask = anchor_mask.permute(0, 2, 1).reshape(b * s, k)

        q = self._split_heads(self.q_proj(flat_anchor))
        key = self._split_heads(self.k_proj(flat_source))
        value = self._split_heads(self.v_proj(flat_source))
        scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(float(self.head_dim))

        source_time = torch.arange(t, device=tokens.device, dtype=torch.long)[None, :].expand(b, t)
        rel_bias = self.relative_time_bias(anchors.time_ids, source_time, dtype=tokens.dtype)
        rel_bias = rel_bias[:, None, :, :, :].expand(b, s, self.num_heads, k, t).reshape(
            b * s, self.num_heads, k, t
        )
        scores = scores + rel_bias
        if saliency_bias is not None:
            if saliency_bias.shape != (b, t):
                raise ValueError(f"saliency_bias must have shape {(b, t)}, got {tuple(saliency_bias.shape)}.")
            sal = saliency_bias.to(device=tokens.device, dtype=tokens.dtype)
            scores = scores + sal[:, None, None, None, :].expand(b, s, self.num_heads, k, t).reshape(
                b * s, self.num_heads, k, t
            )

        weights = safe_masked_softmax(scores, flat_key_mask[:, None, None, :], dim=-1)
        weights = self.dropout(weights)
        x_pool = torch.matmul(weights, value)
        x_pool = self._merge_heads(x_pool)
        gate = torch.sigmoid(self.gate_mlp(torch.cat([flat_anchor, x_pool], dim=-1)))
        out = flat_anchor + gate * self.pool_proj(x_pool)
        out = zero_invalid_tokens(out, flat_query_mask)
        out = out.reshape(b, s, k, d).permute(0, 2, 1, 3)
        return out, anchor_mask

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(b, n, self.d_model)


class GeometryTokenCompressor(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        target_tokens: int = 16,
        min_tokens: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}.")
        self.head_dim = self.d_model // self.num_heads
        self.target_tokens = int(target_tokens)
        self.min_tokens = int(min_tokens)
        self.queries = nn.Parameter(torch.randn(self.target_tokens, self.d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        entity_ids: torch.Tensor,
        target_tokens: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"geometry tokens must be [B,G,D], got {tuple(tokens.shape)}.")
        b, g, d = tokens.shape
        if d != self.d_model:
            raise ValueError(f"geometry token dim must be {self.d_model}, got {d}.")
        if mask.shape != (b, g):
            raise ValueError(f"geometry mask must be {(b, g)}, got {tuple(mask.shape)}.")
        target = self.target_tokens if target_tokens is None else int(target_tokens)
        target = max(self.min_tokens, min(target, self.target_tokens))
        if g <= target:
            return zero_invalid_tokens(tokens, mask), mask, entity_ids

        queries = self.queries[:target][None, :, :].expand(b, target, d)
        q = self._split_heads(self.q_proj(queries))
        key = self._split_heads(self.k_proj(tokens))
        value = self._split_heads(self.v_proj(tokens))
        scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        weights = safe_masked_softmax(scores, mask[:, None, None, :], dim=-1)
        weights = self.dropout(weights)
        out = self.out_proj(self._merge_heads(torch.matmul(weights, value)))
        out_mask = mask.any(dim=1, keepdim=True).expand(b, target)
        out = zero_invalid_tokens(out, out_mask)
        out_entity_ids = torch.arange(target, dtype=torch.long, device=tokens.device)[None, :].expand(b, target)
        return out, out_mask, out_entity_ids

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(b, n, self.d_model)


class SharedSpaceOutputAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        adapter_keys: tuple[str, ...],
        residual_gamma_init: float = 1e-3,
        output_norm: bool = False,
    ) -> None:
        super().__init__()
        self.pre_norm = nn.ModuleDict({key: nn.LayerNorm(d_model) for key in adapter_keys})
        self.projection = nn.ModuleDict({key: nn.Linear(d_model, d_model) for key in adapter_keys})
        self.gamma = nn.ParameterDict(
            {key: nn.Parameter(torch.tensor(float(residual_gamma_init))) for key in adapter_keys}
        )
        self.optional_output_norm = (
            nn.ModuleDict({key: nn.LayerNorm(d_model) for key in adapter_keys}) if output_norm else None
        )
        for projection in self.projection.values():
            nn.init.eye_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, out: TokenizerOutput, adapter_key: str) -> TokenizerOutput:
        if adapter_key not in self.pre_norm:
            raise KeyError(f"Unknown shared-space adapter key {adapter_key!r}.")
        h = out.tokens
        delta = self.projection[adapter_key](self.pre_norm[adapter_key](h))
        residual = self.gamma[adapter_key].to(dtype=h.dtype) * delta
        x = h + residual
        if self.optional_output_norm is not None:
            x = self.optional_output_norm[adapter_key](x)
        x = zero_invalid_tokens(x, out.mask)
        aux = dict(out.aux)
        aux["shared_space_adapter_diagnostics"] = _adapter_diagnostics(
            tokens=x,
            residual=residual,
            original=h,
            mask=out.mask,
            gamma=self.gamma[adapter_key],
            adapter_key=adapter_key,
        )
        return TokenizerOutput(
            tokens=x,
            mask=out.mask,
            modality_ids=out.modality_ids,
            token_type_ids=out.token_type_ids,
            time_ids=out.time_ids,
            entity_ids=out.entity_ids,
            time_mask=out.time_mask,
            aux=aux,
        )


def build_stage1_token_compressor_from_config(config: dict) -> Stage1TokenCompressorAndBudgeter:
    stage1_cfg = config.get("model", {}).get("stage1", {})
    tokenizer_cfg = stage1_cfg.get("tokenizers", {})
    compressor_cfg = stage1_cfg.get("token_compressor", {})
    temporal_cfg = compressor_cfg.get("temporal", {})
    anchor_quota_cfg = compressor_cfg.get("anchor_quota", {})
    saliency_cfg = compressor_cfg.get("saliency", {})
    geometry_cfg = compressor_cfg.get("geometry", {})
    pooling_cfg = compressor_cfg.get("pooling", {})
    budget_cfg = compressor_cfg.get("budget", {})
    shared_cfg = compressor_cfg.get("shared_space", {})
    human_slots = 1 + int(tokenizer_cfg.get("human", {}).get("num_body_slots", 8)) + 2
    relation_slots = int(tokenizer_cfg.get("relation", {}).get("num_relation_slots", 8)) + 2
    return Stage1TokenCompressorAndBudgeter(
        d_model=int(stage1_cfg.get("d_model", compressor_cfg.get("d_model", 256))),
        num_heads=int(compressor_cfg.get("num_heads", stage1_cfg.get("num_heads", 4))),
        human_slots_per_frame=int(compressor_cfg.get("human_slots_per_frame", human_slots)),
        relation_slots_per_frame=int(compressor_cfg.get("relation_slots_per_frame", relation_slots)),
        compression_factor=int(temporal_cfg.get("compression_factor", 4)),
        max_temporal_anchors=int(temporal_cfg.get("max_temporal_anchors", 32)),
        min_temporal_anchors=int(temporal_cfg.get("min_temporal_anchors", 1)),
        geometry_target_tokens=int(geometry_cfg.get("target_tokens", 16)),
        geometry_min_tokens=int(geometry_cfg.get("min_tokens", 1)),
        global_cap=int(budget_cfg.get("global_cap", 768)),
        require_raw_relation_dist=bool(saliency_cfg.get("require_raw_relation_dist", True)),
        use_relation_distance=bool(saliency_cfg.get("use_relation_distance", True)),
        use_relation_velocity=bool(saliency_cfg.get("use_relation_velocity", True)),
        use_relation_acceleration=bool(saliency_cfg.get("use_relation_acceleration", True)),
        use_motion_energy=bool(saliency_cfg.get("use_motion_energy", True)),
        use_stem_learned_saliency=bool(saliency_cfg.get("use_stem_learned_saliency", True)),
        normalize_per_source=bool(saliency_cfg.get("normalize_per_source", True)),
        anchor_mode=str(temporal_cfg.get("anchor_mode", "hybrid")),
        anchor_uniform_quota=float(anchor_quota_cfg.get("uniform", 0.50)),
        anchor_relation_quota=float(anchor_quota_cfg.get("relation", 0.30)),
        anchor_motion_quota=float(anchor_quota_cfg.get("motion", 0.20)),
        anchor_stride=temporal_cfg.get("anchor_stride"),
        anchor_grid_origin=int(temporal_cfg.get("anchor_grid_origin", 0)),
        relative_time_alpha_init=float(temporal_cfg.get("relative_time_alpha_init", 0.1)),
        gate_bias_init=float(pooling_cfg.get("gate_bias_init", 0.0)),
        residual_gamma_init=float(shared_cfg.get("residual_gamma_init", 1e-3)),
        adapter_output_norm=_parse_output_norm(shared_cfg.get("output_norm", None)),
        dropout=float(compressor_cfg.get("dropout", stage1_cfg.get("dropout", 0.0))),
    )


def _view_time_slots(
    out: TokenizerOutput,
    slots_per_frame: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    b, n, d = out.tokens.shape
    if n % slots_per_frame != 0:
        raise ValueError(f"{name} token count {n} is not divisible by slots_per_frame={slots_per_frame}.")
    t = n // slots_per_frame
    return (
        out.tokens.reshape(b, t, slots_per_frame, d),
        out.mask.reshape(b, t, slots_per_frame),
        {
            "modality_ids": out.modality_ids.reshape(b, t, slots_per_frame),
            "token_type_ids": out.token_type_ids.reshape(b, t, slots_per_frame),
            "time_ids": out.time_ids.reshape(b, t, slots_per_frame),
            "entity_ids": out.entity_ids.reshape(b, t, slots_per_frame),
        },
    )


def _static_type_index(out: TokenizerOutput, token_type: TokenTypeId, module_name: str) -> torch.Tensor:
    token_type_ids = out.token_type_ids
    if token_type_ids.ndim != 2:
        raise ValueError(f"{module_name}.token_type_ids must be [B,N], got {tuple(token_type_ids.shape)}.")
    first = token_type_ids[0]
    if not torch.equal(token_type_ids, first[None, :].expand_as(token_type_ids)):
        raise ValueError(f"{module_name} requires token_type_ids layout to be identical across batch.")
    return first == int(token_type)


def _select_meta(out: TokenizerOutput, index: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "modality_ids": out.modality_ids[:, index],
        "token_type_ids": out.token_type_ids[:, index],
        "time_ids": out.time_ids[:, index],
        "entity_ids": out.entity_ids[:, index],
    }


def _gather_by_anchor(x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        b, _, s = x.shape
        idx = anchors[:, :, None].expand(b, anchors.shape[1], s)
        return torch.gather(x, dim=1, index=idx)
    if x.ndim == 4:
        b, _, s, d = x.shape
        idx = anchors[:, :, None, None].expand(b, anchors.shape[1], s, d)
        return torch.gather(x, dim=1, index=idx)
    raise ValueError(f"Cannot gather tensor with shape {tuple(x.shape)}.")


def _window_start_from_batch(batch: Optional[dict], *, batch_size: int, device: torch.device) -> torch.Tensor:
    if batch is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    meta = batch.get("meta") if isinstance(batch, dict) else None
    start = meta.get("start") if isinstance(meta, dict) else None
    return _window_start_or_zeros(start, batch_size=batch_size).to(device=device)


def _window_start_or_zeros(start: Optional[torch.Tensor], *, batch_size: int) -> torch.Tensor:
    if start is None:
        return torch.zeros(batch_size, dtype=torch.long)
    if not isinstance(start, torch.Tensor):
        start = torch.as_tensor(start)
    start = start.detach().to(dtype=torch.long).reshape(-1)
    if start.numel() == 1 and batch_size > 1:
        start = start.expand(batch_size)
    if int(start.numel()) != int(batch_size):
        raise ValueError(f"window_start must have {batch_size} values, got {int(start.numel())}.")
    return start


def _normalize_frame_score(score: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    score = score.masked_fill(~valid_mask, 0.0)
    large = torch.finfo(score.dtype).max / 4
    min_val = score.masked_fill(~valid_mask, large).amin(dim=1, keepdim=True)
    max_val = score.masked_fill(~valid_mask, -large).amax(dim=1, keepdim=True)
    denom = (max_val - min_val).clamp_min(1e-6)
    normalized = (score - min_val) / denom
    any_valid = valid_mask.any(dim=1, keepdim=True)
    return torch.where(any_valid, normalized, torch.zeros_like(score)).masked_fill(~valid_mask, 0.0)


def _token_delta_score(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, t = tokens.shape[:2]
    if t == 1:
        return torch.zeros(b, t, dtype=tokens.dtype, device=tokens.device)
    delta = (tokens[:, 1:] - tokens[:, :-1]).norm(dim=-1)
    delta_mask = mask[:, 1:] & mask[:, :-1]
    if delta.ndim == 3:
        delta = delta.masked_fill(~delta_mask, 0.0).amax(dim=-1)
    score = torch.zeros(b, t, dtype=tokens.dtype, device=tokens.device)
    score[:, 1:] = delta
    frame_mask = mask.any(dim=2) if mask.ndim == 3 else mask
    return _normalize_frame_score(score, frame_mask)


def _frame_vector_norm_score(
    value: Optional[torch.Tensor],
    valid_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    value = value.to(device=device, dtype=dtype)
    if value.shape[:2] != valid_mask.shape:
        raise ValueError(
            f"Motion saliency tensor must start with shape {tuple(valid_mask.shape)}, got {tuple(value.shape)}."
        )
    score = value.norm(dim=-1)
    while score.ndim > 2:
        score = score.amax(dim=-1)
    return _normalize_frame_score(score, valid_mask)


def _get_nested(mapping: Optional[dict], path: tuple[str, ...]) -> Optional[torch.Tensor]:
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, torch.Tensor) else None


def _uniform_pick(valid_times: torch.Tensor, count: int) -> list[int]:
    if count <= 0 or valid_times.numel() == 0:
        return []
    count = min(int(count), int(valid_times.numel()))
    if count == 1:
        return [int(valid_times[0].item())]
    positions = torch.linspace(0, valid_times.numel() - 1, count, device=valid_times.device).round().long()
    return [int(item) for item in valid_times[positions].tolist()]


def _topk_with_gap(
    score: torch.Tensor,
    valid_mask: torch.Tensor,
    count: int,
    min_gap: int,
    blocked: set[int],
) -> list[int]:
    if count <= 0:
        return []
    candidates = torch.argsort(score.masked_fill(~valid_mask, -torch.inf), descending=True)
    selected: list[int] = []
    for raw_time in candidates.tolist():
        time = int(raw_time)
        if not bool(valid_mask[time].item()) or time in blocked:
            continue
        if any(abs(time - other) < min_gap for other in selected):
            continue
        selected.append(time)
        if len(selected) >= count:
            return selected
    for raw_time in candidates.tolist():
        time = int(raw_time)
        if not bool(valid_mask[time].item()) or time in blocked or time in selected:
            continue
        selected.append(time)
        if len(selected) >= count:
            break
    return selected


def _coverage_gap_max(time_ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    gaps = []
    for batch_idx in range(time_ids.shape[0]):
        valid = time_ids[batch_idx][valid_mask[batch_idx]]
        if valid.numel() <= 1:
            gaps.append(torch.zeros((), dtype=time_ids.dtype, device=time_ids.device))
            continue
        gaps.append((valid[1:] - valid[:-1]).max())
    if not gaps:
        return torch.zeros((), dtype=time_ids.dtype, device=time_ids.device)
    return torch.stack(gaps).max()


def _near_contact_recall(
    batch: Optional[dict],
    anchors: Stage1TemporalAnchors,
    distance_threshold: float = 0.05,
) -> Optional[torch.Tensor]:
    dist = _get_nested(batch, ("raw", "relation", "dist")) if batch is not None else None
    if dist is None:
        return None
    if dist.ndim != 4:
        return None
    frame_dist = dist.detach().squeeze(-1).amin(dim=-1)
    near = frame_dist <= float(distance_threshold)
    if near.shape[0] != anchors.time_ids.shape[0]:
        return None
    num_frames = near.shape[1]
    selected = torch.nn.functional.one_hot(
        anchors.time_ids.clamp(min=0, max=max(num_frames - 1, 0)),
        num_classes=num_frames,
    ).to(dtype=torch.bool)
    selected = (selected & anchors.valid_mask.unsqueeze(-1)).any(dim=1)
    denom = near.sum(dim=1).clamp_min(1).to(dtype=frame_dist.dtype)
    recall = (near & selected).sum(dim=1).to(dtype=frame_dist.dtype) / denom
    has_near = near.any(dim=1)
    return torch.where(has_near, recall, torch.ones_like(recall))


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1e-6)
    return torch.log(torch.expm1(value))


def _adapter_diagnostics(
    tokens: torch.Tensor,
    residual: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
    gamma: torch.Tensor,
    adapter_key: str,
) -> dict[str, torch.Tensor | str]:
    valid = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    denom = valid.sum().mul(tokens.shape[-1]).clamp_min(1.0)
    mean = (tokens * valid).sum() / denom
    var = ((tokens - mean) * valid).square().sum() / denom
    original_norm = (original * valid).norm().clamp_min(1e-6)
    residual_norm = (residual * valid).norm() / original_norm
    return {
        "adapter_key": adapter_key,
        "post_mean": mean.detach(),
        "post_std": var.clamp_min(0.0).sqrt().detach(),
        "residual_norm": residual_norm.detach(),
        "gamma": gamma.detach(),
    }


def _parse_output_norm(value: object) -> bool:
    if value is None or value is False:
        return False
    if value is True:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null", "false", "off"}:
            return False
        if normalized in {"layernorm", "layer_norm", "true", "on"}:
            return True
    raise ValueError(f"Unsupported shared_space.output_norm value: {value!r}.")
