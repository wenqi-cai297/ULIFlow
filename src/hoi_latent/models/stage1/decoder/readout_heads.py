from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hoi_latent.data.representations.interact_marker_metadata import (
    NUM_MARKERS,
    NUM_RELATION_TOKENS,
    PART_GROUND,
    PART_UNKNOWN,
    build_stage1_ids,
)
from hoi_latent.models.stage1.decoder.temporal_densifier import (
    dense_substream_major_to_frame_major,
    validate_dense_canonical_layout,
)
from hoi_latent.models.stage1.encoder.transformer_blocks import FeedForward, MultiHeadAttention
from hoi_latent.models.stage1.mask_utils import safe_masked_softmax
from hoi_latent.models.stage1.tokenizer_types import (
    NUM_TOKEN_TYPES,
    ModalityId,
    TokenTypeId,
    TokenizerOutput,
    validate_tokenizer_output,
)


@dataclass
class Stage1CoarsePrediction:
    human: dict[str, torch.Tensor]
    object: dict[str, torch.Tensor]
    relation: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class DenseTokenViews:
    human: torch.Tensor
    object_pose: torch.Tensor
    geometry: torch.Tensor
    relation: torch.Tensor
    human_mask: torch.Tensor
    object_pose_mask: torch.Tensor
    geometry_mask: torch.Tensor
    relation_mask: torch.Tensor
    frame_mask: torch.Tensor


@dataclass
class RelationEntryReadout:
    tokens: torch.Tensor
    mask: torch.Tensor
    attention_weights: torch.Tensor | None
    aux: dict[str, Any]


@dataclass
class HumanMarkerReadout:
    tokens: torch.Tensor
    root_token: torch.Tensor
    left_foot_token: torch.Tensor
    right_foot_token: torch.Tensor
    mask: torch.Tensor
    aux: dict[str, Any]


@dataclass
class ObjectPoseReadoutOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    aux: dict[str, Any]


class DenseTokenViewBuilder(nn.Module):
    def __init__(self, target_num_frames: int = 128, human_slots: int = 11, relation_slots: int = 10) -> None:
        super().__init__()
        if target_num_frames <= 0:
            raise ValueError(f"target_num_frames must be positive, got {target_num_frames}.")
        if human_slots < 3:
            raise ValueError("human_slots must include root, body slots, and left/right foot slots.")
        if relation_slots < 2:
            raise ValueError("relation_slots must include body slots plus ground/root-object slots.")
        self.target_num_frames = int(target_num_frames)
        self.human_slots = int(human_slots)
        self.relation_slots = int(relation_slots)

    def forward(self, dense_tokens: TokenizerOutput, validate: bool = False) -> DenseTokenViews:
        validate_tokenizer_output(dense_tokens, "coarse_readout.input")
        validate_dense_canonical_layout(
            dense_tokens,
            target_num_frames=self.target_num_frames,
            human_slots=self.human_slots,
            relation_slots=self.relation_slots,
        )
        frame = dense_substream_major_to_frame_major(
            dense_tokens,
            target_num_frames=self.target_num_frames,
            human_slots=self.human_slots,
            relation_slots=self.relation_slots,
        )
        human = frame["tokens"][:, :, : self.human_slots]
        object_pose = frame["tokens"][:, :, self.human_slots : self.human_slots + 1]
        relation = frame["tokens"][:, :, self.human_slots + 1 : self.human_slots + 1 + self.relation_slots]
        human_mask = frame["mask"][:, :, : self.human_slots].to(dtype=torch.bool)
        object_pose_mask = frame["mask"][:, :, self.human_slots : self.human_slots + 1].to(dtype=torch.bool)
        relation_mask = frame["mask"][:, :, self.human_slots + 1 : self.human_slots + 1 + self.relation_slots].to(
            dtype=torch.bool
        )

        geometry_start = self.target_num_frames * self.human_slots + self.target_num_frames
        geometry_end = dense_tokens.tokens.shape[1] - self.target_num_frames * self.relation_slots
        geometry = dense_tokens.tokens[:, geometry_start:geometry_end]
        geometry_mask = dense_tokens.mask[:, geometry_start:geometry_end].to(dtype=torch.bool)
        if geometry.shape[1] > 0:
            if not (dense_tokens.token_type_ids[:, geometry_start:geometry_end] == int(TokenTypeId.OBJECT_GEOMETRY)).all():
                raise ValueError("Dense geometry range has incorrect token_type ids.")
            if dense_tokens.time_mask is not None and dense_tokens.time_mask[:, geometry_start:geometry_end].any():
                raise ValueError("Dense geometry tokens must have time_mask=False.")
        frame_mask = object_pose_mask.squeeze(-1)
        if validate:
            _validate_view_masks(human_mask, object_pose_mask, relation_mask, frame_mask)
        return DenseTokenViews(
            human=human,
            object_pose=object_pose,
            geometry=geometry,
            relation=relation,
            human_mask=human_mask,
            object_pose_mask=object_pose_mask,
            geometry_mask=geometry_mask,
            relation_mask=relation_mask,
            frame_mask=frame_mask,
        )


class FrameCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.0, ffn_mult: int = 2) -> None:
        super().__init__()
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, ffn_mult=ffn_mult, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_mask: torch.Tensor,
        memory_mask: torch.Tensor,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.ndim != 4 or memory.ndim != 4:
            raise ValueError("FrameCrossAttentionBlock expects query/memory shaped [B,T,N,D].")
        b, t, q_len, d = query.shape
        if memory.shape[:2] != (b, t) or memory.shape[-1] != d or d != self.d_model:
            raise ValueError("FrameCrossAttentionBlock query/memory shapes are incompatible.")
        if query_mask.shape != query.shape[:3] or memory_mask.shape != memory.shape[:3]:
            raise ValueError("FrameCrossAttentionBlock masks must match query/memory token dimensions.")

        q_flat = query.reshape(b * t, q_len, d)
        m_flat = memory.reshape(b * t, memory.shape[2], d)
        q_mask = query_mask.reshape(b * t, q_len).to(dtype=torch.bool)
        m_mask = memory_mask.reshape(b * t, memory.shape[2]).to(dtype=torch.bool)

        q = self._split_heads(self.q_proj(self.query_norm(q_flat)))
        key = self._split_heads(self.k_proj(self.memory_norm(m_flat)))
        value = self._split_heads(self.v_proj(self.memory_norm(m_flat)))
        scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        weights = safe_masked_softmax(scores, m_mask[:, None, None, :], dim=-1)
        weights = self.dropout(weights)
        attn = torch.matmul(weights, value)
        attn = self.out_proj(self._merge_heads(attn))
        valid = q_mask & m_mask.any(dim=1, keepdim=True)
        x = _zero_invalid_3d(q_flat + attn * valid.unsqueeze(-1).to(dtype=attn.dtype), q_mask)
        x = _zero_invalid_3d(x + self.ffn(self.ffn_norm(x)), q_mask)
        out = x.reshape(b, t, q_len, d)
        if return_weights:
            return out, weights.reshape(b, t, self.num_heads, q_len, memory.shape[2])
        return out, None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(b, n, self.d_model)


class RelationEntryUnpooler(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        target_num_frames: int = 128,
        num_marker_parts: int = 100,
        relation_part_ids: torch.Tensor | list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.target_num_frames = int(target_num_frames)
        relation_part_ids = _as_id_tensor(
            relation_part_ids,
            NUM_RELATION_TOKENS,
            "relation_part_ids",
            default_value=PART_UNKNOWN,
        )
        relation_part_ids[-1] = PART_GROUND
        _check_id_range(relation_part_ids, num_marker_parts, "relation_part_ids")
        self.register_buffer("relation_ids", torch.arange(NUM_RELATION_TOKENS, dtype=torch.long), persistent=True)
        self.register_buffer("relation_part_ids", relation_part_ids.to(dtype=torch.long), persistent=True)
        relation_type_ids = torch.zeros(NUM_RELATION_TOKENS, dtype=torch.long)
        relation_type_ids[-1] = 1
        self.register_buffer("relation_type_ids", relation_type_ids, persistent=True)
        relation_token_types = torch.full((NUM_RELATION_TOKENS,), int(TokenTypeId.RELATION_BODY_SLOT), dtype=torch.long)
        relation_token_types[-1] = int(TokenTypeId.RELATION_GROUND)
        self.register_buffer("relation_token_type_ids", relation_token_types, persistent=True)

        self.relation_id_embedding = nn.Embedding(NUM_RELATION_TOKENS, d_model)
        self.relation_type_embedding = nn.Embedding(2, d_model)
        self.relation_part_embedding = nn.Embedding(num_marker_parts, d_model)
        self.time_embedding = nn.Embedding(target_num_frames, d_model)
        self.token_type_embedding = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.cross_attention = FrameCrossAttentionBlock(d_model, num_heads, dropout=dropout)

    def forward(self, views: DenseTokenViews, return_diagnostics: bool = False) -> RelationEntryReadout:
        b, t, _, d = views.relation.shape
        query = self._build_query(b, t, views.relation.device)
        memory, memory_mask, segments = _concat_frame_memory(
            [
                ("relation_slots", views.relation, views.relation_mask),
                ("human_slots", views.human, views.human_mask),
                ("object_pose", views.object_pose, views.object_pose_mask),
                ("geometry", _expand_geometry(views.geometry, t), _expand_geometry_mask(views.geometry_mask, t)),
            ]
        )
        mask = views.frame_mask[:, :, None].expand(b, t, NUM_RELATION_TOKENS)
        tokens, weights = self.cross_attention(query, memory, mask, memory_mask, return_weights=return_diagnostics)
        tokens = _zero_invalid_4d(tokens, mask)
        aux: dict[str, Any] = {
            "relation_part_ids_degenerate": _part_ids_degenerate(self.relation_part_ids[:NUM_MARKERS]),
            "relation_query_attention_to_matching_part_available": False,
            "relation_query_attention_to_matching_part_reason": (
                "RelationEntryUnpooler memory uses learned relation/human slots without explicit slot-to-part ids."
            ),
        }
        if return_diagnostics and weights is not None:
            aux.update(_attention_mass_by_segments(weights, mask, segments, "relation"))
        return RelationEntryReadout(tokens=tokens, mask=mask, attention_weights=weights, aux=aux)

    def _build_query(self, batch_size: int, num_frames: int, device: torch.device) -> torch.Tensor:
        if num_frames > self.target_num_frames:
            raise ValueError(f"num_frames={num_frames} exceeds configured target_num_frames={self.target_num_frames}.")
        relation_ids = self.relation_ids.to(device=device)
        part_ids = self.relation_part_ids.to(device=device)
        type_ids = self.relation_type_ids.to(device=device)
        token_types = self.relation_token_type_ids.to(device=device)
        time_ids = torch.arange(num_frames, dtype=torch.long, device=device)
        base = (
            self.relation_id_embedding(relation_ids)
            + self.relation_type_embedding(type_ids)
            + self.relation_part_embedding(part_ids)
            + self.token_type_embedding(token_types)
        )
        query = base[None, None, :, :] + self.time_embedding(time_ids)[None, :, None, :]
        return query.expand(batch_size, num_frames, NUM_RELATION_TOKENS, self.d_model).clone()


class HumanMarkerUnpooler(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        target_num_frames: int = 128,
        num_marker_parts: int = 100,
        marker_part_ids: torch.Tensor | list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.target_num_frames = int(target_num_frames)
        marker_part_ids = _as_id_tensor(marker_part_ids, NUM_MARKERS, "marker_part_ids", default_value=PART_UNKNOWN)
        _check_id_range(marker_part_ids, num_marker_parts, "marker_part_ids")
        self.register_buffer("marker_ids", torch.arange(NUM_MARKERS, dtype=torch.long), persistent=True)
        self.register_buffer("marker_part_ids", marker_part_ids.to(dtype=torch.long), persistent=True)
        self.marker_id_embedding = nn.Embedding(NUM_MARKERS, d_model)
        self.marker_part_embedding = nn.Embedding(num_marker_parts, d_model)
        self.time_embedding = nn.Embedding(target_num_frames, d_model)
        self.token_type_embedding = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.cross_attention = FrameCrossAttentionBlock(d_model, num_heads, dropout=dropout)

    def forward(
        self,
        views: DenseTokenViews,
        relation_entry: torch.Tensor,
        relation_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> HumanMarkerReadout:
        b, t, _, _ = views.human.shape
        query = self._build_query(b, t, views.human.device)
        memory, memory_mask, segments = _concat_frame_memory(
            [
                ("human_slots", views.human, views.human_mask),
                ("relation_entries", relation_entry, relation_mask),
                ("object_pose", views.object_pose, views.object_pose_mask),
                ("geometry", _expand_geometry(views.geometry, t), _expand_geometry_mask(views.geometry_mask, t)),
            ]
        )
        mask = views.frame_mask[:, :, None].expand(b, t, NUM_MARKERS)
        tokens, weights = self.cross_attention(query, memory, mask, memory_mask, return_weights=return_diagnostics)
        tokens = _zero_invalid_4d(tokens, mask)
        aux: dict[str, Any] = {"marker_part_ids_degenerate": _part_ids_degenerate(self.marker_part_ids)}
        if return_diagnostics and weights is not None:
            aux.update(_attention_mass_by_segments(weights, mask, segments, "human"))
            relation_segment = _find_segment(segments, "relation_entries")
            aux["marker_query_attention_to_matching_part"] = _matched_part_attention_mass(
                weights=weights,
                query_mask=mask,
                query_part_ids=self.marker_part_ids,
                key_part_ids=self.marker_part_ids.new_tensor(
                    self.marker_part_ids.detach().cpu().tolist() + [PART_GROUND],
                    dtype=torch.long,
                    device=weights.device,
                ),
                key_segment=relation_segment,
            )
        return HumanMarkerReadout(
            tokens=tokens,
            root_token=_zero_invalid_3d(views.human[:, :, 0], views.human_mask[:, :, 0]),
            left_foot_token=_zero_invalid_3d(views.human[:, :, -2], views.human_mask[:, :, -2]),
            right_foot_token=_zero_invalid_3d(views.human[:, :, -1], views.human_mask[:, :, -1]),
            mask=mask,
            aux=aux,
        )

    def _build_query(self, batch_size: int, num_frames: int, device: torch.device) -> torch.Tensor:
        if num_frames > self.target_num_frames:
            raise ValueError(f"num_frames={num_frames} exceeds configured target_num_frames={self.target_num_frames}.")
        marker_ids = self.marker_ids.to(device=device)
        part_ids = self.marker_part_ids.to(device=device)
        time_ids = torch.arange(num_frames, dtype=torch.long, device=device)
        token_type = torch.full((NUM_MARKERS,), int(TokenTypeId.HUMAN_BODY_SLOT), dtype=torch.long, device=device)
        base = (
            self.marker_id_embedding(marker_ids)
            + self.marker_part_embedding(part_ids)
            + self.token_type_embedding(token_type)
        )
        query = base[None, None, :, :] + self.time_embedding(time_ids)[None, :, None, :]
        return query.expand(batch_size, num_frames, NUM_MARKERS, self.d_model).clone()


class ObjectPoseReadout(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 4, target_num_frames: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.target_num_frames = int(target_num_frames)
        self.time_embedding = nn.Embedding(target_num_frames, d_model)
        self.token_type_embedding = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.cross_attention = FrameCrossAttentionBlock(d_model, num_heads, dropout=dropout)

    def forward(
        self,
        views: DenseTokenViews,
        relation_entry: torch.Tensor,
        relation_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> ObjectPoseReadoutOutput:
        b, t, _, _ = views.object_pose.shape
        if t > self.target_num_frames:
            raise ValueError(f"num_frames={t} exceeds configured target_num_frames={self.target_num_frames}.")
        time_ids = torch.arange(t, dtype=torch.long, device=views.object_pose.device)
        token_type = torch.full((b, t, 1), int(TokenTypeId.OBJECT_POSE), dtype=torch.long, device=views.object_pose.device)
        query = views.object_pose + self.time_embedding(time_ids)[None, :, None, :] + self.token_type_embedding(token_type)
        memory, memory_mask, segments = _concat_frame_memory(
            [
                ("object_pose", views.object_pose, views.object_pose_mask),
                ("relation_entries", relation_entry, relation_mask),
                ("human_slots", views.human, views.human_mask),
                ("geometry", _expand_geometry(views.geometry, t), _expand_geometry_mask(views.geometry_mask, t)),
            ]
        )
        tokens, weights = self.cross_attention(
            query,
            memory,
            views.object_pose_mask,
            memory_mask,
            return_weights=return_diagnostics,
        )
        tokens = _zero_invalid_4d(tokens, views.object_pose_mask).squeeze(2)
        aux: dict[str, Any] = {}
        if return_diagnostics and weights is not None:
            aux.update(_attention_mass_by_segments(weights, views.object_pose_mask, segments, "object"))
        return ObjectPoseReadoutOutput(tokens=tokens, mask=views.object_pose_mask.squeeze(-1), aux=aux)


class PerFrameEntityCouplingBlock(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 4, depth: int = 1, dropout: float = 0.0, ffn_mult: int = 2) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}.")
        self.blocks = nn.ModuleList(
            [_EntitySelfAttentionBlock(d_model, num_heads, dropout=dropout, ffn_mult=ffn_mult) for _ in range(depth)]
        )

    def forward(
        self,
        marker_tokens: torch.Tensor,
        object_token: torch.Tensor,
        relation_entry: torch.Tensor,
        human_mask: torch.Tensor,
        object_mask: torch.Tensor,
        relation_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _, d = marker_tokens.shape
        entities = torch.cat([marker_tokens, object_token[:, :, None, :], relation_entry], dim=2)
        mask = torch.cat([human_mask, object_mask[:, :, None], relation_mask], dim=2)
        x = entities.reshape(b * t, entities.shape[2], d)
        m = mask.reshape(b * t, mask.shape[2]).to(dtype=torch.bool)
        for block in self.blocks:
            x = block(x, m)
        x = _zero_invalid_3d(x, m).reshape(b, t, entities.shape[2], d)
        return x[:, :, :NUM_MARKERS], x[:, :, NUM_MARKERS], x[:, :, NUM_MARKERS + 1 :]


class Stage1FieldHeads(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.0, hidden_mult: int = 2) -> None:
        super().__init__()
        self.human_pos_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.human_vel_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.human_acc_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.root_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.root_vel_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.root_acc_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.left_height_head = FieldHead(d_model, 7, hidden_mult=hidden_mult, dropout=dropout)
        self.right_height_head = FieldHead(d_model, 7, hidden_mult=hidden_mult, dropout=dropout)
        self.object_rot6d_head = FieldHead(d_model, 6, hidden_mult=hidden_mult, dropout=dropout)
        self.object_trans_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.object_trans_vel_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.object_trans_acc_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.relation_delta_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.relation_dist_head = FieldHead(d_model, 1, hidden_mult=hidden_mult, dropout=dropout)
        self.relation_vel_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.relation_acc_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.relation_point_head = FieldHead(d_model, 3, hidden_mult=hidden_mult, dropout=dropout)
        root_object_dim = 3 * d_model
        self.root_object_delta_head = FieldHead(root_object_dim, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.root_object_dist_head = FieldHead(root_object_dim, 1, hidden_mult=hidden_mult, dropout=dropout)
        self.root_object_vel_head = FieldHead(root_object_dim, 3, hidden_mult=hidden_mult, dropout=dropout)
        self.root_object_acc_head = FieldHead(root_object_dim, 3, hidden_mult=hidden_mult, dropout=dropout)

    def forward(
        self,
        marker_tokens: torch.Tensor,
        object_token: torch.Tensor,
        relation_entry: torch.Tensor,
        root_token: torch.Tensor,
        left_foot_token: torch.Tensor,
        right_foot_token: torch.Tensor,
        relation_root_object_slot: torch.Tensor,
        human_mask: torch.Tensor,
        object_mask: torch.Tensor,
        relation_mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        human = {
            "pos_rootrel": _mask_last_dim(self.human_pos_head(marker_tokens), human_mask),
            "vel_rootrel": _mask_last_dim(self.human_vel_head(marker_tokens), human_mask),
            "acc_rootrel": _mask_last_dim(self.human_acc_head(marker_tokens), human_mask),
            "root": _mask_last_dim(self.root_head(root_token), object_mask),
            "root_vel": _mask_last_dim(self.root_vel_head(root_token), object_mask),
            "root_acc": _mask_last_dim(self.root_acc_head(root_token), object_mask),
            "left_heights": _mask_last_dim(self.left_height_head(left_foot_token), object_mask),
            "right_heights": _mask_last_dim(self.right_height_head(right_foot_token), object_mask),
        }
        obj = {
            "rot6d": _mask_last_dim(self.object_rot6d_head(object_token), object_mask),
            "trans": _mask_last_dim(self.object_trans_head(object_token), object_mask),
            "trans_vel": _mask_last_dim(self.object_trans_vel_head(object_token), object_mask),
            "trans_acc": _mask_last_dim(self.object_trans_acc_head(object_token), object_mask),
        }
        root_object_source = torch.cat([relation_root_object_slot, object_token, root_token], dim=-1)
        relation = {
            "delta_m2o": _mask_last_dim(self.relation_delta_head(relation_entry), relation_mask),
            "dist": _mask_last_dim(self.relation_dist_head(relation_entry), relation_mask),
            "delta_vel": _mask_last_dim(self.relation_vel_head(relation_entry), relation_mask),
            "delta_acc": _mask_last_dim(self.relation_acc_head(relation_entry), relation_mask),
            "point_canon": _mask_last_dim(self.relation_point_head(relation_entry), relation_mask),
            "root_object_delta": _mask_last_dim(self.root_object_delta_head(root_object_source), object_mask),
            "root_object_dist": _mask_last_dim(self.root_object_dist_head(root_object_source), object_mask),
            "root_object_delta_vel": _mask_last_dim(self.root_object_vel_head(root_object_source), object_mask),
            "root_object_delta_acc": _mask_last_dim(self.root_object_acc_head(root_object_source), object_mask),
        }
        return human, obj, relation


class Stage1CoarseReadoutHeads(nn.Module):
    output_key = "coarse_prediction"

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
        target_num_frames: int = 128,
        human_slots: int = 11,
        relation_slots: int = 10,
        num_marker_parts: int = 100,
        marker_part_ids: torch.Tensor | list[int] | None = None,
        relation_part_ids: torch.Tensor | list[int] | None = None,
        allow_degenerate_part_ids_for_synthetic_tests_only: bool = False,
        coupling_depth: int = 1,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        marker_part_ids = _as_id_tensor(marker_part_ids, NUM_MARKERS, "marker_part_ids", default_value=PART_UNKNOWN)
        if relation_part_ids is None:
            relation_part_ids = torch.cat([marker_part_ids, torch.tensor([PART_GROUND], dtype=torch.long)])
        relation_part_ids = _as_id_tensor(relation_part_ids, NUM_RELATION_TOKENS, "relation_part_ids", default_value=PART_UNKNOWN)
        relation_part_ids[-1] = PART_GROUND
        _validate_part_ids(marker_part_ids, relation_part_ids, allow_degenerate_part_ids_for_synthetic_tests_only)
        _check_id_range(marker_part_ids, num_marker_parts, "marker_part_ids")
        _check_id_range(relation_part_ids, num_marker_parts, "relation_part_ids")
        self.d_model = int(d_model)
        self.target_num_frames = int(target_num_frames)
        self.human_slots = int(human_slots)
        self.relation_slots = int(relation_slots)
        self.num_marker_parts = int(num_marker_parts)

        self.view_builder = DenseTokenViewBuilder(target_num_frames, human_slots, relation_slots)
        self.relation_unpooler = RelationEntryUnpooler(
            d_model=d_model,
            num_heads=num_heads,
            target_num_frames=target_num_frames,
            num_marker_parts=num_marker_parts,
            relation_part_ids=relation_part_ids,
            dropout=dropout,
        )
        self.human_unpooler = HumanMarkerUnpooler(
            d_model=d_model,
            num_heads=num_heads,
            target_num_frames=target_num_frames,
            num_marker_parts=num_marker_parts,
            marker_part_ids=marker_part_ids,
            dropout=dropout,
        )
        self.object_readout = ObjectPoseReadout(d_model, num_heads, target_num_frames, dropout=dropout)
        self.entity_coupling = PerFrameEntityCouplingBlock(
            d_model=d_model,
            num_heads=num_heads,
            depth=coupling_depth,
            dropout=dropout,
            ffn_mult=ffn_mult,
        )
        self.field_heads = Stage1FieldHeads(d_model=d_model, dropout=dropout)

    def forward(
        self,
        dense_tokens: TokenizerOutput,
        return_diagnostics: bool = False,
        validate: bool = False,
    ) -> Stage1CoarsePrediction:
        views = self.view_builder(dense_tokens, validate=validate)
        relation_entry = self.relation_unpooler(views, return_diagnostics=return_diagnostics)
        human_marker = self.human_unpooler(
            views,
            relation_entry=relation_entry.tokens,
            relation_mask=relation_entry.mask,
            return_diagnostics=return_diagnostics,
        )
        object_pose = self.object_readout(
            views,
            relation_entry=relation_entry.tokens,
            relation_mask=relation_entry.mask,
            return_diagnostics=return_diagnostics,
        )
        marker_tokens, object_token, relation_tokens = self.entity_coupling(
            marker_tokens=human_marker.tokens,
            object_token=object_pose.tokens,
            relation_entry=relation_entry.tokens,
            human_mask=human_marker.mask,
            object_mask=object_pose.mask,
            relation_mask=relation_entry.mask,
        )
        root_object_slot = _zero_invalid_3d(views.relation[:, :, -1], views.relation_mask[:, :, -1])
        human, obj, relation = self.field_heads(
            marker_tokens=marker_tokens,
            object_token=object_token,
            relation_entry=relation_tokens,
            root_token=human_marker.root_token,
            left_foot_token=human_marker.left_foot_token,
            right_foot_token=human_marker.right_foot_token,
            relation_root_object_slot=root_object_slot,
            human_mask=human_marker.mask,
            object_mask=object_pose.mask,
            relation_mask=relation_entry.mask,
        )
        masks = {
            "frame": views.frame_mask,
            "human": human_marker.mask,
            "object": object_pose.mask,
            "relation": relation_entry.mask,
            "velocity": views.frame_mask,
            "acceleration": views.frame_mask,
        }
        aux = dict(dense_tokens.aux)
        aux["coarse_readout_heads"] = self._build_aux(
            relation_entry=relation_entry,
            human_marker=human_marker,
            object_pose=object_pose,
            prediction=(human, obj, relation),
            masks=masks,
            return_diagnostics=return_diagnostics,
        )
        pred = Stage1CoarsePrediction(human=human, object=obj, relation=relation, masks=masks, aux=aux)
        if validate:
            validate_stage1_coarse_prediction(pred)
        return pred

    def _build_aux(
        self,
        relation_entry: RelationEntryReadout,
        human_marker: HumanMarkerReadout,
        object_pose: ObjectPoseReadoutOutput,
        prediction: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]],
        masks: dict[str, torch.Tensor],
        return_diagnostics: bool,
    ) -> dict[str, Any]:
        human, obj, relation = prediction
        invalid_max = _invalid_prediction_max_abs(human, obj, relation, masks)
        aux: dict[str, Any] = {
            "output_key": self.output_key,
            "diagnostics_enabled": bool(return_diagnostics),
            "target_num_frames": self.target_num_frames,
            "human_slots": self.human_slots,
            "relation_slots": self.relation_slots,
            "invalid_output_max_abs": invalid_max.detach(),
            "marker_part_ids_degenerate": human_marker.aux["marker_part_ids_degenerate"],
            "relation_part_ids_degenerate": relation_entry.aux["relation_part_ids_degenerate"],
        }
        if return_diagnostics:
            aux.update(_prefix_keys(human_marker.aux, "human"))
            aux.update(_prefix_keys(object_pose.aux, "object"))
            aux.update(_prefix_keys(relation_entry.aux, "relation"))
            aux.update(
                {
                    "relation_entry_variance": _masked_variance(relation_entry.tokens, relation_entry.mask).detach(),
                    "relation_head_delta_std": relation["delta_m2o"][masks["relation"]].std(unbiased=False).detach()
                    if masks["relation"].any()
                    else relation["delta_m2o"].new_tensor(0.0),
                    "relation_head_dist_std": relation["dist"][masks["relation"]].std(unbiased=False).detach()
                    if masks["relation"].any()
                    else relation["dist"].new_tensor(0.0),
                    "relation_output_nonzero_ratio": _nonzero_ratio(relation, masks["relation"]).detach(),
                    "frame_mask_valid_ratio": masks["frame"].to(dtype=torch.float32).mean().detach(),
                }
            )
        return aux


class _EntitySelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, ffn_mult: int = 2) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, use_rope=False)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, ffn_mult=ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = _zero_invalid_3d(x + self.attn(self.attn_norm(x), key_mask=mask, query_mask=mask), mask)
        x = _zero_invalid_3d(x + self.ffn(self.ffn_norm(x)), mask)
        return x


class FieldHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, hidden_mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(hidden_mult) * int(d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_stage1_coarse_readout_heads_from_config(config: dict) -> Stage1CoarseReadoutHeads:
    stage1_cfg = config.get("model", {}).get("stage1", {})
    tokenizer_cfg = stage1_cfg.get("tokenizers", {})
    decoder_cfg = stage1_cfg.get("decoder", {})
    readout_cfg = decoder_cfg.get("readout_heads", {})
    _validate_readout_config(readout_cfg)
    ids_cfg = readout_cfg.get("ids", {})
    require_marker_part_ids = bool(ids_cfg.get("require_marker_part_ids", True))
    allow_degenerate = bool(ids_cfg.get("allow_degenerate_part_ids_for_synthetic_tests_only", False))
    map_path = ids_cfg.get("marker_part_map_path") or config.get("data", {}).get("structured", {}).get("marker_part_map_path")
    if require_marker_part_ids and not map_path and not allow_degenerate:
        raise ValueError("Stage1CoarseReadoutHeads requires ids.marker_part_map_path or data.structured.marker_part_map_path.")
    if map_path and not Path(map_path).exists():
        raise ValueError(f"marker_part_map_path does not exist: {map_path}")
    static_ids = build_stage1_ids(map_path)
    human_slots = int(readout_cfg.get("human_slots", 1 + int(tokenizer_cfg.get("human", {}).get("num_body_slots", 8)) + 2))
    relation_slots = int(readout_cfg.get("relation_slots", int(tokenizer_cfg.get("relation", {}).get("num_relation_slots", 8)) + 2))
    coupling_cfg = readout_cfg.get("coupling", {})
    return Stage1CoarseReadoutHeads(
        d_model=int(readout_cfg.get("d_model", stage1_cfg.get("d_model", 256))),
        num_heads=int(readout_cfg.get("num_heads", stage1_cfg.get("num_heads", 4))),
        dropout=float(readout_cfg.get("dropout", stage1_cfg.get("dropout", 0.0))),
        target_num_frames=int(readout_cfg.get("target_num_frames", stage1_cfg.get("max_seq_len", 128))),
        human_slots=human_slots,
        relation_slots=relation_slots,
        num_marker_parts=int(stage1_cfg.get("num_marker_parts", 100)),
        marker_part_ids=static_ids["marker_part_ids"],
        relation_part_ids=static_ids["relation_part_ids"],
        allow_degenerate_part_ids_for_synthetic_tests_only=allow_degenerate,
        coupling_depth=int(coupling_cfg.get("per_frame_entity_attention_depth", 1)),
    )


def validate_stage1_coarse_prediction(prediction: Stage1CoarsePrediction) -> None:
    required_human = {
        "pos_rootrel": (77, 3),
        "vel_rootrel": (77, 3),
        "acc_rootrel": (77, 3),
        "root": (3,),
        "root_vel": (3,),
        "root_acc": (3,),
        "left_heights": (7,),
        "right_heights": (7,),
    }
    required_object = {"rot6d": (6,), "trans": (3,), "trans_vel": (3,), "trans_acc": (3,)}
    required_relation = {
        "delta_m2o": (78, 3),
        "dist": (78, 1),
        "delta_vel": (78, 3),
        "delta_acc": (78, 3),
        "point_canon": (78, 3),
        "root_object_delta": (3,),
        "root_object_dist": (1,),
        "root_object_delta_vel": (3,),
        "root_object_delta_acc": (3,),
    }
    if "frame" not in prediction.masks:
        raise ValueError("prediction.masks must include frame mask.")
    b, t = prediction.masks["frame"].shape
    _validate_field_group(prediction.human, required_human, b, t, "human")
    _validate_field_group(prediction.object, required_object, b, t, "object")
    _validate_field_group(prediction.relation, required_relation, b, t, "relation")
    _validate_masked_zero(prediction.human, prediction.masks["human"], ["pos_rootrel", "vel_rootrel", "acc_rootrel"])
    _validate_masked_zero(
        prediction.human,
        prediction.masks["object"],
        ["root", "root_vel", "root_acc", "left_heights", "right_heights"],
    )
    _validate_masked_zero(prediction.relation, prediction.masks["relation"], ["delta_m2o", "dist", "delta_vel", "delta_acc", "point_canon"])
    _validate_masked_zero(
        prediction.relation,
        prediction.masks["object"],
        ["root_object_delta", "root_object_dist", "root_object_delta_vel", "root_object_delta_acc"],
    )
    _validate_masked_zero(prediction.object, prediction.masks["object"], ["rot6d", "trans", "trans_vel", "trans_acc"])


def _concat_frame_memory(
    pieces: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, slice]]]:
    tokens = []
    masks = []
    segments: list[tuple[str, slice]] = []
    cursor = 0
    for name, value, mask in pieces:
        if value.shape[:3] != mask.shape:
            raise ValueError(f"Memory piece {name!r} mask shape {tuple(mask.shape)} does not match tokens {tuple(value.shape)}.")
        tokens.append(value)
        masks.append(mask)
        segments.append((name, slice(cursor, cursor + value.shape[2])))
        cursor += value.shape[2]
    return torch.cat(tokens, dim=2), torch.cat(masks, dim=2), segments


def _expand_geometry(geometry: torch.Tensor, num_frames: int) -> torch.Tensor:
    return geometry[:, None, :, :].expand(geometry.shape[0], num_frames, geometry.shape[1], geometry.shape[2])


def _expand_geometry_mask(geometry_mask: torch.Tensor, num_frames: int) -> torch.Tensor:
    return geometry_mask[:, None, :].expand(geometry_mask.shape[0], num_frames, geometry_mask.shape[1])


def _attention_mass_by_segments(
    weights: torch.Tensor,
    query_mask: torch.Tensor,
    segments: list[tuple[str, slice]],
    prefix: str,
) -> dict[str, torch.Tensor]:
    query_valid = query_mask[:, :, None, :].to(dtype=weights.dtype)
    denom = query_valid.sum().mul(weights.shape[2]).clamp_min(1.0)
    out = {}
    for name, segment in segments:
        out[f"{prefix}_attention_to_{name}_mass"] = (
            weights[..., segment].sum(dim=-1) * query_valid
        ).sum().detach() / denom
    return out


def _find_segment(segments: list[tuple[str, slice]], name: str) -> slice:
    for segment_name, segment in segments:
        if segment_name == name:
            return segment
    raise ValueError(f"Missing attention segment {name!r}.")


def _matched_part_attention_mass(
    *,
    weights: torch.Tensor,
    query_mask: torch.Tensor,
    query_part_ids: torch.Tensor,
    key_part_ids: torch.Tensor,
    key_segment: slice,
) -> torch.Tensor:
    query_part_ids = query_part_ids.to(device=weights.device, dtype=torch.long)
    key_part_ids = key_part_ids.to(device=weights.device, dtype=torch.long)
    segment_width = key_segment.stop - key_segment.start
    if key_part_ids.numel() != segment_width:
        raise ValueError(
            f"key_part_ids length {key_part_ids.numel()} does not match attention segment width {segment_width}."
        )
    if query_part_ids.numel() != weights.shape[-2]:
        raise ValueError(
            f"query_part_ids length {query_part_ids.numel()} does not match attention query width {weights.shape[-2]}."
        )
    match = query_part_ids[:, None] == key_part_ids[None, :]
    match = match & (query_part_ids[:, None] != PART_UNKNOWN)
    selected = weights[..., key_segment] * match[None, None, None, :, :].to(dtype=weights.dtype)
    query_valid = query_mask[:, :, None, :].to(dtype=weights.dtype)
    denom = query_valid.sum().mul(weights.shape[2]).clamp_min(1.0)
    return (selected.sum(dim=-1) * query_valid).sum().detach() / denom


def _as_id_tensor(ids: torch.Tensor | list[int] | None, expected: int, name: str, default_value: int) -> torch.Tensor:
    if ids is None:
        return torch.full((expected,), int(default_value), dtype=torch.long)
    value = torch.as_tensor(ids, dtype=torch.long).clone()
    if tuple(value.shape) != (expected,):
        raise ValueError(f"{name} must have shape ({expected},), got {tuple(value.shape)}.")
    return value


def _validate_part_ids(marker_part_ids: torch.Tensor, relation_part_ids: torch.Tensor, allow_degenerate: bool) -> None:
    if relation_part_ids[-1].item() != PART_GROUND:
        raise ValueError(f"relation_part_ids[-1] must be PART_GROUND={PART_GROUND}.")
    if not torch.equal(relation_part_ids[:NUM_MARKERS], marker_part_ids):
        raise ValueError("relation_part_ids[:77] must match marker_part_ids.")
    if not allow_degenerate and _part_ids_degenerate(marker_part_ids):
        raise ValueError("Stage1CoarseReadoutHeads requires non-degenerate marker_part_ids for training/real data.")


def _part_ids_degenerate(ids: torch.Tensor) -> bool:
    ids = ids.detach().cpu()
    return bool(ids.numel() == 0 or torch.unique(ids).numel() <= 1 or torch.all(ids == PART_UNKNOWN))


def _check_id_range(ids: torch.Tensor, num_ids: int, name: str) -> None:
    if (ids < 0).any() or (ids >= int(num_ids)).any():
        raise ValueError(f"{name} must be in [0, {num_ids}), got min={ids.min().item()} max={ids.max().item()}.")


def _zero_invalid_3d(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return tokens * mask.to(dtype=tokens.dtype).unsqueeze(-1)


def _zero_invalid_4d(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return tokens * mask.to(dtype=tokens.dtype).unsqueeze(-1)


def _mask_last_dim(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values * mask.to(dtype=values.dtype).unsqueeze(-1)


def _validate_view_masks(
    human_mask: torch.Tensor,
    object_pose_mask: torch.Tensor,
    relation_mask: torch.Tensor,
    frame_mask: torch.Tensor,
) -> None:
    if not torch.equal(object_pose_mask.squeeze(-1), frame_mask):
        raise ValueError("object pose mask must define the frame mask.")
    expected_human = frame_mask[:, :, None].expand_as(human_mask)
    expected_relation = frame_mask[:, :, None].expand_as(relation_mask)
    if not torch.equal(human_mask, expected_human):
        raise ValueError("human dense masks are not aligned with frame mask.")
    if not torch.equal(relation_mask, expected_relation):
        raise ValueError("relation dense masks are not aligned with frame mask.")


def _validate_field_group(group: dict[str, torch.Tensor], expected: dict[str, tuple[int, ...]], b: int, t: int, name: str) -> None:
    for key, trailing in expected.items():
        if key not in group:
            raise ValueError(f"{name} missing required field {key!r}.")
        value = group[key]
        if tuple(value.shape) != (b, t, *trailing):
            raise ValueError(f"{name}.{key} expected shape {(b, t, *trailing)}, got {tuple(value.shape)}.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}.{key} contains non-finite values.")


def _validate_masked_zero(group: dict[str, torch.Tensor], mask: torch.Tensor, keys: list[str]) -> None:
    invalid = ~mask
    if not invalid.any():
        return
    for key in keys:
        value = group[key]
        if value[invalid].abs().max().item() != 0.0:
            raise ValueError(f"{key} has non-zero invalid outputs.")


def _invalid_prediction_max_abs(
    human: dict[str, torch.Tensor],
    obj: dict[str, torch.Tensor],
    relation: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> torch.Tensor:
    values = []
    if (~masks["human"]).any():
        values.extend([human[key][~masks["human"]].abs().max() for key in ["pos_rootrel", "vel_rootrel", "acc_rootrel"]])
    if (~masks["object"]).any():
        values.extend([obj[key][~masks["object"]].abs().max() for key in ["rot6d", "trans", "trans_vel", "trans_acc"]])
        values.extend(
            [
                human[key][~masks["object"]].abs().max()
                for key in ["root", "root_vel", "root_acc", "left_heights", "right_heights"]
            ]
        )
        values.extend(
            [
                relation[key][~masks["object"]].abs().max()
                for key in ["root_object_delta", "root_object_dist", "root_object_delta_vel", "root_object_delta_acc"]
            ]
        )
    if (~masks["relation"]).any():
        values.extend([relation[key][~masks["relation"]].abs().max() for key in ["delta_m2o", "dist", "delta_vel", "delta_acc", "point_canon"]])
    if not values:
        return next(iter(human.values())).new_tensor(0.0)
    return torch.stack(values).max()


def _masked_variance(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool)
    if not valid.any():
        return tokens.new_tensor(0.0)
    selected = tokens[valid]
    return selected.var(unbiased=False)


def _nonzero_ratio(group: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    valid_values = []
    for value in group.values():
        if value.shape[: mask.ndim] == mask.shape and mask.any():
            valid_values.append((value[mask].abs() > 1e-8).to(dtype=torch.float32).mean())
    if not valid_values:
        return next(iter(group.values())).new_tensor(0.0)
    return torch.stack(valid_values).mean()


def _prefix_keys(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _validate_readout_config(readout_cfg: dict) -> None:
    if readout_cfg.get("mode") in {"flat_motion_962_readout", "human_object_only_readout"}:
        raise ValueError(f"Unsupported Stage1CoarseReadoutHeads mode: {readout_cfg.get('mode')!r}.")
    relation_cfg = readout_cfg.get("relation", {})
    if relation_cfg.get("output_required", True) is not True:
        raise ValueError("Stage1CoarseReadoutHeads requires relation.output_required=true.")
    if relation_cfg.get("dist_parameterization", "linear_model_space") != "linear_model_space":
        raise ValueError("Stage1CoarseReadoutHeads only supports linear_model_space distance readout.")
    coupling_cfg = readout_cfg.get("coupling", {})
    if coupling_cfg.get("relation_entry_memory_required", True) is not True:
        raise ValueError("Stage1CoarseReadoutHeads requires relation_entry_memory_required=true.")
