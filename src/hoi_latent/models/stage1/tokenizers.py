from __future__ import annotations

"""Stage 1 tokenizer modules.

The current learned absolute TimeEmbedding supports fixed-window sequences with
T <= max_seq_len. Full-sequence evaluation requires increasing max_seq_len or
switching to a different time encoding later.
"""

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from hoi_latent.models.stage1.attention_pool import CrossAttentionPool
from hoi_latent.models.stage1.embeddings import TimeEmbedding, TokenTypeEmbedding
from hoi_latent.models.stage1.mask_utils import repeat_valid_over_slots, zero_invalid_tokens
from hoi_latent.models.stage1.tokenizer_types import (
    NUM_TOKEN_TYPES,
    ModalityId,
    TokenizerOutput,
    TokenTypeId,
    assert_shape,
    concat_tokenizer_outputs,
    validate_tokenizer_output,
)


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, d_model: int, hidden_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(hidden_mult) * int(d_model)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HumanTokenizer(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        num_heads: int,
        num_body_slots: int,
        num_marker_parts: int,
        dropout: float = 0.0,
        use_marker_part_bias: bool = True,
        include_official_vel: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_body_slots = int(num_body_slots)
        self.num_marker_parts = int(num_marker_parts)
        self.use_marker_part_bias = bool(use_marker_part_bias)
        self.include_official_vel = bool(include_official_vel)

        marker_in_dim = 12 if self.include_official_vel else 9
        self.marker_proj = MLPProjector(marker_in_dim, d_model, dropout=dropout)
        self.root_proj = MLPProjector(9, d_model, dropout=dropout)
        self.left_foot_proj = MLPProjector(7, d_model, dropout=dropout)
        self.right_foot_proj = MLPProjector(7, d_model, dropout=dropout)

        self.time_embedding = TimeEmbedding(max_seq_len, d_model)
        self.token_type_embedding = TokenTypeEmbedding(NUM_TOKEN_TYPES, d_model)
        self.marker_id_embedding = nn.Embedding(77, d_model)
        self.marker_part_embedding = nn.Embedding(num_marker_parts, d_model)
        self.body_slot_queries = nn.Parameter(torch.randn(num_body_slots, d_model) * 0.02)
        self.body_slot_output_embedding = nn.Embedding(num_body_slots, d_model)
        self.body_pool = CrossAttentionPool(d_model, num_heads, dropout=dropout)
        self.body_part_bias = nn.Embedding(num_marker_parts, num_body_slots)
        nn.init.zeros_(self.body_part_bias.weight)

    def forward(
        self,
        batch: dict,
        return_aux: bool = False,
        return_aux_detached: bool = True,
    ) -> TokenizerOutput:
        human = _require_group(batch, "model", "human")
        masks = _require_dict(batch, "masks")
        ids = _require_dict(batch, "ids")

        pos_rootrel = _require_tensor(human, "pos_rootrel", "model.human.pos_rootrel")
        vel_rootrel = _require_tensor(human, "vel_rootrel", "model.human.vel_rootrel")
        acc_rootrel = _require_tensor(human, "acc_rootrel", "model.human.acc_rootrel")
        root = _require_tensor(human, "root", "model.human.root")
        root_vel = _require_tensor(human, "root_vel", "model.human.root_vel")
        root_acc = _require_tensor(human, "root_acc", "model.human.root_acc")
        left_heights = _require_tensor(human, "left_heights", "model.human.left_heights")
        right_heights = _require_tensor(human, "right_heights", "model.human.right_heights")

        assert_shape(pos_rootrel, 4, "model.human.pos_rootrel")
        b, t, num_markers, _ = pos_rootrel.shape
        _check_seq_len(self.__class__.__name__, t, self.time_embedding.max_len)
        if num_markers != 77 or pos_rootrel.shape[-1] != 3:
            raise ValueError(f"model.human.pos_rootrel must have shape [B,T,77,3], got {tuple(pos_rootrel.shape)}.")
        _assert_same_shape(vel_rootrel, pos_rootrel, "model.human.vel_rootrel")
        _assert_same_shape(acc_rootrel, pos_rootrel, "model.human.acc_rootrel")
        _assert_shape_tuple(root, (b, t, 3), "model.human.root")
        _assert_shape_tuple(root_vel, (b, t, 3), "model.human.root_vel")
        _assert_shape_tuple(root_acc, (b, t, 3), "model.human.root_acc")
        _assert_shape_tuple(left_heights, (b, t, 7), "model.human.left_heights")
        _assert_shape_tuple(right_heights, (b, t, 7), "model.human.right_heights")

        device = pos_rootrel.device
        valid_mask = _valid_mask(masks, b, t, device)
        human_mask = _human_mask(masks, b, t, device)
        marker_ids = _static_ids(ids, "marker_ids", torch.arange(77, device=device), device, 77)
        marker_part_ids = _static_ids(
            ids,
            "marker_part_ids",
            torch.zeros(77, dtype=torch.long, device=device),
            device,
            77,
        )
        _check_id_range(marker_ids, 77, "ids.marker_ids")
        _check_id_range(marker_part_ids, self.num_marker_parts, "ids.marker_part_ids")

        marker_features = [pos_rootrel, vel_rootrel, acc_rootrel]
        if self.include_official_vel:
            official = human.get("vel_official")
            if official is None:
                official = torch.zeros_like(pos_rootrel)
            else:
                _assert_same_shape(official, pos_rootrel, "model.human.vel_official")
            marker_features.append(official)
        marker_feat = torch.cat(marker_features, dim=-1)
        marker_low = self.marker_proj(marker_feat)

        frame_time_ids = _frame_time_ids(b, t, device)
        time_emb = self.time_embedding(frame_time_ids)
        body_type_emb = self._token_type_emb(TokenTypeId.HUMAN_BODY_SLOT, device)
        marker_low = marker_low + time_emb[:, :, None, :]
        marker_low = marker_low + self.marker_id_embedding(marker_ids)[None, None, :, :]
        marker_low = marker_low + self.marker_part_embedding(marker_part_ids)[None, None, :, :]
        marker_low = marker_low + body_type_emb

        p = self.num_body_slots
        kv = marker_low.reshape(b * t, num_markers, self.d_model)
        key_mask = human_mask.reshape(b * t, num_markers)
        body_q = self.body_slot_queries[None, None, :, :].expand(b, t, p, self.d_model)
        body_q = body_q + time_emb[:, :, None, :]
        body_q = body_q.reshape(b * t, p, self.d_model)
        query_mask = valid_mask.reshape(b * t, 1).expand(b * t, p)
        marker_part_bias, used_part_bias = self._marker_part_bias(marker_part_ids, b, t, num_markers, device)
        body, body_attn = self.body_pool(
            query=body_q,
            key_value=kv,
            key_mask=key_mask,
            query_mask=query_mask,
            attn_bias=marker_part_bias,
            need_weights=return_aux,
        )
        body = body.reshape(b, t, p, self.d_model)
        body_slot_ids = torch.arange(p, dtype=torch.long, device=device)
        body = body + self.body_slot_output_embedding(body_slot_ids)[None, None, :, :]
        body = body + self._token_type_emb(TokenTypeId.HUMAN_BODY_SLOT, device)
        body = body * valid_mask[:, :, None, None].to(dtype=body.dtype)

        root_feat = torch.cat([root, root_vel, root_acc], dim=-1)
        root_token = self.root_proj(root_feat)
        root_token = root_token + time_emb + self._token_type_emb(TokenTypeId.HUMAN_ROOT, device)
        left_token = self.left_foot_proj(left_heights)
        left_token = left_token + time_emb + self._token_type_emb(TokenTypeId.HUMAN_FOOT_LEFT, device)
        right_token = self.right_foot_proj(right_heights)
        right_token = right_token + time_emb + self._token_type_emb(TokenTypeId.HUMAN_FOOT_RIGHT, device)

        frame = torch.cat(
            [root_token[:, :, None, :], body, left_token[:, :, None, :], right_token[:, :, None, :]],
            dim=2,
        )
        slots_per_frame = 1 + p + 2
        tokens = frame.reshape(b, t * slots_per_frame, self.d_model)
        mask = repeat_valid_over_slots(valid_mask, slots_per_frame)
        tokens = zero_invalid_tokens(tokens, mask)

        token_type_ids = _repeat_per_frame(
            torch.tensor(
                [int(TokenTypeId.HUMAN_ROOT)] + [int(TokenTypeId.HUMAN_BODY_SLOT)] * p + [
                    int(TokenTypeId.HUMAN_FOOT_LEFT),
                    int(TokenTypeId.HUMAN_FOOT_RIGHT),
                ],
                dtype=torch.long,
                device=device,
            ),
            b,
            t,
        )
        out = TokenizerOutput(
            tokens=tokens,
            mask=mask,
            modality_ids=torch.full_like(token_type_ids, int(ModalityId.HUMAN)),
            token_type_ids=token_type_ids,
            time_ids=frame_time_ids[:, :, None].expand(b, t, slots_per_frame).reshape(b, -1),
            entity_ids=_repeat_per_frame(torch.arange(slots_per_frame, dtype=torch.long, device=device), b, t),
            time_mask=mask.clone(),
            aux={},
        )
        if return_aux:
            out.aux = {
                "marker_low": _maybe_detach(marker_low, return_aux_detached),
                "body_pool_attn": _maybe_detach(body_attn, return_aux_detached),
                "num_body_slots": p,
                "marker_part_ids_unique": marker_part_ids.unique().detach().cpu().tolist(),
                "used_marker_part_bias": used_part_bias,
            }
        return out

    def _token_type_emb(self, token_type: TokenTypeId, device: torch.device) -> torch.Tensor:
        ids = torch.as_tensor(int(token_type), dtype=torch.long, device=device)
        return self.token_type_embedding(ids)

    def _marker_part_bias(
        self,
        marker_part_ids: torch.Tensor,
        batch_size: int,
        seq_len: int,
        num_markers: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], bool]:
        if not self.use_marker_part_bias or marker_part_ids.unique().numel() <= 1:
            return None, False
        part_bias = self.body_part_bias(marker_part_ids).transpose(0, 1)
        part_bias = part_bias[None, None, :, :].expand(batch_size * seq_len, 1, self.num_body_slots, num_markers)
        return part_bias.to(device=device), True


class ObjectTokenizer(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        num_heads: int,
        num_geometry_slots: int,
        geometry_source: str = "bps",
        dropout: float = 0.0,
        include_official_vel9: bool = True,
        use_object_id_embedding: bool = False,
        num_objects: Optional[int] = None,
        max_bps_points: int = 1024,
    ):
        super().__init__()
        if geometry_source not in {"bps", "points", "both"}:
            raise ValueError(f"geometry_source must be 'bps', 'points', or 'both', got {geometry_source!r}.")
        self.d_model = int(d_model)
        self.num_geometry_slots = int(num_geometry_slots)
        self.geometry_source = geometry_source
        self.include_official_vel9 = bool(include_official_vel9)
        self.use_object_id_embedding = bool(use_object_id_embedding)
        self.max_bps_points = int(max_bps_points)

        pose_in_dim = 24 if self.include_official_vel9 else 15
        self.pose_proj = MLPProjector(pose_in_dim, d_model, dropout=dropout)
        self.bps_proj = MLPProjector(3, d_model, dropout=dropout)
        self.point_proj = MLPProjector(3, d_model, dropout=dropout)
        self.time_embedding = TimeEmbedding(max_seq_len, d_model)
        self.token_type_embedding = TokenTypeEmbedding(NUM_TOKEN_TYPES, d_model)
        self.bps_index_embedding = nn.Embedding(self.max_bps_points, d_model)
        self.geometry_source_embedding = nn.Embedding(2, d_model)
        self.geometry_slot_queries = nn.Parameter(torch.randn(num_geometry_slots, d_model) * 0.02)
        self.geometry_slot_output_embedding = nn.Embedding(num_geometry_slots, d_model)
        self.geometry_pool = CrossAttentionPool(d_model, num_heads, dropout=dropout)
        if self.use_object_id_embedding:
            if num_objects is None or num_objects <= 0:
                raise ValueError("num_objects must be provided when use_object_id_embedding=True.")
            self.object_id_embedding = nn.Embedding(num_objects, d_model)
        else:
            self.object_id_embedding = None

    def forward(
        self,
        batch: dict,
        return_aux: bool = False,
        return_aux_detached: bool = True,
    ) -> TokenizerOutput:
        obj = _require_group(batch, "model", "object")
        geometry = _require_dict(batch, "geometry")
        masks = _require_dict(batch, "masks")
        ids = _require_dict(batch, "ids")

        rot6d = _require_tensor(obj, "rot6d", "model.object.rot6d")
        trans = _require_tensor(obj, "trans", "model.object.trans")
        trans_vel = _require_tensor(obj, "trans_vel", "model.object.trans_vel")
        trans_acc = _require_tensor(obj, "trans_acc", "model.object.trans_acc")
        assert_shape(rot6d, 3, "model.object.rot6d")
        b, t, _ = rot6d.shape
        _check_seq_len(self.__class__.__name__, t, self.time_embedding.max_len)
        _assert_shape_tuple(rot6d, (b, t, 6), "model.object.rot6d")
        _assert_shape_tuple(trans, (b, t, 3), "model.object.trans")
        _assert_shape_tuple(trans_vel, (b, t, 3), "model.object.trans_vel")
        _assert_shape_tuple(trans_acc, (b, t, 3), "model.object.trans_acc")

        device = rot6d.device
        object_mask = _object_mask(masks, b, t, device)
        pose_features = [rot6d, trans, trans_vel, trans_acc]
        if self.include_official_vel9:
            vel9 = obj.get("vel9_official")
            if vel9 is None:
                vel9 = torch.zeros(b, t, 9, dtype=rot6d.dtype, device=device)
            else:
                _assert_shape_tuple(vel9, (b, t, 9), "model.object.vel9_official")
            pose_features.append(vel9)
        pose_feat = torch.cat(pose_features, dim=-1)
        pose = self.pose_proj(pose_feat)
        frame_time_ids = _frame_time_ids(b, t, device)
        pose = pose + self.time_embedding(frame_time_ids)
        pose = pose + self._token_type_emb(TokenTypeId.OBJECT_POSE, device)

        geometry_low, geometry_mask, geometry_source = self._build_geometry_low(geometry, b, device)
        g = self.num_geometry_slots
        geo_query = self.geometry_slot_queries[None, :, :].expand(b, g, self.d_model)
        geo_query_mask = geometry_mask.any(dim=1, keepdim=True).expand(b, g)
        geo, geo_attn = self.geometry_pool(
            query=geo_query,
            key_value=geometry_low,
            key_mask=geometry_mask,
            query_mask=geo_query_mask,
            need_weights=return_aux,
        )
        geo_slot_ids = torch.arange(g, dtype=torch.long, device=device)
        geo = geo + self.geometry_slot_output_embedding(geo_slot_ids)[None, :, :]
        geo = geo + self._token_type_emb(TokenTypeId.OBJECT_GEOMETRY, device)
        geo = geo * geo_query_mask.to(dtype=geo.dtype).unsqueeze(-1)

        if self.object_id_embedding is not None:
            object_ids = _batch_ids(ids, ["object_ids", "object_id"], b, device)
            _check_id_range(object_ids, self.object_id_embedding.num_embeddings, "ids.object_id")
            object_emb = self.object_id_embedding(object_ids)
            pose = pose + object_emb[:, None, :]
            geo = geo + object_emb[:, None, :]

        tokens = torch.cat([pose, geo], dim=1)
        mask = torch.cat([object_mask, geo_query_mask], dim=1)
        tokens = zero_invalid_tokens(tokens, mask)
        token_type_ids = torch.cat(
            [
                torch.full((b, t), int(TokenTypeId.OBJECT_POSE), dtype=torch.long, device=device),
                torch.full((b, g), int(TokenTypeId.OBJECT_GEOMETRY), dtype=torch.long, device=device),
            ],
            dim=1,
        )
        time_ids = torch.cat(
            [
                frame_time_ids,
                torch.zeros((b, g), dtype=torch.long, device=device),
            ],
            dim=1,
        )
        time_mask = torch.cat(
            [
                object_mask,
                torch.zeros((b, g), dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        entity_ids = torch.cat(
            [
                torch.zeros((b, t), dtype=torch.long, device=device),
                torch.arange(1, g + 1, dtype=torch.long, device=device)[None, :].expand(b, g),
            ],
            dim=1,
        )
        out = TokenizerOutput(
            tokens=tokens,
            mask=mask,
            modality_ids=torch.full_like(token_type_ids, int(ModalityId.OBJECT)),
            token_type_ids=token_type_ids,
            time_ids=time_ids,
            entity_ids=entity_ids,
            time_mask=time_mask,
            aux={},
        )
        if return_aux:
            out.aux = {
                "geometry_low_shape": tuple(geometry_low.shape),
                "geometry_source": geometry_source,
                "geometry_pool_attn": _maybe_detach(geo_attn, return_aux_detached),
            }
        return out

    def _build_geometry_low(self, geometry: dict, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, str]:
        lows = []
        masks = []
        sources = []
        if self.geometry_source in {"bps", "both"}:
            bps = _require_tensor(geometry, "object_bps", "geometry.object_bps")
            if bps.ndim == 4 and bps.shape[1] == 1:
                bps = bps[:, 0]
            if bps.ndim != 3 or bps.shape[0] != batch_size or bps.shape[-1] != 3 or bps.shape[1] == 0:
                raise ValueError(f"geometry.object_bps must have shape [B,N,3] with N>0, got {tuple(bps.shape)}.")
            n_bps = int(bps.shape[1])
            if n_bps > self.max_bps_points:
                raise ValueError(
                    f"geometry.object_bps has {n_bps} points, but max_bps_points={self.max_bps_points}."
                )
            bps_mask = geometry.get("object_bps_mask")
            if bps_mask is None:
                bps_mask = torch.ones(batch_size, bps.shape[1], dtype=torch.bool, device=device)
            else:
                bps_mask = bps_mask.to(device=device, dtype=torch.bool)
                if bps_mask.ndim == 3 and bps_mask.shape[1] == 1:
                    bps_mask = bps_mask[:, 0]
                _assert_shape_tuple(bps_mask, (batch_size, bps.shape[1]), "geometry.object_bps_mask")
            bps_ids = torch.arange(n_bps, dtype=torch.long, device=device)
            bps_low = self.bps_proj(bps)
            bps_low = bps_low + self.bps_index_embedding(bps_ids)[None, :, :]
            bps_low = bps_low + self.geometry_source_embedding(torch.as_tensor(0, dtype=torch.long, device=device))
            bps_low = bps_low + self._token_type_emb(TokenTypeId.OBJECT_GEOMETRY, device)
            lows.append(bps_low)
            masks.append(bps_mask)
            sources.append("bps")
        if self.geometry_source in {"points", "both"}:
            points = _require_tensor(geometry, "object_points_local", "geometry.object_points_local")
            if points.ndim != 3 or points.shape[0] != batch_size or points.shape[-1] != 3 or points.shape[1] == 0:
                raise ValueError(
                    f"geometry.object_points_local must have shape [B,N,3] with N>0, got {tuple(points.shape)}."
                )
            point_mask = geometry.get("object_points_mask")
            if point_mask is None:
                point_mask = torch.ones(batch_size, points.shape[1], dtype=torch.bool, device=device)
            else:
                point_mask = point_mask.to(device=device, dtype=torch.bool)
                _assert_shape_tuple(point_mask, (batch_size, points.shape[1]), "geometry.object_points_mask")
            point_low = self.point_proj(points)
            point_low = point_low + self.geometry_source_embedding(torch.as_tensor(1, dtype=torch.long, device=device))
            point_low = point_low + self._token_type_emb(TokenTypeId.OBJECT_GEOMETRY, device)
            lows.append(point_low)
            masks.append(point_mask)
            sources.append("points")
        if not lows:
            raise KeyError(f"No geometry source selected for geometry_source={self.geometry_source!r}.")
        return torch.cat(lows, dim=1), torch.cat(masks, dim=1), "+".join(sources)

    def _token_type_emb(self, token_type: TokenTypeId, device: torch.device) -> torch.Tensor:
        ids = torch.as_tensor(int(token_type), dtype=torch.long, device=device)
        return self.token_type_embedding(ids)


class RelationTokenizer(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        num_heads: int,
        num_relation_slots: int,
        num_marker_parts: int,
        num_relation_types: int,
        dropout: float = 0.0,
        use_proximity_bias: bool = True,
        use_relation_part_bias: bool = True,
        use_raw_distance_for_bias: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_relation_slots = int(num_relation_slots)
        self.num_marker_parts = int(num_marker_parts)
        self.num_relation_types = int(num_relation_types)
        self.use_proximity_bias = bool(use_proximity_bias)
        self.use_relation_part_bias = bool(use_relation_part_bias)
        self.use_raw_distance_for_bias = bool(use_raw_distance_for_bias)

        self.local_proj = MLPProjector(13, d_model, dropout=dropout)
        self.ground_proj = MLPProjector(13, d_model, dropout=dropout)
        self.global_proj = MLPProjector(10, d_model, dropout=dropout)
        self.time_embedding = TimeEmbedding(max_seq_len, d_model)
        self.token_type_embedding = TokenTypeEmbedding(NUM_TOKEN_TYPES, d_model)
        self.relation_id_embedding = nn.Embedding(78, d_model)
        self.relation_type_embedding = nn.Embedding(num_relation_types, d_model)
        self.relation_part_embedding = nn.Embedding(num_marker_parts, d_model)
        self.relation_slot_queries = nn.Parameter(torch.randn(num_relation_slots, d_model) * 0.02)
        self.relation_slot_output_embedding = nn.Embedding(num_relation_slots, d_model)
        self.relation_pool = CrossAttentionPool(d_model, num_heads, dropout=dropout)
        self.proximity_scale = nn.Parameter(torch.ones(num_heads))
        self.relation_part_bias = nn.Embedding(num_marker_parts, num_relation_slots)
        nn.init.zeros_(self.relation_part_bias.weight)

    def forward(
        self,
        batch: dict,
        return_aux: bool = False,
        return_aux_detached: bool = True,
    ) -> TokenizerOutput:
        relation = _require_group(batch, "model", "relation")
        masks = _require_dict(batch, "masks")
        ids = _require_dict(batch, "ids")

        delta = _require_tensor(relation, "delta_m2o", "model.relation.delta_m2o")
        dist = _require_tensor(relation, "dist", "model.relation.dist")
        delta_vel = _require_tensor(relation, "delta_vel", "model.relation.delta_vel")
        delta_acc = _require_tensor(relation, "delta_acc", "model.relation.delta_acc")
        point_canon = _require_tensor(relation, "point_canon", "model.relation.point_canon")
        root_object_delta = _require_tensor(relation, "root_object_delta", "model.relation.root_object_delta")
        root_object_dist = _require_tensor(relation, "root_object_dist", "model.relation.root_object_dist")
        root_object_delta_vel = _require_tensor(
            relation,
            "root_object_delta_vel",
            "model.relation.root_object_delta_vel",
        )
        root_object_delta_acc = _require_tensor(
            relation,
            "root_object_delta_acc",
            "model.relation.root_object_delta_acc",
        )

        assert_shape(delta, 4, "model.relation.delta_m2o")
        b, t, num_relations, _ = delta.shape
        _check_seq_len(self.__class__.__name__, t, self.time_embedding.max_len)
        if num_relations != 78 or delta.shape[-1] != 3:
            raise ValueError(f"model.relation.delta_m2o must have shape [B,T,78,3], got {tuple(delta.shape)}.")
        _assert_shape_tuple(dist, (b, t, 78, 1), "model.relation.dist")
        _assert_shape_tuple(delta_vel, (b, t, 78, 3), "model.relation.delta_vel")
        _assert_shape_tuple(delta_acc, (b, t, 78, 3), "model.relation.delta_acc")
        _assert_shape_tuple(point_canon, (b, t, 78, 3), "model.relation.point_canon")
        _assert_shape_tuple(root_object_delta, (b, t, 3), "model.relation.root_object_delta")
        _assert_shape_tuple(root_object_dist, (b, t, 1), "model.relation.root_object_dist")
        _assert_shape_tuple(root_object_delta_vel, (b, t, 3), "model.relation.root_object_delta_vel")
        _assert_shape_tuple(root_object_delta_acc, (b, t, 3), "model.relation.root_object_delta_acc")

        device = delta.device
        valid_mask = _valid_mask(masks, b, t, device)
        relation_mask = _relation_mask(masks, b, t, device)
        relation_ids = _static_ids(ids, "relation_ids", torch.arange(78, device=device), device, 78)
        relation_type_ids = _static_ids(
            ids,
            "relation_type_ids",
            torch.cat(
                [
                    torch.zeros(77, dtype=torch.long, device=device),
                    torch.ones(1, dtype=torch.long, device=device),
                ]
            ),
            device,
            78,
        )
        relation_part_ids = _static_ids(
            ids,
            "relation_part_ids",
            torch.zeros(78, dtype=torch.long, device=device),
            device,
            78,
        )
        _check_id_range(relation_ids, 78, "ids.relation_ids")
        _check_id_range(relation_type_ids, self.num_relation_types, "ids.relation_type_ids")
        _check_id_range(relation_part_ids, self.num_marker_parts, "ids.relation_part_ids")

        local_feat = torch.cat([delta, dist, delta_vel, delta_acc, point_canon], dim=-1)
        local_low = self.local_proj(local_feat)
        frame_time_ids = _frame_time_ids(b, t, device)
        time_emb = self.time_embedding(frame_time_ids)
        local_low = local_low + time_emb[:, :, None, :]
        local_low = local_low + self.relation_id_embedding(relation_ids)[None, None, :, :]
        local_low = local_low + self.relation_type_embedding(relation_type_ids)[None, None, :, :]
        local_low = local_low + self.relation_part_embedding(relation_part_ids)[None, None, :, :]
        local_low = local_low + self._token_type_emb(TokenTypeId.RELATION_BODY_SLOT, device)

        human_relation_low = local_low[:, :, :77, :]
        human_relation_mask = relation_mask[:, :, :77]
        p = self.num_relation_slots
        kv = human_relation_low.reshape(b * t, 77, self.d_model)
        key_mask = human_relation_mask.reshape(b * t, 77)
        rel_q = self.relation_slot_queries[None, None, :, :].expand(b, t, p, self.d_model)
        rel_q = rel_q + time_emb[:, :, None, :]
        rel_q = rel_q.reshape(b * t, p, self.d_model)
        query_mask = valid_mask.reshape(b * t, 1).expand(b * t, p)
        attn_bias, used_proximity, used_part_bias = self._relation_attn_bias(batch, relation_part_ids, b, t, device)
        rel_body, rel_attn = self.relation_pool(
            query=rel_q,
            key_value=kv,
            key_mask=key_mask,
            query_mask=query_mask,
            attn_bias=attn_bias,
            need_weights=return_aux,
        )
        rel_body = rel_body.reshape(b, t, p, self.d_model)
        rel_slot_ids = torch.arange(p, dtype=torch.long, device=device)
        rel_body = rel_body + self.relation_slot_output_embedding(rel_slot_ids)[None, None, :, :]
        rel_body = rel_body + self._token_type_emb(TokenTypeId.RELATION_BODY_SLOT, device)
        rel_body = rel_body * valid_mask[:, :, None, None].to(dtype=rel_body.dtype)

        ground_feat = local_feat[:, :, 77, :]
        ground_mask = relation_mask[:, :, 77]
        rel_ground = self.ground_proj(ground_feat)
        rel_ground = rel_ground + time_emb + self._token_type_emb(TokenTypeId.RELATION_GROUND, device)
        rel_ground = rel_ground * ground_mask.to(dtype=rel_ground.dtype).unsqueeze(-1)

        global_feat = torch.cat(
            [root_object_delta, root_object_dist, root_object_delta_vel, root_object_delta_acc],
            dim=-1,
        )
        rel_global = self.global_proj(global_feat)
        rel_global = rel_global + time_emb + self._token_type_emb(TokenTypeId.RELATION_ROOT_OBJECT, device)
        rel_global = rel_global * valid_mask.to(dtype=rel_global.dtype).unsqueeze(-1)

        frame = torch.cat([rel_body, rel_ground[:, :, None, :], rel_global[:, :, None, :]], dim=2)
        slots_per_frame = p + 2
        tokens = frame.reshape(b, t * slots_per_frame, self.d_model)
        per_frame_mask = torch.cat(
            [
                valid_mask[:, :, None].expand(b, t, p),
                ground_mask[:, :, None],
                valid_mask[:, :, None],
            ],
            dim=2,
        )
        mask = per_frame_mask.reshape(b, t * slots_per_frame)
        tokens = zero_invalid_tokens(tokens, mask)
        token_type_ids = _repeat_per_frame(
            torch.tensor(
                [int(TokenTypeId.RELATION_BODY_SLOT)] * p
                + [int(TokenTypeId.RELATION_GROUND), int(TokenTypeId.RELATION_ROOT_OBJECT)],
                dtype=torch.long,
                device=device,
            ),
            b,
            t,
        )
        out = TokenizerOutput(
            tokens=tokens,
            mask=mask,
            modality_ids=torch.full_like(token_type_ids, int(ModalityId.RELATION)),
            token_type_ids=token_type_ids,
            time_ids=frame_time_ids[:, :, None].expand(b, t, slots_per_frame).reshape(b, -1),
            entity_ids=_repeat_per_frame(torch.arange(slots_per_frame, dtype=torch.long, device=device), b, t),
            time_mask=mask.clone(),
            aux={},
        )
        if return_aux:
            out.aux = {
                "local_low": _maybe_detach(local_low, return_aux_detached),
                "relation_pool_attn": _maybe_detach(rel_attn, return_aux_detached),
                "used_proximity_bias": used_proximity,
                "used_relation_part_bias": used_part_bias,
                "ground_relation_index": 77,
            }
        return out

    def _relation_attn_bias(
        self,
        batch: dict,
        relation_part_ids: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], bool, bool]:
        attn_bias = None
        used_proximity = False
        if self.use_proximity_bias:
            if self.use_raw_distance_for_bias:
                if "raw" not in batch or "relation" not in batch["raw"] or "dist" not in batch["raw"]["relation"]:
                    raise KeyError(
                        "RelationTokenizer requires batch['raw']['relation']['dist'] "
                        "when use_raw_distance_for_bias=True."
                    )
                dist_for_bias = batch["raw"]["relation"]["dist"]
            else:
                dist_for_bias = _require_group(batch, "model", "relation")["dist"]
            dist_human = dist_for_bias[:, :, :77, 0].to(device=device)
            log_dist = torch.log1p(dist_human.clamp_min(0.0)).reshape(batch_size * seq_len, 77)
            scale = F.softplus(self.proximity_scale)[None, :, None, None]
            attn_bias = -scale * log_dist[:, None, None, :]
            used_proximity = True

        used_part_bias = False
        human_part_ids = relation_part_ids[:77]
        if self.use_relation_part_bias and human_part_ids.unique().numel() > 1:
            part_bias = self.relation_part_bias(human_part_ids).transpose(0, 1)
            part_bias = part_bias[None, None, :, :].expand(batch_size * seq_len, 1, self.num_relation_slots, 77)
            attn_bias = _add_attn_bias(attn_bias, part_bias.to(device=device))
            used_part_bias = True
        return attn_bias, used_proximity, used_part_bias

    def _token_type_emb(self, token_type: TokenTypeId, device: torch.device) -> torch.Tensor:
        ids = torch.as_tensor(int(token_type), dtype=torch.long, device=device)
        return self.token_type_embedding(ids)


class Stage1TokenizerModule(nn.Module):
    """Official Stage 1 tokenizer boundary.

    The module is intentionally checkpointable as one independent component:
    save it under ``checkpoint[checkpoint_key]`` while keeping submodule state
    dict keys stable as ``human.*``, ``object.*``, and ``relation.*``.
    """

    checkpoint_key = "tokenizer"
    output_keys = ("human", "object", "relation")
    concat_key = "concat"

    def __init__(self, human: HumanTokenizer, object: ObjectTokenizer, relation: RelationTokenizer):
        super().__init__()
        self.human = human
        self.object = object
        self.relation = relation

    def forward(
        self,
        batch: dict,
        return_aux: bool = False,
        return_aux_detached: bool = True,
        return_concat: bool = False,
        validate: bool = False,
    ) -> dict[str, TokenizerOutput]:
        outputs = {
            "human": self.human(batch, return_aux=return_aux, return_aux_detached=return_aux_detached),
            "object": self.object(batch, return_aux=return_aux, return_aux_detached=return_aux_detached),
            "relation": self.relation(batch, return_aux=return_aux, return_aux_detached=return_aux_detached),
        }
        if validate:
            self.validate_outputs(outputs)
        if return_concat:
            outputs[self.concat_key] = self.concat_outputs(outputs)
            if validate:
                validate_tokenizer_output(outputs[self.concat_key], f"{self.__class__.__name__}.{self.concat_key}")
        return outputs

    def concat_outputs(self, outputs: dict[str, TokenizerOutput]) -> TokenizerOutput:
        missing = [key for key in self.output_keys if key not in outputs]
        if missing:
            raise KeyError(f"Cannot concatenate tokenizer outputs; missing keys: {missing}.")
        return concat_tokenizer_outputs(outputs[key] for key in self.output_keys)

    def validate_outputs(self, outputs: dict[str, TokenizerOutput]) -> None:
        for key in self.output_keys:
            if key not in outputs:
                raise KeyError(f"Tokenizer output is missing key {key!r}.")
            validate_tokenizer_output(outputs[key], f"{self.__class__.__name__}.{key}")


class Stage1HOITokenizer(Stage1TokenizerModule):
    """Backward-compatible name for the formal Stage1TokenizerModule."""

    pass


def build_stage1_tokenizer_from_config(config: dict, batch: Optional[dict] = None) -> Stage1TokenizerModule:
    stage1_cfg = config.get("model", {}).get("stage1", {})
    d_model = int(stage1_cfg.get("d_model", 256))
    num_heads = int(stage1_cfg.get("num_heads", 4))
    max_seq_len = int(stage1_cfg.get("max_seq_len", 128))
    dropout = float(stage1_cfg.get("dropout", 0.0))
    tok_cfg = stage1_cfg.get("tokenizers", {})
    num_marker_parts = int(stage1_cfg.get("num_marker_parts", _infer_num_marker_parts(batch)))
    num_relation_types = int(stage1_cfg.get("num_relation_types", _infer_num_relation_types(batch)))
    num_objects = stage1_cfg.get("num_objects")
    num_objects = None if num_objects is None else int(num_objects)

    human_cfg = tok_cfg.get("human", {})
    object_cfg = tok_cfg.get("object", {})
    relation_cfg = tok_cfg.get("relation", {})
    data_window_size = config.get("data", {}).get("window_size")
    if data_window_size is not None:
        validate_stage1_tokenizer_config(int(data_window_size), max_seq_len)
    return Stage1TokenizerModule(
        human=HumanTokenizer(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            num_body_slots=int(human_cfg.get("num_body_slots", 8)),
            num_marker_parts=num_marker_parts,
            dropout=float(human_cfg.get("dropout", dropout)),
            use_marker_part_bias=bool(human_cfg.get("use_marker_part_bias", True)),
            include_official_vel=bool(human_cfg.get("include_official_vel", False)),
        ),
        object=ObjectTokenizer(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            num_geometry_slots=int(object_cfg.get("num_geometry_slots", 4)),
            geometry_source=object_cfg.get("geometry_source", "bps"),
            dropout=float(object_cfg.get("dropout", dropout)),
            include_official_vel9=bool(object_cfg.get("include_official_vel9", True)),
            use_object_id_embedding=bool(object_cfg.get("use_object_id_embedding", False)),
            num_objects=num_objects,
            max_bps_points=int(object_cfg.get("max_bps_points", 1024)),
        ),
        relation=RelationTokenizer(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            num_relation_slots=int(relation_cfg.get("num_relation_slots", 8)),
            num_marker_parts=num_marker_parts,
            num_relation_types=num_relation_types,
            dropout=float(relation_cfg.get("dropout", dropout)),
            use_proximity_bias=bool(relation_cfg.get("use_proximity_bias", True)),
            use_relation_part_bias=bool(relation_cfg.get("use_relation_part_bias", True)),
            use_raw_distance_for_bias=bool(relation_cfg.get("use_raw_distance_for_bias", True)),
        ),
    )


def validate_stage1_tokenizer_config(data_window_size: int, max_seq_len: int) -> None:
    if max_seq_len < data_window_size:
        raise ValueError(
            f"max_seq_len={max_seq_len} must be >= data.window_size={data_window_size}."
        )


def _require_dict(mapping: dict, key: str) -> dict:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"batch[{key!r}] is required and must be a dict.")
    return value


def _require_group(batch: dict, top_key: str, group_key: str) -> dict:
    top = _require_dict(batch, top_key)
    group = top.get(group_key)
    if not isinstance(group, dict):
        raise KeyError(f"batch[{top_key!r}][{group_key!r}] is required and must be a dict.")
    return group


def _require_tensor(mapping: dict, key: str, name: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"{name} is required and must be a torch.Tensor.")
    return value


def _assert_same_shape(x: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    if tuple(x.shape) != tuple(ref.shape):
        raise ValueError(f"{name} must have shape {tuple(ref.shape)}, got {tuple(x.shape)}.")


def _assert_shape_tuple(x: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(x.shape)}.")


def _valid_mask(masks: dict, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    valid = masks.get("valid")
    if valid is None:
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    valid = valid.to(device=device, dtype=torch.bool)
    _assert_shape_tuple(valid, (batch_size, seq_len), "masks.valid")
    return valid


def _human_mask(masks: dict, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = masks.get("human")
    if mask is None:
        return _valid_mask(masks, batch_size, seq_len, device)[:, :, None].expand(batch_size, seq_len, 77)
    mask = mask.to(device=device, dtype=torch.bool)
    _assert_shape_tuple(mask, (batch_size, seq_len, 77), "masks.human")
    return mask


def _object_mask(masks: dict, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = masks.get("object")
    if mask is None:
        return _valid_mask(masks, batch_size, seq_len, device)
    mask = mask.to(device=device, dtype=torch.bool)
    _assert_shape_tuple(mask, (batch_size, seq_len), "masks.object")
    return mask


def _relation_mask(masks: dict, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = masks.get("relation")
    if mask is None:
        return _valid_mask(masks, batch_size, seq_len, device)[:, :, None].expand(batch_size, seq_len, 78)
    mask = mask.to(device=device, dtype=torch.bool)
    _assert_shape_tuple(mask, (batch_size, seq_len, 78), "masks.relation")
    return mask


def _static_ids(
    ids: dict,
    key: str,
    default: torch.Tensor,
    device: torch.device,
    expected_len: int,
) -> torch.Tensor:
    value = ids.get(key)
    if value is None:
        value = default
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.to(device=device, dtype=torch.long)
    if value.ndim == 2:
        value = value[0]
    _assert_shape_tuple(value, (expected_len,), f"ids.{key}")
    return value


def _batch_ids(ids: dict, keys: list[str], batch_size: int, device: torch.device) -> torch.Tensor:
    value = None
    used_key = None
    for key in keys:
        if key in ids:
            value = ids[key]
            used_key = key
            break
    if value is None:
        raise KeyError(f"One of {keys} is required for object id embedding.")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.to(device=device, dtype=torch.long)
    _assert_shape_tuple(value, (batch_size,), f"ids.{used_key}")
    return value


def _check_id_range(ids: torch.Tensor, num_embeddings: int, name: str) -> None:
    if ids.numel() == 0:
        return
    min_id = int(ids.min().item())
    max_id = int(ids.max().item())
    if min_id < 0 or max_id >= num_embeddings:
        raise ValueError(f"{name} ids must be in [0, {num_embeddings - 1}], got min={min_id}, max={max_id}.")


def _check_seq_len(module_name: str, seq_len: int, max_seq_len: int) -> None:
    if seq_len > max_seq_len:
        raise ValueError(
            f"{module_name} received T={seq_len}, but max_seq_len={max_seq_len}. "
            "Use fixed-window data with T <= max_seq_len, or increase max_seq_len."
        )


def _frame_time_ids(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, dtype=torch.long, device=device)[None, :].expand(batch_size, seq_len)


def _repeat_per_frame(values: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    return values[None, None, :].expand(batch_size, seq_len, values.shape[0]).reshape(batch_size, seq_len * values.shape[0])


def _maybe_detach(value: Any, detach: bool) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach() if detach else value
    return value


def _add_attn_bias(a: Optional[torch.Tensor], b: torch.Tensor) -> torch.Tensor:
    return b if a is None else a + b


def _infer_num_marker_parts(batch: Optional[dict]) -> int:
    value = 100
    if batch and isinstance(batch.get("ids"), dict):
        for key in ("marker_part_ids", "relation_part_ids"):
            ids = batch["ids"].get(key)
            if isinstance(ids, torch.Tensor) and ids.numel() > 0:
                value = max(value, int(ids.max().item()) + 1)
    return value


def _infer_num_relation_types(batch: Optional[dict]) -> int:
    value = 2
    if batch and isinstance(batch.get("ids"), dict):
        ids = batch["ids"].get("relation_type_ids")
        if isinstance(ids, torch.Tensor) and ids.numel() > 0:
            value = max(value, int(ids.max().item()) + 1)
    return value
