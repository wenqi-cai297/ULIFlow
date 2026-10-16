from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from hoi_latent.models.stage1.encoder.hoi_packing import (
    HOI_SUBSTREAM_NAMES,
    HOISubstreamId,
    derive_hoi_substream_ids,
)
from hoi_latent.models.stage1.encoder.transformer_blocks import FeedForward, MultiHeadAttention
from hoi_latent.models.stage1.mask_utils import safe_masked_softmax, zero_invalid_tokens
from hoi_latent.models.stage1.time_bias import RelativeTimeBias
from hoi_latent.models.stage1.tokenizer_types import (
    NUM_TOKEN_TYPES,
    ModalityId,
    TokenTypeId,
    TokenizerOutput,
    validate_tokenizer_output,
)


class TopologyAwareDenseQueryBuilder(nn.Module):
    """Build dense dynamic H/O/R queries with metadata-aligned topology."""

    def __init__(
        self,
        d_model: int = 256,
        target_num_frames: int = 128,
        human_slots: int = 11,
        relation_slots: int = 10,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if target_num_frames <= 0:
            raise ValueError(f"target_num_frames must be positive, got {target_num_frames}.")
        if human_slots < 3:
            raise ValueError("human_slots must be at least 3: root, body slots, left/right feet.")
        if relation_slots < 2:
            raise ValueError("relation_slots must be at least 2: body slots plus ground/root-object relations.")
        self.d_model = int(d_model)
        self.target_num_frames = int(target_num_frames)
        self.human_slots = int(human_slots)
        self.relation_slots = int(relation_slots)
        self.max_slots = max(self.human_slots, self.relation_slots, 1)

        self.base_query = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.time_embedding = nn.Embedding(self.target_num_frames, d_model)
        self.slot_embedding = nn.Embedding(self.max_slots, d_model)
        self.modality_embedding = nn.Embedding(len(ModalityId), d_model)
        self.token_type_embedding = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        for embedding in [
            self.base_query,
            self.time_embedding,
            self.slot_embedding,
            self.modality_embedding,
            self.token_type_embedding,
        ]:
            nn.init.normal_(embedding.weight, mean=0.0, std=float(init_std))

    def forward(
        self,
        decoder_joint: TokenizerOutput,
        target_num_frames: int,
        target_frame_mask: torch.Tensor,
    ) -> TokenizerOutput:
        if target_num_frames > self.target_num_frames:
            raise ValueError(
                f"target_num_frames={target_num_frames} exceeds configured maximum {self.target_num_frames}."
            )
        b, _, d = decoder_joint.tokens.shape
        if d != self.d_model:
            raise ValueError(f"QueryBuilder d_model={self.d_model} but decoder_joint dim is {d}.")
        if target_frame_mask.shape != (b, target_num_frames):
            raise ValueError(f"target_frame_mask must have shape {(b, target_num_frames)}.")
        device = decoder_joint.tokens.device
        target_frame_mask = target_frame_mask.to(device=device, dtype=torch.bool)

        human_types = _human_slot_token_types(self.human_slots, device)
        relation_types = _relation_slot_token_types(self.relation_slots, device)
        human_entities = torch.arange(self.human_slots, dtype=torch.long, device=device)
        relation_entities = torch.arange(self.relation_slots, dtype=torch.long, device=device)
        times = torch.arange(target_num_frames, dtype=torch.long, device=device)

        human = self._make_group(
            batch_size=b,
            time_ids=times,
            slot_ids=human_entities,
            token_type_ids=human_types,
            modality_id=int(ModalityId.HUMAN),
            valid_mask=target_frame_mask,
        )
        object_pose = self._make_group(
            batch_size=b,
            time_ids=times,
            slot_ids=torch.zeros(1, dtype=torch.long, device=device),
            token_type_ids=torch.full((1,), int(TokenTypeId.OBJECT_POSE), dtype=torch.long, device=device),
            modality_id=int(ModalityId.OBJECT),
            valid_mask=target_frame_mask,
        )
        relation = self._make_group(
            batch_size=b,
            time_ids=times,
            slot_ids=relation_entities,
            token_type_ids=relation_types,
            modality_id=int(ModalityId.RELATION),
            valid_mask=target_frame_mask,
        )

        out = _dynamic_frame_major_to_substream_major(
            tokens_ts=torch.cat([human["tokens"], object_pose["tokens"], relation["tokens"]], dim=2),
            mask_ts=torch.cat([human["mask"], object_pose["mask"], relation["mask"]], dim=2),
            modality_ids_ts=torch.cat([human["modality_ids"], object_pose["modality_ids"], relation["modality_ids"]], dim=2),
            token_type_ids_ts=torch.cat(
                [human["token_type_ids"], object_pose["token_type_ids"], relation["token_type_ids"]], dim=2
            ),
            time_ids_ts=torch.cat([human["time_ids"], object_pose["time_ids"], relation["time_ids"]], dim=2),
            entity_ids_ts=torch.cat([human["entity_ids"], object_pose["entity_ids"], relation["entity_ids"]], dim=2),
            time_mask_ts=torch.cat([human["time_mask"], object_pose["time_mask"], relation["time_mask"]], dim=2),
            human_slots=self.human_slots,
            relation_slots=self.relation_slots,
            aux={
                "query_builder": {
                    "mode": "topology_aware_dense_queries",
                    "target_num_frames": int(target_num_frames),
                    "human_slots": self.human_slots,
                    "relation_slots": self.relation_slots,
                }
            },
        )
        return out

    def _make_group(
        self,
        batch_size: int,
        time_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        modality_id: int,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t = time_ids.numel()
        s = slot_ids.numel()
        time_grid = time_ids[:, None].expand(t, s)
        slot_grid = slot_ids[None, :].expand(t, s)
        type_grid = token_type_ids[None, :].expand(t, s)
        modality_grid = torch.full((t, s), int(modality_id), dtype=torch.long, device=time_ids.device)
        tokens = (
            self.base_query(type_grid)
            + self.time_embedding(time_grid)
            + self.slot_embedding(slot_grid)
            + self.modality_embedding(modality_grid)
            + self.token_type_embedding(type_grid)
        )
        tokens = tokens[None, :, :, :].expand(batch_size, t, s, self.d_model).clone()
        mask = valid_mask[:, :, None].expand(batch_size, t, s)
        return {
            "tokens": tokens * mask.to(dtype=tokens.dtype).unsqueeze(-1),
            "mask": mask,
            "modality_ids": modality_grid[None, :, :].expand(batch_size, t, s),
            "token_type_ids": type_grid[None, :, :].expand(batch_size, t, s),
            "time_ids": time_grid[None, :, :].expand(batch_size, t, s),
            "entity_ids": slot_grid[None, :, :].expand(batch_size, t, s),
            "time_mask": mask.clone(),
        }


class MemoryCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
        relative_time_alpha_init: float = 0.1,
        substream_bias: bool = True,
    ) -> None:
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
        self.relative_time_bias = RelativeTimeBias(alpha_init=relative_time_alpha_init)
        self.substream_bias = nn.Parameter(torch.zeros(len(HOI_SUBSTREAM_NAMES), len(HOI_SUBSTREAM_NAMES))) if substream_bias else None

    def forward(
        self,
        dense: TokenizerOutput,
        memory: TokenizerOutput,
        return_diagnostics: bool = False,
    ) -> tuple[TokenizerOutput, dict]:
        b, q_len, d = dense.tokens.shape
        if d != self.d_model:
            raise ValueError(f"MemoryCrossAttention d_model={self.d_model} but dense dim is {d}.")
        if memory.tokens.shape[0] != b or memory.tokens.shape[-1] != self.d_model:
            raise ValueError("MemoryCrossAttention memory batch/d_model mismatch.")
        k_len = memory.tokens.shape[1]
        dense_mask = dense.mask.to(dtype=torch.bool)
        memory_mask = memory.mask.to(device=dense.tokens.device, dtype=torch.bool)
        dense_substream = derive_hoi_substream_ids(dense.modality_ids, dense.token_type_ids)
        memory_substream = derive_hoi_substream_ids(memory.modality_ids, memory.token_type_ids)

        q = self._split_heads(self.q_proj(self.query_norm(dense.tokens)))
        key = self._split_heads(self.k_proj(self.memory_norm(memory.tokens)))
        value = self._split_heads(self.v_proj(self.memory_norm(memory.tokens)))
        scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(float(self.head_dim))

        memory_time_mask = memory.time_mask if memory.time_mask is not None else memory.mask
        geometry_key = memory.token_type_ids == int(TokenTypeId.OBJECT_GEOMETRY)
        rel_bias = self.relative_time_bias(
            dense.time_ids,
            memory.time_ids,
            dtype=scores.dtype,
            key_time_mask=memory_time_mask,
            zero_bias_key_mask=geometry_key,
        )
        scores = scores + rel_bias

        if self.substream_bias is not None:
            bias = self.substream_bias[dense_substream[:, :, None], memory_substream[:, None, :]]
            scores = scores + bias[:, None, :, :]

        weights = safe_masked_softmax(scores, memory_mask[:, None, None, :], dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, value)
        out = self._merge_heads(out)
        valid_output = dense_mask & memory_mask.any(dim=1, keepdim=True)
        out = self.out_proj(out) * valid_output.to(dtype=out.dtype).unsqueeze(-1)
        tokens = zero_invalid_tokens(dense.tokens + out, dense_mask)
        aux = dict(dense.aux)
        result = _replace_tokens(dense, tokens, aux=aux)
        diagnostics = {}
        if return_diagnostics:
            diagnostics = _cross_attention_diagnostics(weights, dense_mask, memory_mask, dense_substream, memory_substream)
            diagnostics["relative_time_alpha"] = torch.nn.functional.softplus(
                self.relative_time_bias.relative_time_alpha
            ).detach()
            if self.substream_bias is not None:
                diagnostics["substream_bias_matrix"] = self.substream_bias.detach()
        return result, diagnostics

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(b, n, self.d_model)


class AnchorKeepResidual(nn.Module):
    def __init__(self, d_model: int = 256, gate_init: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError(f"gate_init must be in (0, 1), got {gate_init}.")
        self.anchor_proj = nn.Linear(d_model, d_model)
        self.gate_mlp = nn.Linear(2 * d_model, d_model)
        nn.init.eye_(self.anchor_proj.weight)
        nn.init.zeros_(self.anchor_proj.bias)
        nn.init.zeros_(self.gate_mlp.weight)
        nn.init.constant_(self.gate_mlp.bias, _logit(gate_init))

    def forward(
        self,
        dense: TokenizerOutput,
        memory: TokenizerOutput,
        return_diagnostics: bool = False,
    ) -> tuple[TokenizerOutput, dict]:
        match, indices = _same_semantic_match(dense, memory)
        gather_index = indices[:, :, None].expand(indices.shape[0], indices.shape[1], memory.tokens.shape[-1])
        anchor_token = torch.gather(memory.tokens, dim=1, index=gather_index)
        anchor_token = anchor_token * match.to(dtype=anchor_token.dtype).unsqueeze(-1)
        gate = torch.sigmoid(self.gate_mlp(torch.cat([dense.tokens, anchor_token], dim=-1)))
        delta = self.anchor_proj(anchor_token)
        tokens = dense.tokens + match.to(dtype=dense.tokens.dtype).unsqueeze(-1) * gate * delta
        tokens = zero_invalid_tokens(tokens, dense.mask)
        diagnostics = {}
        if return_diagnostics:
            valid = dense.mask.to(dtype=torch.bool)
            denom = valid.sum().clamp_min(1).to(dtype=tokens.dtype)
            diagnostics = {
                "anchor_match_ratio": (match & valid).sum().to(dtype=tokens.dtype) / denom,
                "anchor_keep_gate_mean": (gate * match.unsqueeze(-1).to(dtype=gate.dtype)).sum()
                / match.sum().mul(gate.shape[-1]).clamp_min(1).to(dtype=gate.dtype),
                "anchor_delta_norm": (delta * match.unsqueeze(-1).to(dtype=delta.dtype)).norm().detach(),
            }
        return _replace_tokens(dense, tokens), diagnostics


class PerFrameSlotCoordinationBlock(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, use_rope=False)

    def forward(
        self,
        dense: TokenizerOutput,
        target_num_frames: int,
        human_slots: int,
        relation_slots: int,
        return_diagnostics: bool = False,
    ) -> tuple[TokenizerOutput, dict]:
        frame = dense_substream_major_to_frame_major(dense, target_num_frames, human_slots, relation_slots)
        b, t, s, d = frame["tokens"].shape
        x = frame["tokens"].reshape(b * t, s, d)
        mask = frame["mask"].reshape(b * t, s)
        h = self.attn(self.norm(x), key_mask=mask, query_mask=mask)
        x = zero_invalid_tokens(x + h, mask)
        frame["tokens"] = x.reshape(b, t, s, d)
        out = dense_frame_major_to_substream_major(frame, human_slots, relation_slots, aux=dense.aux)
        diagnostics = {}
        if return_diagnostics:
            diagnostics = {"per_frame_valid_ratio": mask.to(dtype=torch.float32).mean().detach()}
        return out, diagnostics


class SlotWiseLocalTemporalBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        radius: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.radius = int(radius)
        self.attn_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, ffn_mult=ffn_mult, dropout=dropout)

    def forward(
        self,
        dense: TokenizerOutput,
        target_num_frames: int,
        human_slots: int,
        relation_slots: int,
        return_diagnostics: bool = False,
    ) -> tuple[TokenizerOutput, dict]:
        frame = dense_substream_major_to_frame_major(dense, target_num_frames, human_slots, relation_slots)
        b, t, s, d = frame["tokens"].shape
        x = frame["tokens"].permute(0, 2, 1, 3).reshape(b * s, t, d)
        mask = frame["mask"].permute(0, 2, 1).reshape(b * s, t)
        q = self._split_heads(self.q_proj(self.attn_norm(x)))
        key = self._split_heads(self.k_proj(self.attn_norm(x)))
        value = self._split_heads(self.v_proj(self.attn_norm(x)))
        scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        local = _local_temporal_mask(t, self.radius, x.device)
        pair_mask = mask[:, None, :] & local[None, :, :]
        weights = safe_masked_softmax(scores, pair_mask[:, None, :, :], dim=-1)
        weights = self.dropout(weights)
        attn_out = torch.matmul(weights, value)
        attn_out = self.out_proj(self._merge_heads(attn_out))
        valid_output = mask & pair_mask.any(dim=-1)
        x = zero_invalid_tokens(x + attn_out * valid_output.unsqueeze(-1).to(dtype=attn_out.dtype), mask)
        x = zero_invalid_tokens(x + self.ffn(self.ffn_norm(x)), mask)
        frame["tokens"] = x.reshape(b, s, t, d).permute(0, 2, 1, 3)
        out = dense_frame_major_to_substream_major(frame, human_slots, relation_slots, aux=dense.aux)
        diagnostics = {}
        if return_diagnostics:
            outside = ~local
            outside_valid = (
                outside[None, None, :, :]
                & mask[:, None, :, None]
                & mask[:, None, None, :]
            )
            outside_mass = weights.masked_select(outside_valid).max() if outside_valid.any() else weights.new_tensor(0.0)
            eps = torch.finfo(weights.dtype).eps
            entropy = -(weights * (weights + eps).log()).sum(dim=-1)
            diagnostics = {
                "local_temporal_attention_entropy": _masked_mean(entropy, mask[:, None, :].expand_as(entropy)).detach(),
                "local_temporal_attention_outside_radius_max_mass": outside_mass.detach(),
            }
        return out, diagnostics

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(b, n, self.d_model)


class DensifierBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        ffn_mult: int = 4,
        local_radius: int = 4,
        dropout: float = 0.0,
        relative_time_alpha_init: float = 0.1,
        substream_bias: bool = True,
        anchor_gate_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.memory_cross_attention = MemoryCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            relative_time_alpha_init=relative_time_alpha_init,
            substream_bias=substream_bias,
        )
        self.anchor_keep = AnchorKeepResidual(d_model=d_model, gate_init=anchor_gate_init)
        self.per_frame_slot_attention = PerFrameSlotCoordinationBlock(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.local_temporal_attention = SlotWiseLocalTemporalBlock(
            d_model=d_model,
            num_heads=num_heads,
            radius=local_radius,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

    def forward(
        self,
        dense: TokenizerOutput,
        memory: TokenizerOutput,
        target_num_frames: int,
        human_slots: int,
        relation_slots: int,
        return_diagnostics: bool = False,
    ) -> tuple[TokenizerOutput, dict]:
        diagnostics: dict[str, Any] = {}
        dense, diag = self.memory_cross_attention(dense, memory, return_diagnostics=return_diagnostics)
        if return_diagnostics:
            diagnostics["memory_cross_attention"] = diag
        dense, diag = self.anchor_keep(dense, memory, return_diagnostics=return_diagnostics)
        if return_diagnostics:
            diagnostics["anchor_keep"] = diag
        dense, diag = self.per_frame_slot_attention(
            dense,
            target_num_frames=target_num_frames,
            human_slots=human_slots,
            relation_slots=relation_slots,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            diagnostics["per_frame_slot_attention"] = diag
        dense, diag = self.local_temporal_attention(
            dense,
            target_num_frames=target_num_frames,
            human_slots=human_slots,
            relation_slots=relation_slots,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            diagnostics["slot_wise_local_temporal_attention"] = diag
        return dense, diagnostics


class Stage1TemporalDensifier(nn.Module):
    output_key = "dense_decoder_tokens"

    def __init__(
        self,
        d_model: int = 256,
        target_num_frames: int = 128,
        human_slots: int = 11,
        relation_slots: int = 10,
        depth: int = 4,
        num_heads: int = 4,
        ffn_mult: int = 4,
        local_radius: int = 4,
        dropout: float = 0.0,
        relative_time_alpha_init: float = 0.1,
        substream_bias: bool = True,
        anchor_gate_init: float = 0.5,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}.")
        self.d_model = int(d_model)
        self.target_num_frames = int(target_num_frames)
        self.human_slots = int(human_slots)
        self.relation_slots = int(relation_slots)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.ffn_mult = int(ffn_mult)
        self.local_radius = int(local_radius)
        self.query_builder = TopologyAwareDenseQueryBuilder(
            d_model=d_model,
            target_num_frames=target_num_frames,
            human_slots=human_slots,
            relation_slots=relation_slots,
        )
        self.blocks = nn.ModuleList(
            [
                DensifierBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    local_radius=local_radius,
                    dropout=dropout,
                    relative_time_alpha_init=relative_time_alpha_init,
                    substream_bias=substream_bias,
                    anchor_gate_init=anchor_gate_init,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        decoder_joint: TokenizerOutput,
        target_num_frames: int | None = None,
        target_frame_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        validate: bool = False,
    ) -> TokenizerOutput:
        if validate:
            validate_tokenizer_output(decoder_joint, "temporal_densifier.input")
        b, _, d = decoder_joint.tokens.shape
        if d != self.d_model:
            raise ValueError(f"Stage1TemporalDensifier d_model={self.d_model} but input dim is {d}.")
        t = self.target_num_frames if target_num_frames is None else int(target_num_frames)
        if t <= 0:
            raise ValueError(f"target_num_frames must be positive, got {t}.")
        if target_frame_mask is None:
            target_frame_mask = torch.ones(b, t, dtype=torch.bool, device=decoder_joint.tokens.device)
        else:
            target_frame_mask = target_frame_mask.to(device=decoder_joint.tokens.device, dtype=torch.bool)
        if target_frame_mask.shape != (b, t):
            raise ValueError(f"target_frame_mask must have shape {(b, t)}, got {tuple(target_frame_mask.shape)}.")

        dense = self.query_builder(decoder_joint, t, target_frame_mask)
        geometry = _extract_geometry_memory(decoder_joint)
        layer_diagnostics = []
        for block in self.blocks:
            dense, diag = block(
                dense,
                decoder_joint,
                target_num_frames=t,
                human_slots=self.human_slots,
                relation_slots=self.relation_slots,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                layer_diagnostics.append(diag)

        out = _attach_geometry(dense, geometry)
        aux = {
            **decoder_joint.aux,
            **dense.aux,
            "temporal_densifier": (
                self._diagnostics(out, geometry, target_frame_mask, layer_diagnostics)
                if return_diagnostics
                else self._minimal_diagnostics(t)
            ),
        }
        out = TokenizerOutput(
            tokens=out.tokens,
            mask=out.mask,
            modality_ids=out.modality_ids,
            token_type_ids=out.token_type_ids,
            time_ids=out.time_ids,
            entity_ids=out.entity_ids,
            time_mask=out.time_mask,
            aux=aux,
        )
        if validate:
            validate_dense_canonical_layout(out, t, self.human_slots, self.relation_slots)
            validate_tokenizer_output(out, "temporal_densifier")
        return out

    def _minimal_diagnostics(self, target_num_frames: int) -> dict:
        return {
            "target_num_frames": int(target_num_frames),
            "depth": self.depth,
            "human_slots": self.human_slots,
            "relation_slots": self.relation_slots,
            "diagnostics_enabled": False,
        }

    def _diagnostics(
        self,
        out: TokenizerOutput,
        geometry: TokenizerOutput,
        target_frame_mask: torch.Tensor,
        layer_diagnostics: list[dict],
    ) -> dict:
        invalid = ~out.mask
        invalid_max = out.tokens[invalid].abs().max() if invalid.any() else out.tokens.new_tensor(0.0)
        memory_diags = [item.get("memory_cross_attention", {}) for item in layer_diagnostics]
        anchor_diags = [item.get("anchor_keep", {}) for item in layer_diagnostics]
        temporal_diags = [item.get("slot_wise_local_temporal_attention", {}) for item in layer_diagnostics]
        return {
            "target_num_frames": int(target_frame_mask.shape[1]),
            "depth": self.depth,
            "human_slots": self.human_slots,
            "relation_slots": self.relation_slots,
            "diagnostics_enabled": True,
            "target_frame_mask_valid_ratio": target_frame_mask.to(dtype=torch.float32).mean().detach(),
            "dense_num_tokens": int(out.tokens.shape[1]),
            "geometry_num_tokens": int(geometry.tokens.shape[1]),
            "invalid_dense_token_max_abs": invalid_max.detach(),
            "layout_metadata_alignment_check": True,
            "memory_cross_attention": _stack_diag(memory_diags),
            "anchor_keep": _stack_diag(anchor_diags),
            "slot_wise_local_temporal_attention": _stack_diag(temporal_diags),
        }


def build_stage1_temporal_densifier_from_config(config: dict) -> Stage1TemporalDensifier:
    stage1_cfg = config.get("model", {}).get("stage1", {})
    tokenizer_cfg = stage1_cfg.get("tokenizers", {})
    decoder_cfg = stage1_cfg.get("decoder", {})
    densifier_cfg = decoder_cfg.get("temporal_densifier", {})
    query_cfg = densifier_cfg.get("queries", {})
    memory_cfg = densifier_cfg.get("memory_cross_attention", {})
    anchor_cfg = densifier_cfg.get("anchor_keep", {})
    frame_cfg = densifier_cfg.get("per_frame_slot_attention", {})
    temporal_cfg = densifier_cfg.get("local_temporal_attention", {})
    _validate_temporal_densifier_config(query_cfg, memory_cfg, anchor_cfg, frame_cfg, temporal_cfg)
    human_slots = int(query_cfg.get("human_slots", 1 + int(tokenizer_cfg.get("human", {}).get("num_body_slots", 8)) + 2))
    relation_slots = int(query_cfg.get("relation_slots", int(tokenizer_cfg.get("relation", {}).get("num_relation_slots", 8)) + 2))
    return Stage1TemporalDensifier(
        d_model=int(densifier_cfg.get("d_model", stage1_cfg.get("d_model", 256))),
        target_num_frames=int(densifier_cfg.get("target_num_frames", 128)),
        human_slots=human_slots,
        relation_slots=relation_slots,
        depth=int(densifier_cfg.get("depth", 4)),
        num_heads=int(densifier_cfg.get("num_heads", stage1_cfg.get("num_heads", 4))),
        ffn_mult=int(densifier_cfg.get("ffn_mult", 4)),
        local_radius=int(temporal_cfg.get("radius", 4)),
        dropout=float(densifier_cfg.get("dropout", stage1_cfg.get("dropout", 0.0))),
        relative_time_alpha_init=float(memory_cfg.get("relative_time_alpha_init", 0.1)),
        substream_bias=bool(memory_cfg.get("substream_bias", True)),
        anchor_gate_init=float(anchor_cfg.get("gate_init", 0.5)),
    )


def validate_dense_canonical_layout(
    out: TokenizerOutput,
    target_num_frames: int,
    human_slots: int,
    relation_slots: int,
) -> None:
    expected_dynamic = target_num_frames * human_slots + target_num_frames + target_num_frames * relation_slots
    if out.tokens.shape[1] < expected_dynamic:
        raise ValueError("Dense output has fewer tokens than expected dynamic topology.")
    human = slice(0, target_num_frames * human_slots)
    obj = slice(human.stop, human.stop + target_num_frames)
    relation = slice(out.tokens.shape[1] - target_num_frames * relation_slots, out.tokens.shape[1])
    geometry = slice(obj.stop, relation.start)
    if not (out.modality_ids[:, human] == int(ModalityId.HUMAN)).all():
        raise ValueError("Dense human canonical range has incorrect modality ids.")
    expected_human_types = _human_slot_token_types(human_slots, out.tokens.device)
    if not torch.equal(
        out.token_type_ids[:, human].reshape(out.tokens.shape[0], target_num_frames, human_slots),
        expected_human_types[None, None, :].expand(out.tokens.shape[0], target_num_frames, human_slots),
    ):
        raise ValueError("Dense human canonical range has incorrect token_type ids.")
    if not (out.token_type_ids[:, obj] == int(TokenTypeId.OBJECT_POSE)).all():
        raise ValueError("Dense object pose canonical range has incorrect token_type ids.")
    if not (out.modality_ids[:, relation] == int(ModalityId.RELATION)).all():
        raise ValueError("Dense relation canonical range has incorrect modality ids.")
    expected_relation_types = _relation_slot_token_types(relation_slots, out.tokens.device)
    if not torch.equal(
        out.token_type_ids[:, relation].reshape(out.tokens.shape[0], target_num_frames, relation_slots),
        expected_relation_types[None, None, :].expand(out.tokens.shape[0], target_num_frames, relation_slots),
    ):
        raise ValueError("Dense relation canonical range has incorrect token_type ids.")
    if geometry.stop > geometry.start:
        if not (out.token_type_ids[:, geometry] == int(TokenTypeId.OBJECT_GEOMETRY)).all():
            raise ValueError("Dense geometry canonical range has incorrect token_type ids.")
        if out.time_mask is not None and out.time_mask[:, geometry].any():
            raise ValueError("Dense geometry tokens must have time_mask=False.")


def select_dense_tokens_by_metadata(
    out: TokenizerOutput,
    modality_id: int | None = None,
    token_type_id: int | None = None,
    entity_id: int | None = None,
) -> torch.Tensor:
    selection = out.mask.clone().to(dtype=torch.bool)
    if modality_id is not None:
        selection = selection & (out.modality_ids == int(modality_id))
    if token_type_id is not None:
        selection = selection & (out.token_type_ids == int(token_type_id))
    if entity_id is not None:
        selection = selection & (out.entity_ids == int(entity_id))
    return selection


def dense_substream_major_to_frame_major(
    out: TokenizerOutput,
    target_num_frames: int,
    human_slots: int,
    relation_slots: int,
) -> dict[str, torch.Tensor]:
    dynamic = _dynamic_slices(target_num_frames, human_slots, relation_slots, total_tokens=out.tokens.shape[1])
    return {
        "tokens": _field_to_frame_major(out.tokens, dynamic, human_slots, relation_slots),
        "mask": _field_to_frame_major(out.mask, dynamic, human_slots, relation_slots),
        "modality_ids": _field_to_frame_major(out.modality_ids, dynamic, human_slots, relation_slots),
        "token_type_ids": _field_to_frame_major(out.token_type_ids, dynamic, human_slots, relation_slots),
        "time_ids": _field_to_frame_major(out.time_ids, dynamic, human_slots, relation_slots),
        "entity_ids": _field_to_frame_major(out.entity_ids, dynamic, human_slots, relation_slots),
        "time_mask": _field_to_frame_major(out.time_mask, dynamic, human_slots, relation_slots),
    }


def dense_frame_major_to_substream_major(
    frame: dict[str, torch.Tensor],
    human_slots: int,
    relation_slots: int,
    aux: dict | None = None,
) -> TokenizerOutput:
    return _dynamic_frame_major_to_substream_major(
        tokens_ts=frame["tokens"],
        mask_ts=frame["mask"],
        modality_ids_ts=frame["modality_ids"],
        token_type_ids_ts=frame["token_type_ids"],
        time_ids_ts=frame["time_ids"],
        entity_ids_ts=frame["entity_ids"],
        time_mask_ts=frame["time_mask"],
        human_slots=human_slots,
        relation_slots=relation_slots,
        aux={} if aux is None else aux,
    )


def _dynamic_frame_major_to_substream_major(
    tokens_ts: torch.Tensor,
    mask_ts: torch.Tensor,
    modality_ids_ts: torch.Tensor,
    token_type_ids_ts: torch.Tensor,
    time_ids_ts: torch.Tensor,
    entity_ids_ts: torch.Tensor,
    time_mask_ts: torch.Tensor,
    human_slots: int,
    relation_slots: int,
    aux: dict | None = None,
) -> TokenizerOutput:
    fields = {
        "tokens": tokens_ts,
        "mask": mask_ts,
        "modality_ids": modality_ids_ts,
        "token_type_ids": token_type_ids_ts,
        "time_ids": time_ids_ts,
        "entity_ids": entity_ids_ts,
        "time_mask": time_mask_ts,
    }
    flat = {
        key: _field_to_substream_major(value, human_slots, relation_slots)
        for key, value in fields.items()
    }
    return TokenizerOutput(
        tokens=zero_invalid_tokens(flat["tokens"], flat["mask"]),
        mask=flat["mask"].to(dtype=torch.bool),
        modality_ids=flat["modality_ids"].to(dtype=torch.long),
        token_type_ids=flat["token_type_ids"].to(dtype=torch.long),
        time_ids=flat["time_ids"].to(dtype=torch.long),
        entity_ids=flat["entity_ids"].to(dtype=torch.long),
        time_mask=flat["time_mask"].to(dtype=torch.bool),
        aux={} if aux is None else aux,
    )


def _field_to_substream_major(value: torch.Tensor, human_slots: int, relation_slots: int) -> torch.Tensor:
    human = value[:, :, :human_slots]
    obj = value[:, :, human_slots : human_slots + 1]
    relation = value[:, :, human_slots + 1 : human_slots + 1 + relation_slots]
    if value.ndim == 4:
        b, t, _, d = value.shape
        return torch.cat(
            [
                human.reshape(b, t * human_slots, d),
                obj.reshape(b, t, d),
                relation.reshape(b, t * relation_slots, d),
            ],
            dim=1,
        )
    b, t, _ = value.shape
    return torch.cat(
        [
            human.reshape(b, t * human_slots),
            obj.reshape(b, t),
            relation.reshape(b, t * relation_slots),
        ],
        dim=1,
    )


def _field_to_frame_major(
    value: torch.Tensor | None,
    dynamic: dict[str, slice],
    human_slots: int,
    relation_slots: int,
) -> torch.Tensor:
    if value is None:
        raise ValueError("Cannot convert missing time_mask to frame-major layout.")
    b = value.shape[0]
    t = (dynamic["object"].stop - dynamic["object"].start)
    if value.ndim == 3:
        d = value.shape[-1]
        human = value[:, dynamic["human"], :].reshape(b, t, human_slots, d)
        obj = value[:, dynamic["object"], :].reshape(b, t, 1, d)
        relation = value[:, dynamic["relation"], :].reshape(b, t, relation_slots, d)
    else:
        human = value[:, dynamic["human"]].reshape(b, t, human_slots)
        obj = value[:, dynamic["object"]].reshape(b, t, 1)
        relation = value[:, dynamic["relation"]].reshape(b, t, relation_slots)
    return torch.cat([human, obj, relation], dim=2)


def _dynamic_slices(
    target_num_frames: int,
    human_slots: int,
    relation_slots: int,
    total_tokens: int | None = None,
) -> dict[str, slice]:
    human_end = target_num_frames * human_slots
    object_end = human_end + target_num_frames
    if total_tokens is None:
        relation_start = object_end
        relation_end = object_end + target_num_frames * relation_slots
    else:
        relation_end = int(total_tokens)
        relation_start = relation_end - target_num_frames * relation_slots
        if relation_start < object_end:
            raise ValueError("Dense output does not have enough tokens for canonical dynamic layout.")
    return {
        "human": slice(0, human_end),
        "object": slice(human_end, object_end),
        "relation": slice(relation_start, relation_end),
    }


def _extract_geometry_memory(memory: TokenizerOutput) -> TokenizerOutput:
    geometry_index = memory.token_type_ids[0] == int(TokenTypeId.OBJECT_GEOMETRY)
    if not torch.equal(memory.token_type_ids == int(TokenTypeId.OBJECT_GEOMETRY), geometry_index[None, :].expand_as(memory.token_type_ids)):
        raise ValueError("TemporalDensifier requires object geometry layout to be identical across batch.")
    return TokenizerOutput(
        tokens=zero_invalid_tokens(memory.tokens[:, geometry_index, :], memory.mask[:, geometry_index]),
        mask=memory.mask[:, geometry_index],
        modality_ids=memory.modality_ids[:, geometry_index],
        token_type_ids=memory.token_type_ids[:, geometry_index],
        time_ids=memory.time_ids[:, geometry_index],
        entity_ids=memory.entity_ids[:, geometry_index],
        time_mask=torch.zeros_like(memory.mask[:, geometry_index], dtype=torch.bool),
        aux={"source": "temporal_densifier_geometry_memory"},
    )


def _attach_geometry(dense: TokenizerOutput, geometry: TokenizerOutput) -> TokenizerOutput:
    # Use metadata masks so geometry can be inserted in the public canonical
    # position without assuming a temporary dynamic layout.
    human = dense.modality_ids == int(ModalityId.HUMAN)
    object_pose = dense.token_type_ids == int(TokenTypeId.OBJECT_POSE)
    relation = dense.modality_ids == int(ModalityId.RELATION)

    def select(field: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if field.ndim == 3:
            return field[index].reshape(dense.tokens.shape[0], -1, field.shape[-1])
        return field[index].reshape(dense.tokens.shape[0], -1)

    return TokenizerOutput(
        tokens=torch.cat(
            [select(dense.tokens, human), select(dense.tokens, object_pose), geometry.tokens, select(dense.tokens, relation)],
            dim=1,
        ),
        mask=torch.cat([select(dense.mask, human), select(dense.mask, object_pose), geometry.mask, select(dense.mask, relation)], dim=1),
        modality_ids=torch.cat(
            [select(dense.modality_ids, human), select(dense.modality_ids, object_pose), geometry.modality_ids, select(dense.modality_ids, relation)],
            dim=1,
        ),
        token_type_ids=torch.cat(
            [
                select(dense.token_type_ids, human),
                select(dense.token_type_ids, object_pose),
                geometry.token_type_ids,
                select(dense.token_type_ids, relation),
            ],
            dim=1,
        ),
        time_ids=torch.cat([select(dense.time_ids, human), select(dense.time_ids, object_pose), geometry.time_ids, select(dense.time_ids, relation)], dim=1),
        entity_ids=torch.cat(
            [select(dense.entity_ids, human), select(dense.entity_ids, object_pose), geometry.entity_ids, select(dense.entity_ids, relation)],
            dim=1,
        ),
        time_mask=torch.cat(
            [select(dense.time_mask, human), select(dense.time_mask, object_pose), geometry.time_mask, select(dense.time_mask, relation)],
            dim=1,
        ),
        aux=dense.aux,
    )


def _same_semantic_match(dense: TokenizerOutput, memory: TokenizerOutput) -> tuple[torch.Tensor, torch.Tensor]:
    dense_valid = dense.mask.to(dtype=torch.bool)
    memory_valid = memory.mask.to(dtype=torch.bool)
    match = (
        dense_valid[:, :, None]
        & memory_valid[:, None, :]
        & (dense.time_ids[:, :, None] == memory.time_ids[:, None, :])
        & (dense.modality_ids[:, :, None] == memory.modality_ids[:, None, :])
        & (dense.token_type_ids[:, :, None] == memory.token_type_ids[:, None, :])
        & (dense.entity_ids[:, :, None] == memory.entity_ids[:, None, :])
    )
    has_match = match.any(dim=-1)
    indices = match.to(dtype=torch.float32).argmax(dim=-1).to(dtype=torch.long)
    return has_match, indices


def _replace_tokens(out: TokenizerOutput, tokens: torch.Tensor, aux: dict | None = None) -> TokenizerOutput:
    return TokenizerOutput(
        tokens=zero_invalid_tokens(tokens, out.mask),
        mask=out.mask,
        modality_ids=out.modality_ids,
        token_type_ids=out.token_type_ids,
        time_ids=out.time_ids,
        entity_ids=out.entity_ids,
        time_mask=out.time_mask,
        aux=out.aux if aux is None else aux,
    )


def _human_slot_token_types(human_slots: int, device: torch.device) -> torch.Tensor:
    body_slots = human_slots - 3
    return torch.tensor(
        [int(TokenTypeId.HUMAN_ROOT)]
        + [int(TokenTypeId.HUMAN_BODY_SLOT)] * body_slots
        + [int(TokenTypeId.HUMAN_FOOT_LEFT), int(TokenTypeId.HUMAN_FOOT_RIGHT)],
        dtype=torch.long,
        device=device,
    )


def _relation_slot_token_types(relation_slots: int, device: torch.device) -> torch.Tensor:
    body_slots = relation_slots - 2
    return torch.tensor(
        [int(TokenTypeId.RELATION_BODY_SLOT)] * body_slots
        + [int(TokenTypeId.RELATION_GROUND), int(TokenTypeId.RELATION_ROOT_OBJECT)],
        dtype=torch.long,
        device=device,
    )


def _cross_attention_diagnostics(
    weights: torch.Tensor,
    dense_mask: torch.Tensor,
    memory_mask: torch.Tensor,
    dense_substream: torch.Tensor,
    memory_substream: torch.Tensor,
) -> dict:
    _, heads, _, _ = weights.shape
    mass = weights.new_zeros(heads, len(HOI_SUBSTREAM_NAMES), len(HOI_SUBSTREAM_NAMES))
    for q_idx in range(len(HOI_SUBSTREAM_NAMES)):
        q_mask = (dense_substream == q_idx) & dense_mask
        q_count = q_mask.sum().clamp_min(1).to(dtype=weights.dtype)
        for k_idx in range(len(HOI_SUBSTREAM_NAMES)):
            k_mask = (memory_substream == k_idx) & memory_mask
            selected = weights * q_mask[:, None, :, None].to(dtype=weights.dtype) * k_mask[:, None, None, :].to(
                dtype=weights.dtype
            )
            mass[:, q_idx, k_idx] = selected.sum(dim=(0, 2, 3)) / q_count
    geometry_idx = int(HOISubstreamId.OBJECT_GEOMETRY)
    eps = torch.finfo(weights.dtype).eps
    entropy = -(weights * (weights + eps).log()).sum(dim=-1)
    return {
        "memory_cross_attention_mass_by_source_substream": mass.detach(),
        "object_geometry_memory_attention_mass": mass[:, :, geometry_idx].mean().detach(),
        "memory_cross_attention_entropy": _masked_mean(
            entropy,
            dense_mask[:, None, :].expand_as(entropy),
        ).detach(),
    }


def _local_temporal_mask(num_frames: int, radius: int, device: torch.device) -> torch.Tensor:
    time = torch.arange(num_frames, device=device)
    return (time[None, :] - time[:, None]).abs() <= int(radius)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _stack_diag(items: list[dict]) -> dict:
    merged: dict[str, Any] = {}
    keys = sorted({key for item in items for key in item.keys()})
    for key in keys:
        values = [item[key] for item in items if key in item]
        if values and isinstance(values[0], torch.Tensor):
            merged[key] = torch.stack(values)
        elif values:
            merged[key] = values
    return merged


def _validate_temporal_densifier_config(
    query_cfg: dict,
    memory_cfg: dict,
    anchor_cfg: dict,
    frame_cfg: dict,
    temporal_cfg: dict,
) -> None:
    _require_value(query_cfg, "mode", "topology_aware_dense_queries")
    _require_value(query_cfg, "include_object_pose", True)
    _require_value(query_cfg, "keep_object_geometry_static", True)
    _require_value(memory_cfg, "enabled", True)
    _require_value(memory_cfg, "attention_scope", "global_memory")
    _require_value(memory_cfg, "relative_time_bias", True)
    _require_value(memory_cfg, "geometry_memory", True)
    _require_value(memory_cfg, "geometry_relative_time_bias", "zero")
    _require_value(anchor_cfg, "enabled", True)
    _require_value(frame_cfg, "enabled", True)
    _require_value(temporal_cfg, "enabled", True)
    _require_value(temporal_cfg, "scope", "local")
    _require_value(temporal_cfg, "full_temporal_attention", "ablation_only")


def _require_value(cfg: dict, key: str, expected) -> None:
    if key in cfg and cfg[key] != expected:
        raise ValueError(f"Stage1TemporalDensifier only supports {key}={expected!r} for v1, got {cfg[key]!r}.")


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))
