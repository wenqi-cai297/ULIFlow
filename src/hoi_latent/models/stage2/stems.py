from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from hoi_latent.models.stage1.attention_pool import CrossAttentionPool
from hoi_latent.models.stage1.encoder.transformer_blocks import FeedForward, MultiHeadAttention
from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.models.stage1.tokenizer_types import NUM_TOKEN_TYPES
from hoi_latent.models.stage1.tokenizers import MLPProjector
from hoi_latent.models.stage2.flow_time import temporal_dynamic_token_mask
from hoi_latent.models.stage2.stem_types import (
    NUM_STAGE2_FLOW_ROUTES,
    NUM_STAGE2_LATENT_SUBSTREAMS,
    Stage2FlowRouteId,
    Stage2GeometryStemOutput,
    Stage2LatentStemOutput,
    Stage2TextStemOutput,
    derive_stage2_latent_substream_ids,
    sanitize_ids_for_embedding,
)


class GatedTextContextBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ffn_mult: int = 4,
        residual_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, use_rope=False)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, ffn_mult=ffn_mult, dropout=dropout)
        self.gate_attn = nn.Parameter(torch.full((d_model,), float(residual_gate_init)))
        self.gate_ffn = nn.Parameter(torch.full((d_model,), float(residual_gate_init)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.attn(self.attn_norm(x), key_mask=mask, query_mask=mask)
        x = zero_invalid_tokens(x + self.gate_attn.view(1, 1, -1) * y, mask)
        y = self.ffn(self.ffn_norm(x))
        return zero_invalid_tokens(x + self.gate_ffn.view(1, 1, -1) * y, mask)


class TextConditionEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 6,
        backbone: str = "openai/clip-vit-base-patch32",
        backend: str = "clip",
        freeze_clip: bool | None = None,
        freeze_text_encoder: bool | None = None,
        max_seq_len: int | None = None,
        clip_dim: int | None = None,
        encoder_dim: int | None = None,
        encoder_dtype: str = "auto",
        contextualizer_depth: int = 2,
        contextualizer_dropout: float = 0.0,
        residual_gate_init: float = 0.0,
        null_prompt: str = "",
        text_model: nn.Module | None = None,
        clip_model: nn.Module | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        if contextualizer_depth < 0:
            raise ValueError(f"contextualizer_depth must be non-negative, got {contextualizer_depth}.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.backbone = backbone
        self.backend = _normalize_text_backend(backend)
        if freeze_text_encoder is None:
            freeze_text_encoder = True if freeze_clip is None else bool(freeze_clip)
        self.freeze_text_encoder = bool(freeze_text_encoder)
        self.freeze_clip = self.freeze_text_encoder
        self.null_prompt = null_prompt
        self.encoder_dtype = str(encoder_dtype or "auto").lower()

        if text_model is not None and clip_model is not None and text_model is not clip_model:
            raise ValueError("Provide either text_model or clip_model, not both.")
        if text_model is None:
            text_model = clip_model
        if (text_model is None) != (tokenizer is None):
            raise ValueError("TextConditionEncoder requires text_model and tokenizer to be provided together.")
        if text_model is None or tokenizer is None:
            text_model, tokenizer = _load_text_components(self.backend, backbone)
        self.text_model = text_model
        self.tokenizer = tokenizer
        encoder_torch_dtype = _text_encoder_dtype_from_config(self.encoder_dtype)
        if encoder_torch_dtype is not None:
            self.text_model.to(dtype=encoder_torch_dtype)
        if self.freeze_text_encoder:
            self.text_model.eval()
            for param in self.text_model.parameters():
                param.requires_grad_(False)

        inferred_max_len = _infer_text_max_length(tokenizer)
        self.max_seq_len = int(max_seq_len or inferred_max_len)
        inferred_encoder_dim = encoder_dim or clip_dim or _infer_text_hidden_size(text_model)
        self.encoder_dim = int(inferred_encoder_dim)
        self.clip_dim = self.encoder_dim

        self.text_token_norm = nn.LayerNorm(self.encoder_dim)
        self.text_token_proj = nn.Linear(self.encoder_dim, d_model)
        self.encoder_pool_proj = nn.Linear(self.encoder_dim, d_model) if self.backend == "clip" else None
        self.token_pool_proj = nn.Linear(d_model, d_model)
        self.text_pool_norm = nn.LayerNorm(d_model)
        self.text_position_embedding = nn.Parameter(torch.zeros(self.max_seq_len, d_model))
        self.flow_route_embedding = nn.Embedding(NUM_STAGE2_FLOW_ROUTES, d_model)
        self.contextualizer = nn.ModuleList(
            [
                GatedTextContextBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=contextualizer_dropout,
                    residual_gate_init=residual_gate_init,
                )
                for _ in range(contextualizer_depth)
            ]
        )
        self.summary_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.summary_pool = CrossAttentionPool(d_model=d_model, num_heads=num_heads, dropout=contextualizer_dropout)

    @property
    def clip_model(self) -> nn.Module:
        return self.text_model

    def forward(
        self,
        batch: dict,
        force_null_text_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> Stage2TextStemOutput:
        texts_raw = batch.get("meta", {}).get("text")
        if texts_raw is None:
            raise KeyError("TextConditionEncoder requires batch['meta']['text'].")
        texts = list(texts_raw)
        batch_size = len(texts)
        missing = torch.tensor([text is None for text in texts], dtype=torch.bool, device=self._device)
        if force_null_text_mask is not None:
            force_null = force_null_text_mask.to(device=self._device, dtype=torch.bool)
            if force_null.shape != (batch_size,):
                raise ValueError(f"force_null_text_mask must have shape {(batch_size,)}, got {tuple(force_null.shape)}.")
        else:
            force_null = torch.zeros(batch_size, dtype=torch.bool, device=self._device)

        texts = [
            self.null_prompt if (text is None or bool(force_null[idx].item())) else str(text)
            for idx, text in enumerate(texts)
        ]
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        input_ids = _to_tensor(encoded["input_ids"], self._device, torch.long)
        attention_mask = _to_tensor(encoded["attention_mask"], self._device, torch.bool)
        if input_ids.shape != attention_mask.shape:
            raise ValueError("Text tokenizer input_ids and attention_mask shapes do not match.")
        if input_ids.shape[0] != batch_size:
            raise ValueError(f"Tokenizer returned batch size {input_ids.shape[0]}, expected {batch_size}.")
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError(f"Tokenizer returned length {input_ids.shape[1]} > max_seq_len={self.max_seq_len}.")

        with torch.set_grad_enabled(not self.freeze_text_encoder):
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask.to(dtype=torch.long),
                output_hidden_states=True,
            )
        encoder_tokens = text_outputs.last_hidden_state
        encoder_pool = getattr(text_outputs, "pooler_output", None)
        if encoder_pool is None and self.encoder_pool_proj is not None:
            encoder_pool = _pool_last_valid_token(encoder_tokens, attention_mask)
        encoder_tokens = encoder_tokens.to(dtype=self.text_token_proj.weight.dtype)
        if encoder_pool is not None:
            encoder_pool = encoder_pool.to(dtype=self.text_token_proj.weight.dtype)
        if encoder_tokens.shape[-1] != self.encoder_dim:
            raise ValueError(
                f"Text encoder hidden size {encoder_tokens.shape[-1]} does not match configured "
                f"encoder_dim={self.encoder_dim}."
            )

        length = encoder_tokens.shape[1]
        if length > self.max_seq_len:
            raise ValueError(f"Text token length {length} exceeds max_seq_len={self.max_seq_len}.")
        x = self.text_token_proj(self.text_token_norm(encoder_tokens))
        x = x + self.text_position_embedding[:length][None, :, :]
        text_route = torch.full((batch_size, length), int(Stage2FlowRouteId.TEXT_CONDITION), dtype=torch.long, device=x.device)
        x = x + self.flow_route_embedding(text_route)
        x = zero_invalid_tokens(x, attention_mask)
        for block in self.contextualizer:
            x = block(x, attention_mask)

        query = self.summary_query[None, :, :].expand(batch_size, 1, self.d_model)
        token_pool, _ = self.summary_pool(query=query, key_value=x, key_mask=attention_mask)
        token_pool = token_pool.squeeze(1)
        pooled = self.token_pool_proj(token_pool)
        if encoder_pool is not None and self.encoder_pool_proj is not None:
            pooled = pooled + self.encoder_pool_proj(encoder_pool)
        pooled = self.text_pool_norm(pooled)

        diagnostics = {}
        if return_diagnostics:
            diagnostics = {
                "missing_text_count": missing.sum().detach(),
                "missing_text_fraction": missing.float().mean().detach() if missing.numel() else torch.tensor(0.0),
                "forced_null_text_count": force_null.sum().detach(),
                "valid_token_count": attention_mask.sum(dim=1).detach(),
            }
        return Stage2TextStemOutput(
            tokens=x,
            mask=attention_mask,
            pooled=pooled,
            input_ids=input_ids,
            attention_mask=attention_mask,
            flow_route_ids=text_route,
            diagnostics=diagnostics,
        )

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device


class ObjectGeometryConditionEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 6,
        max_bps_points: int = 1024,
        num_geometry_tokens: int = 16,
        dropout: float = 0.0,
        allow_missing_bps: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        if max_bps_points <= 0:
            raise ValueError(f"max_bps_points must be positive, got {max_bps_points}.")
        if num_geometry_tokens <= 0:
            raise ValueError(f"num_geometry_tokens must be positive, got {num_geometry_tokens}.")
        self.d_model = int(d_model)
        self.max_bps_points = int(max_bps_points)
        self.num_geometry_tokens = int(num_geometry_tokens)
        self.allow_missing_bps = bool(allow_missing_bps)

        self.bps_proj = MLPProjector(3, d_model, dropout=dropout)
        self.bps_index_embedding = nn.Embedding(self.max_bps_points, d_model)
        self.geometry_queries = nn.Parameter(torch.randn(num_geometry_tokens, d_model) * 0.02)
        self.geometry_slot_embedding = nn.Embedding(num_geometry_tokens, d_model)
        self.flow_route_embedding = nn.Embedding(NUM_STAGE2_FLOW_ROUTES, d_model)
        self.geometry_pool = CrossAttentionPool(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.summary_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.summary_pool = CrossAttentionPool(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.geometry_output_norm = nn.LayerNorm(d_model)
        self.geometry_pool_norm = nn.LayerNorm(d_model)
        self.null_geometry_tokens = nn.Parameter(torch.zeros(num_geometry_tokens, d_model))
        self.null_pooled_geometry = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        batch: dict,
        force_null_geometry_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> Stage2GeometryStemOutput:
        geometry = batch.get("geometry")
        if geometry is None:
            raise KeyError("ObjectGeometryConditionEncoder requires batch['geometry'].")
        if "object_bps" not in geometry:
            raise KeyError("ObjectGeometryConditionEncoder requires batch['geometry']['object_bps'].")
        bps = geometry["object_bps"]
        if bps.ndim == 4 and bps.shape[1] == 1:
            bps = bps[:, 0]
        if bps.ndim != 3 or bps.shape[-1] != 3:
            raise ValueError(f"geometry.object_bps must have shape [B,N,3], got {tuple(bps.shape)}.")
        batch_size, n_bps, _ = bps.shape
        if n_bps <= 0:
            raise ValueError("geometry.object_bps must contain at least one BPS point.")
        if n_bps > self.max_bps_points:
            raise ValueError(f"geometry.object_bps has {n_bps} points, max_bps_points={self.max_bps_points}.")
        bps = bps.to(device=self._device, dtype=self.bps_index_embedding.weight.dtype)
        bps_mask = geometry.get("object_bps_mask")
        if bps_mask is None:
            bps_mask = torch.ones(batch_size, n_bps, dtype=torch.bool, device=self._device)
        else:
            bps_mask = bps_mask.to(device=self._device, dtype=torch.bool)
            if bps_mask.ndim == 3 and bps_mask.shape[1] == 1:
                bps_mask = bps_mask[:, 0]
            if bps_mask.shape != (batch_size, n_bps):
                raise ValueError(f"geometry.object_bps_mask must have shape {(batch_size, n_bps)}, got {tuple(bps_mask.shape)}.")

        has_bps = _normalize_optional_bool_vector(geometry.get("has_object_bps"), batch_size, self._device, default=True)
        if force_null_geometry_mask is not None:
            force_null = force_null_geometry_mask.to(device=self._device, dtype=torch.bool)
            if force_null.shape != (batch_size,):
                raise ValueError(
                    f"force_null_geometry_mask must have shape {(batch_size,)}, got {tuple(force_null.shape)}."
                )
            has_bps = has_bps & ~force_null
        else:
            force_null = torch.zeros(batch_size, dtype=torch.bool, device=self._device)
        if not self.allow_missing_bps and (~has_bps).any():
            raise ValueError("Missing object BPS is not allowed by this encoder configuration.")

        bps_low = self.bps_proj(bps)
        bps_ids = torch.arange(n_bps, dtype=torch.long, device=self._device)
        bps_low = bps_low + self.bps_index_embedding(bps_ids)[None, :, :]
        bps_low = zero_invalid_tokens(bps_low, bps_mask)
        key_mask = bps_mask & has_bps[:, None]
        queries = self.geometry_queries[None, :, :].expand(batch_size, self.num_geometry_tokens, self.d_model)
        geometry_tokens, _ = self.geometry_pool(query=queries, key_value=bps_low, key_mask=key_mask)
        slot_ids = torch.arange(self.num_geometry_tokens, dtype=torch.long, device=self._device)
        route_ids = torch.full(
            (batch_size, self.num_geometry_tokens),
            int(Stage2FlowRouteId.OBJECT_GEOMETRY_CONDITION),
            dtype=torch.long,
            device=self._device,
        )
        geometry_tokens = geometry_tokens + self.geometry_slot_embedding(slot_ids)[None, :, :]
        geometry_tokens = geometry_tokens + self.flow_route_embedding(route_ids)
        geometry_tokens = self.geometry_output_norm(geometry_tokens)

        geometry_mask = torch.ones(batch_size, self.num_geometry_tokens, dtype=torch.bool, device=self._device)
        summary_query = self.summary_query[None, :, :].expand(batch_size, 1, self.d_model)
        pooled, _ = self.summary_pool(query=summary_query, key_value=geometry_tokens, key_mask=geometry_mask)
        pooled = self.geometry_pool_norm(pooled.squeeze(1))

        null_tokens = self.null_geometry_tokens[None, :, :].expand(batch_size, self.num_geometry_tokens, self.d_model)
        null_pooled = self.null_pooled_geometry[None, :].expand(batch_size, self.d_model)
        geometry_tokens = torch.where(has_bps[:, None, None], geometry_tokens, null_tokens)
        pooled = torch.where(has_bps[:, None], pooled, null_pooled)
        geometry_tokens = zero_invalid_tokens(geometry_tokens, geometry_mask)

        missing = ~has_bps
        diagnostics = {}
        if return_diagnostics:
            diagnostics = {
                "missing_geometry_count": missing.sum().detach(),
                "missing_geometry_fraction": missing.float().mean().detach() if missing.numel() else torch.tensor(0.0),
                "forced_null_geometry_count": force_null.sum().detach(),
                "geometry_source": "bps",
            }
        return Stage2GeometryStemOutput(
            tokens=geometry_tokens,
            mask=geometry_mask,
            pooled=pooled,
            source_mask=bps_mask,
            missing=missing,
            flow_route_ids=route_ids,
            diagnostics=diagnostics,
        )

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device


class LatentNoisyStreamEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        latent_dim: int = 16,
        num_entity_embeddings: int = 1024,
        use_latent_substream_embedding: bool = False,
        allow_unknown_token_type_ids: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if num_entity_embeddings <= 0:
            raise ValueError(f"num_entity_embeddings must be positive, got {num_entity_embeddings}.")
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.num_entity_embeddings = int(num_entity_embeddings)
        self.use_latent_substream_embedding = bool(use_latent_substream_embedding)
        self.allow_unknown_token_type_ids = bool(allow_unknown_token_type_ids)

        self.latent_in_proj = nn.Linear(latent_dim, d_model)
        self.flow_route_embedding = nn.Embedding(NUM_STAGE2_FLOW_ROUTES, d_model)
        self.token_type_embedding = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.entity_embedding = nn.Embedding(num_entity_embeddings, d_model)
        self.latent_substream_embedding = (
            nn.Embedding(NUM_STAGE2_LATENT_SUBSTREAMS, d_model) if self.use_latent_substream_embedding else None
        )
        self.latent_output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z_white_t: torch.Tensor,
        latent_mask: torch.Tensor,
        modality_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        time_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        time_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Stage2LatentStemOutput:
        if z_white_t.ndim != 3 or z_white_t.shape[-1] != self.latent_dim:
            raise ValueError(f"z_white_t must have shape [B,N,{self.latent_dim}], got {tuple(z_white_t.shape)}.")
        batch_size, num_tokens, _ = z_white_t.shape
        expected = (batch_size, num_tokens)
        for name, value in {
            "latent_mask": latent_mask,
            "modality_ids": modality_ids,
            "token_type_ids": token_type_ids,
            "time_ids": time_ids,
            "entity_ids": entity_ids,
            "time_mask": time_mask,
        }.items():
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}.")

        device = z_white_t.device
        latent_mask = latent_mask.to(device=device, dtype=torch.bool)
        h_z = self.latent_in_proj(z_white_t)
        h_z = zero_invalid_tokens(h_z, latent_mask)
        modality_ids = modality_ids.to(device=device, dtype=torch.long)
        raw_token_type_ids = token_type_ids.to(device=device, dtype=torch.long)
        token_type_ids = raw_token_type_ids.clone()
        if self.allow_unknown_token_type_ids:
            token_type_ids[(token_type_ids < 0) & latent_mask] = 0
        token_type_ids = sanitize_ids_for_embedding(
            token_type_ids,
            latent_mask,
            num_embeddings=NUM_TOKEN_TYPES,
            name="token_type_ids",
        )
        output_token_type_ids = raw_token_type_ids.clone()
        output_token_type_ids[~latent_mask] = 0
        entity_ids = sanitize_ids_for_embedding(
            entity_ids.to(device=device),
            latent_mask,
            num_embeddings=self.num_entity_embeddings,
            name="entity_ids",
        )
        time_ids = time_ids.to(device=device, dtype=torch.long)
        time_mask = time_mask.to(device=device, dtype=torch.bool)

        route_ids = torch.full(
            expected,
            int(Stage2FlowRouteId.LATENT_GENERATED),
            dtype=torch.long,
            device=device,
        )
        h_z = h_z + self.flow_route_embedding(route_ids)
        h_z = h_z + self.token_type_embedding(token_type_ids)
        h_z = h_z + self.entity_embedding(entity_ids)
        if self.latent_substream_embedding is not None:
            substream_ids = derive_stage2_latent_substream_ids(modality_ids, output_token_type_ids, mask=latent_mask)
            h_z = h_z + self.latent_substream_embedding(substream_ids)
        h_z = self.latent_output_norm(h_z)
        h_z = zero_invalid_tokens(h_z, latent_mask)

        apply_temporal_rope = temporal_dynamic_token_mask(time_mask, latent_mask, output_token_type_ids)
        diagnostics = {}
        if return_diagnostics:
            diagnostics = {
                "valid_token_count": latent_mask.sum(dim=1).detach(),
                "apply_temporal_rope_count": apply_temporal_rope.sum(dim=1).detach(),
                "use_latent_substream_embedding": self.use_latent_substream_embedding,
                "allow_unknown_token_type_ids": self.allow_unknown_token_type_ids,
            }
        return Stage2LatentStemOutput(
            tokens=h_z,
            mask=latent_mask,
            time_ids=time_ids,
            token_type_ids=output_token_type_ids,
            entity_ids=entity_ids,
            flow_route_ids=route_ids,
            apply_temporal_rope=apply_temporal_rope,
            diagnostics=diagnostics,
        )


def _normalize_text_backend(backend: str) -> str:
    value = str(backend or "clip").lower()
    if value in {"clip", "clip_text", "clip_text_model"}:
        return "clip"
    if value in {"t5", "t5_encoder", "flan_t5", "flan-t5"}:
        return "t5_encoder"
    raise ValueError(f"Unsupported text encoder backend {backend!r}; expected 'clip' or 't5_encoder'.")


def _load_text_components(backend: str, backbone: str) -> tuple[nn.Module, Any]:
    if backend == "clip":
        return _load_clip_text_components(backbone)
    if backend == "t5_encoder":
        return _load_t5_text_components(backbone)
    raise ValueError(f"Unsupported text encoder backend {backend!r}.")


def _load_clip_text_components(backbone: str) -> tuple[nn.Module, Any]:
    try:
        from transformers import CLIPTextModel, CLIPTokenizerFast
    except ImportError as exc:
        raise ImportError(
            "TextConditionEncoder backend='clip' requires transformers. "
            "Install project dependencies or inject text_model/tokenizer for tests."
        ) from exc
    return CLIPTextModel.from_pretrained(backbone), CLIPTokenizerFast.from_pretrained(backbone)


def _load_t5_text_components(backbone: str) -> tuple[nn.Module, Any]:
    try:
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as exc:
        raise ImportError(
            "TextConditionEncoder backend='t5_encoder' requires transformers. "
            "Install project dependencies or inject text_model/tokenizer for tests."
        ) from exc
    return T5EncoderModel.from_pretrained(backbone), AutoTokenizer.from_pretrained(backbone, use_fast=True)


def _infer_text_max_length(tokenizer: Any) -> int:
    value = int(getattr(tokenizer, "model_max_length", 77))
    if value <= 0 or value > 10000:
        return 77
    return value


def _infer_text_hidden_size(text_model: nn.Module) -> int:
    config = getattr(text_model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "d_model", None)
    if hidden_size is None:
        raise ValueError("encoder_dim must be provided when text_model.config hidden size is unavailable.")
    return int(hidden_size)


def _text_encoder_dtype_from_config(value: str) -> torch.dtype | None:
    dtype = str(value or "auto").lower()
    if dtype in {"auto", "none", "default"}:
        return None
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported text encoder dtype {value!r}; expected auto, bf16, fp16, or fp32.")


def _pool_last_valid_token(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.to(dtype=torch.long).sum(dim=1).clamp_min(1)
    index = (lengths - 1).to(device=tokens.device)
    batch = torch.arange(tokens.shape[0], device=tokens.device)
    return tokens[batch, index]


def _to_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=dtype)


def _normalize_optional_bool_vector(value: Any, batch_size: int, device: torch.device, default: bool) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), bool(default), dtype=torch.bool, device=device)
    if isinstance(value, bool):
        return torch.full((batch_size,), value, dtype=torch.bool, device=device)
    tensor = torch.as_tensor(value, dtype=torch.bool, device=device)
    if tensor.ndim == 0:
        return tensor.expand(batch_size)
    if tensor.shape != (batch_size,):
        raise ValueError(f"Expected bool vector with shape {(batch_size,)}, got {tuple(tensor.shape)}.")
    return tensor


def fake_clip_outputs(last_hidden_state: torch.Tensor, pooler_output: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(last_hidden_state=last_hidden_state, pooler_output=pooler_output)
