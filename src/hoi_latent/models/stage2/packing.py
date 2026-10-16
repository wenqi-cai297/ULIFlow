from __future__ import annotations

import torch
from torch import nn

from hoi_latent.models.stage1.mask_utils import zero_invalid_tokens
from hoi_latent.models.stage2.stem_types import (
    Stage2GeometryStemOutput,
    Stage2LatentStemOutput,
    Stage2PackedStreams,
    Stage2TextStemOutput,
)


class Stage2StreamPacker(nn.Module):
    def forward(
        self,
        latent: Stage2LatentStemOutput,
        text: Stage2TextStemOutput,
        geometry: Stage2GeometryStemOutput,
    ) -> Stage2PackedStreams:
        _validate_rank3(latent.tokens, "latent.tokens")
        _validate_rank3(text.tokens, "text.tokens")
        _validate_rank3(geometry.tokens, "geometry.tokens")
        batch_size, latent_len, d_model = latent.tokens.shape
        text_len = text.tokens.shape[1]
        geometry_len = geometry.tokens.shape[1]
        if text.tokens.shape[0] != batch_size or geometry.tokens.shape[0] != batch_size:
            raise ValueError("All Stage2 stream outputs must have the same batch size.")
        if text.tokens.shape[-1] != d_model or geometry.tokens.shape[-1] != d_model:
            raise ValueError("All Stage2 stream outputs must have the same d_model.")

        _validate_mask(latent.mask, (batch_size, latent_len), "latent.mask")
        _validate_mask(text.mask, (batch_size, text_len), "text.mask")
        _validate_mask(geometry.mask, (batch_size, geometry_len), "geometry.mask")
        _validate_mask(latent.flow_route_ids, (batch_size, latent_len), "latent.flow_route_ids")
        _validate_mask(text.flow_route_ids, (batch_size, text_len), "text.flow_route_ids")
        _validate_mask(geometry.flow_route_ids, (batch_size, geometry_len), "geometry.flow_route_ids")
        _validate_mask(latent.time_ids, (batch_size, latent_len), "latent.time_ids")
        _validate_mask(latent.apply_temporal_rope, (batch_size, latent_len), "latent.apply_temporal_rope")

        latent_slice = slice(0, latent_len)
        text_slice = slice(latent_len, latent_len + text_len)
        geometry_slice = slice(latent_len + text_len, latent_len + text_len + geometry_len)

        tokens = torch.cat([latent.tokens, text.tokens, geometry.tokens], dim=1)
        attention_mask = torch.cat(
            [
                latent.mask.to(dtype=torch.bool),
                text.mask.to(dtype=torch.bool),
                geometry.mask.to(dtype=torch.bool),
            ],
            dim=1,
        )
        flow_route_ids = torch.cat(
            [
                latent.flow_route_ids.to(dtype=torch.long),
                text.flow_route_ids.to(dtype=torch.long),
                geometry.flow_route_ids.to(dtype=torch.long),
            ],
            dim=1,
        )

        text_time_ids = torch.zeros((batch_size, text_len), dtype=latent.time_ids.dtype, device=tokens.device)
        geometry_time_ids = torch.zeros((batch_size, geometry_len), dtype=latent.time_ids.dtype, device=tokens.device)
        time_ids = torch.cat([latent.time_ids.to(device=tokens.device), text_time_ids, geometry_time_ids], dim=1)
        apply_temporal_rope = torch.cat(
            [
                latent.apply_temporal_rope.to(device=tokens.device, dtype=torch.bool),
                torch.zeros((batch_size, text_len), dtype=torch.bool, device=tokens.device),
                torch.zeros((batch_size, geometry_len), dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )
        tokens = zero_invalid_tokens(tokens, attention_mask)

        diagnostics = {
            "latent_tokens": latent_len,
            "text_tokens": text_len,
            "geometry_tokens": geometry_len,
            "total_tokens": tokens.shape[1],
        }
        return Stage2PackedStreams(
            tokens=tokens,
            attention_mask=attention_mask,
            flow_route_ids=flow_route_ids,
            time_ids=time_ids,
            apply_temporal_rope=apply_temporal_rope,
            latent_slice=latent_slice,
            text_slice=text_slice,
            geometry_slice=geometry_slice,
            pooled_text=text.pooled,
            pooled_geometry=geometry.pooled,
            diagnostics=diagnostics,
        )


def _validate_rank3(value: torch.Tensor, name: str) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B,N,D], got {tuple(value.shape)}.")


def _validate_mask(value: torch.Tensor, expected: tuple[int, int], name: str) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}.")
