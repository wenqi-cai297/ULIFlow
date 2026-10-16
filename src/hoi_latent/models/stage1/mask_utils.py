from __future__ import annotations

import torch


def zero_invalid_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [B,N,D], got {tuple(tokens.shape)}.")
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"mask must have shape {tuple(tokens.shape[:2])}, got {tuple(mask.shape)}.")
    return tokens * mask.to(dtype=tokens.dtype).unsqueeze(-1)


def flatten_time_entity_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"Expected [B,T,N] mask, got {tuple(mask.shape)}.")
    return mask.reshape(mask.shape[0], mask.shape[1] * mask.shape[2])


def repeat_valid_over_slots(valid: torch.Tensor, num_slots: int) -> torch.Tensor:
    if valid.ndim != 2:
        raise ValueError(f"valid must have shape [B,T], got {tuple(valid.shape)}.")
    if num_slots <= 0:
        raise ValueError(f"num_slots must be positive, got {num_slots}.")
    b, t = valid.shape
    return valid[:, :, None].expand(b, t, num_slots).reshape(b, t * num_slots)


def make_time_mask_from_token_mask(mask: torch.Tensor, static: bool = False) -> torch.Tensor:
    if static:
        return torch.zeros_like(mask, dtype=torch.bool)
    return mask.to(dtype=torch.bool)


def safe_masked_softmax(scores: torch.Tensor, key_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = key_mask.to(torch.bool)
    scores = scores.masked_fill(~mask, -1e4)
    probs = torch.softmax(scores, dim=dim)
    probs = probs * mask.to(dtype=probs.dtype)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    probs = probs / denom
    any_valid = mask.any(dim=dim, keepdim=True)
    return probs * any_valid.to(dtype=probs.dtype)
