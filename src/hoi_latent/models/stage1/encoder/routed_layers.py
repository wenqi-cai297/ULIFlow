from __future__ import annotations

import copy

import torch
from torch import nn

from hoi_latent.models.stage1.encoder.hoi_packing import HOI_SUBSTREAM_NAMES
from hoi_latent.models.stage1.encoder.transformer_blocks import FeedForward


class SubstreamEmbedding(nn.Module):
    def __init__(self, d_model: int, num_substreams: int = 4, init_std: float = 0.02) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_substreams, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=float(init_std))

    def forward(self, substream_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(substream_ids.to(dtype=torch.long))


class RoutedLayerNorm(nn.Module):
    def __init__(self, d_model: int, route_names: tuple[str, ...] = HOI_SUBSTREAM_NAMES) -> None:
        super().__init__()
        self.route_names = tuple(route_names)
        self.norms = nn.ModuleDict({name: nn.LayerNorm(d_model) for name in self.route_names})

    def forward(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B,N,D], got {tuple(x.shape)}.")
        if route_ids.shape != x.shape[:2]:
            raise ValueError(f"route_ids must have shape {tuple(x.shape[:2])}, got {tuple(route_ids.shape)}.")
        out = torch.zeros_like(x)
        for idx, name in enumerate(self.route_names):
            route_mask = (route_ids == idx).unsqueeze(-1).to(dtype=x.dtype)
            if route_mask.any():
                out = out + self.norms[name](x).to(dtype=x.dtype) * route_mask
        return out


class RoutedFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        route_names: tuple[str, ...] = HOI_SUBSTREAM_NAMES,
    ) -> None:
        super().__init__()
        self.route_names = tuple(route_names)
        self.ffns = nn.ModuleDict(
            {
                name: FeedForward(d_model=d_model, ffn_mult=ffn_mult, dropout=dropout)
                for name in self.route_names
            }
        )

    def forward(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B,N,D], got {tuple(x.shape)}.")
        if route_ids.shape != x.shape[:2]:
            raise ValueError(f"route_ids must have shape {tuple(x.shape[:2])}, got {tuple(route_ids.shape)}.")
        out = torch.zeros_like(x)
        for idx, name in enumerate(self.route_names):
            route_mask = (route_ids == idx).unsqueeze(-1).to(dtype=x.dtype)
            if route_mask.any():
                out = out + self.ffns[name](x).to(dtype=x.dtype) * route_mask
        return out


class SharedAttentionProjection(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def q(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x)

    def k(self, x: torch.Tensor) -> torch.Tensor:
        return self.k_proj(x)

    def v(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_proj(x)

    def o(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


class RoutedAttentionProjection(nn.Module):
    """Route-aware Q/K/V/O projection with safe full-mode initialization."""

    allowed_init_modes = ("shared_plus_delta", "tie_at_init")

    def __init__(
        self,
        d_model: int,
        init_mode: str = "shared_plus_delta",
        gamma_init: float = 0.0,
        route_names: tuple[str, ...] = HOI_SUBSTREAM_NAMES,
    ) -> None:
        super().__init__()
        if init_mode not in self.allowed_init_modes:
            raise ValueError(
                f"full routed Q/K/V/O init_mode must be one of {self.allowed_init_modes}, got {init_mode!r}."
            )
        self.d_model = int(d_model)
        self.init_mode = init_mode
        self.route_names = tuple(route_names)
        if init_mode == "shared_plus_delta":
            self.shared = SharedAttentionProjection(d_model)
            self.delta_q = nn.ModuleDict({name: nn.Linear(d_model, d_model) for name in self.route_names})
            self.delta_k = nn.ModuleDict({name: nn.Linear(d_model, d_model) for name in self.route_names})
            self.delta_v = nn.ModuleDict({name: nn.Linear(d_model, d_model) for name in self.route_names})
            self.delta_o = nn.ModuleDict({name: nn.Linear(d_model, d_model) for name in self.route_names})
            self.gamma_q = nn.ParameterDict(
                {name: nn.Parameter(torch.tensor(float(gamma_init))) for name in self.route_names}
            )
            self.gamma_k = nn.ParameterDict(
                {name: nn.Parameter(torch.tensor(float(gamma_init))) for name in self.route_names}
            )
            self.gamma_v = nn.ParameterDict(
                {name: nn.Parameter(torch.tensor(float(gamma_init))) for name in self.route_names}
            )
            self.gamma_o = nn.ParameterDict(
                {name: nn.Parameter(torch.tensor(float(gamma_init))) for name in self.route_names}
            )
        else:
            base = SharedAttentionProjection(d_model)
            self.q_proj = nn.ModuleDict({name: copy.deepcopy(base.q_proj) for name in self.route_names})
            self.k_proj = nn.ModuleDict({name: copy.deepcopy(base.k_proj) for name in self.route_names})
            self.v_proj = nn.ModuleDict({name: copy.deepcopy(base.v_proj) for name in self.route_names})
            self.o_proj = nn.ModuleDict({name: copy.deepcopy(base.o_proj) for name in self.route_names})

    def q(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        return self._project(x, route_ids, "q")

    def k(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        return self._project(x, route_ids, "k")

    def v(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        return self._project(x, route_ids, "v")

    def o(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        return self._project(x, route_ids, "o")

    def _project(self, x: torch.Tensor, route_ids: torch.Tensor, kind: str) -> torch.Tensor:
        if self.init_mode == "shared_plus_delta":
            shared = getattr(self.shared, kind)(x)
            out = shared
            module_dict = getattr(self, f"delta_{kind}")
            gamma_dict = getattr(self, f"gamma_{kind}")
            for idx, name in enumerate(self.route_names):
                route_mask = (route_ids == idx).unsqueeze(-1).to(dtype=shared.dtype)
                if route_mask.any():
                    delta = module_dict[name](x).to(dtype=shared.dtype)
                    gamma = gamma_dict[name].to(device=x.device, dtype=shared.dtype)
                    out = out + gamma * delta * route_mask
            return out
        module_dict = getattr(self, f"{kind}_proj")
        out = torch.zeros_like(x)
        for idx, name in enumerate(self.route_names):
            route_mask = (route_ids == idx).unsqueeze(-1).to(dtype=x.dtype)
            if route_mask.any():
                out = out + module_dict[name](x).to(dtype=x.dtype) * route_mask
        return out
