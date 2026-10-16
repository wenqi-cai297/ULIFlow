from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from hoi_latent.models.stage1.autoencoder.latent_bottleneck import Stage1LatentFieldOutput
from hoi_latent.models.stage1.decoder.readout_heads import Stage1CoarsePrediction
from hoi_latent.models.stage1.tokenizer_types import TokenizerOutput


@dataclass
class Stage1ArchitectureAblationOutput:
    latent: Stage1LatentFieldOutput
    prediction: Stage1CoarsePrediction
    aux: dict[str, Any] = field(default_factory=dict)
    tokenizer_outputs: dict[str, TokenizerOutput] | None = None

    @property
    def coarse(self) -> Stage1CoarsePrediction:
        return self.prediction

    @property
    def refined(self) -> Stage1CoarsePrediction:
        return self.prediction

    @property
    def human(self) -> dict[str, torch.Tensor]:
        return self.prediction.human

    @property
    def object(self) -> dict[str, torch.Tensor]:
        return self.prediction.object

    @property
    def masks(self) -> dict[str, torch.Tensor]:
        return self.prediction.masks
