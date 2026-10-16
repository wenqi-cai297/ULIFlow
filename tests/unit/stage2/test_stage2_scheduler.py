from __future__ import annotations

import pytest
import torch

from hoi_latent.training.stage2 import (
    build_stage2_optimizer_from_config,
    build_stage2_scheduler_from_config,
)


def test_stage2_scheduler_warmup_constant_cooldown_reaches_final_lr() -> None:
    model = torch.nn.Linear(2, 2)
    cfg = {
        "training": {
            "stage2": {
                "max_steps": 6,
                "optimizer": {"type": "adamw", "lr": 1.0e-4},
                "scheduler": {
                    "type": "warmup_constant_cooldown",
                    "warmup_steps": 2,
                    "cooldown_start_step": 4,
                    "cooldown_steps": 2,
                    "cooldown_type": "cosine",
                    "final_lr": 1.0e-5,
                },
            }
        }
    }
    optimizer = build_stage2_optimizer_from_config(cfg, model)
    scheduler = build_stage2_scheduler_from_config(cfg, optimizer)
    assert scheduler is not None
    metadata = getattr(scheduler, "phase1_metadata")
    assert metadata["max_steps"] == 6
    assert metadata["min_lr"] == pytest.approx(1.0e-5)
    assert metadata["cooldown_start_step"] == 4

    observed = []
    for _ in range(6):
        observed.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert observed == pytest.approx([5.0e-5, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-5])
