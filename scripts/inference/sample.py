from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hoi_latent.config.stage1_anchor_overrides import apply_stage1_anchor_override
from hoi_latent.inference.stage2 import (
    Stage2FlowForcingRolloutConfig,
    Stage2GuidanceConfig,
    decode_stage2_z_white_windows,
    extract_stage2_inference_latent_template_spec,
    prediction_to_world_markers_and_object_points,
    sample_stage2_flow_forcing_rollout,
    stitch_stage1_predictions_preferred_slot,
)
from hoi_latent.models.stage1.autoencoder.latent_whitening import load_stage1_latent_whitening_transform
from hoi_latent.models.stage2 import build_stage2_tri_stream_mot_model_from_config
from hoi_latent.training.stage1.phase1_warmup import (
    build_stage1_phase1_dataloaders,
    load_or_compute_stage1_normalizers,
    move_to_device,
)
from hoi_latent.training.stage1.source_autoencoder import (
    build_stage1_source_autoencoder_from_config,
    encode_stage1_source_latent,
    load_stage1_source_component_state_dicts,
    load_stage1_source_config,
)
from hoi_latent.training.stage2.rectified_flow import (
    _load_stage2_state_without_frozen_clip,
    load_stage2_rectified_flow_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a human-object sequence from a trained ULIFlow model.")
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-ckpt", default="best_val.pt")
    parser.add_argument("--stage2-run", type=Path, required=True)
    parser.add_argument("--stage2-ckpt", default="final.pt")
    parser.add_argument("--whitening", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-block", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    _, _, config = load_stage2_rectified_flow_training_config(
        args.data_config, args.train_config, overrides={"batch_size": 1, "num_workers": 0}
    )
    train_config = config["training"]["stage2"]
    anchor_override = apply_stage1_anchor_override(
        config,
        anchor_mode=train_config.get("stage1_anchor_mode"),
        anchor_stride=train_config.get("stage1_anchor_stride"),
        anchor_grid_origin=train_config.get("stage1_anchor_grid_origin"),
    )
    config["loader"]["batch_size"] = 1
    config["loader"]["num_workers"] = 0
    normalizer, structured_normalizer, _ = load_or_compute_stage1_normalizers(config)
    train_loader, val_loader = build_stage1_phase1_dataloaders(
        config, normalizer=normalizer, structured_normalizer=structured_normalizer
    )
    batch = move_to_device(next(iter(train_loader if args.split == "train" else val_loader)), device)
    condition_batch = {
        "meta": {"text": batch["meta"]["text"]},
        "geometry": {
            key: batch["geometry"][key]
            for key in ("object_bps", "object_bps_mask", "has_object_bps")
            if key in batch["geometry"]
        },
    }

    source_path = args.source_run / "checkpoints" / args.source_ckpt
    source_config = load_stage1_source_config(args.source_run, fallback_config=config)
    apply_stage1_anchor_override(
        source_config,
        anchor_mode=train_config.get("stage1_anchor_mode"),
        anchor_stride=train_config.get("stage1_anchor_stride"),
        anchor_grid_origin=train_config.get("stage1_anchor_grid_origin"),
    )
    stage1 = build_stage1_source_autoencoder_from_config(source_config)
    load_stage1_source_component_state_dicts(
        stage1, torch.load(source_path, map_location="cpu", weights_only=False)["components"]
    )
    stage1.to(device).eval()

    stage2 = build_stage2_tri_stream_mot_model_from_config(config)
    checkpoint = torch.load(
        args.stage2_run / "checkpoints" / args.stage2_ckpt,
        map_location="cpu",
        weights_only=False,
    )
    _load_stage2_state_without_frozen_clip(stage2, checkpoint["stage2"])
    shadow = checkpoint.get("ema", {}).get("shadow", {})
    for name, parameter in stage2.named_parameters():
        if name in shadow:
            parameter.data.copy_(shadow[name].to(dtype=parameter.dtype))
    stage2.to(device).eval()
    whitening = load_stage1_latent_whitening_transform(args.whitening, map_location="cpu").to(device)
    whitening.eval()

    with torch.inference_mode():
        latent = encode_stage1_source_latent(stage1, batch)
        digest = hashlib.sha256()
        with source_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        spec = extract_stage2_inference_latent_template_spec(
            latent,
            window_frames=int(config["data"]["window_size"]),
            anchor_stride_frames=int(
                source_config["model"]["stage1"].get("token_compressor", {})
                .get("temporal", {}).get("anchor_stride", 4)
            ),
            block_size_anchors=int(config["training"]["stage2"]["flow_forcing"]["block_size_anchors"]),
            source_stage1_checkpoint_sha256=digest.hexdigest(),
            stage1_anchor_override=anchor_override,
        )
        rollout = sample_stage2_flow_forcing_rollout(
            stage2,
            batch=condition_batch,
            template_spec=spec,
            requested_frames=args.frames,
            guidance=Stage2GuidanceConfig(enabled=False),
            solver="euler",
            seed=args.seed,
            config=Stage2FlowForcingRolloutConfig(steps_per_block=args.steps_per_block),
        )
        decoded = decode_stage2_z_white_windows(
            stage1_decoder=stage1.decoder,
            whitening=whitening,
            rollout=rollout,
            template_spec=spec,
            requested_frames=args.frames,
        )
        prediction = stitch_stage1_predictions_preferred_slot(
            window_predictions=decoded.predictions,
            decode_window_starts=decoded.decode_window_starts,
            preferred_decode_window_start=rollout.preferred_decode_window_start,
            preferred_local_slot=rollout.preferred_local_slot,
            requested_frames=args.frames,
            block_size_frames=spec.block_size_anchors * spec.anchor_stride_frames,
        )
        world = prediction_to_world_markers_and_object_points(
            prediction,
            structured_normalizer=structured_normalizer,
            object_points_local=batch["geometry"]["object_points_local"],
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        human_markers=world.human_markers,
        object_points=world.object_points,
        object_trans=world.object_trans,
        object_rotmat=world.object_rotmat,
        frame_mask=world.frame_mask,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
