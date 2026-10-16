"""Integration check for the whitening transform on the REAL Stage1 checkpoint.

The unit tests (tests/unit/stage1/test_stage1_latent_whitening.py) confirm the ZCA math on
synthetic data. They CANNOT confirm two things that need the real frozen model + data:

  A. on the actual converged z_rec (cond# ~52 at 100k), whitening drives cond# -> ~1 / Cov -> I.
  B. the round trip is LOSSLESS THROUGH THE FROZEN DECODER:
        decode(unwhiten(whiten(z_rec)))  ==  decode(z_rec)        (design doc acceptance #7)
     i.e. the decoded MOTION is unchanged (not just the latent).

Fits W on the TRAIN split (no eval leakage), verifies on a held-out EVAL batch. Doubles as the
real whitening fit (optional --out writes the artifact).

See README.md for the whitening fit command and dataset setup.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import torch

from hoi_latent.config.io import load_yaml
from hoi_latent.config.stage1_anchor_overrides import apply_stage1_anchor_override
from hoi_latent.models.stage1.autoencoder.latent_whitening import (
    Stage1LatentWhiteningTransform,
    fit_stage1_latent_whitening_from_values,
    save_stage1_latent_whitening_stats,
)
from hoi_latent.training.stage1.phase1_warmup import (
    build_stage1_phase1_dataloaders,
    infer_frame_mask,
    load_or_compute_stage1_normalizers,
    move_to_device,
)
from hoi_latent.training.stage1.source_autoencoder import (
    build_stage1_source_autoencoder_from_config,
    decode_stage1_source_prediction,
    encode_stage1_source_latent,
    load_stage1_source_component_state_dicts,
)


def encode(model, batch):
    return encode_stage1_source_latent(model, batch, return_diagnostics=False, validate=False)


def decode_world(model, latent, batch, sn):
    fm = infer_frame_mask(batch)
    c = decode_stage1_source_prediction(
        model,
        latent,
        batch,
        target_frame_mask=fm,
        return_diagnostics=False,
        validate=False,
    )
    pr = sn.denormalize_group_torch("human", "pos_rootrel", c.human["pos_rootrel"].cpu(), require_stats=True)
    rt = sn.denormalize_group_torch("human", "root", c.human["root"].cpu(), require_stats=True)
    return pr + rt[:, :, None, :]


def cov_stats(z):  # z [M, D] -> cond#, offdiag_energy
    c = torch.cov(z.T)
    if z.shape[1] == 1:
        return 1.0, 0.0
    ev = torch.linalg.eigvalsh(c).clamp_min(1e-12)
    cond = float(ev.max() / ev.min())
    d = c.shape[0]
    off = c - torch.diag(torch.diagonal(c))
    return cond, float(off.pow(2).sum() / (d * (d - 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fit-batches", type=int, default=24, help="train batches to accumulate for covariance fit")
    ap.add_argument("--out", default=None, help="optional path to save the fitted whitening artifact")
    ap.add_argument(
        "--fail-on-verification-error",
        action="store_true",
        help="exit non-zero and do not write --out unless every verification check passes",
    )
    ap.add_argument(
        "--allow-rank-deficient",
        action="store_true",
        help=(
            "accept epsilon-regularized ZCA when the raw covariance is rank deficient and "
            "the active covariance subspace is whitened"
        ),
    )
    ap.add_argument(
        "--max-active-post-condition",
        type=float,
        default=1.05,
        help="maximum fit-set post-whitening condition number in the active covariance subspace",
    )
    ap.add_argument(
        "--max-decoded-roundtrip-cm",
        type=float,
        default=1.0e-3,
        help="maximum decoded MPJPE difference for the whitening round trip, in centimeters",
    )
    ap.add_argument("--anchor-mode", choices=["hybrid", "absolute_grid"], default=None)
    ap.add_argument("--anchor-stride", type=int, default=None)
    ap.add_argument("--anchor-grid-origin", type=int, default=None)
    args = ap.parse_args()
    if args.max_active_post_condition < 1.0:
        ap.error("--max-active-post-condition must be at least 1.0")
    if args.max_decoded_roundtrip_cm <= 0.0:
        ap.error("--max-decoded-roundtrip-cm must be positive")

    torch.set_grad_enabled(False)
    run = Path(args.run_dir)
    cfg = load_yaml(str(run / "resolved_config.yaml"))
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    anchor_override = apply_stage1_anchor_override(
        cfg,
        anchor_mode=args.anchor_mode,
        anchor_stride=args.anchor_stride,
        anchor_grid_origin=args.anchor_grid_origin,
    )
    dev = torch.device(args.device)

    normalizer, sn, _ = load_or_compute_stage1_normalizers(cfg)
    train_loader, eval_loader = build_stage1_phase1_dataloaders(
        cfg, normalizer=normalizer, structured_normalizer=sn
    )
    model = build_stage1_source_autoencoder_from_config(cfg).to(dev)
    ck = torch.load(str(run / "checkpoints" / args.ckpt), map_location="cpu", weights_only=False)
    load_stage1_source_component_state_dicts(model, ck["components"])
    model.eval()
    print(f"[whiten-verify] step={ck['global_step']} fit_batches={args.fit_batches}")
    if anchor_override:
        print(f"  anchor_override={anchor_override}")
    print()

    # --- fit W on TRAIN split ---
    pool = []
    it = iter(train_loader)
    for _ in range(args.fit_batches):
        try:
            batch = move_to_device(next(it), dev)
        except StopIteration:
            break
        latent = encode(model, batch)
        pool.append(latent.z_rec.float()[latent.mask.bool()].cpu())
    values = torch.cat(pool, 0)
    stats = fit_stage1_latent_whitening_from_values(values)
    transform = Stage1LatentWhiteningTransform(stats).to(dev)
    diag = transform.diagnostics()
    active_count = int(diag["active_dimension_count"])
    floored_count = int(diag["floored_dimension_count"])
    rank_deficient = bool(diag["rank_deficient"])
    active_post_cond = float(diag["active_post_condition_number"])
    print(f"fit on M={values.shape[0]} train tokens, d_z={stats.z_dim}")
    print(f"  transform.diagnostics: pre_cond={float(diag['pre_condition_number']):.2f} -> "
          f"post_cond={float(diag['post_condition_number']):.3f}  "
          f"post_eff_rank={float(diag['post_effective_rank']):.2f}")
    print(f"  raw_eff_rank={float(diag['raw_effective_rank']):.2f}  "
          f"active_dims={active_count}/{stats.z_dim}  floored_dims={floored_count}  "
          f"active_post_cond={active_post_cond:.3f}\n")

    # --- verify ---
    # A. held-out generalization: accumulate eval cov over several batches (stable cond#)
    ev_pool, first_batch = [], None
    it = iter(eval_loader)
    for bi in range(8):
        try:
            batch = move_to_device(next(it), dev)
        except StopIteration:
            break
        latent = encode(model, batch)
        ev_pool.append(transform.whiten(latent.z_rec.float(), latent.mask.bool())[latent.mask.bool()].cpu())
        if bi == 0:
            first_batch = (batch, latent)
    eval_values = torch.cat(ev_pool, 0)
    post_cond_ho, post_off_ho = cov_stats(eval_values)
    active_mask_cpu = diag["active_dimension_mask"].detach().cpu().bool()
    active_basis_cpu = transform.eigvecs.detach().cpu()[:, active_mask_cpu]
    if active_count > 0:
        active_post_cond_ho, active_post_off_ho = cov_stats(eval_values @ active_basis_cpu)
    else:
        active_post_cond_ho, active_post_off_ho = math.inf, math.inf
    fit_post_cond = float(diag["post_condition_number"])
    print("=== A. covariance after whitening ===")
    print(f"  fit (train) post cond#   = {fit_post_cond:.3f}             (W correctness; want ->1)")
    print(f"  fit active-subspace cond# = {active_post_cond:.3f}  "
          f"({active_count} active, {floored_count} epsilon-floored directions)")
    print(f"  held-out eval post cond# = {post_cond_ho:.3f}  offdiag={post_off_ho:.4f}   "
          f"(generalization; >1 = train/eval gap + finite sample)")
    print(f"  held-out active cond#    = {active_post_cond_ho:.3f}  offdiag={active_post_off_ho:.4f}")
    full_space_ok = (
        not rank_deficient
        and math.isfinite(fit_post_cond)
        and fit_post_cond < 1.01
    )
    rank_deficient_ok = (
        args.allow_rank_deficient
        and rank_deficient
        and active_count > 0
        and math.isfinite(active_post_cond)
        and active_post_cond <= args.max_active_post_condition
    )
    A_ok = full_space_ok or rank_deficient_ok
    if full_space_ok:
        whitening_acceptance = "full_rank_zca"
    elif rank_deficient_ok:
        whitening_acceptance = "regularized_rank_deficient_zca"
    else:
        whitening_acceptance = "rejected"

    # B. latent round-trip + DECODED-motion round-trip (lossless through frozen decoder)
    batch, latent = first_batch
    mask = latent.mask.bool()
    z_rec = latent.z_rec.float()
    z_white = transform.whiten(z_rec, mask)
    z_rt = transform.unwhiten(z_white, mask)
    latent_err = (z_rt[mask] - z_rec[mask]).abs().max().item()
    motion_orig = decode_world(model, latent, batch, sn)
    motion_rt = decode_world(model, dataclasses.replace(latent, z_rec=z_rt), batch, sn)
    hmask = batch["masks"].get("human")
    hmask = hmask.bool().cpu() if hmask is not None else infer_frame_mask(batch)[:, :, None].cpu().expand(motion_orig.shape[:3])
    decode_diff_cm = float((motion_orig - motion_rt).norm(dim=-1)[hmask].mean() * 100)
    print("\n=== B. round trip: latent + DECODED motion (frozen decoder) ===")
    print(f"  max |z_rt - z_rec| (valid)       = {latent_err:.2e}")
    print(f"  decoded MPJPE(orig vs roundtrip) = {decode_diff_cm:.6f} cm   (want ~0 = lossless)")
    B_ok = latent_err < 1e-3 and decode_diff_cm < args.max_decoded_roundtrip_cm

    # C. invalid tokens stay exactly zero (guard fully-valid batches)
    n_inv = int((~mask).sum())
    inv_max = z_white[~mask].abs().max().item() if n_inv > 0 else 0.0
    inv_ok = inv_max == 0.0
    print(f"\n=== C. invalid tokens (n={n_inv}) stay zero: max={inv_max} ===")

    print("\n=== VERDICT ===")
    print(f"  A whitening ({whitening_acceptance}): {'PASS' if A_ok else 'FAIL'}")
    print(f"  B decode-lossless:        {'PASS' if B_ok else 'FAIL'}")
    print(f"  C invalid stays zero:     {'PASS' if inv_ok else 'FAIL'}")
    all_ok = A_ok and B_ok and inv_ok
    print(f"  ALL: {'PASS' if all_ok else 'FAIL'}")

    if args.out and (all_ok or not args.fail_on_verification_error):
        verification = {
            "status": "passed" if all_ok else "failed",
            "whitening_acceptance": whitening_acceptance,
            "allow_rank_deficient": bool(args.allow_rank_deficient),
            "checks": {
                "whitening": bool(A_ok),
                "decode_roundtrip": bool(B_ok),
                "invalid_tokens_zero": bool(inv_ok),
            },
            "thresholds": {
                "max_full_post_condition": 1.01,
                "max_active_post_condition": float(args.max_active_post_condition),
                "max_latent_roundtrip_error": 1.0e-3,
                "max_decoded_roundtrip_cm": float(args.max_decoded_roundtrip_cm),
            },
            "fit": {
                "full_post_condition": fit_post_cond,
                "active_post_condition": active_post_cond,
                "raw_effective_rank": float(diag["raw_effective_rank"]),
                "post_effective_rank": float(diag["post_effective_rank"]),
                "active_dimension_count": active_count,
                "floored_dimension_count": floored_count,
                "active_dimension_fraction": float(diag["active_dimension_fraction"]),
            },
            "held_out": {
                "full_post_condition": post_cond_ho,
                "full_offdiag_energy": post_off_ho,
                "active_post_condition": active_post_cond_ho,
                "active_offdiag_energy": active_post_off_ho,
            },
            "roundtrip": {
                "max_latent_error": latent_err,
                "decoded_mpjpe_cm": decode_diff_cm,
            },
        }
        save_stage1_latent_whitening_stats(
            stats, args.out,
            metadata={"source_checkpoint": str(run / "checkpoints" / args.ckpt),
                      "source_run_dir": str(run), "global_step": int(ck["global_step"]),
                      "fit_tokens": int(values.shape[0]),
                      "anchor_override": anchor_override,
                      "verification": verification},
        )
        print(f"\nwrote whitening artifact -> {args.out}")
    elif args.out:
        print(f"\nrefusing to write failed whitening artifact -> {args.out}")

    if args.fail_on_verification_error and not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
