# ULIFlow

ULIFlow generates variable-length human-object interaction motion from text and object geometry. This repository contains the core Stage 1 MoT autoencoder, latent whitening and decoder post-training, the Stage 2 conditional flow model, dataset-conditioned inference, and the evaluation code needed to inspect generated motion. It is a clean code release: no dataset, pretrained weights, checkpoints, generated samples, W&B credentials, or third-party baseline implementations are included.

## Requirements

- Linux, Python 3.10, and a CUDA-compatible PyTorch installation for training. The supplied environment pins PyTorch 2.7.1 with CUDA 12.8 wheels; adapt the CUDA wheel to your driver if needed.
- The processed InterAct motion-962 dataset with text, object sample points, and object BPS features. Obtain the dataset and any required processing tools under their respective terms. The expected layout is `data/InterAct_official_combination/<dataset>/sequences_canonical/<sequence>/motion.npy`, alongside `text.txt` and `object.npz`, plus `<dataset>/objects/<object>/sample_points.npy` and `<dataset>/objects_bps/<object>/<object>_1024.npy`.
- Access to `google/flan-t5-xl` through Hugging Face for Stage 2. Its weights are downloaded separately by Transformers.

From the repository root:

```bash
conda env create -f environment.yml
conda activate uliflow
python -m pytest tests -q
```

Set `data.root` in `configs/stage1/data_interact_motion962_all_v0.yaml` to your processed InterAct directory. For a machine-specific path, make a `*_local.yaml` copy; local config files and all run outputs are ignored by Git. The published configs default to offline W&B logging. Set `--wandb-mode online` explicitly if you intend to upload runs.

## Train

The following commands reproduce the released full-MoT configuration. Run them in order from the repository root. Choose `--num_processes` for the GPUs actually available; changing batch size or accumulation changes the optimization setup. The commands below use the values in the configs unless an override is shown.

```bash
DATA_CONFIG=configs/stage1/data_interact_motion962_all_v0.yaml
AE_RUN=runs/ablations/stage1_architecture/full_mot_reference_v0
POST_RUN=runs/ablations/stage1_architecture/full_mot_reference_v0_post_jitter_human_object_smooth
WHITE=runs/training/stage1/full_mot_reference_whitening.pt
STAGE2_RUN=runs/training/stage2/stage2_ff1_full_mot_post_jitter_flan_t5_xl_v0

accelerate launch --multi_gpu --num_processes 3 \
  scripts/ablations/stage1_architecture/train_stage1_architecture_ablation.py \
  --data-config "$DATA_CONFIG" \
  --train-config configs/ablations/stage1_architecture/train_full_mot_reference_v0.yaml

python scripts/diagnostics/stage1_latent/verify_latent_whitening_on_ckpt.py \
  --run-dir "$AE_RUN" --ckpt final.pt --fit-batches 512 \
  --allow-rank-deficient --fail-on-verification-error --out "$WHITE"

accelerate launch --multi_gpu --num_processes 3 \
  scripts/training/stage1/train_decoder_post_jitter.py \
  --data-config "$DATA_CONFIG" \
  --train-config configs/ablations/stage1_architecture/train_full_mot_reference_post_jitter_human_object_smooth_v0.yaml \
  --source-run "$AE_RUN" --source-ckpt final.pt --whitening "$WHITE"

accelerate launch --multi_gpu --num_processes 3 \
  scripts/training/stage2/train_rectified_flow.py \
  --data-config "$DATA_CONFIG" \
  --train-config configs/stage2/train_rectified_flow_full_mot_post_jitter_v0.yaml \
  --source-run "$POST_RUN" --source-ckpt best_val.pt --whitening "$WHITE"
```

The whitening command fits on training records and verifies the decoder round trip before writing the artifact. It exits without an artifact if verification fails. Stage 2 freezes the Stage 1 source and learns a flow in the whitened latent space. Run directories contain the resolved configuration, metadata, logs, and checkpoints.

## Generate

This entry point takes the text and object geometry from one InterAct record, then samples the requested duration. It uses the record's encoded latent **layout only** to build the fixed inference template; future motion values are not fed to the flow model. The output `.npz` contains world-space human markers, object points and rigid pose, and a frame mask.

```bash
python scripts/inference/sample.py \
  --data-config "$DATA_CONFIG" \
  --train-config configs/stage2/train_rectified_flow_full_mot_post_jitter_v0.yaml \
  --source-run "$POST_RUN" --source-ckpt best_val.pt \
  --stage2-run "$STAGE2_RUN" --stage2-ckpt final.pt \
  --whitening "$WHITE" --split val --frames 256 \
  --output runs/inference/example_256.npz
```

The script samples a dataset condition, not a free-form prompt supplied on the command line. No model weights are shipped with this repository; train first or supply your own compatible checkpoints.

## Evaluate

To generate a small local set of decoded records and motion diagnostics without an external feature evaluator:

```bash
python scripts/evaluation/stage2/evaluate_flow_forcing_quantitative.py \
  --data-config "$DATA_CONFIG" \
  --train-config configs/stage2/train_rectified_flow_full_mot_post_jitter_v0.yaml \
  --stage2-run "$STAGE2_RUN" --stage2-ckpt final.pt \
  --source-run "$POST_RUN" --source-ckpt best_val.pt \
  --whitening "$WHITE" --stage1-anchor-mode source \
  --eval-protocol eval_128_dev --split val --num-conditions 8 \
  --feature-evaluator-mode disabled --run-name local_eval
```

FID, retrieval precision, and other official-comparable feature metrics additionally require the InterAct Markers+BPS evaluator artifacts, which are not redistributed here. Place them under `assets/evaluation/interact_markersbps/`, use `--feature-evaluator-mode official_artifact`, and run the evaluator's parity and split checks before reporting those metrics. The local diagnostic command above is **not** a reproduction of the paper's official metric table.

## Scope

`src/hoi_latent/` contains reusable model, data, loss, training, inference, and evaluation code; `scripts/` contains runnable entry points. `tests/` covers the released pipeline. Research-only ablations, visualization polishing, third-party adapters, paper assets, and private experimental outputs are intentionally omitted. Keep downloaded data, evaluator artifacts, credentials, and all generated results out of Git.
