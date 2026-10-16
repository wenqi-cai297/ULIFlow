from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from hoi_latent.inference.stage2.decode_render import object_points_world as decode_object_points_world

from .decoded_records import Stage2WorldMotionArrays
from .feature_preprocess import InterActFeaturePreprocessOutput, preprocess_interact_marker_bps_samples


DEFAULT_INTERACT_FEATURE_ARTIFACT_ROOT = Path("assets/evaluation/interact_markersbps")


class InterActFeatureEvaluatorUnavailable(RuntimeError):
    """Raised when official-artifact feature metrics cannot be emitted safely."""


@dataclass(frozen=True)
class InterActFeatureArtifactPaths:
    checkpoint: Path
    mean: Path
    std: Path
    bps_basis: Path
    state_dict: Path | None = None
    manifest: Path | None = None


@dataclass
class InterActFeatureArtifactBundle:
    paths: InterActFeatureArtifactPaths
    mean: np.ndarray
    std: np.ndarray
    bps_basis: np.ndarray
    checkpoint_state: dict[str, torch.Tensor]
    sha256: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    manifest_valid: bool = False


@dataclass
class InterActFeatureParityReport:
    preprocessing_max_abs_raw: float
    preprocessing_max_abs_normalized: float
    motion_loc_max_abs: float
    motion_loc_cosine: float
    text_loc_max_abs: float
    text_loc_cosine: float
    passed: bool
    golden_source: str = "unknown"
    fixture_sha256: dict[str, str] = field(default_factory=dict)
    golden_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocessing_max_abs_raw": self.preprocessing_max_abs_raw,
            "preprocessing_max_abs_normalized": self.preprocessing_max_abs_normalized,
            "motion_loc_max_abs": self.motion_loc_max_abs,
            "motion_loc_cosine": self.motion_loc_cosine,
            "text_loc_max_abs": self.text_loc_max_abs,
            "text_loc_cosine": self.text_loc_cosine,
            "passed": self.passed,
            "golden_source": self.golden_source,
            "fixture_sha256": dict(self.fixture_sha256),
            "golden_metadata": dict(self.golden_metadata),
        }


class InterActPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False) -> None:
        super().__init__()
        self.batch_first = bool(batch_first)
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class InterActMotionActorAgnosticEncoder(nn.Module):
    def __init__(
        self,
        *,
        nfeats: int = 231 + 128 * 3,
        vae: bool = True,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.vae = bool(vae)
        self.skel_embedding = nn.Linear(nfeats, latent_dim)
        if self.vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))
        self.sequence_pos_encoding = InterActPositionalEncoding(latent_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, features: torch.Tensor, lengths: Sequence[int] | torch.Tensor | None = None):
        if features.ndim != 3:
            raise ValueError(f"features must have shape [B,T,F], got {tuple(features.shape)}.")
        if lengths is None:
            lengths = [features.shape[1]] * features.shape[0]
        mask = lengths_to_mask(lengths, features.device, max_len=features.shape[1])
        batch_size = features.shape[0]
        x = self.skel_embedding(features.float()).permute(1, 0, 2)
        if self.vae:
            mu_token = self.mu_token.tile((batch_size,)).reshape(batch_size, -1)
            logvar_token = self.logvar_token.tile((batch_size,)).reshape(batch_size, -1)
            xseq = torch.cat((mu_token[None], logvar_token[None], x), dim=0)
            aug_mask = torch.cat((torch.ones((batch_size, 2), dtype=torch.bool, device=features.device), mask), dim=1)
        else:
            emb_token = self.emb_token.tile((batch_size,)).reshape(batch_size, -1)
            xseq = torch.cat((emb_token[None], x), dim=0)
            aug_mask = torch.cat((torch.ones((batch_size, 1), dtype=torch.bool, device=features.device), mask), dim=1)
        final = self.seqTransEncoder(self.sequence_pos_encoding(xseq), src_key_padding_mask=~aug_mask)
        if self.vae:
            mu, logvar = final[0], final[1]
            return torch.distributions.Normal(mu, logvar.exp().pow(0.5))
        return final[0]


class InterActDistilbertActorAgnosticTextEncoder(nn.Module):
    def __init__(
        self,
        modelpath: str = "distilbert-base-uncased",
        *,
        finetune: bool = False,
        vae: bool = True,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        tokenizer: Any | None = None,
        text_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.finetune = bool(finetune)
        if tokenizer is None or text_model is None:
            from transformers import AutoModel, AutoTokenizer
            from transformers import logging as transformers_logging

            transformers_logging.set_verbosity_error()
            tokenizer = AutoTokenizer.from_pretrained(modelpath)
            text_model = AutoModel.from_pretrained(modelpath)
        self.tokenizer = tokenizer
        self.text_model = text_model
        if not self.finetune:
            self.text_model.eval()
            for param in self.text_model.parameters():
                param.requires_grad_(False)
        hidden_size = int(self.text_model.config.hidden_size)
        self.projection = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, latent_dim))
        self.vae = bool(vae)
        if self.vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))
        self.sequence_pos_encoding = InterActPositionalEncoding(latent_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.finetune:
            self.text_model.eval()
        return self

    def forward(self, texts: Sequence[str]):
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask=True)
        x = self.projection(text_encoded.float())
        batch_size = x.shape[0]
        x = x.permute(1, 0, 2)
        if self.vae:
            mu_token = self.mu_token.tile((batch_size,)).reshape(batch_size, -1)
            logvar_token = self.logvar_token.tile((batch_size,)).reshape(batch_size, -1)
            xseq = torch.cat((mu_token[None], logvar_token[None], x), dim=0)
            aug_mask = torch.cat((torch.ones((batch_size, 2), dtype=torch.bool, device=x.device), mask), dim=1)
        else:
            emb_token = self.emb_token.tile((batch_size,)).reshape(batch_size, -1)
            xseq = torch.cat((emb_token[None], x), dim=0)
            aug_mask = torch.cat((torch.ones((batch_size, 1), dtype=torch.bool, device=x.device), mask), dim=1)
        final = self.seqTransEncoder(self.sequence_pos_encoding(xseq), src_key_padding_mask=~aug_mask)
        if self.vae:
            mu, logvar = final[0], final[1]
            return torch.distributions.Normal(mu, logvar.exp().pow(0.5))
        return final[0]

    def get_last_hidden_state(self, texts: Sequence[str], *, return_mask: bool = False):
        encoded = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        device = _module_device(self.text_model)
        encoded = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in dict(encoded).items()}
        output = self.text_model(**encoded)
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, encoded["attention_mask"].to(dtype=torch.bool)


class InterActFeatureEvaluator(nn.Module):
    def __init__(
        self,
        *,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
        bps_basis: np.ndarray | torch.Tensor,
        motion_encoder: InterActMotionActorAgnosticEncoder | None = None,
        text_encoder: InterActDistilbertActorAgnosticTextEncoder | None = None,
    ) -> None:
        super().__init__()
        self.motion_encoder = motion_encoder or InterActMotionActorAgnosticEncoder()
        self.text_encoder = text_encoder or InterActDistilbertActorAgnosticTextEncoder()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).flatten())
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).flatten())
        self.register_buffer("bps_basis", torch.as_tensor(bps_basis, dtype=torch.float32))
        if self.mean.shape != self.std.shape:
            raise ValueError(f"mean/std shapes must match, got {tuple(self.mean.shape)} and {tuple(self.std.shape)}.")
        if torch.any(self.std <= 0):
            raise ValueError("std must be strictly positive.")

    @torch.no_grad()
    def preprocess(self, samples: Sequence[Stage2WorldMotionArrays]) -> InterActFeaturePreprocessOutput:
        return preprocess_interact_marker_bps_samples(samples, mean=self.mean, std=self.std, bps_basis=self.bps_basis, device=self.mean.device)

    @torch.no_grad()
    def encode_motion(self, normalized_features: torch.Tensor, lengths: Sequence[int] | torch.Tensor) -> torch.Tensor:
        self.motion_encoder.eval()
        return self.motion_encoder(normalized_features.to(device=self.mean.device), lengths).loc

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        self.text_encoder.eval()
        return self.text_encoder(texts).loc


def default_interact_feature_artifact_paths(
    artifact_root: str | Path = DEFAULT_INTERACT_FEATURE_ARTIFACT_ROOT,
) -> InterActFeatureArtifactPaths:
    root = Path(artifact_root)
    return InterActFeatureArtifactPaths(
        checkpoint=root / "markersbps.ckpt",
        mean=root / "mean_train_markersbps.npy",
        std=root / "std_train_markersbps.npy",
        bps_basis=root / "bps_basis_set_128_1.npy",
        state_dict=root / "markersbps_state_dict.pt",
        manifest=root / "manifest.json",
    )


def load_interact_feature_artifacts(paths: InterActFeatureArtifactPaths) -> InterActFeatureArtifactBundle:
    has_state_dict = paths.state_dict is not None and paths.state_dict.exists()
    required = [paths.mean, paths.std, paths.bps_basis]
    if not has_state_dict:
        required.append(paths.checkpoint)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise InterActFeatureEvaluatorUnavailable(
            "InterAct official-artifact feature evaluator is unavailable; missing artifact(s): " + ", ".join(missing)
        )
    mean = np.load(paths.mean).astype(np.float32, copy=False)
    std = np.load(paths.std).astype(np.float32, copy=False)
    bps_basis = np.load(paths.bps_basis).astype(np.float32, copy=False)
    _validate_official_artifact_shapes(mean=mean, std=std, bps_basis=bps_basis)
    state_source = paths.state_dict if has_state_dict else paths.checkpoint
    try:
        payload = torch.load(state_source, map_location="cpu", weights_only=True)
    except Exception:
        try:
            payload = torch.load(state_source, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as exc:
            if state_source == paths.checkpoint:
                raise InterActFeatureEvaluatorUnavailable(
                    f"{paths.checkpoint} requires external Python modules to unpickle. "
                    "Create a repo-local pure state_dict artifact at "
                    f"{paths.state_dict or '<artifact_root>/markersbps_state_dict.pt'} "
                    "from the trusted official checkpoint, then rerun evaluation."
                ) from exc
            raise
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise ValueError(f"{paths.checkpoint} does not contain a state_dict mapping.")
    manifest: dict[str, Any] = {}
    manifest_valid = False
    if paths.manifest is not None and paths.manifest.exists():
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    sha = {
        "mean": sha256_file(paths.mean),
        "std": sha256_file(paths.std),
        "bps_basis": sha256_file(paths.bps_basis),
    }
    if paths.checkpoint.exists():
        sha["checkpoint"] = sha256_file(paths.checkpoint)
    if paths.state_dict is not None and paths.state_dict.exists():
        sha["state_dict"] = sha256_file(paths.state_dict)
    if paths.manifest is not None and paths.manifest.exists():
        sha["manifest"] = sha256_file(paths.manifest)
        _validate_interact_feature_artifact_manifest(manifest, sha)
        manifest_valid = True
    return InterActFeatureArtifactBundle(
        paths=paths,
        mean=mean,
        std=std,
        bps_basis=bps_basis,
        checkpoint_state={str(k): v for k, v in state.items()},
        sha256=sha,
        manifest=manifest,
        manifest_valid=manifest_valid,
    )


def build_interact_feature_evaluator_from_artifacts(
    bundle: InterActFeatureArtifactBundle,
    *,
    modelpath: str = "distilbert-base-uncased",
    strict: bool = True,
) -> InterActFeatureEvaluator:
    evaluator = InterActFeatureEvaluator(
        mean=bundle.mean,
        std=bundle.std,
        bps_basis=bundle.bps_basis,
        motion_encoder=InterActMotionActorAgnosticEncoder(nfeats=int(bundle.mean.reshape(-1).shape[0])),
        text_encoder=InterActDistilbertActorAgnosticTextEncoder(modelpath=modelpath),
    )
    text_state, motion_state = split_interact_markersbps_checkpoint_state(bundle.checkpoint_state)
    evaluator.text_encoder.load_state_dict(text_state, strict=strict)
    evaluator.motion_encoder.load_state_dict(motion_state, strict=strict)
    evaluator.eval()
    return evaluator


def split_interact_markersbps_checkpoint_state(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    text_state: dict[str, torch.Tensor] = {}
    motion_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("textencoder."):
            text_state[key[len("textencoder.") :]] = value
        elif key.startswith("motionencoder."):
            motion_state[key[len("motionencoder.") :]] = value
    if not text_state or not motion_state:
        raise ValueError("checkpoint state must contain textencoder.* and motionencoder.* keys.")
    return text_state, motion_state


def run_interact_feature_parity_check(
    evaluator: InterActFeatureEvaluator,
    *,
    parity_dir: str | Path,
    motion_max_abs_tol: float = 1e-5,
    motion_cosine_min: float = 0.999999,
    text_max_abs_tol: float = 1e-4,
    text_cosine_min: float = 0.99999,
    preprocess_max_abs_tol: float = 1e-5,
) -> InterActFeatureParityReport:
    root = Path(parity_dir)
    inputs_path = root / "fixture_inputs.npz"
    texts_path = root / "fixture_texts.json"
    golden_path = root / "golden_features.npz"
    metadata_path = root / "golden_metadata.json"
    missing = [str(path) for path in (inputs_path, texts_path, golden_path) if not path.exists()]
    if missing:
        raise InterActFeatureEvaluatorUnavailable("missing feature parity fixture file(s): " + ", ".join(missing))
    golden_metadata = _load_golden_metadata(metadata_path)
    golden_source = str(golden_metadata.get("golden_source", "unknown"))
    fixture_sha = {
        "fixture_inputs": sha256_file(inputs_path),
        "fixture_texts": sha256_file(texts_path),
        "golden_features": sha256_file(golden_path),
    }
    if metadata_path.exists():
        fixture_sha["golden_metadata"] = sha256_file(metadata_path)
    with np.load(inputs_path) as inputs:
        samples = _samples_from_fixture_inputs(inputs)
    texts = json.loads(texts_path.read_text(encoding="utf-8"))
    if not isinstance(texts, list) or len(texts) != len(samples):
        raise ValueError("fixture_texts.json must contain one text string per sample.")
    with np.load(golden_path) as golden:
        preprocess = evaluator.preprocess(samples)
        motion_loc = evaluator.encode_motion(preprocess.normalized_features, preprocess.lengths)
        text_loc = evaluator.encode_text([str(text) for text in texts])
        raw_max = _max_abs(preprocess.raw_features.cpu().numpy(), golden["raw_feature_615"])
        norm_max = _max_abs(preprocess.normalized_features.cpu().numpy(), golden["normalized_feature_615"])
        motion_max = _max_abs(motion_loc.cpu().numpy(), golden["motion_loc"])
        text_max = _max_abs(text_loc.cpu().numpy(), golden["text_loc"])
        motion_cos = _mean_cosine(motion_loc.cpu().numpy(), golden["motion_loc"])
        text_cos = _mean_cosine(text_loc.cpu().numpy(), golden["text_loc"])
    passed = (
        raw_max <= preprocess_max_abs_tol
        and norm_max <= preprocess_max_abs_tol
        and motion_max <= motion_max_abs_tol
        and motion_cos >= motion_cosine_min
        and text_max <= text_max_abs_tol
        and text_cos >= text_cosine_min
    )
    return InterActFeatureParityReport(
        preprocessing_max_abs_raw=raw_max,
        preprocessing_max_abs_normalized=norm_max,
        motion_loc_max_abs=motion_max,
        motion_loc_cosine=motion_cos,
        text_loc_max_abs=text_max,
        text_loc_cosine=text_cos,
        passed=passed,
        golden_source=golden_source,
        fixture_sha256=fixture_sha,
        golden_metadata=golden_metadata,
    )


def lengths_to_mask(lengths: Sequence[int] | torch.Tensor, device: torch.device, max_len: int | None = None) -> torch.Tensor:
    if isinstance(lengths, torch.Tensor):
        values = lengths.to(device=device, dtype=torch.long).flatten()
    else:
        values = torch.as_tensor(list(lengths), device=device, dtype=torch.long)
    if values.numel() == 0:
        raise ValueError("lengths must not be empty.")
    if torch.any(values <= 0):
        raise ValueError("all lengths must be positive.")
    limit = int(max_len) if max_len is not None else int(values.max().item())
    return torch.arange(limit, device=device).expand(values.numel(), limit) < values[:, None]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_official_artifact_shapes(*, mean: np.ndarray, std: np.ndarray, bps_basis: np.ndarray) -> None:
    mean_dim = int(np.asarray(mean).reshape(-1).shape[0])
    std_dim = int(np.asarray(std).reshape(-1).shape[0])
    if mean_dim != 231 + 128 * 3 or std_dim != 231 + 128 * 3:
        raise ValueError(f"official InterAct markers+BPS mean/std must have 615 dims, got {mean_dim} and {std_dim}.")
    if tuple(np.asarray(bps_basis).shape) != (128, 3):
        raise ValueError(f"official InterAct markers+BPS basis must have shape [128,3], got {np.asarray(bps_basis).shape}.")


def _validate_interact_feature_artifact_manifest(manifest: Mapping[str, Any], sha256: Mapping[str, str]) -> None:
    if not isinstance(manifest, Mapping):
        raise InterActFeatureEvaluatorUnavailable("InterAct feature artifact manifest must be a JSON object.")
    schema_version = manifest.get("schema_version")
    if schema_version not in (1, "1"):
        raise InterActFeatureEvaluatorUnavailable(
            f"InterAct feature artifact manifest schema_version must be 1, got {schema_version!r}."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise InterActFeatureEvaluatorUnavailable("InterAct feature artifact manifest must contain an 'artifacts' mapping.")
    for name, entry in artifacts.items():
        if isinstance(entry, Mapping):
            expected = entry.get("sha256")
        else:
            expected = entry
        if not isinstance(expected, str) or not expected:
            raise InterActFeatureEvaluatorUnavailable(f"manifest artifact {name!r} must provide a non-empty sha256.")
        if name not in sha256:
            raise InterActFeatureEvaluatorUnavailable(f"manifest references unknown artifact {name!r}.")
        if sha256[name] != expected:
            raise InterActFeatureEvaluatorUnavailable(
                f"manifest sha256 mismatch for {name!r}: expected {expected}, got {sha256[name]}."
            )


def _load_golden_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"golden_source": "unknown"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise InterActFeatureEvaluatorUnavailable(f"{path} must contain a JSON object.")
    metadata = dict(payload)
    metadata.setdefault("golden_source", "unknown")
    return metadata


def _samples_from_fixture_inputs(inputs: Mapping[str, np.ndarray]) -> list[Stage2WorldMotionArrays]:
    markers = np.asarray(inputs["human_markers_world"], dtype=np.float32)
    if "object_points_world" in inputs:
        object_points = np.asarray(inputs["object_points_world"], dtype=np.float32)
    else:
        canonical = np.asarray(inputs["object_points_canonical"], dtype=np.float32)
        rotmat = np.asarray(inputs["object_rotmat"], dtype=np.float32)
        trans = np.asarray(inputs["object_trans"], dtype=np.float32)
        object_points = np.stack(
            [decode_object_points_world(canonical[index], rotmat[index], trans[index]) for index in range(canonical.shape[0])],
            axis=0,
        )
    trans = np.asarray(inputs["object_trans"], dtype=np.float32)
    frame_mask = np.asarray(inputs["frame_mask"], dtype=bool) if "frame_mask" in inputs else np.ones(markers.shape[:2], dtype=bool)
    samples = []
    for index in range(markers.shape[0]):
        samples.append(
            Stage2WorldMotionArrays(
                human_markers=markers[index],
                object_points=object_points[index],
                frame_mask=frame_mask[index],
                object_trans=trans[index],
                object_rotmat=np.asarray(inputs["object_rotmat"], dtype=np.float32)[index] if "object_rotmat" in inputs else None,
            )
        )
    return samples


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))))


def _mean_cosine(actual: np.ndarray, expected: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64).reshape(actual.shape[0], -1)
    b = np.asarray(expected, dtype=np.float64).reshape(expected.shape[0], -1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.clip(denom, 1e-12, None)
    return float(((a * b).sum(axis=1) / denom).mean())
