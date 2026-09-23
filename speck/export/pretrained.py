"""Load pinned Hugging Face weights into a local Speck model."""

import json
from dataclasses import fields
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from speck.data.loader import manifest_fingerprint
from speck.model.architecture import ArchitectureConfig
from speck.provenance.io import file_sha256 as _sha256
from speck.training.checkpoint import is_assistant_checkpoint, load_metadata, load_model


def checkpoint_tokenizer_fingerprint(metadata):
    """Recover a native checkpoint's tokenizer identity without trusting its current path."""

    resolved = metadata["resolved"]
    if is_assistant_checkpoint(metadata):
        return resolved["tokenizer"]["base_fingerprint"]
    expected = resolved.get("tokenizer_fingerprint")
    if expected is not None:
        return expected
    manifest_path = Path(resolved["data_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest_fingerprint(manifest) != metadata["manifest"]:
        raise ValueError("original packed manifest differs from the checkpoint")
    return manifest["tokenizer"]["fingerprint"]


def native_pretrained_source(checkpoint_dir, step):
    """Bind the exact weights and metadata needed for a local SFT parent."""

    directory = Path(checkpoint_dir).expanduser().resolve()
    load_metadata(directory, step)
    return {
        "checkpoint_dir": str(directory),
        "step": step,
        "model_sha256": _sha256(directory / f"model_{step:06d}.pt"),
        "metadata_sha256": _sha256(directory / f"metadata_{step:06d}.json"),
    }


def pretrained_source_matches(provenance, settings):
    """Compare configured parent identity without requiring it again during SFT resume."""

    if "checkpoint_dir" in settings:
        required = {"checkpoint_dir", "step", "model_sha256", "metadata_sha256"}
        if set(settings) != required or type(settings["step"]) is not int or settings["step"] < 0:
            return False
        normalized = {
            **settings,
            "checkpoint_dir": str(Path(settings["checkpoint_dir"]).expanduser().resolve()),
        }
        return {key: provenance.get(key) for key in required} == normalized
    return {key: provenance.get(key) for key in ("repo", "revision", "filename")} == {
        "filename": "model.safetensors",
        **settings,
    }


def load_pretrained(
    model,
    repo=None,
    revision=None,
    filename="model.safetensors",
    *,
    checkpoint_dir=None,
    step=None,
    model_sha256=None,
    metadata_sha256=None,
    tokenizer_fingerprint=None,
):
    """Load pinned Hub weights or an explicitly identified completed native checkpoint."""

    if checkpoint_dir is not None:
        if repo is not None or revision is not None or filename != "model.safetensors":
            raise ValueError("native and Hub pretrained sources cannot be combined")
        settings = {
            "checkpoint_dir": str(Path(checkpoint_dir).expanduser().resolve()),
            "step": step,
            "model_sha256": model_sha256,
            "metadata_sha256": metadata_sha256,
        }
        identity = native_pretrained_source(checkpoint_dir, step)
        if identity != settings:
            raise ValueError("native pretrained checkpoint identity mismatch")
        metadata = load_metadata(checkpoint_dir, step)
        if (
            tokenizer_fingerprint is not None
            and checkpoint_tokenizer_fingerprint(metadata) != tokenizer_fingerprint
        ):
            raise ValueError("pretrained tokenizer does not match the SFT tokenizer")
        if ArchitectureConfig.from_dict(metadata["config"]).settings() != model.config.settings():
            raise ValueError("pretrained model architecture does not match the experiment")
        model.load_state_dict(load_model(checkpoint_dir, step, "cpu"), strict=True)
        return {"kind": "native_checkpoint", **identity}
    if any(value is not None for value in (step, model_sha256, metadata_sha256)):
        raise ValueError("native checkpoint fields require checkpoint_dir")

    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("pretrained revision must be a full commit hash")
    config_path = hf_hub_download(repo, "config.json", revision=revision)
    if tokenizer_fingerprint is not None:
        tokenizer_path = hf_hub_download(repo, "tokenizer.model", revision=revision)
        if _sha256(tokenizer_path) != tokenizer_fingerprint:
            raise ValueError("pretrained tokenizer does not match the SFT tokenizer")
    weights_path = hf_hub_download(repo, filename, revision=revision)
    remote = json.loads(Path(config_path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(ArchitectureConfig)}
    remote_config = ArchitectureConfig.from_dict(
        {key: value for key, value in remote.items() if key in allowed}
    )
    if remote_config.settings() != model.config.settings():
        raise ValueError("pretrained model architecture does not match the experiment")

    state = load_file(weights_path, device="cpu")
    if "lm_head.weight" not in state:
        state["lm_head.weight"] = state["embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    if model.lm_head.weight is not model.embed_tokens.weight:
        raise RuntimeError("pretrained model embeddings are not tied")
    return {
        "repo": repo,
        "revision": revision,
        "filename": filename,
        "config_sha256": _sha256(config_path),
        "weights_sha256": _sha256(weights_path),
    }
