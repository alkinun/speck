"""Load pinned Hugging Face weights into a local Speck model."""

import hashlib
import json
from dataclasses import fields
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from speck.architecture import ArchitectureConfig


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_pretrained(model, repo, revision, filename="model.safetensors"):
    """Load an immutable Hub model revision and return its provenance."""

    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("pretrained revision must be a full commit hash")
    config_path = hf_hub_download(repo, "config.json", revision=revision)
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
