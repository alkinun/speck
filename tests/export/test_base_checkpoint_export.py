import json

import pytest
import torch

from speck.data.loader import manifest_fingerprint
from speck.export.checkpoint import checkpoint_tokenizer, load_source
from speck.training.checkpoint import save


def test_base_export_loads_checkpoint_with_provenance(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    metadata = {
        "step": 3,
        "training_phase": "base",
        "config": {},
        "resolved": {"steps": 3},
        "partial": False,
    }
    save(checkpoints, 3, {"weight": torch.tensor(3.0)}, {}, metadata)

    state, loaded, source, provenance = load_source(checkpoints, None)
    assert state["weight"].item() == 3.0
    assert loaded == metadata
    assert source == "step 3"
    assert provenance["type"] == "checkpoint"
    assert provenance["checkpoint"]["step"] == 3


@pytest.mark.parametrize(
    "metadata",
    (
        {"step": 2, "training_phase": "base", "resolved": {"steps": 3}, "partial": True},
        {"step": 2, "training_phase": "base", "resolved": {"steps": 3}, "partial": False},
        {"step": 2, "training_phase": "base", "resolved": {}, "partial": False},
    ),
)
def test_base_export_rejects_partial_or_nonfinal_checkpoint(tmp_path, metadata):
    checkpoints = tmp_path / "checkpoints"
    save(checkpoints, 2, {"weight": torch.tensor(2.0)}, {}, metadata)

    with pytest.raises(ValueError, match="non-partial resolved final"):
        load_source(checkpoints, 2)


@pytest.mark.parametrize("legacy", [False, True])
def test_base_export_tokenizer_identity_survives_relocation_and_rejects_drift(
    tmp_path, monkeypatch, legacy
):
    class Tokenizer:
        def fingerprint(self):
            return "original"

    def load(*, directory):
        assert directory == str(tmp_path)
        return Tokenizer()

    monkeypatch.setattr("speck.export.checkpoint.get_tokenizer", load)
    manifest = {"tokenizer": {"fingerprint": "original"}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    metadata = {
        "manifest": manifest_fingerprint(manifest),
        "resolved": {"tokenizer": {"directory": "old-machine"}, "data_dir": str(tmp_path)},
    }
    if not legacy:
        metadata["resolved"]["tokenizer_fingerprint"] = "original"
        path.unlink()  # New checkpoints need neither packed data nor the old machine's path.
    assert checkpoint_tokenizer(metadata, tmp_path).fingerprint() == "original"
    if legacy:
        path.write_text(json.dumps({"tokenizer": {"fingerprint": "other"}}))
        message = "manifest differs"
    else:
        metadata["resolved"]["tokenizer_fingerprint"] = "other"
        message = "tokenizer differs"
    with pytest.raises(ValueError, match=message):
        checkpoint_tokenizer(metadata, tmp_path)


@pytest.mark.parametrize("cached", [True, False])
def test_offline_export_template_uses_only_staged_cache(tmp_path, monkeypatch, cached):
    from huggingface_hub import HfApi, constants
    from huggingface_hub.errors import LocalEntryNotFoundError

    from speck.export.checkpoint import (
        TEMPLATE_FILES,
        TEMPLATE_REPO,
        TEMPLATE_REVISION,
        template_snapshot,
    )

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("offline export attempted a Hub API call")

    monkeypatch.setattr(HfApi, "repo_info", forbidden)
    monkeypatch.setattr(HfApi, "list_repo_tree", forbidden)
    snapshot = (
        tmp_path / ("models--" + TEMPLATE_REPO.replace("/", "--")) / "snapshots" / TEMPLATE_REVISION
    )
    if cached:
        snapshot.mkdir(parents=True)
        for name in TEMPLATE_FILES:
            (snapshot / name).write_text(name)
        assert template_snapshot() == snapshot
    else:
        with pytest.raises(LocalEntryNotFoundError):
            template_snapshot()
