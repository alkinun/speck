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


def test_exports_ship_the_code_licence_and_tokenizer_attribution(tmp_path):
    from speck.export.transformers import copy_licenses

    copy_licenses(tmp_path)
    assert (tmp_path / "LICENSE").read_text().startswith("MIT License")
    assert "Mistral-7B-v0.1" in (tmp_path / "LICENSE.tokenizer").read_text()
