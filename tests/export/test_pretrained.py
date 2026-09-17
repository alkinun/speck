import pytest
import torch

from speck.export.pretrained import (
    load_pretrained,
    native_pretrained_source,
    pretrained_source_matches,
)
from speck.model import SpeckForCausalLM
from speck.training.checkpoint import save
from tests.evaluation.test_infer import tiny_config


def test_native_sft_parent_loads_without_hub_or_optimizer(tmp_path, monkeypatch):
    source = SpeckForCausalLM(tiny_config())
    source.init_weights()
    save(
        tmp_path,
        2,
        source.state_dict(),
        {},
        {
            "step": 2,
            "config": source.config.settings(),
            "resolved": {"tokenizer_fingerprint": "original"},
        },
    )
    parent = native_pretrained_source(tmp_path, 2)
    (tmp_path / "optimizer_000002.pt").write_bytes(
        b"optimizer is unnecessary for SFT initialization"
    )
    target = SpeckForCausalLM(tiny_config())
    monkeypatch.setattr(
        "speck.export.pretrained.hf_hub_download",
        lambda *a, **k: pytest.fail("unexpected Hub access"),
    )
    with pytest.raises(ValueError, match="tokenizer does not match"):
        load_pretrained(target, **parent, tokenizer_fingerprint="different-same-vocab-size")
    provenance = load_pretrained(target, **parent, tokenizer_fingerprint="original")
    assert pretrained_source_matches(provenance, parent)
    assert all(
        torch.equal(value, target.state_dict()[key]) for key, value in source.state_dict().items()
    )
    assert not pretrained_source_matches(provenance, {**parent, "step": 3})
    # Continuing SFT can verify its declared parent without reopening a pruned base checkpoint.
    (tmp_path / "model_000002.pt").unlink()
    assert pretrained_source_matches(provenance, parent)


def test_native_sft_parent_rejects_changed_or_incomplete_weights(tmp_path):
    model = SpeckForCausalLM(tiny_config())
    save(tmp_path, 2, model.state_dict(), {}, {"step": 2, "config": model.config.settings()})
    parent = native_pretrained_source(tmp_path, 2)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_pretrained(model, **{**parent, "model_sha256": "0" * 64})
    with pytest.raises(ValueError, match="cannot be combined"):
        load_pretrained(model, repo="unexpected/repo", **parent)
    (tmp_path / "complete_000002").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_pretrained(model, **parent)
