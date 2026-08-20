import json

import speck.hub as hub


def test_checkpoint_commit_payload(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    tokenizer_dir = tmp_path / "tokenizer"
    checkpoint_dir.mkdir()
    tokenizer_dir.mkdir()
    (checkpoint_dir / "model_000003.pt").write_bytes(b"model")
    (checkpoint_dir / "optimizer_000003.pt").write_bytes(b"optimizer")
    (tokenizer_dir / "tokenizer.model").write_bytes(b"tokenizer")
    (tokenizer_dir / "tokenizer_metadata.json").write_text("{}")
    monkeypatch.setenv("speck_base_dir", str(tmp_path))

    operations = []

    class Operation:
        def __init__(self, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj
            operations.append(self)

    class Api:
        def create_repo(self, *args, **kwargs):
            pass

        def create_commit(self, **kwargs):
            self.commit = kwargs
            return type("commit", (), {"commit_url": "https://huggingface.co/test/commit/1"})()

    monkeypatch.setattr(hub, "CommitOperationAdd", Operation)
    monkeypatch.setattr(hub, "HfApi", Api)
    metadata = {
        "resolved": {"model": {"max_position_embeddings": 4096, "model_type": "speck"}},
        "step": 3,
    }
    url = hub.upload("owner/model", checkpoint_dir, 3, metadata, optimizer=True)
    assert url.endswith("/commit/1")
    paths = {operation.path_in_repo for operation in operations}
    assert paths == {
        "pytorch_model.bin",
        "config.json",
        "training/state.json",
        "training/optimizer.pt",
        "configuration_speck.py",
        "modeling_speck.py",
        "tokenizer.model",
        "tokenizer_config.json",
        "artifacts/tokenizer_metadata.json",
    }
    config = next(operation for operation in operations if operation.path_in_repo == "config.json")
    assert json.loads(config.path_or_fileobj) == {
        "max_position_embeddings": 4096,
        "model_type": "speck",
    }
