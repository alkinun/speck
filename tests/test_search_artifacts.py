import json

import pytest

from speck.search.artifacts import (
    ArtifactEdge,
    ArtifactManifest,
    ArtifactStore,
    file_digest,
)


def test_artifact_store_deduplicates_and_verifies_bytes(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    first = store.put_bytes("worker_result", b"result")
    second = store.put_bytes("worker_result", b"result")
    assert first == second
    assert store.read_bytes(first) == b"result"
    assert store.verify(first)


def test_artifact_store_streams_files(tmp_path):
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"checkpoint")
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put_file("checkpoint", source)
    assert artifact.digest == file_digest(source)
    assert store.read_bytes(artifact) == b"checkpoint"


def test_artifact_store_detects_tampering(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put_bytes("worker_result", b"result")
    store.path(artifact).write_bytes(b"changed")
    with pytest.raises(ValueError, match="size does not match"):
        store.verify(artifact)


def test_artifact_manifest_records_lineage(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    parent = store.put_json("worker_input", {"trial": 1})
    child = store.put_json("worker_result", {"loss": 2.0})
    manifest = ArtifactManifest(
        (parent, child),
        (ArtifactEdge(parent.digest, child.digest, "produced"),),
    )
    artifact = store.put_manifest(manifest)
    stored = json.loads(store.read_bytes(artifact))
    assert stored == manifest.export()
    assert len(manifest.digest) == 64


def test_artifact_manifest_rejects_unknown_lineage(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put_bytes("worker_input", b"input")
    with pytest.raises(ValueError, match="reference manifest"):
        ArtifactManifest(
            (artifact,),
            (ArtifactEdge(artifact.digest, "0" * 64, "produced"),),
        )
