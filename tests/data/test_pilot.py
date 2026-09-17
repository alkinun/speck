import hashlib
import json

import pytest

from speck.data.pilot import cached_documents, code_documents, pack_candidates
from speck.provenance.io import atomic_json, file_sha256


class Tokenizer:
    def fingerprint(self):
        return "frozen-tokenizer"

    def encode_batch(self, texts, **kwargs):
        return [[1, *range(len(text)), 2] for text in texts]


def test_cached_document_identity_and_selected_token_counts(tmp_path):
    text = "A retained document."
    digest = hashlib.sha256(text.encode()).hexdigest()
    row = {"text": text, "content_id": "doc-1", "released_content_sha256": digest}
    source, index, parent, manifest = [
        tmp_path / name
        for name in ("source.jsonl", "documents.jsonl", "parent.json", "manifest.json")
    ]
    source.write_text(json.dumps(row) + "\n")
    index.write_text(
        json.dumps(
            {
                "ordinal": 0,
                "content_id": "doc-1",
                "released_content_sha256": digest,
                "token_count": 7,
            }
        )
        + "\n"
    )
    atomic_json(parent, {"status": "retained"})
    stock = {
        "status": "complete_document_token_cache_not_training_view",
        "plan": {
            "tokenizer": {"sha256": Tokenizer().fingerprint()},
            "parent_manifest": {"path": str(parent), "sha256": file_sha256(parent)},
            "input": {"path": str(source), "sha256": file_sha256(source)},
        },
        "documents": {"path": index.name, "sha256": file_sha256(index)},
    }
    atomic_json(manifest, stock)
    assert list(cached_documents(manifest, file_sha256(manifest), Tokenizer())) == [(row, 7)]
    source.write_text(source.read_text() + "changed")
    with pytest.raises(ValueError, match="changed"):
        list(cached_documents(manifest, file_sha256(manifest), Tokenizer()))


def test_code_reader_binds_manifest_and_payload_before_using_text(tmp_path):
    text = "print(42)"
    source, path = tmp_path / "records.jsonl", tmp_path / "manifest.json"
    source.write_text(
        json.dumps(
            {"text": text, "released_content_sha256": hashlib.sha256(text.encode()).hexdigest()}
        )
        + "\n"
    )
    manifest = {
        "language": "Python",
        "output": {"path": source.name, "sha256": file_sha256(source)},
    }
    atomic_json(path, manifest)
    receipt = {
        "status": "content_acquisition_complete",
        "units": [
            {
                "manifest": manifest,
                "manifest_identity": {"path": str(path), "sha256": file_sha256(path)},
            }
        ],
    }
    opened = []
    result = list(code_documents(receipt, Tokenizer(), "Python", opened))
    assert result[0][1] == len(text) + 2
    assert opened == [{"path": str(source), "sha256": file_sha256(source)}]
    source.write_text("changed")
    with pytest.raises(ValueError, match="payload changed"):
        list(code_documents(receipt, Tokenizer(), "Python", []))


def test_packing_order_is_independent_of_acquisition_language_grouping(tmp_path):
    from speck.data.pilot import ordered_documents

    rows = [
        {
            "text": str(i),
            "content_id": str(i),
            "language": language,
            "released_content_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
        }
        for i, language in enumerate(["Python"] * 10 + ["Rust"] * 10)
    ]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps(row) + "\n" for row in rows))
    b.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    left, right = list(ordered_documents(a, 42)), list(ordered_documents(b, 42))
    assert left == right
    assert {row["content"] for row in left} == {str(i) for i in range(20)}
    assert len({row["metadata"]["language"] for row in left[:10]}) == 2


@pytest.mark.parametrize("changed", ["data", "tokenizer", "inputs"])
def test_packing_rejects_configuration_drift_after_selection(tmp_path, changed):
    experiment, work = tmp_path / "experiment", tmp_path / "work"
    identities = {}
    for name in ("data", "tokenizer", "inputs"):
        path = experiment / f"{name}.json"
        atomic_json(path, {"original": True})
        identities[name] = file_sha256(path)
    atomic_json(work / "selection.json", {"contract": {"configs": identities}})
    atomic_json(experiment / f"{changed}.json", {"original": False})

    with pytest.raises(ValueError, match="contract changed before packing"):
        pack_candidates(experiment, work)
