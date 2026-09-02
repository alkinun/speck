import json

from speck import dataset
from speck.dataloader import packed_loader
from speck.long_data import derive_long_document_dataset


class FakeTokenizer:
    vocab_size = 256
    bos_id = 1
    eos_id = 2

    def encode_batch(self, texts, bos=False, eos=False):
        return [
            ([self.bos_id] if bos else [])
            + [3 + byte % 200 for byte in text.encode()]
            + ([self.eos_id] if eos else [])
            for text in texts
        ]

    def fingerprint(self):
        return "long-data-test"


def test_long_document_derivation_preserves_only_complete_filtered_documents(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        dataset,
        "_is_validation_document",
        lambda content, seed, fraction: content.startswith("val-"),
    )
    parent = tmp_path / "parent"
    source = {
        "id": "papers",
        "repo": "test/papers",
        "revision": "abc",
        "tree_path": "data",
        "content_column": "text",
        "metadata_columns": {},
        "filters": {},
    }
    documents = [
        {"content": f"val-{index}-" + "v" * 80} for index in range(2)
    ] + [{"content": f"train-{index}-" + "t" * 80} for index in range(10)]
    dataset.prepare_dataset(
        sources=[source],
        mixture={"phases": [{"end_tokens": 400, "weights": {"papers": 100}}]},
        requested_train_tokens=400,
        validation_tokens_per_source=100,
        validation_fraction=0.1,
        filtering={"min_chars": 0, "max_chars": 1_000},
        dedup={
            "normalization": "NFKC+lower+whitespace",
            "hash": "blake2b-128",
            "scope": "global",
        },
        shards={"tokens": 100, "maximum_loader_microbatch_tokens": 0},
        output_dir=parent,
        tokenizer=FakeTokenizer(),
        document_iterators={"papers": documents},
        check_disk=False,
    )

    output = tmp_path / "long"
    manifest = derive_long_document_dataset(
        parent,
        output,
        source_weights={"papers": 100},
        requested_train_tokens=200,
        validation_tokens_per_source=50,
        minimum_document_tokens=50,
        shard_tokens=80,
        maximum_loader_microbatch_tokens=10,
    )

    assert dataset.load_manifest(output) == manifest
    dataset.verify_shards(output, manifest)
    records = [
        json.loads(line)
        for line in (output / "sources" / "papers" / "documents.jsonl").read_text().splitlines()
    ]
    assert records
    assert all(record["tokens"] >= 50 for record in records)
    assert manifest["preparation"]["parent_manifest"]
    inputs, targets, _ = next(
        packed_loader(FakeTokenizer(), 1, 16, "train", data_dir=output, device="cpu")
    )
    assert inputs.shape == targets.shape == (1, 16)
