"""Pack a ladder experiment's corpus from preprocessed sources and their family partition.

Every source is a preprocessed, firewall-excluded text file whose documents the joint family
graph assigned to whole-family buckets. Train-bucket documents may be trained on,
development-bucket documents form validation, and final or held documents are never read into a
corpus. Each source is streamed in SHA-256(seed:content) order, so validation selection does not
depend on the mixture and every arm sharing a source validates on the same documents.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from speck.config import load_experiment
from speck.data.dataset import prepare_dataset, resolve_data_dir, verify_shards
from speck.provenance.io import atomic_json, file_sha256
from speck.tokenization.tokenizer import get_tokenizer

INPUTS_FORMAT = "speck_ladder_inputs"
SPLITS = {"train": "train", "development": "val"}


def _verified(entry, description):
    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"{description} changed: {path}")
    return path


def load_inputs(path):
    inputs = json.loads(Path(path).read_text())
    if inputs.get("format") != INPUTS_FORMAT or inputs.get("format_version") != 1:
        raise ValueError("unsupported ladder inputs")
    ids = [source["id"] for source in inputs["sources"]]
    if len(ids) != len(set(ids)):
        raise ValueError("ladder input source ids must be unique")
    return inputs


def partition_buckets(partitions, source_id):
    """Map one source's content hashes to their family bucket."""
    buckets = {}
    with Path(partitions).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["source"] == source_id:
                buckets[row["released_content_sha256"]] = row["candidate_partition"]
    if not buckets:
        raise ValueError(f"the partition assigns no documents to {source_id}")
    return buckets


def family_documents(text, buckets, seed, counts):
    """Yield train and development documents of one source in seeded hash order."""
    positions = []
    with Path(text).open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            content = json.loads(raw)["released_content_sha256"]
            priority = hashlib.sha256(f"{seed}:{content}".encode()).digest()
            positions.append((priority, offset))
        positions.sort()
        for ordinal, (_, offset) in enumerate(positions):
            handle.seek(offset)
            row = json.loads(handle.readline())
            content = row["released_content_sha256"]
            if hashlib.sha256(row["text"].encode()).hexdigest() != content:
                raise ValueError("document text differs from its content hash")
            bucket = buckets.get(content)
            if bucket is None:
                raise ValueError("a document has no family assignment")
            counts[bucket] += 1
            if bucket not in SPLITS:
                continue
            yield {
                "content": row["text"],
                "row": ordinal,
                "split": SPLITS[bucket],
                "metadata": {"content_id": row.get("content_id"), "bucket": bucket},
            }


def prepare(experiment, inputs_path):
    """Pack the experiment's data.json from the bound inputs and write a receipt beside it."""
    experiment = Path(experiment)
    configs = load_experiment(experiment, "data", "tokenizer")
    data = configs["data"]
    inputs = load_inputs(inputs_path)
    partitions = _verified(inputs["partitions"], "family partition")
    by_id = {source["id"]: source for source in inputs["sources"]}
    missing = [source["id"] for source in data["sources"] if source["id"] not in by_id]
    if missing:
        raise ValueError(f"sources without bound inputs: {missing}")
    counts = {source["id"]: Counter() for source in data["sources"]}
    iterators = {}
    for source in data["sources"]:
        bound = by_id[source["id"]]
        text = _verified(bound["text"], f"{source['id']} text")
        buckets = partition_buckets(partitions, bound.get("partition_source", source["id"]))
        iterators[source["id"]] = family_documents(
            text, buckets, data["seed"], counts[source["id"]]
        )
    try:
        manifest = prepare_dataset(
            **data, tokenizer=get_tokenizer(**configs["tokenizer"]), document_iterators=iterators
        )
    finally:
        for stream in iterators.values():
            stream.close()
    output = resolve_data_dir(data.get("output_dir"), data.get("output_name"))
    verify_shards(output, manifest)
    receipt = {
        "format": "speck_ladder_data",
        "format_version": 1,
        "experiment": str(experiment),
        "data_config_sha256": file_sha256(experiment / "data.json"),
        "inputs": {"path": str(inputs_path), "sha256": file_sha256(inputs_path)},
        "manifest": {
            "path": str(output / "manifest.json"),
            "sha256": file_sha256(output / "manifest.json"),
        },
        "data_order": "per-source SHA-256(seed:released_content_sha256) ascending",
        "split_rule": "train bucket to train, development bucket to validation, others never read",
        "documents_streamed_by_bucket": {key: dict(value) for key, value in counts.items()},
        "training_admitted": False,
    }
    atomic_json(output / "ladder-data.json", receipt)
    return receipt
