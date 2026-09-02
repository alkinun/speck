"""Derive a compact long-document dataset from indexed packed token streams."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from speck.dataloader import PackedTokenSource, manifest_fingerprint
from speck.dataset import TokenShardWriter, derive_source_quotas, load_manifest, verify_shards


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _source_weights(parent, weights):
    if not isinstance(weights, dict) or not weights:
        raise ValueError("long-document source weights must be a non-empty object")
    available = {source["id"] for source in parent["sources"]}
    if set(weights) - available:
        raise ValueError("long-document source weights name an unknown packed source")
    if any(
        isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0
        for weight in weights.values()
    ):
        raise ValueError("long-document source weights must be positive numbers")
    if sum(weights.values()) != 100:
        raise ValueError("long-document source weights must sum to 100")
    return dict(weights)


def _selected_source(
    parent_directory,
    parent_source,
    output_directory,
    *,
    train_target,
    train_requested,
    train_reserve,
    validation_requested,
    minimum_tokens,
    shard_tokens,
    dedup_file,
    dedup_start,
):
    source_id = parent_source["id"]
    source_directory = output_directory / "sources" / source_id
    source_directory.mkdir(parents=True)
    streams = {
        split: PackedTokenSource(parent_directory, parent_source, split)
        for split in ("train", "val")
    }
    writers = {
        split: TokenShardWriter(source_directory, split, shard_tokens)
        for split in ("train", "val")
    }
    targets = {"train": train_target, "val": validation_requested}
    document_counts = {"train": 0, "val": 0}
    index_path = source_directory / "documents.jsonl"
    index_digest = hashlib.sha256()
    dedup_digest = hashlib.sha256()
    parent_index = parent_directory / parent_source["document_index"]["path"]
    with parent_index.open(encoding="utf-8") as source_index, index_path.open("wb") as output_index:
        for line in source_index:
            record = json.loads(line)
            split = record["split"]
            token_count = record["end_token"] - record["start_token"]
            if token_count < minimum_tokens or writers[split].total_tokens >= targets[split]:
                continue
            tokens = streams[split].read(
                record["start_token"],
                token_count,
                dtype=np.uint16,
            )
            start = writers[split].total_tokens
            writers[split].write(tokens)
            selected = {
                **record,
                "start_token": start,
                "end_token": start + token_count,
                "tokens": token_count,
            }
            encoded = (json.dumps(selected, sort_keys=True, separators=(",", ":")) + "\n").encode()
            output_index.write(encoded)
            index_digest.update(encoded)
            dedup_hash = bytes.fromhex(record["dedup_hash"])
            dedup_file.write(dedup_hash)
            dedup_digest.update(dedup_hash)
            document_counts[split] += 1
            if all(writers[name].total_tokens >= target for name, target in targets.items()):
                break
    missing = [
        f"{split} {writers[split].total_tokens:,}/{target:,}"
        for split, target in targets.items()
        if writers[split].total_tokens < target
    ]
    if missing:
        raise ValueError(f"long-document source {source_id} is too small: {', '.join(missing)}")
    for writer in writers.values():
        writer.finish()
    documents = sum(document_counts.values())
    dedup_end = dedup_start + documents * 16
    splits = {}
    for split, writer in writers.items():
        requested = train_requested if split == "train" else validation_requested
        target = train_target if split == "train" else validation_requested
        splits[split] = {
            "requested_tokens": requested,
            "preparation_target_tokens": target,
            "tokens": writer.total_tokens,
            "reserve_tokens": train_reserve if split == "train" else 0,
            "overshoot_tokens": writer.total_tokens - target,
            "documents": document_counts[split],
            "shards": [
                {**shard, "path": f"sources/{source_id}/{shard['path']}"}
                for shard in writer.shards
            ],
        }
    return {
        **parent_source,
        "filters": {**parent_source.get("filters", {}), "min_tokens": minimum_tokens},
        "documents": documents,
        "document_index": {
            "path": f"sources/{source_id}/documents.jsonl",
            "records": documents,
            "bytes": index_path.stat().st_size,
            "sha256": index_digest.hexdigest(),
        },
        "dedup_journal": {
            "start_byte": dedup_start,
            "end_byte": dedup_end,
            "hashes": documents,
            "sha256": dedup_digest.hexdigest(),
        },
        "splits": splits,
    }


def derive_long_document_dataset(
    parent_directory,
    output_directory,
    *,
    source_weights,
    requested_train_tokens,
    validation_tokens_per_source,
    minimum_document_tokens,
    shard_tokens,
    maximum_loader_microbatch_tokens,
    restart=False,
):
    """Copy complete long-document spans into a new verified packed dataset."""

    parent_directory = Path(parent_directory).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    if parent_directory == output_directory:
        raise ValueError("long-document output must differ from its parent dataset")
    parent = load_manifest(parent_directory)
    source_weights = _source_weights(parent, source_weights)
    requested_train_tokens = _positive_integer(requested_train_tokens, "requested train tokens")
    validation_tokens_per_source = _positive_integer(
        validation_tokens_per_source, "validation tokens per source"
    )
    minimum_document_tokens = _positive_integer(
        minimum_document_tokens, "minimum document tokens"
    )
    shard_tokens = _positive_integer(shard_tokens, "shard tokens")
    maximum_loader_microbatch_tokens = _positive_integer(
        maximum_loader_microbatch_tokens, "maximum loader microbatch tokens"
    )
    phase = {"end_tokens": requested_train_tokens, "weights": source_weights}
    quotas, phases = derive_source_quotas(
        list(source_weights), {"phases": [phase]}, requested_train_tokens
    )
    reserve = 2 * maximum_loader_microbatch_tokens
    estimated_tokens = requested_train_tokens + len(source_weights) * (
        reserve + validation_tokens_per_source + minimum_document_tokens
    )
    required_bytes = 2 * estimated_tokens
    free_bytes = shutil.disk_usage(output_directory.parent).free
    if free_bytes < required_bytes:
        raise OSError(
            f"long-document derivation requires {required_bytes:,} bytes but only "
            f"{free_bytes:,} are free"
        )

    building = output_directory.with_name(output_directory.name + ".building")
    if output_directory.exists():
        raise FileExistsError(f"long-document dataset already exists: {output_directory}")
    if building.exists():
        if not restart:
            raise FileExistsError(f"incomplete long-document build exists: {building}")
        shutil.rmtree(building)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    (building / "sources").mkdir(parents=True)
    parent_sources = {source["id"]: source for source in parent["sources"]}
    summaries = []
    dedup_path = building / "dedup_hashes.bin"
    try:
        with dedup_path.open("wb") as dedup_file:
            dedup_start = 0
            for source_id in source_weights:
                summary = _selected_source(
                    parent_directory,
                    parent_sources[source_id],
                    building,
                    train_target=quotas[source_id] + reserve,
                    train_requested=quotas[source_id],
                    train_reserve=reserve,
                    validation_requested=validation_tokens_per_source,
                    minimum_tokens=minimum_document_tokens,
                    shard_tokens=shard_tokens,
                    dedup_file=dedup_file,
                    dedup_start=dedup_start,
                )
                summaries.append(summary)
                dedup_start = summary["dedup_journal"]["end_byte"]
        splits = {
            split: {
                "requested_tokens": (
                    requested_train_tokens
                    if split == "train"
                    else validation_tokens_per_source * len(summaries)
                ),
                "tokens": sum(source["splits"][split]["tokens"] for source in summaries),
                "documents": sum(source["splits"][split]["documents"] for source in summaries),
            }
            for split in ("train", "val")
        }
        manifest = {
            "format": "speck_packed_tokens",
            "format_version": parent["format_version"],
            "dtype": parent["dtype"],
            "requested_train_tokens": requested_train_tokens,
            "validation_tokens_per_source": validation_tokens_per_source,
            "mixture": {"phases": phases, "source_quotas": quotas},
            "preparation": {
                "kind": "long_document_derivation",
                "parent_directory": str(parent_directory),
                "parent_manifest": manifest_fingerprint(parent),
                "minimum_document_tokens": minimum_document_tokens,
                "shard_tokens": shard_tokens,
                "maximum_loader_microbatch_tokens": maximum_loader_microbatch_tokens,
                "train_reserve_tokens_per_source": reserve,
                "disk_preflight": {
                    "required_bytes": required_bytes,
                    "free_bytes": free_bytes,
                },
            },
            "dedup": {
                "normalization": "NFKC+lower+whitespace",
                "hash": "blake2b-128",
                "scope": "global",
                "path": dedup_path.name,
                "accepted_hashes": sum(source["documents"] for source in summaries),
                "sha256": _sha256(dedup_path),
                "collision_policy": "128-bit collisions are treated as duplicates",
            },
            "tokenizer": parent["tokenizer"],
            "documents": sum(source["documents"] for source in summaries),
            "sources": summaries,
            "splits": splits,
        }
        manifest_path = building / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        load_manifest(building)
        verify_shards(building, manifest)
        os.replace(building, output_directory)
        return manifest
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
