"""atomic document-index upgrade for persisted format-one packed datasets."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from speck.dataloader import manifest_fingerprint
from speck.dataset import load_manifest
from speck.search.segments import load_document_index


def _write_bytes_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _split_boundaries(data_dir, split, manifest, chunk_bytes=8 * 1024 * 1024):
    bos = manifest["tokenizer"]["bos_token_id"]
    eos = manifest["tokenizer"]["eos_token_id"]
    bos_positions = []
    eos_positions = []
    split_offset = 0
    for shard in manifest["splits"][split]["shards"]:
        path = data_dir / shard["path"]
        expected_bytes = shard["tokens"] * np.dtype("<u2").itemsize
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(f"invalid packed token shard: {path}")
        digest = hashlib.sha256()
        shard_offset = 0
        with path.open("rb") as source:
            while raw := source.read(chunk_bytes):
                digest.update(raw)
                values = np.frombuffer(raw, dtype="<u2")
                bos_positions.append(
                    np.flatnonzero(values == bos).astype(np.int64)
                    + split_offset
                    + shard_offset
                )
                eos_positions.append(
                    np.flatnonzero(values == eos).astype(np.int64)
                    + split_offset
                    + shard_offset
                )
                shard_offset += len(values)
        if shard_offset != shard["tokens"]:
            raise ValueError(f"packed shard token count does not match: {path}")
        if digest.hexdigest() != shard["sha256"]:
            raise ValueError(f"packed shard checksum mismatch: {path}")
        split_offset += shard["tokens"]
    if split_offset != manifest["splits"][split]["tokens"]:
        raise ValueError(f"packed {split} token count does not match")
    starts = np.concatenate(bos_positions) if bos_positions else np.empty(0, np.int64)
    ends = np.concatenate(eos_positions) if eos_positions else np.empty(0, np.int64)
    if len(starts) != len(ends):
        raise ValueError(f"packed {split} bos and eos counts do not match")
    if not len(starts):
        raise ValueError(f"packed {split} split contains no documents")
    if starts[0] != 0 or ends[-1] != split_offset - 1:
        raise ValueError(f"packed {split} boundaries do not cover the split")
    if np.any(starts > ends) or np.any(ends[:-1] + 1 != starts[1:]):
        raise ValueError(f"packed {split} document boundaries are malformed")
    return starts, ends + 1


def upgrade_document_index(data_dir):
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = load_manifest(data_dir)
    if manifest.get("format_version") == 2:
        records = load_document_index(data_dir, manifest)
        return {
            "documents": len(records),
            "index": str((data_dir / manifest["document_index"]["path"]).resolve()),
            "sha256": manifest["document_index"]["sha256"],
            "upgraded": False,
        }
    if manifest.get("format_version") != 1:
        raise ValueError("document-index upgrade requires packed format one")
    backup = data_dir / "manifest.v1.json"
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError("format-one manifest backup does not match")
    else:
        _write_bytes_atomic(backup, original)
    dataset_digest = manifest_fingerprint(manifest)
    index_path = data_dir / "documents.jsonl"
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    index_digest = hashlib.sha256()
    records = 0
    source = json.dumps("legacy_packed")
    try:
        with temporary.open("wb") as output:
            for split in ("train", "val"):
                starts, ends = _split_boundaries(data_dir, split, manifest)
                for start, end in zip(starts, ends):
                    start = int(start)
                    end = int(end)
                    identity = hashlib.sha256(
                        f"{dataset_digest}:{split}:{start}:{end}".encode()
                    ).hexdigest()
                    line = (
                        f'{{"content_hash":"{identity}","end_token":{end},'
                        f'"score":null,"source":{source},"split":"{split}",'
                        f'"start_token":{start}}}\n'
                    ).encode()
                    output.write(line)
                    index_digest.update(line)
                    records += 1
            output.flush()
            os.fsync(output.fileno())
        if records != manifest.get("documents"):
            raise ValueError("recovered document count does not match the manifest")
        os.replace(temporary, index_path)
        upgraded = dict(manifest)
        upgraded["format_version"] = 2
        upgraded["document_index"] = {
            "path": index_path.name,
            "records": records,
            "sha256": index_digest.hexdigest(),
            "identity": "verified_dataset_split_token_range",
        }
        encoded = json.dumps(upgraded, indent=2, sort_keys=True).encode()
        _write_bytes_atomic(manifest_path, encoded)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "documents": records,
        "index": str(index_path.resolve()),
        "sha256": index_digest.hexdigest(),
        "upgraded": True,
    }
