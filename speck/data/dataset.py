"""Stream, deduplicate, tokenize, and pack source-separated training data."""

import hashlib
import json
import math
import os
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import HfApi

from speck.data.acquisition import (
    _dataset_url as _dataset_url,
)
from speck.data.acquisition import (
    _detect_language as _detect_language,
)
from speck.data.acquisition import (
    _download_file as _download_file,
)
from speck.data.acquisition import (
    _is_string_type as _is_string_type,
)
from speck.data.acquisition import (
    _metadata_value as _metadata_value,
)
from speck.data.acquisition import (
    _py3langid_identifier as _py3langid_identifier,
)
from speck.data.acquisition import (
    _score_passes as _score_passes,
)
from speck.data.acquisition import (
    _shuffle_seed as _shuffle_seed,
)
from speck.data.acquisition import (
    _validate_parquet_schema as _validate_parquet_schema,
)
from speck.data.acquisition import (
    discover_source_files as discover_source_files,
)
from speck.data.acquisition import (
    iter_jsonl_gzip_documents as iter_jsonl_gzip_documents,
)
from speck.data.acquisition import (
    iter_parquet_documents as iter_parquet_documents,
)
from speck.data.acquisition import (
    iter_source_file_documents as iter_source_file_documents,
)
from speck.data.configuration import (
    _DEDUP_BYTES as _DEDUP_BYTES,
)
from speck.data.configuration import (
    _DEDUP_SETTINGS as _DEDUP_SETTINGS,
)
from speck.data.configuration import (
    _LANGUAGE_DETECTORS as _LANGUAGE_DETECTORS,
)
from speck.data.configuration import (
    _MAX_DOCUMENT_CHARACTERS as _MAX_DOCUMENT_CHARACTERS,
)
from speck.data.configuration import (
    _MAX_TOKENIZER_BATCH_CHARACTERS as _MAX_TOKENIZER_BATCH_CHARACTERS,
)
from speck.data.configuration import (
    _MAX_TOKENIZER_DOCUMENTS as _MAX_TOKENIZER_DOCUMENTS,
)
from speck.data.configuration import (
    _MIN_INDEX_DEDUP_HEADROOM_BYTES as _MIN_INDEX_DEDUP_HEADROOM_BYTES,
)
from speck.data.configuration import (
    _RAW_SHARD_ALLOWANCE_BYTES as _RAW_SHARD_ALLOWANCE_BYTES,
)
from speck.data.configuration import (
    _SOURCE_FIELDS as _SOURCE_FIELDS,
)
from speck.data.configuration import (
    _SOURCE_FILE_SUFFIXES as _SOURCE_FILE_SUFFIXES,
)
from speck.data.configuration import (
    _integer as _integer,
)
from speck.data.configuration import (
    _tree_size as _tree_size,
)
from speck.data.configuration import (
    _validate_source as _validate_source,
)
from speck.data.configuration import (
    default_data_dir as default_data_dir,
)
from speck.data.configuration import (
    derive_source_quotas as derive_source_quotas,
)
from speck.data.configuration import (
    disk_preflight as disk_preflight,
)
from speck.data.configuration import (
    estimate_disk_requirement as estimate_disk_requirement,
)
from speck.data.configuration import (
    format_version as format_version,
)
from speck.data.configuration import (
    resolve_data_dir as resolve_data_dir,
)
from speck.data.configuration import (
    validate_data_settings as validate_data_settings,
)
from speck.data.packing import BestFitRows
from speck.data.packing import TokenShardWriter as TokenShardWriter
from speck.data.validation import fingerprint, slice_sha256
from speck.provenance.io import file_sha256 as _file_hash
from speck.provenance.io import lines_sha256 as _line_hash
from speck.tokenization.chat import ChatFormatError
from speck.tokenization.tokenizer import get_tokenizer
from speck.tokenization.tools import encode_conversation


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path):
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def normalize_for_dedup(content):
    """Normalize text for deterministic lightweight exact deduplication."""

    return " ".join(unicodedata.normalize("NFKC", content).lower().split())


def dedup_hash(content):
    normalized = normalize_for_dedup(content)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=_DEDUP_BYTES).digest()


def _is_validation_document(content, seed, fraction):
    person = hashlib.sha256(str(seed).encode()).digest()[:16]
    digest = hashlib.blake2b(content.encode("utf-8"), digest_size=8, person=person).digest()
    return int.from_bytes(digest, "big") / 2**64 < fraction


def _prefixed_shards(shards, source_id):
    return [{**shard, "path": f"sources/{source_id}/{shard['path']}"} for shard in shards]


def _sync_file(handle):
    handle.flush()
    os.fsync(handle.fileno())


def _truncate(path, size):
    with Path(path).open("r+b") as handle:
        handle.truncate(size)
        _sync_file(handle)


def _encode_messages(tokenizer, content):
    """Encode one JSON chat record with its assistant mask, or None if it is unusable."""

    try:
        tokens, mask = encode_conversation(tokenizer, json.loads(content))
    except (ChatFormatError, ValueError, TypeError, KeyError):
        return None, None
    if not any(mask):
        return None, None
    return tokens, [int(value) for value in mask]


class SourceBuilder:
    """Append one source and expose durable remote-file checkpoints."""

    def __init__(
        self,
        *,
        directory,
        source,
        resolved,
        tokenizer,
        accepted_hashes,
        dedup_file,
        dedup_start,
        train_requested,
        train_reserve,
        validation_requested,
        validation_fraction,
        shard_tokens,
        filtering,
        seed,
        progress=None,
    ):
        self.directory = Path(directory)
        self.source = source
        self.source_id = source["id"]
        self.resolved = resolved
        self.tokenizer = tokenizer
        self.accepted_hashes = accepted_hashes
        self.dedup_file = dedup_file
        self.dedup_start = dedup_start
        self.train_requested = train_requested
        self.train_reserve = train_reserve
        self.validation_requested = validation_requested
        self.validation_fraction = validation_fraction
        self.filtering = filtering
        self.seed = seed
        # A source that declares passes streams only its unique pool, which finish() repeats.
        self.passes = source.get("passes", 1)
        self.exposure_target = train_requested + train_reserve
        self.targets = {
            "train": -(-self.exposure_target // self.passes),
            "val": validation_requested,
        }
        restored_splits = (progress or {}).get("splits", {})
        self.writers = {
            split: TokenShardWriter(
                self.directory,
                split,
                shard_tokens,
                shards=restored_splits.get(split, {}).get("shards", []),
                total_tokens=restored_splits.get(split, {}).get("tokens", 0),
            )
            for split in ("train", "val")
        }
        # Row-packed sources keep whole records in fixed rows beside a parallel loss mask.
        self.packing = source.get("packing")
        self.masks = {}
        self.packers = {}
        self.record_format = source.get("record_format", "text")
        self.rejected = {
            split: restored_splits.get(split, {}).get(
                "rejected_records", {"too_long": 0, "invalid": 0}
            )
            for split in ("train", "val")
        }
        if self.packing is not None:
            for split in ("train", "val"):
                restored = restored_splits.get(split, {})
                self.masks[split] = TokenShardWriter(
                    self.directory,
                    split,
                    shard_tokens,
                    shards=restored.get("mask_shards", []),
                    total_tokens=restored.get("tokens", 0),
                    dtype="u1",
                    name=f"{split}_mask",
                )
                self.packers[split] = BestFitRows(
                    self.packing["row_tokens"], self.packing["open_rows"], tokenizer.eos_id
                )
        self._synced_shards = {
            shard["path"]
            for writer in (*self.writers.values(), *self.masks.values())
            for shard in writer.shards
        }
        self.document_counts = {
            split: restored_splits.get(split, {}).get("documents", 0) for split in ("train", "val")
        }
        self.index_path = self.directory / "documents.jsonl"
        self.index_hash = hashlib.sha256()
        if self.index_path.exists():
            with self.index_path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    self.index_hash.update(chunk)
        self.index_file = self.index_path.open("ab")
        self.dedup_hash = hashlib.sha256()
        if self.dedup_file.tell() > dedup_start:
            with Path(self.dedup_file.name).open("rb") as handle:
                handle.seek(dedup_start)
                remaining = self.dedup_file.tell() - dedup_start
                while remaining:
                    chunk = handle.read(min(8 * 1024 * 1024, remaining))
                    self.dedup_hash.update(chunk)
                    remaining -= len(chunk)

    def _tokens(self, split):
        pending = self.packers[split].pending_tokens if self.packers else 0
        return self.writers[split].total_tokens + pending

    @property
    def complete(self):
        return all(self._tokens(split) >= self.targets[split] for split in self.writers)

    def _process(self, batch):
        pending = set()
        rows = []
        for document in batch:
            content = document.get("content")
            if not isinstance(content, str) or not content:
                continue
            if not self.filtering["min_chars"] <= len(content) <= self.filtering["max_chars"]:
                continue
            normalized = normalize_for_dedup(content)
            digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=_DEDUP_BYTES).digest()
            digest_integer = int.from_bytes(digest, "big")
            if digest_integer in self.accepted_hashes or digest_integer in pending:
                continue
            # A caller that knows document families assigns the split itself, so validation
            # families stay disjoint from training; otherwise the content hash decides.
            preferred = document.get("split") or (
                "val"
                if _is_validation_document(normalized, self.seed, self.validation_fraction)
                else "train"
            )
            if preferred not in self.targets:
                raise ValueError(f"unknown document split: {preferred!r}")
            if self._tokens(preferred) >= self.targets[preferred]:
                continue
            pending.add(digest_integer)
            rows.append((document, digest, digest_integer, preferred))
        if not rows:
            return
        if self.record_format == "messages":
            encoded = [_encode_messages(self.tokenizer, row[0]["content"]) for row in rows]
        else:
            token_rows = self.tokenizer.encode_batch(
                [row[0]["content"] for row in rows], bos=True, eos=True
            )
            if len(token_rows) != len(rows):
                raise ValueError("tokenizer returned the wrong number of encoded documents")
            # The record's BOS is context, never a target.
            encoded = [(tokens, [0] + [1] * (len(tokens) - 1)) for tokens in token_rows]
        for (document, digest, digest_integer, split), (token_ids, mask) in zip(rows, encoded):
            if self.complete:
                break
            if self._tokens(split) >= self.targets[split]:
                continue
            if token_ids is None:
                self.rejected[split]["invalid"] += 1
                continue
            minimum_tokens = self.source["filters"].get("min_tokens", 1)
            maximum_tokens = self.source["filters"].get("max_tokens", math.inf)
            if not minimum_tokens <= len(token_ids) <= maximum_tokens:
                continue
            if self.packing is not None and len(token_ids) > self.packing["row_tokens"]:
                # Whole records only: an over-long record is counted and left out, never cut.
                self.rejected[split]["too_long"] += 1
                continue
            self.accepted_hashes.add(digest_integer)
            self.dedup_file.write(digest)
            self.dedup_hash.update(digest)
            metadata = dict(document.get("metadata") or {})
            if document.get("file") is not None:
                metadata["file"] = document["file"]
            if document.get("row") is not None:
                metadata["row"] = document["row"]
            record = {
                "content_hash": hashlib.sha256(document["content"].encode()).hexdigest(),
                "dedup_hash": digest.hex(),
                "source_id": self.source_id,
                "split": split,
            }
            score = document.get("score")
            if score is not None:
                record["score"] = float(score)
            if metadata:
                record["metadata"] = {
                    key: _metadata_value(value)
                    for key, value in metadata.items()
                    if value is not None
                }
            self.document_counts[split] += 1
            if self.packing is None:
                start_token = self.writers[split].total_tokens
                self._index(record, start_token, self.writers[split].write(token_ids))
            else:
                self._write_rows(split, self.packers[split].add(token_ids, mask, record))

    def _index(self, record, start_token, tokens):
        record = {**record, "start_token": start_token, "end_token": start_token + tokens}
        record["tokens"] = tokens
        line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self.index_file.write(line)
        self.index_hash.update(line)

    def _write_rows(self, split, rows):
        for tokens, mask, records in rows:
            start = self.writers[split].total_tokens
            self.writers[split].write(tokens)
            self.masks[split].write(mask)
            for record, offset, length in records:
                self._index(record, start + offset, length)

    def _flush_rows(self, tail=False):
        for split, packer in self.packers.items():
            self._write_rows(split, packer.flush())
            if tail:
                # One unsupervised token lets the loader read its row plus one lookahead token.
                self.writers[split].write([self.tokenizer.eos_id])
                self.masks[split].write([0])

    def consume(self, documents):
        batch = []
        characters = 0
        for document in documents:
            content = document.get("content")
            length = len(content) if isinstance(content, str) else 0
            if batch and (
                len(batch) >= _MAX_TOKENIZER_DOCUMENTS
                or characters + length > _MAX_TOKENIZER_BATCH_CHARACTERS
            ):
                self._process(batch)
                batch.clear()
                characters = 0
                if self.complete:
                    return
            batch.append(document)
            characters += length
            if (
                len(batch) >= _MAX_TOKENIZER_DOCUMENTS
                or characters >= _MAX_TOKENIZER_BATCH_CHARACTERS
            ):
                self._process(batch)
                batch.clear()
                characters = 0
                if self.complete:
                    return
        if batch and not self.complete:
            self._process(batch)

    def _sync_outputs(self):
        for writer in (*self.writers.values(), *self.masks.values()):
            writer.finish()
            for shard in writer.shards:
                if shard["path"] in self._synced_shards:
                    continue
                with (self.directory / shard["path"]).open("rb") as handle:
                    os.fsync(handle.fileno())
                self._synced_shards.add(shard["path"])
        _fsync_directory(self.directory)
        _sync_file(self.index_file)
        _sync_file(self.dedup_file)

    def progress(self, next_file_index):
        # Open rows are closed at every checkpoint so a resumed build restores exact state.
        self._flush_rows()
        self._sync_outputs()
        files = self.resolved["files"]
        journal_end = self.dedup_file.tell()
        records = sum(self.document_counts.values())
        if journal_end - self.dedup_start != records * _DEDUP_BYTES:
            raise ValueError("source dedup journal count does not match document records")
        value = {
            "format_version": 1,
            "source_id": self.source_id,
            "file_list_sha256": self.resolved["file_list_sha256"],
            "next_file_index": next_file_index,
            "next_file_path": files[next_file_index] if next_file_index < len(files) else None,
            "splits": {
                split: {
                    "tokens": self.writers[split].total_tokens,
                    "documents": self.document_counts[split],
                    "shards": list(self.writers[split].shards),
                    **(
                        {
                            "mask_shards": list(self.masks[split].shards),
                            "rejected_records": dict(self.rejected[split]),
                        }
                        if self.packing is not None
                        else {}
                    ),
                }
                for split in ("train", "val")
            },
            "document_index": {
                "path": self.index_path.name,
                "bytes": self.index_path.stat().st_size,
                "records": records,
                "sha256": self.index_hash.hexdigest(),
            },
            "dedup_journal": {
                "start_byte": self.dedup_start,
                "end_byte": journal_end,
                "hashes": records,
                "sha256": self.dedup_hash.hexdigest(),
            },
        }
        _atomic_json(self.directory / "source_progress.json", value)
        return value

    def finish(self, *, files_completed=0, final_file=None):
        if not self.complete:
            missing = ", ".join(
                f"{split} {self.writers[split].total_tokens:,}/{self.targets[split]:,}"
                for split in ("train", "val")
                if self.writers[split].total_tokens < self.targets[split]
            )
            pool = (
                f" (train is the unique pool for {self.passes} passes)" if self.passes > 1 else ""
            )
            raise RuntimeError(
                f"source {self.source_id} was exhausted before meeting its budgets: {missing}{pool}"
            )
        self._flush_rows(tail=True)
        repetition = self._repeat_passes() if self.passes > 1 else None
        self._sync_outputs()
        self.index_file.close()
        final_index = self.index_path
        journal_end = self.dedup_file.tell()
        records = sum(self.document_counts.values())
        split_summaries = {}
        for split in ("train", "val"):
            requested = self.train_requested if split == "train" else self.validation_requested
            target = self.exposure_target if split == "train" else self.targets[split]
            split_summaries[split] = {
                "requested_tokens": requested,
                "preparation_target_tokens": target,
                "tokens": self.writers[split].total_tokens,
                "reserve_tokens": self.train_reserve if split == "train" else 0,
                "overshoot_tokens": self.writers[split].total_tokens - target,
                "documents": self.document_counts[split],
                "shards": _prefixed_shards(self.writers[split].shards, self.source_id),
            }
            if self.packing is not None:
                split_summaries[split]["mask_shards"] = _prefixed_shards(
                    self.masks[split].shards, self.source_id
                )
                split_summaries[split]["rejected_records"] = dict(self.rejected[split])
        summary = {
            "id": self.source_id,
            "repo": self.source["repo"],
            "revision": self.resolved["revision"],
            "tree_path": self.source["tree_path"],
            "content_column": self.source["content_column"],
            "file_format": self.source["file_format"],
            "score_column": self.source.get("score_column"),
            "language_column": self.source.get("language_column"),
            "metadata_columns": self.source["metadata_columns"],
            "filters": {**self.filtering, **self.source["filters"]},
            "file_count": len(self.resolved["files"]),
            "files_completed": files_completed,
            "final_file": final_file,
            "file_list_sha256": self.resolved["file_list_sha256"],
            "documents": records,
            "document_index": {
                "path": f"sources/{self.source_id}/{final_index.name}",
                "records": records,
                "bytes": final_index.stat().st_size,
                "sha256": self.index_hash.hexdigest(),
            },
            "dedup_journal": {
                "start_byte": self.dedup_start,
                "end_byte": journal_end,
                "hashes": records,
                "sha256": self.dedup_hash.hexdigest(),
            },
            "splits": split_summaries,
        }
        if self.source.get("language_detector") is not None:
            summary["language_detector"] = self.source["language_detector"]
        if self.packing is not None:
            summary["packing"] = {**self.packing, "kind": "best_fit_rows"}
            summary["record_format"] = self.record_format
        if repetition is not None:
            summary["repetition"] = repetition
        _atomic_json(self.directory / "source.json", summary)
        return summary

    def _pass_priority(self, pass_number, dedup_hex):
        return hashlib.sha256(
            f"{self.seed}:{self.source_id}:{pass_number}:{dedup_hex}".encode()
        ).digest()

    def _repeat_passes(self):
        """Append passes 2..k of the unique train pool, each in its own seeded order.

        Pass 1 is the pool as streamed. Every later pass copies the same documents' tokens from
        the pass-1 shards in SHA-256(seed:source:pass:dedup_hash) order, stopping at the exposure
        target, so only the final pass may be partial. Each pass starts on a fresh shard.
        """

        writer = self.writers["train"]
        writer.finish()
        self.index_file.flush()
        documents = []
        with self.index_path.open("rb") as handle:
            for line in handle:
                record = json.loads(line)
                if record["split"] == "train":
                    documents.append(
                        (record["dedup_hash"], record["start_token"], record["end_token"])
                    )
        unique_tokens = writer.total_tokens
        if sum(end - start for _, start, end in documents) != unique_tokens:
            raise ValueError(f"source {self.source_id} train index does not cover its pool")
        pool = _ShardReader(self.directory, writer.shards)
        passes = [
            {
                "pass": 1,
                "documents": len(documents),
                "tokens": unique_tokens,
                "first_shard": 0,
                "order_sha256": _line_hash([digest for digest, _, _ in documents]),
            }
        ]
        for pass_number in range(2, self.passes + 1):
            order = sorted(
                documents, key=lambda document: self._pass_priority(pass_number, document[0])
            )
            first_shard = len(writer.shards)
            start_tokens = writer.total_tokens
            written = []
            for digest, start, end in order:
                if writer.total_tokens >= self.exposure_target:
                    break
                writer.write(pool.read(start, end))
                written.append(digest)
            if not written:
                raise ValueError(
                    f"source {self.source_id} reaches its exposure before pass {pass_number}; "
                    f"its {unique_tokens:,}-token unique pool cannot fill {self.passes} passes"
                )
            writer.finish()
            passes.append(
                {
                    "pass": pass_number,
                    "documents": len(written),
                    "tokens": writer.total_tokens - start_tokens,
                    "first_shard": first_shard,
                    "order_sha256": _line_hash(written),
                }
            )
        del pool
        if writer.total_tokens < self.exposure_target:
            raise ValueError(f"source {self.source_id} repeated passes fall short of exposure")
        return {
            "passes": self.passes,
            "unique_target_tokens": self.targets["train"],
            "unique_tokens": unique_tokens,
            "unique_documents": len(documents),
            "exposure_tokens": writer.total_tokens,
            "exposure_documents": sum(entry["documents"] for entry in passes),
            "order": (
                "pass 1 in source stream order; pass p > 1 in "
                "SHA-256(seed:source_id:p:dedup_hash) ascending order, truncated at exposure"
            ),
            "pass_orders": passes,
        }


class _ShardReader:
    """Read token ranges of a finished split stream without loading it whole."""

    def __init__(self, directory, shards):
        self.arrays = [
            np.memmap(Path(directory) / shard["path"], mode="r", dtype="<u2") for shard in shards
        ]
        self.starts = np.cumsum([0] + [shard["tokens"] for shard in shards])

    def read(self, start, end):
        parts = []
        index = int(np.searchsorted(self.starts, start, side="right")) - 1
        while start < end:
            offset = int(self.starts[index])
            stop = min(end, int(self.starts[index + 1]))
            parts.append(self.arrays[index][start - offset : stop - offset])
            start = stop
            index += 1
        return np.concatenate(parts)


def _prepare_injected_source(*, documents, **builder_settings):
    builder = SourceBuilder(**builder_settings)
    try:
        builder.consume(documents)
    finally:
        close = getattr(documents, "close", None)
        if close is not None:
            close()
    return builder.finish()


def _load_hashes(path):
    path = Path(path)
    if not path.exists():
        return set()
    if path.stat().st_size % _DEDUP_BYTES:
        raise ValueError("staged dedup hash journal is truncated")
    values = set()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            values.update(
                int.from_bytes(chunk[index : index + _DEDUP_BYTES], "big")
                for index in range(0, len(chunk), _DEDUP_BYTES)
            )
    if len(values) * _DEDUP_BYTES != path.stat().st_size:
        raise ValueError("staged dedup hash journal contains duplicate accepted hashes")
    return values


def _source_summary(path):
    path = Path(path) / "source.json"
    if not path.is_file():
        raise ValueError(f"completed staged source is missing provenance: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_committed_shards(directory, splits, *, prefixed=False):
    directory = Path(directory)
    committed = set()
    for split in ("train", "val"):
        split_manifest = splits[split]
        if sum(shard["tokens"] for shard in split_manifest["shards"]) != split_manifest["tokens"]:
            raise ValueError(f"staged {split} shard totals are inconsistent")
        for shard in split_manifest["shards"]:
            name = Path(shard["path"]).name if prefixed else shard["path"]
            path = directory / name
            expected_bytes = shard["tokens"] * np.dtype("<u2").itemsize
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"invalid committed packed shard: {path}")
            if _file_hash(path) != shard["sha256"]:
                raise ValueError(f"committed packed shard checksum mismatch: {path}")
            committed.add(name)
    return committed


def _recover_source_progress(directory, progress, resolved, dedup_path, dedup_start):
    directory = Path(directory)
    if (
        progress.get("format_version") != 1
        or progress.get("source_id") is None
        or progress.get("file_list_sha256") != resolved["file_list_sha256"]
    ):
        raise ValueError("staged source progress does not match its resolved file list")
    next_index = progress.get("next_file_index")
    if isinstance(next_index, bool) or not isinstance(next_index, int):
        raise ValueError("staged source progress has an invalid file index")
    files = resolved["files"]
    if not 0 <= next_index <= len(files):
        raise ValueError("staged source progress file index is out of range")
    expected_path = files[next_index] if next_index < len(files) else None
    if progress.get("next_file_path") != expected_path:
        raise ValueError("staged source progress file path is inconsistent")

    journal = progress["dedup_journal"]
    records = progress["document_index"]["records"]
    if journal["start_byte"] != dedup_start or journal["hashes"] != records:
        raise ValueError("staged source dedup boundary is inconsistent")
    if journal["end_byte"] - dedup_start != records * _DEDUP_BYTES:
        raise ValueError("staged source dedup length is inconsistent")
    if dedup_path.stat().st_size < journal["end_byte"]:
        raise ValueError("staged dedup journal is shorter than committed progress")
    if dedup_path.stat().st_size > journal["end_byte"]:
        _truncate(dedup_path, journal["end_byte"])
    if slice_sha256(dedup_path, dedup_start, journal["end_byte"]) != journal["sha256"]:
        raise ValueError("staged source dedup slice checksum mismatch")

    index = progress["document_index"]
    index_path = directory / index["path"]
    if not index_path.exists() and index["bytes"] == 0:
        index_path.touch()
    if not index_path.is_file() or index_path.stat().st_size < index["bytes"]:
        raise ValueError("staged document index is shorter than committed progress")
    if index_path.stat().st_size > index["bytes"]:
        _truncate(index_path, index["bytes"])
    if _file_hash(index_path) != index["sha256"]:
        raise ValueError("staged document index checksum mismatch")
    if records != sum(progress["splits"][split]["documents"] for split in ("train", "val")):
        raise ValueError("staged source document counts are inconsistent")

    committed = _verify_committed_shards(directory, progress["splits"])
    for split in ("train", "val"):
        for path in directory.glob(f"{split}_*.bin*"):
            if path.name not in committed:
                path.unlink()
    _fsync_directory(directory)
    return progress


def _verify_source_integrity(
    directory,
    summary,
    dedup_path,
    expected_start,
    *,
    require_journal_end=False,
):
    directory = Path(directory)
    journal = summary["dedup_journal"]
    if journal["start_byte"] != expected_start:
        raise ValueError(f"source {summary['id']} dedup journal is not contiguous")
    if journal["end_byte"] - expected_start != journal["hashes"] * _DEDUP_BYTES:
        raise ValueError(f"source {summary['id']} dedup journal length is invalid")
    size = dedup_path.stat().st_size
    if size < journal["end_byte"] or (require_journal_end and size != journal["end_byte"]):
        raise ValueError(f"source {summary['id']} dedup journal boundary is invalid")
    if slice_sha256(dedup_path, expected_start, journal["end_byte"]) != journal["sha256"]:
        raise ValueError(f"source {summary['id']} dedup journal checksum mismatch")
    if journal["hashes"] != summary["documents"]:
        raise ValueError(f"source {summary['id']} dedup count is invalid")
    index = summary["document_index"]
    index_path = directory / Path(index["path"]).name
    if not index_path.is_file() or index_path.stat().st_size != index["bytes"]:
        raise ValueError(f"source {summary['id']} document index size is invalid")
    if _file_hash(index_path) != index["sha256"]:
        raise ValueError(f"source {summary['id']} document index checksum mismatch")
    _verify_committed_shards(directory, summary["splits"], prefixed=True)
    return journal["end_byte"]


@dataclass(frozen=True)
class _DatasetBuildRequest:
    requested_train_tokens: int
    validation_tokens_per_source: int
    validation_fraction: float
    seed: int
    output_dir: Path
    restart: bool
    tokenizer: Any
    document_iterators: Any
    api: Any
    check_disk: bool
    disk_usage: Any


class _DatasetBuild:
    """Own the durable state and explicit phases of one packed-dataset build."""

    def __init__(self, settings, request):
        self.settings = settings
        self.requested_train_tokens = request.requested_train_tokens
        self.validation_tokens_per_source = request.validation_tokens_per_source
        self.validation_fraction = request.validation_fraction
        self.seed = request.seed
        self.output_dir = request.output_dir
        self.restart = request.restart
        self.tokenizer = request.tokenizer
        self.document_iterators = request.document_iterators or {}
        self.api = request.api
        self.check_disk = request.check_disk
        self.disk_usage = request.disk_usage
        self.staging = self.output_dir.with_name(self.output_dir.name + ".building")
        self.state_path = self.staging / "build_state.json"
        self.dedup_path = self.staging / "dedup_hashes.bin"
        self.raw_directory = self.staging / ".raw"

    def _prepare_staging(self):
        if self.output_dir.exists():
            if any(self.output_dir.iterdir()):
                raise FileExistsError(f"dataset already exists: {self.output_dir}")
            self.output_dir.rmdir()
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.staging.exists() and self.restart:
            shutil.rmtree(self.staging)
        self.disk_report = disk_preflight(
            self.output_dir,
            self.settings,
            self.requested_train_tokens,
            check=self.check_disk,
            disk_usage=self.disk_usage,
        )
        if self.check_disk:
            print(
                f"Disk preflight: required {self.disk_report['required_bytes']:,} bytes, "
                f"free {self.disk_report['free_bytes']:,} bytes"
            )
        self.tokenizer_manifest = {
            "fingerprint": self.tokenizer.fingerprint(),
            "vocab_size": self.tokenizer.vocab_size,
            "bos_token_id": self.tokenizer.bos_id,
            "eos_token_id": self.tokenizer.eos_id,
        }
        self._load_or_create_state()
        unknown_iterators = set(self.document_iterators) - {
            source["id"] for source in self.settings["sources"]
        }
        if unknown_iterators:
            raise ValueError(
                f"document iterators have unknown sources: {', '.join(unknown_iterators)}"
            )

    def _load_or_create_state(self):
        contract = {
            "format_version": format_version,
            "sources": self.settings["sources"],
            "mixture": {"phases": self.settings["phases"]},
            "requested_train_tokens": self.requested_train_tokens,
            "validation_tokens_per_source": self.validation_tokens_per_source,
            "validation_fraction": self.validation_fraction,
            "filtering": self.settings["filtering"],
            "dedup": self.settings["dedup"],
            "shards": self.settings["shards"],
            "seed": self.seed,
            "tokenizer": self.tokenizer_manifest,
        }
        contract_hash = fingerprint(contract)
        if not self.staging.exists():
            self.staging.mkdir(parents=True)
            (self.staging / "sources").mkdir()
            self.state = {
                "format_version": format_version,
                "contract": contract_hash,
                "completed_sources": [],
                "current_source": None,
                "resolved_sources": {},
                "disk_preflight": self.disk_report,
            }
            _atomic_json(self.state_path, self.state)
            return
        if not self.state_path.is_file():
            raise ValueError(f"invalid staged build: {self.staging}; pass --restart to replace it")
        self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if self.state.get("contract") != contract_hash:
            raise ValueError("staged build settings changed; pass --restart to replace it")
        self.state["disk_preflight"] = self.disk_report
        _atomic_json(self.state_path, self.state)

    def _resolve_sources(self):
        api = self.api or HfApi()
        for source in self.settings["sources"]:
            source_id = source["id"]
            if source_id in self.state["resolved_sources"]:
                continue
            if source_id in self.document_iterators:
                resolved = {
                    "revision": "injected",
                    "files": [],
                    "file_list_sha256": _line_hash([]),
                }
            else:
                resolved = discover_source_files(source, self.seed, api)
            self.state["resolved_sources"][source_id] = resolved
            _atomic_json(self.state_path, self.state)

    def _verify_completed_sources(self):
        self.dedup_path.touch(exist_ok=True)
        self.source_ids = [source["id"] for source in self.settings["sources"]]
        self.completed = self.state["completed_sources"]
        if self.completed != self.source_ids[: len(self.completed)]:
            raise ValueError("staged completed sources are not a source-order prefix")
        self.summaries = {}
        self.journal_boundary = 0
        for source_id in self.completed:
            directory = self.staging / "sources" / source_id
            summary = _source_summary(directory)
            if summary.get("id") != source_id:
                raise ValueError("completed staged source ID is inconsistent")
            self.journal_boundary = _verify_source_integrity(
                directory,
                summary,
                self.dedup_path,
                self.journal_boundary,
            )
            self.summaries[source_id] = summary

    def _recover_current_source(self):
        current = self.state.get("current_source")
        if not current:
            if self.dedup_path.stat().st_size != self.journal_boundary:
                raise ValueError(
                    "staged dedup journal does not end at the completed-source boundary"
                )
            return
        source_id = current["id"]
        expected_source = (
            self.source_ids[len(self.completed)]
            if len(self.completed) < len(self.source_ids)
            else None
        )
        if source_id != expected_source or current["dedup_bytes_before"] != self.journal_boundary:
            raise ValueError("staged current source boundary is inconsistent")
        final_directory = self.staging / "sources" / source_id
        temporary_directory = self.staging / "sources" / f"{source_id}.building"
        if final_directory.is_dir():
            self._recover_published_source(source_id, final_directory)
        elif (
            current.get("mode") == "files"
            and (temporary_directory / "source_progress.json").is_file()
        ):
            self._recover_file_progress(source_id, temporary_directory)
        else:
            self._discard_uncommitted_source(temporary_directory)

    def _recover_published_source(self, source_id, final_directory):
        summary = _source_summary(final_directory)
        self.journal_boundary = _verify_source_integrity(
            final_directory,
            summary,
            self.dedup_path,
            self.journal_boundary,
            require_journal_end=True,
        )
        self.completed.append(source_id)
        self.state["current_source"] = None
        _atomic_json(self.state_path, self.state)
        (final_directory / "source_progress.json").unlink(missing_ok=True)
        _fsync_directory(final_directory)
        self.summaries[source_id] = summary

    def _recover_file_progress(self, source_id, temporary_directory):
        progress_path = temporary_directory / "source_progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("source_id") != source_id:
            raise ValueError("staged source progress has the wrong source ID")
        _recover_source_progress(
            temporary_directory,
            progress,
            self.state["resolved_sources"][source_id],
            self.dedup_path,
            self.journal_boundary,
        )

    def _discard_uncommitted_source(self, temporary_directory):
        if self.dedup_path.stat().st_size < self.journal_boundary:
            raise ValueError("staged dedup journal is shorter than completed sources")
        if self.dedup_path.stat().st_size > self.journal_boundary:
            _truncate(self.dedup_path, self.journal_boundary)
        shutil.rmtree(temporary_directory, ignore_errors=True)
        self.state["current_source"] = None
        _atomic_json(self.state_path, self.state)

    def _build_remaining_sources(self):
        self.accepted_hashes = _load_hashes(self.dedup_path)
        for source in self.settings["sources"]:
            if source["id"] not in self.completed:
                self._build_source(source)

    def _build_source(self, source):
        source_id = source["id"]
        final_directory = self.staging / "sources" / source_id
        temporary_directory = self.staging / "sources" / f"{source_id}.building"
        resuming = (self.state.get("current_source") or {}).get("id") == source_id
        if not resuming:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            temporary_directory.mkdir(parents=True)
            dedup_start = self.dedup_path.stat().st_size
            self.state["current_source"] = {
                "id": source_id,
                "dedup_bytes_before": dedup_start,
                "mode": "injected" if source_id in self.document_iterators else "files",
            }
            _atomic_json(self.state_path, self.state)
        else:
            dedup_start = self.state["current_source"]["dedup_bytes_before"]
        resolved = self.state["resolved_sources"][source_id]
        progress_path = temporary_directory / "source_progress.json"
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.is_file()
            else None
        )
        builder_settings = self._builder_settings(
            source,
            resolved,
            temporary_directory,
            dedup_start,
        )
        with self.dedup_path.open("ab") as dedup_file:
            builder_settings["dedup_file"] = dedup_file
            if source_id in self.document_iterators:
                summary = _prepare_injected_source(
                    documents=iter(self.document_iterators[source_id]),
                    **builder_settings,
                )
            else:
                summary = self._build_source_files(source, resolved, progress, builder_settings)
        self._commit_source(source_id, temporary_directory, final_directory, summary)

    def _builder_settings(self, source, resolved, temporary_directory, dedup_start):
        source_id = source["id"]
        return {
            "directory": temporary_directory,
            "source": source,
            "resolved": resolved,
            "tokenizer": self.tokenizer,
            "accepted_hashes": self.accepted_hashes,
            "dedup_start": dedup_start,
            "train_requested": self.settings["quotas"][source_id],
            "train_reserve": self.settings["train_reserve_tokens_per_source"],
            "validation_requested": self.settings["validation_tokens_per_source"],
            "validation_fraction": self.settings["validation_fraction"],
            "shard_tokens": self.settings["shards"]["tokens"],
            "filtering": self.settings["filtering"],
            "seed": self.seed,
        }

    def _build_source_files(self, source, resolved, progress, builder_settings):
        builder = SourceBuilder(progress=progress, **builder_settings)
        if progress is None:
            progress = builder.progress(0)
        next_file = progress["next_file_index"]
        final_file = None
        for file_index in range(next_file, len(resolved["files"])):
            filename = resolved["files"][file_index]
            documents = iter_source_file_documents(
                source=source,
                revision=resolved["revision"],
                filename=filename,
                filtering=self.settings["filtering"],
                cache_dir=self.raw_directory,
                description=(f"{source['id']} {file_index + 1}/{len(resolved['files'])}"),
            )
            try:
                builder.consume(documents)
            finally:
                documents.close()
            if builder.complete:
                final_file = filename
                break
            progress = builder.progress(file_index + 1)
        return builder.finish(
            files_completed=progress["next_file_index"],
            final_file=final_file,
        )

    def _commit_source(self, source_id, temporary_directory, final_directory, summary):
        temporary_directory.replace(final_directory)
        _fsync_directory(final_directory.parent)
        (final_directory / "source_progress.json").unlink(missing_ok=True)
        _fsync_directory(final_directory)
        self.completed.append(source_id)
        self.state["current_source"] = None
        _atomic_json(self.state_path, self.state)
        self.summaries[source_id] = summary
        self.journal_boundary = summary["dedup_journal"]["end_byte"]

    def _finalize(self):
        if self.dedup_path.stat().st_size != self.journal_boundary:
            raise ValueError("dedup journal does not end at the final source boundary")
        ordered_summaries = [self.summaries[source["id"]] for source in self.settings["sources"]]
        manifest = self._manifest(ordered_summaries)
        _atomic_json(self.staging / "manifest.json", manifest)
        shutil.rmtree(self.raw_directory, ignore_errors=True)
        self.staging.replace(self.output_dir)
        _fsync_directory(self.output_dir.parent)
        (self.output_dir / self.state_path.name).unlink()
        _fsync_directory(self.output_dir)
        self._print_summary(manifest, ordered_summaries)
        return manifest

    def _manifest(self, ordered_summaries):
        aggregate_splits = {}
        for split in ("train", "val"):
            aggregate_splits[split] = {
                "requested_tokens": (
                    self.requested_train_tokens
                    if split == "train"
                    else self.validation_tokens_per_source * len(ordered_summaries)
                ),
                "tokens": sum(source["splits"][split]["tokens"] for source in ordered_summaries),
                "documents": sum(
                    source["splits"][split]["documents"] for source in ordered_summaries
                ),
            }
        return {
            "format": "speck_packed_tokens",
            "format_version": format_version,
            "dtype": "<u2",
            "requested_train_tokens": self.requested_train_tokens,
            "validation_tokens_per_source": self.validation_tokens_per_source,
            "mixture": {
                "phases": self.settings["phases"],
                "source_quotas": self.settings["quotas"],
            },
            "preparation": {
                "seed": self.seed,
                "validation_fraction": self.settings["validation_fraction"],
                "filtering": self.settings["filtering"],
                "shards": self.settings["shards"],
                "train_reserve_tokens_per_source": self.settings["train_reserve_tokens_per_source"],
                "reserve_basis": "(phase_count + 1) * maximum_loader_microbatch_tokens",
                "tokenizer_batch": {
                    "maximum_documents": _MAX_TOKENIZER_DOCUMENTS,
                    "maximum_characters": _MAX_TOKENIZER_BATCH_CHARACTERS,
                    "maximum_document_characters": _MAX_DOCUMENT_CHARACTERS,
                },
                "disk_preflight": self.disk_report,
            },
            "dedup": {
                **self.settings["dedup"],
                "path": self.dedup_path.name,
                "accepted_hashes": self.dedup_path.stat().st_size // _DEDUP_BYTES,
                "sha256": _file_hash(self.dedup_path),
                "collision_policy": "128-bit collisions are treated as duplicates",
            },
            "tokenizer": self.tokenizer_manifest,
            "documents": sum(source["documents"] for source in ordered_summaries),
            "sources": ordered_summaries,
            "splits": aggregate_splits,
        }

    def _print_summary(self, manifest, ordered_summaries):
        print(
            f"Prepared {manifest['splits']['train']['tokens']:,} train and "
            f"{manifest['splits']['val']['tokens']:,} validation tokens"
        )
        for source in ordered_summaries:
            train = source["splits"]["train"]
            print(
                f"{source['id']}: requested {train['requested_tokens']:,}, "
                f"reserve {train['reserve_tokens']:,}, actual {train['tokens']:,}"
            )
            repetition = source.get("repetition")
            if repetition is not None:
                print(
                    f"{source['id']}: {repetition['passes']} passes over "
                    f"{repetition['unique_tokens']:,} unique tokens"
                )
        print(f"Manifest: {self.output_dir / 'manifest.json'}")

    def run(self):
        self._prepare_staging()
        self._resolve_sources()
        self._verify_completed_sources()
        self._recover_current_source()
        self._build_remaining_sources()
        return self._finalize()


def prepare_dataset(
    *,
    sources,
    mixture,
    requested_train_tokens,
    validation_tokens_per_source,
    validation_fraction,
    filtering,
    dedup,
    shards,
    seed=42,
    output_dir=None,
    output_name=None,
    restart=False,
    tokenizer=None,
    document_iterators=None,
    api=None,
    check_disk=True,
    disk_usage=None,
):
    """Build a resumable v3 source-separated packed dataset and manifest."""

    settings = validate_data_settings(
        sources=sources,
        mixture=mixture,
        requested_train_tokens=requested_train_tokens,
        validation_tokens_per_source=validation_tokens_per_source,
        validation_fraction=validation_fraction,
        filtering=filtering,
        dedup=dedup,
        shards=shards,
    )
    seed = _integer(seed, "seed")
    tokenizer = tokenizer or get_tokenizer()
    if tokenizer.vocab_size > 65536:
        raise ValueError("packed uint16 data requires vocab_size <= 65536")
    request = _DatasetBuildRequest(
        requested_train_tokens=requested_train_tokens,
        validation_tokens_per_source=validation_tokens_per_source,
        validation_fraction=validation_fraction,
        seed=seed,
        output_dir=resolve_data_dir(output_dir, output_name),
        restart=restart,
        tokenizer=tokenizer,
        document_iterators=document_iterators,
        api=api,
        check_disk=check_disk,
        disk_usage=disk_usage,
    )
    build = _DatasetBuild(settings, request)
    return build.run()


def _validate_text_manifest(manifest):
    if manifest.get("format") != "speck_packed_tokens":
        raise ValueError("invalid packed dataset format")
    if manifest.get("format_version") != format_version:
        raise ValueError(f"unsupported packed dataset version: {manifest.get('format_version')}")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("packed dataset manifest has no sources")
    source_ids = [source.get("id") for source in sources]
    if any(not isinstance(source_id, str) for source_id in source_ids) or len(source_ids) != len(
        set(source_ids)
    ):
        raise ValueError("packed dataset manifest has invalid source IDs")
    quotas, phases = derive_source_quotas(
        source_ids, manifest.get("mixture", {}), manifest.get("requested_train_tokens")
    )
    if manifest["mixture"].get("source_quotas") != quotas:
        raise ValueError("packed dataset source quotas do not match its phases")
    if manifest["mixture"]["phases"] != phases:
        raise ValueError("packed dataset phases are not canonical")
    dedup = manifest.get("dedup", {})
    if any(dedup.get(key) != value for key, value in _DEDUP_SETTINGS.items()):
        raise ValueError("packed dataset dedup settings are invalid")
    journal_boundary = 0
    for source in sources:
        source_id = source["id"]
        if source.get("splits", {}).get("train", {}).get("requested_tokens") != quotas[source_id]:
            raise ValueError(f"packed dataset source quota differs for {source_id}")
        for split in ("train", "val"):
            split_manifest = source.get("splits", {}).get(split)
            if not isinstance(split_manifest, dict) or split_manifest.get("tokens", 0) < 1:
                raise ValueError(f"packed dataset source {source_id} has invalid {split} data")
            if (
                sum(shard.get("tokens", 0) for shard in split_manifest.get("shards", []))
                != split_manifest["tokens"]
            ):
                raise ValueError(f"packed dataset source {source_id} has invalid {split} shards")
        packing = source.get("packing")
        if packing is not None:
            for split in ("train", "val"):
                split_manifest = source["splits"][split]
                masks = split_manifest.get("mask_shards", [])
                if sum(shard.get("tokens", 0) for shard in masks) != split_manifest["tokens"]:
                    raise ValueError(f"packed dataset source {source_id} has invalid {split} masks")
                # Whole rows plus one unsupervised lookahead token.
                if split_manifest["tokens"] % packing.get("row_tokens", 0) != 1:
                    raise ValueError(f"packed dataset source {source_id} has partial {split} rows")
        repetition = source.get("repetition")
        if repetition is not None:
            train = source["splits"]["train"]
            orders = repetition.get("pass_orders", [])
            if (
                len(orders) != repetition.get("passes")
                or repetition.get("exposure_tokens") != train["tokens"]
                or sum(entry.get("tokens", 0) for entry in orders) != train["tokens"]
                or repetition.get("unique_documents") != train["documents"]
            ):
                raise ValueError(f"packed dataset source {source_id} repetition is inconsistent")
        if source.get("documents") != source.get("document_index", {}).get("records"):
            raise ValueError(f"packed dataset source {source_id} document index is invalid")
        journal = source.get("dedup_journal", {})
        if (
            journal.get("start_byte") != journal_boundary
            or journal.get("hashes") != source["documents"]
            or journal.get("end_byte", -1) - journal_boundary != source["documents"] * _DEDUP_BYTES
            or not isinstance(journal.get("sha256"), str)
        ):
            raise ValueError(f"packed dataset source {source_id} dedup journal is invalid")
        journal_boundary = journal["end_byte"]
    if (
        dedup.get("accepted_hashes") != sum(source["documents"] for source in sources)
        or journal_boundary != dedup.get("accepted_hashes", -1) * _DEDUP_BYTES
    ):
        raise ValueError("packed dataset aggregate dedup count is invalid")
    if manifest.get("documents") != sum(source["documents"] for source in sources):
        raise ValueError("packed dataset aggregate document count is invalid")
    for split in ("train", "val"):
        total = sum(source["splits"][split]["tokens"] for source in sources)
        if manifest.get("splits", {}).get(split, {}).get("tokens") != total:
            raise ValueError(f"packed dataset aggregate {split} token count is invalid")
    return manifest


def _validate_manifest(manifest):
    if manifest.get("format") == "speck_packed_tokens":
        return _validate_text_manifest(manifest)
    raise ValueError("invalid packed dataset format")


def load_manifest(data_dir=None):
    data_dir = Path(data_dir or default_data_dir / "packed")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"packed dataset not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _validate_manifest(manifest)


def _verify_file(path, expected):
    if _file_hash(path) != expected:
        raise ValueError(f"packed data checksum mismatch: {path}")


def verify_shards(data_dir=None, manifest=None):
    data_dir = Path(data_dir or default_data_dir / "packed")
    manifest = _validate_manifest(manifest) if manifest is not None else load_manifest(data_dir)
    for source in manifest["sources"]:
        for split in source["splits"].values():
            shards = [(shard, "<u2") for shard in split["shards"]]
            shards += [(shard, "u1") for shard in split.get("mask_shards", [])]
            for shard, dtype in shards:
                path = data_dir / shard["path"]
                expected_bytes = shard["tokens"] * np.dtype(dtype).itemsize
                if not path.is_file() or path.stat().st_size != expected_bytes:
                    raise ValueError(f"invalid packed shard: {path}")
                _verify_file(path, shard["sha256"])
        index = source["document_index"]
        index_path = data_dir / index["path"]
        if not index_path.is_file() or index_path.stat().st_size != index["bytes"]:
            raise ValueError(f"invalid packed document index: {index_path}")
        _verify_file(index_path, index["sha256"])
    dedup_path = data_dir / manifest["dedup"]["path"]
    expected_bytes = manifest["dedup"]["accepted_hashes"] * _DEDUP_BYTES
    if not dedup_path.is_file() or dedup_path.stat().st_size != expected_bytes:
        raise ValueError(f"invalid packed dedup journal: {dedup_path}")
    _verify_file(dedup_path, manifest["dedup"]["sha256"])
    for source in manifest["sources"]:
        journal = source["dedup_journal"]
        if (
            slice_sha256(dedup_path, journal["start_byte"], journal["end_byte"])
            != journal["sha256"]
        ):
            raise ValueError(f"packed dedup slice checksum mismatch: {source['id']}")
