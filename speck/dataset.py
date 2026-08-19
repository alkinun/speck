"""stream ultra-fineweb and build local packed token shards."""

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import requests

from speck.common import base_dir
from speck.tokenizer import get_tokenizer


dataset_repo = "openbmb/Ultra-FineWeb"
dataset_language = "en"
dataset_parts = 2048
format_version = 1
default_data_dir = Path(base_dir()) / "ultra_fineweb"


def get_parquet_urls(revision=None):
    revision = revision or get_dataset_revision()
    root = f"https://huggingface.co/datasets/{dataset_repo}/resolve/{revision}/data/ultrafineweb_en"
    return [f"{root}/ultrafineweb-en-part-{index:04d}-of-{dataset_parts}.parquet" for index in range(1, dataset_parts + 1)]


def get_dataset_revision():
    response = requests.get(f"https://huggingface.co/api/datasets/{dataset_repo}", timeout=60)
    response.raise_for_status()
    return response.json()["sha"]


def _download_file(url, destination, attempts=5):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    for attempt in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                reported = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            downloaded += len(chunk)
                            if downloaded - reported >= 256 * 1024 * 1024:
                                reported = downloaded
                                suffix = f"/{total / 2**30:.1f}gb" if total else "gb"
                                print(f"downloaded {downloaded / 2**30:.1f}{suffix}")
            temporary.replace(destination)
            return
        except (OSError, requests.RequestException):
            temporary.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def iter_documents(
    *,
    seed=42,
    min_score=0.8,
    min_chars=200,
    max_chars=100_000,
    cache_dir=None,
    keep_raw=False,
    urls=None,
):
    """yield filtered documents while keeping at most one remote parquet shard locally."""
    urls = list(get_parquet_urls() if urls is None else urls)
    random.Random(seed).shuffle(urls)
    cache_dir = Path(cache_dir or default_data_dir / "raw")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for shard_index, url in enumerate(urls):
        cache_key = hashlib.sha256(url.encode()).hexdigest()[:20]
        local_path = cache_dir / f"{cache_key}.parquet"
        if not local_path.exists():
            print(f"downloading ultra-fineweb shard {shard_index + 1}/{len(urls)}")
            _download_file(url, local_path)
        try:
            parquet = pq.ParquetFile(local_path)
            available = set(parquet.schema_arrow.names)
            required = {"content", "score", "source"}
            if not required.issubset(available):
                raise ValueError(f"unexpected ultra-fineweb columns: {sorted(available)}")
            for batch in parquet.iter_batches(columns=["content", "score", "source"], batch_size=2048):
                contents = batch.column(0).to_pylist()
                scores = batch.column(1).to_pylist()
                sources = batch.column(2).to_pylist()
                for content, score, source in zip(contents, scores, sources):
                    if not isinstance(content, str) or not content:
                        continue
                    if score is None or float(score) < min_score:
                        continue
                    if not min_chars <= len(content) <= max_chars:
                        continue
                    yield {"content": content, "score": float(score), "source": source or "unknown"}
        finally:
            if not keep_raw:
                local_path.unlink(missing_ok=True)


class TokenShardWriter:
    def __init__(self, directory, split, shard_tokens):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_tokens = shard_tokens
        self.shards = []
        self.total_tokens = 0
        self._array: np.memmap | None = None
        self._path: Path | None = None
        self._position = 0
        self._hasher: Any = None

    def _open(self):
        index = len(self.shards)
        self._path = self.directory / f"{self.split}_{index:05d}.bin.tmp"
        self._array = np.memmap(self._path, mode="w+", dtype="<u2", shape=(self.shard_tokens,))
        self._position = 0
        self._hasher = hashlib.sha256()

    def write(self, token_ids):
        values = np.asarray(token_ids, dtype=np.int64)
        if values.size == 0:
            return 0
        if values.min() < 0 or values.max() > np.iinfo(np.uint16).max:
            raise ValueError("token ids must fit in uint16")
        written = 0
        while written < values.size:
            if self._array is None:
                self._open()
            assert self._array is not None and self._hasher is not None
            count = min(values.size - written, self.shard_tokens - self._position)
            chunk = values[written:written + count].astype("<u2", copy=False)
            self._array[self._position:self._position + count] = chunk
            self._hasher.update(chunk.tobytes())
            self._position += count
            self.total_tokens += count
            written += count
            if self._position == self.shard_tokens:
                self._close()
        return written

    def _close(self):
        if self._array is None:
            return
        assert self._path is not None and self._hasher is not None
        self._array.flush()
        del self._array
        self._array = None
        with self._path.open("r+b") as handle:
            handle.truncate(self._position * np.dtype("<u2").itemsize)
        final_path = self._path.with_suffix("")
        self._path.replace(final_path)
        self.shards.append({
            "path": final_path.name,
            "tokens": self._position,
            "sha256": self._hasher.hexdigest(),
        })
        self._path = None
        self._position = 0
        self._hasher = None

    def finish(self):
        self._close()
        return self.shards


def _is_validation_document(content, seed, fraction):
    digest = hashlib.blake2b(content.encode("utf-8"), digest_size=8, person=str(seed).encode()[:16]).digest()
    value = int.from_bytes(digest, "big") / 2**64
    return value < fraction


def prepare_dataset(
    *,
    train_tokens=10_000_524_288,
    validation_tokens=20_000_000,
    shard_tokens=100_000_000,
    validation_fraction=0.002,
    seed=42,
    min_score=0.8,
    output_dir=None,
    document_iterator=None,
):
    tokenizer = get_tokenizer()
    if tokenizer.vocab_size > 65536:
        raise ValueError("packed uint16 data requires vocab_size <= 65536")
    output_dir = Path(output_dir or default_data_dir / "packed")
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"dataset already exists: {output_dir}")
        output_dir.rmdir()
    staging = output_dir.with_name(output_dir.name + ".building")
    if staging.exists():
        raise FileExistsError(f"incomplete dataset build exists: {staging}")
    staging.mkdir(parents=True)
    train_writer = TokenShardWriter(staging, "train", shard_tokens)
    val_writer = TokenShardWriter(staging, "val", shard_tokens)
    bos = tokenizer.bos_id
    eos = tokenizer.eos_id
    if document_iterator is None:
        source_revision = get_dataset_revision()
        parquet_urls = get_parquet_urls(source_revision)
        parquet_list_hash = hashlib.sha256("\n".join(parquet_urls).encode()).hexdigest()
        documents = iter_documents(seed=seed, min_score=min_score, urls=parquet_urls)
    else:
        source_revision = "injected"
        parquet_list_hash = None
        documents = document_iterator
    document_count = 0
    source_counts = {}

    for document in documents:
        content = document["content"]
        is_validation = _is_validation_document(content, seed, validation_fraction)
        writer = val_writer if is_validation else train_writer
        target = validation_tokens if is_validation else train_tokens
        if writer.total_tokens >= target:
            continue
        token_ids = tokenizer.encode(content, bos=True, eos=True)
        writer.write(token_ids)
        document_count += 1
        source = document.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        if train_writer.total_tokens >= train_tokens and val_writer.total_tokens >= validation_tokens:
            break
    else:
        raise RuntimeError("ultra-fineweb was exhausted before reaching the requested token budgets")

    manifest = {
        "format": "speck_packed_tokens",
        "format_version": format_version,
        "dtype": "<u2",
        "dataset": {
            "repo": dataset_repo,
            "language": dataset_language,
            "revision": source_revision,
            "parquet_list_hash": parquet_list_hash,
            "seed": seed,
            "min_score": min_score,
            "validation_fraction": validation_fraction,
        },
        "tokenizer": {
            "fingerprint": tokenizer.fingerprint(),
            "vocab_size": tokenizer.vocab_size,
            "bos_token_id": bos,
            "eos_token_id": eos,
        },
        "documents": document_count,
        "sources": source_counts,
        "splits": {
            "train": {"tokens": train_writer.total_tokens, "shards": train_writer.finish()},
            "val": {"tokens": val_writer.total_tokens, "shards": val_writer.finish()},
        },
    }
    manifest_path = staging / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path)
    staging.replace(output_dir)
    manifest_path = output_dir / "manifest.json"
    print(f"prepared {train_writer.total_tokens:,} train and {val_writer.total_tokens:,} validation tokens")
    print(f"manifest: {manifest_path}")
    return manifest


def load_manifest(data_dir=None):
    data_dir = Path(data_dir or default_data_dir / "packed")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"packed dataset not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != format_version:
        raise ValueError(f"unsupported packed dataset version: {manifest.get('format_version')}")
    return manifest


def verify_shards(data_dir=None, manifest=None):
    data_dir = Path(data_dir or default_data_dir / "packed")
    manifest = manifest or load_manifest(data_dir)
    for split in manifest["splits"].values():
        for shard in split["shards"]:
            path = data_dir / shard["path"]
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    hasher.update(chunk)
            if hasher.hexdigest() != shard["sha256"]:
                raise ValueError(f"packed shard checksum mismatch: {path}")
