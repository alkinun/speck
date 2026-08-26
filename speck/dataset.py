"""Stream, deduplicate, tokenize, and pack source-separated training data."""

import hashlib
import json
import math
import os
import random
import shutil
import time
import unicodedata
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from speck.common import base_dir
from speck.tokenizer import get_tokenizer

format_version = 3
default_data_dir = Path(base_dir()) / "data"

_SOURCE_FIELDS = {
    "id",
    "repo",
    "revision",
    "tree_path",
    "content_column",
    "language_detector",
    "score_column",
    "language_column",
    "metadata_columns",
    "filters",
}
_DEDUP_SETTINGS = {
    "normalization": "NFKC+lower+whitespace",
    "hash": "blake2b-128",
    "scope": "global",
}
_DEDUP_BYTES = 16
_MAX_TOKENIZER_DOCUMENTS = 1024
_MAX_TOKENIZER_CHARACTERS = 2_000_000
_RAW_SHARD_ALLOWANCE_BYTES = 20 * 1024**3
_MIN_INDEX_DEDUP_HEADROOM_BYTES = 5 * 1024**3
_LANGUAGE_DETECTORS = {"py3langid"}


def resolve_data_dir(output_dir=None, output_name=None):
    """Resolve an explicit path or an isolated name under the Speck data cache."""

    if output_dir is not None:
        return Path(output_dir).expanduser()
    if output_name is None:
        return default_data_dir / "packed"
    if not isinstance(output_name, str) or not output_name or Path(output_name).name != output_name:
        raise ValueError("data output_name must be one nonempty path component")
    return default_data_dir / output_name


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


def _fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _line_hash(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def derive_source_quotas(sources, mixture, requested_train_tokens):
    """Derive exact per-source token quotas from integer phase weights."""

    requested_train_tokens = _integer(requested_train_tokens, "requested_train_tokens", minimum=1)
    source_ids = [source["id"] if isinstance(source, dict) else str(source) for source in sources]
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("data sources must have unique IDs")
    if not isinstance(mixture, dict) or not isinstance(mixture.get("phases"), list):
        raise ValueError("mixture.phases must be a list")
    phases = mixture["phases"]
    if not phases:
        raise ValueError("mixture requires at least one phase")

    quotas = {source_id: 0 for source_id in source_ids}
    previous_end = 0
    normalized_phases = []
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise ValueError(f"mixture phase {index} must be an object")
        end = _integer(phase.get("end_tokens"), f"mixture phase {index} end_tokens", minimum=1)
        if end <= previous_end:
            raise ValueError("mixture phase ends must increase monotonically")
        weights = phase.get("weights")
        if not isinstance(weights, dict):
            raise ValueError(f"mixture phase {index} weights must be an object")
        unknown = set(weights) - set(source_ids)
        missing = set(source_ids) - set(weights)
        if unknown:
            raise ValueError(
                f"mixture phase {index} has unknown sources: {', '.join(sorted(unknown))}"
            )
        if missing:
            raise ValueError(
                f"mixture phase {index} is missing sources: {', '.join(sorted(missing))}"
            )
        ordered_weights = {}
        fractional_weights = {}
        for source_id in source_ids:
            value = weights[source_id]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"mixture weight for {source_id} must be a number >= 0")
            ordered_weights[source_id] = value
            fractional_weights[source_id] = Fraction(str(value))
        if sum(fractional_weights.values()) != 100:
            raise ValueError(f"mixture phase {index} weights must sum to 100")
        duration = end - previous_end
        for source_id, weight in fractional_weights.items():
            tokens = duration * weight / 100
            if tokens.denominator != 1:
                raise ValueError(
                    f"mixture phase {index} produces a fractional quota for {source_id}"
                )
            quotas[source_id] += tokens.numerator
        normalized_phases.append({"end_tokens": end, "weights": ordered_weights})
        previous_end = end
    if previous_end != requested_train_tokens:
        raise ValueError("final mixture phase must end at requested_train_tokens")
    if sum(quotas.values()) != requested_train_tokens:
        raise ValueError("derived source quotas do not equal requested_train_tokens")
    return quotas, normalized_phases


def _validate_source(source):
    if not isinstance(source, dict):
        raise ValueError("each data source must be an object")
    unknown = set(source) - _SOURCE_FIELDS
    if unknown:
        raise ValueError(f"unknown source settings: {', '.join(sorted(unknown))}")
    required = {"id", "repo", "tree_path", "content_column"}
    missing = required - source.keys()
    if missing:
        raise ValueError(f"data source is missing settings: {', '.join(sorted(missing))}")
    source_id = source["id"]
    if not isinstance(source_id, str) or not source_id or Path(source_id).name != source_id:
        raise ValueError("source ID must be one nonempty path component")
    for key in ("repo", "tree_path", "content_column"):
        if not isinstance(source[key], str) or (key != "tree_path" and not source[key]):
            raise ValueError(f"source {source_id} {key} must be a string")
    for key in ("score_column", "language_column", "revision"):
        if source.get(key) is not None and not isinstance(source[key], str):
            raise ValueError(f"source {source_id} {key} must be null or a string")
    language_detector = source.get("language_detector")
    if language_detector is not None and language_detector not in _LANGUAGE_DETECTORS:
        raise ValueError(f"source {source_id} has unsupported language_detector")
    metadata = source.get("metadata_columns", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(alias, str) or not isinstance(column, str)
        for alias, column in metadata.items()
    ):
        raise ValueError(f"source {source_id} metadata_columns must map names to columns")
    filters = source.get("filters", {})
    if not isinstance(filters, dict) or set(filters) - {
        "min_score",
        "score_operator",
        "language",
    }:
        raise ValueError(f"source {source_id} has unsupported filters")
    if "min_score" in filters:
        if not source.get("score_column"):
            raise ValueError(f"source {source_id} score filter requires score_column")
        minimum = filters["min_score"]
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, (int, float))
            or not math.isfinite(minimum)
        ):
            raise ValueError(f"source {source_id} min_score must be numeric")
    score_operator = filters.get("score_operator")
    if score_operator is not None:
        if "min_score" not in filters:
            raise ValueError(f"source {source_id} score_operator requires min_score")
        if score_operator not in {">", ">="}:
            raise ValueError(f"source {source_id} has unsupported score_operator")
    if language_detector is not None and "language" not in filters:
        raise ValueError(f"source {source_id} language_detector requires a language filter")
    if source.get("language_column") and language_detector:
        raise ValueError(
            f"source {source_id} cannot use both language_column and language_detector"
        )
    if "language" in filters and not (source.get("language_column") or language_detector):
        raise ValueError(
            f"source {source_id} language filter requires language_column or language_detector"
        )
    return {
        **source,
        "revision": source.get("revision"),
        "score_column": source.get("score_column"),
        "language_column": source.get("language_column"),
        "metadata_columns": dict(metadata),
        "filters": dict(filters),
    }


def validate_data_settings(
    *,
    sources,
    mixture,
    requested_train_tokens,
    validation_tokens_per_source,
    validation_fraction,
    filtering,
    dedup,
    shards,
):
    """Validate preparation settings and return their normalized derived values."""

    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    normalized_sources = [_validate_source(source) for source in sources]
    quotas, phases = derive_source_quotas(normalized_sources, mixture, requested_train_tokens)
    validation_tokens_per_source = _integer(
        validation_tokens_per_source, "validation_tokens_per_source", minimum=1
    )
    if not isinstance(validation_fraction, (int, float)) or not 0 < validation_fraction <= 1:
        raise ValueError("validation_fraction must be in (0, 1]")
    if not isinstance(filtering, dict) or set(filtering) != {"min_chars", "max_chars"}:
        raise ValueError("filtering must contain exactly min_chars and max_chars")
    min_chars = _integer(filtering["min_chars"], "filtering.min_chars", minimum=0)
    max_chars = _integer(filtering["max_chars"], "filtering.max_chars", minimum=1)
    if min_chars > max_chars:
        raise ValueError("filtering.min_chars cannot exceed filtering.max_chars")
    if max_chars > _MAX_TOKENIZER_CHARACTERS:
        raise ValueError(f"filtering.max_chars cannot exceed {_MAX_TOKENIZER_CHARACTERS:,}")
    if dedup != _DEDUP_SETTINGS:
        raise ValueError(f"dedup must be {_DEDUP_SETTINGS}")
    if not isinstance(shards, dict) or set(shards) != {
        "tokens",
        "maximum_loader_microbatch_tokens",
    }:
        raise ValueError("shards must contain tokens and maximum_loader_microbatch_tokens")
    shard_tokens = _integer(shards["tokens"], "shards.tokens", minimum=1)
    maximum_microbatch = _integer(
        shards["maximum_loader_microbatch_tokens"],
        "shards.maximum_loader_microbatch_tokens",
    )
    reserve = (len(phases) + 1) * maximum_microbatch if maximum_microbatch else 0
    return {
        "sources": normalized_sources,
        "phases": phases,
        "quotas": quotas,
        "validation_tokens_per_source": validation_tokens_per_source,
        "validation_fraction": float(validation_fraction),
        "filtering": {"min_chars": min_chars, "max_chars": max_chars},
        "dedup": dict(dedup),
        "shards": {
            "tokens": shard_tokens,
            "maximum_loader_microbatch_tokens": maximum_microbatch,
        },
        "train_reserve_tokens_per_source": reserve,
    }


def estimate_disk_requirement(settings, requested_train_tokens):
    """Conservatively estimate bytes needed to stage and publish packed data."""

    source_count = len(settings["sources"])
    packed_tokens = (
        requested_train_tokens
        + settings["train_reserve_tokens_per_source"] * source_count
        + settings["validation_tokens_per_source"] * source_count
    )
    packed_bytes = packed_tokens * np.dtype("<u2").itemsize
    index_dedup_headroom = max(
        _MIN_INDEX_DEDUP_HEADROOM_BYTES,
        packed_bytes // 2,
    )
    components = {
        "packed_uint16_bytes": packed_bytes,
        "temporary_raw_shard_bytes": _RAW_SHARD_ALLOWANCE_BYTES,
        "dedup_index_headroom_bytes": index_dedup_headroom,
    }
    return {"required_bytes": sum(components.values()), "components": components}


def _tree_size(path):
    path = Path(path)
    if not path.is_dir():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def disk_preflight(
    output_dir,
    settings,
    requested_train_tokens,
    *,
    check=True,
    disk_usage=None,
):
    """Check live capacity, crediting reusable bytes in an existing staged build."""

    output_dir = Path(output_dir)
    estimate = estimate_disk_requirement(settings, requested_train_tokens)
    staging = output_dir.with_name(output_dir.name + ".building")
    reusable = _tree_size(staging)
    usage = (disk_usage or shutil.disk_usage)(output_dir.parent)
    report = {
        **estimate,
        "checked": bool(check),
        "free_bytes": usage.free,
        "reusable_staged_bytes": reusable,
        "effective_available_bytes": usage.free + reusable,
    }
    if check and report["effective_available_bytes"] < report["required_bytes"]:
        raise OSError(
            "insufficient disk space for packed data: "
            f"required {report['required_bytes']:,} bytes, "
            f"available {report['free_bytes']:,} bytes "
            f"(+ {reusable:,} reusable staged bytes)"
        )
    return report


def _shuffle_seed(source_id, seed):
    payload = f"{source_id}\0{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def discover_parquet_files(source, seed, api=None):
    """Resolve one source revision and discover its raw repository Parquet files."""

    source = _validate_source(source)
    api = api or HfApi()
    info = api.dataset_info(source["repo"], revision=source["revision"])
    revision = info.sha
    entries = api.list_repo_tree(
        source["repo"],
        path_in_repo=source["tree_path"] or None,
        recursive=True,
        revision=revision,
        repo_type="dataset",
    )
    files = sorted(
        {entry.path for entry in entries if getattr(entry, "path", "").lower().endswith(".parquet")}
    )
    if not files:
        raise RuntimeError(f"source {source['id']} repository tree contains no Parquet files")
    random.Random(_shuffle_seed(source["id"], seed)).shuffle(files)
    return {
        "revision": revision,
        "files": files,
        "file_list_sha256": _line_hash(files),
    }


def _dataset_url(repo, revision, filename):
    return (
        f"https://huggingface.co/datasets/{repo}/resolve/{quote(revision, safe='')}"
        f"/{quote(filename, safe='/')}"
    )


def _download_file(url, destination, description, attempts=5, repo=None):
    """Download one revision-pinned dataset file through the HF/Xet cache."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = unquote(urlparse(url).path)
    try:
        prefix, resolved = path.split("/resolve/", 1)
        revision, filename = resolved.split("/", 1)
        repo = repo or prefix.split("/datasets/", 1)[1]
    except (IndexError, ValueError) as error:
        raise ValueError(f"unexpected hugging face dataset url: {url}") from error
    cache_dir = destination.parent / f".{destination.stem}.download"
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    for attempt in range(attempts):
        try:
            print(f"{description}: {filename}")
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_dir,
            )
            shutil.move(Path(downloaded).resolve(), destination)
            shutil.rmtree(cache_dir, ignore_errors=True)
            return
        except (OSError, HfHubHTTPError):
            destination.unlink(missing_ok=True)
            shutil.rmtree(cache_dir, ignore_errors=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)


def _metadata_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _is_string_type(value):
    return pa.types.is_string(value) or pa.types.is_large_string(value)


def _score_passes(score, minimum, operator):
    if score is None or not math.isfinite(score):
        return False
    return score > minimum if operator == ">" else score >= minimum


@lru_cache(maxsize=1)
def _py3langid_identifier():
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    return LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)


def _detect_language(content, detector):
    if detector == "py3langid":
        return _py3langid_identifier().classify(content)[0]
    raise ValueError(f"unsupported language detector: {detector}")


def _validate_parquet_schema(parquet, source, filename):
    schema = parquet.schema_arrow
    available = set(schema.names)
    content_column = source["content_column"]
    required = {content_column}
    if source.get("language_column"):
        required.add(source["language_column"])
    if "min_score" in source["filters"]:
        required.add(source["score_column"])
    missing = required - available
    if missing:
        raise ValueError(f"{filename} is missing configured columns: {sorted(missing)}")
    content_type = schema.field(content_column).type
    if not _is_string_type(content_type):
        raise ValueError(f"{filename} content column must contain strings")
    language_column = source.get("language_column")
    if language_column:
        language_type = schema.field(language_column).type
        if not _is_string_type(language_type):
            raise ValueError(f"{filename} language column must contain strings")
    score_column = source.get("score_column")
    if score_column and score_column in available:
        score_type = schema.field(score_column).type
        if not (
            pa.types.is_integer(score_type)
            or pa.types.is_floating(score_type)
            or pa.types.is_decimal(score_type)
            or pa.types.is_string(score_type)
            or pa.types.is_large_string(score_type)
        ):
            raise ValueError(f"{filename} score column must be numeric or numeric text")
    columns = [content_column]
    optional = [score_column, language_column, *source["metadata_columns"].values()]
    for column in optional:
        if column and column in available and column not in columns:
            columns.append(column)
    return columns


def iter_parquet_documents(
    *,
    source,
    revision,
    filename,
    filtering,
    cache_dir=None,
    keep_raw=False,
    description=None,
):
    """Yield filtered rows from one downloaded repository Parquet file."""

    source = _validate_source(source)
    cache_dir = Path(cache_dir or default_data_dir / "raw")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(f"{source['repo']}\0{revision}\0{filename}".encode()).hexdigest()[
        :20
    ]
    local_path = cache_dir / f"{cache_key}.parquet"
    if not local_path.exists():
        _download_file(
            _dataset_url(source["repo"], revision, filename),
            local_path,
            description or source["id"],
            repo=source["repo"],
        )
    try:
        parquet = pq.ParquetFile(local_path)
        columns = _validate_parquet_schema(parquet, source, filename)
        row_number = 0
        for batch in parquet.iter_batches(columns=columns, batch_size=2048):
            values = {
                column: batch.column(index).to_pylist() for index, column in enumerate(columns)
            }
            for row_index, content in enumerate(values[source["content_column"]]):
                absolute_row = row_number + row_index
                metadata = {
                    alias: _metadata_value(values[column][row_index])
                    for alias, column in source["metadata_columns"].items()
                    if column in values and values[column][row_index] is not None
                }
                if not isinstance(content, str) or not content:
                    continue
                if not filtering["min_chars"] <= len(content) <= filtering["max_chars"]:
                    continue
                score = None
                score_column = source.get("score_column")
                if score_column and score_column in values:
                    raw_score = values[score_column][row_index]
                    try:
                        score = None if raw_score is None else float(raw_score)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"{filename} score at row {absolute_row} is not numeric"
                        ) from error
                    if score is not None and not math.isfinite(score):
                        continue
                    minimum = source["filters"].get("min_score")
                    if minimum is not None and not _score_passes(
                        score,
                        minimum,
                        source["filters"].get("score_operator", ">="),
                    ):
                        continue
                language = source["filters"].get("language")
                language_column = source.get("language_column")
                if language and language_column and values[language_column][row_index] != language:
                    continue
                detector = source.get("language_detector")
                if language and detector and _detect_language(content, detector) != language:
                    continue
                yield {
                    "content": content,
                    "score": score,
                    "metadata": metadata,
                    "file": filename,
                    "row": absolute_row,
                }
            row_number += len(batch)
    finally:
        if not keep_raw:
            local_path.unlink(missing_ok=True)


def iter_documents(
    *,
    source,
    revision,
    files,
    filtering,
    cache_dir=None,
    keep_raw=False,
):
    """Yield files sequentially while retaining at most one raw Parquet shard."""

    files = list(files)
    for shard_index, filename in enumerate(files):
        yield from iter_parquet_documents(
            source=source,
            revision=revision,
            filename=filename,
            filtering=filtering,
            cache_dir=cache_dir,
            keep_raw=keep_raw,
            description=f"{source['id']} {shard_index + 1}/{len(files)}",
        )


class TokenShardWriter:
    """Write token IDs into bounded, checksummed uint16 shards."""

    def __init__(self, directory, split, shard_tokens, *, shards=None, total_tokens=0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_tokens = shard_tokens
        self.shards = list(shards or [])
        self.total_tokens = total_tokens
        if sum(shard["tokens"] for shard in self.shards) != self.total_tokens:
            raise ValueError(f"invalid restored {split} shard totals")
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
            chunk = values[written : written + count].astype("<u2", copy=False)
            self._array[self._position : self._position + count] = chunk
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
        self.shards.append(
            {
                "path": final_path.name,
                "tokens": self._position,
                "sha256": self._hasher.hexdigest(),
            }
        )
        self._path = None
        self._position = 0
        self._hasher = None

    def finish(self):
        self._close()
        return self.shards


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


def _slice_hash(path, start, end):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        handle.seek(start)
        remaining = end - start
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"file ended before integrity boundary: {path}")
            hasher.update(chunk)
            remaining -= len(chunk)
    return hasher.hexdigest()


def _truncate(path, size):
    with Path(path).open("r+b") as handle:
        handle.truncate(size)
        _sync_file(handle)


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
        self.targets = {
            "train": train_requested + train_reserve,
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
        self._synced_shards = {
            shard["path"] for writer in self.writers.values() for shard in writer.shards
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

    @property
    def complete(self):
        return all(
            self.writers[split].total_tokens >= self.targets[split] for split in self.writers
        )

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
            preferred = (
                "val"
                if _is_validation_document(normalized, self.seed, self.validation_fraction)
                else "train"
            )
            if self.writers[preferred].total_tokens >= self.targets[preferred]:
                continue
            pending.add(digest_integer)
            rows.append((document, digest, digest_integer, preferred))
        if not rows:
            return
        token_rows = self.tokenizer.encode_batch(
            [row[0]["content"] for row in rows], bos=True, eos=True
        )
        if len(token_rows) != len(rows):
            raise ValueError("tokenizer returned the wrong number of encoded documents")
        for (document, digest, digest_integer, split), token_ids in zip(rows, token_rows):
            if self.complete:
                break
            writer = self.writers[split]
            if writer.total_tokens >= self.targets[split]:
                continue
            start_token = writer.total_tokens
            written = writer.write(token_ids)
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
                "end_token": start_token + written,
                "source_id": self.source_id,
                "split": split,
                "start_token": start_token,
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
            line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            self.index_file.write(line)
            self.index_hash.update(line)
            self.document_counts[split] += 1

    def consume(self, documents):
        batch = []
        characters = 0
        for document in documents:
            content = document.get("content")
            length = len(content) if isinstance(content, str) else 0
            if batch and (
                len(batch) >= _MAX_TOKENIZER_DOCUMENTS
                or characters + length > _MAX_TOKENIZER_CHARACTERS
            ):
                self._process(batch)
                batch.clear()
                characters = 0
                if self.complete:
                    return
            batch.append(document)
            characters += length
            if len(batch) >= _MAX_TOKENIZER_DOCUMENTS or characters >= _MAX_TOKENIZER_CHARACTERS:
                self._process(batch)
                batch.clear()
                characters = 0
                if self.complete:
                    return
        if batch and not self.complete:
            self._process(batch)

    def _sync_outputs(self):
        for writer in self.writers.values():
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
            raise RuntimeError(
                f"source {self.source_id} was exhausted before its budgets: {missing}"
            )
        self._sync_outputs()
        self.index_file.close()
        final_index = self.index_path
        journal_end = self.dedup_file.tell()
        records = sum(self.document_counts.values())
        split_summaries = {}
        for split in ("train", "val"):
            requested = self.train_requested if split == "train" else self.validation_requested
            target = self.targets[split]
            split_summaries[split] = {
                "requested_tokens": requested,
                "preparation_target_tokens": target,
                "tokens": self.writers[split].total_tokens,
                "reserve_tokens": self.train_reserve if split == "train" else 0,
                "overshoot_tokens": self.writers[split].total_tokens - target,
                "documents": self.document_counts[split],
                "shards": _prefixed_shards(self.writers[split].shards, self.source_id),
            }
        summary = {
            "id": self.source_id,
            "repo": self.source["repo"],
            "revision": self.resolved["revision"],
            "tree_path": self.source["tree_path"],
            "content_column": self.source["content_column"],
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
        _atomic_json(self.directory / "source.json", summary)
        return summary


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
    if _slice_hash(dedup_path, dedup_start, journal["end_byte"]) != journal["sha256"]:
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
    if _slice_hash(dedup_path, expected_start, journal["end_byte"]) != journal["sha256"]:
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
    output_dir = resolve_data_dir(output_dir, output_name)
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"dataset already exists: {output_dir}")
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(output_dir.name + ".building")
    if staging.exists() and restart:
        shutil.rmtree(staging)
    disk_report = disk_preflight(
        output_dir,
        settings,
        requested_train_tokens,
        check=check_disk,
        disk_usage=disk_usage,
    )
    if check_disk:
        print(
            f"disk preflight: required {disk_report['required_bytes']:,} bytes, "
            f"free {disk_report['free_bytes']:,} bytes"
        )

    tokenizer_manifest = {
        "fingerprint": tokenizer.fingerprint(),
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_id,
        "eos_token_id": tokenizer.eos_id,
    }
    contract = {
        "format_version": format_version,
        "sources": settings["sources"],
        "mixture": {"phases": settings["phases"]},
        "requested_train_tokens": requested_train_tokens,
        "validation_tokens_per_source": validation_tokens_per_source,
        "validation_fraction": validation_fraction,
        "filtering": settings["filtering"],
        "dedup": settings["dedup"],
        "shards": settings["shards"],
        "seed": seed,
        "tokenizer": tokenizer_manifest,
    }
    contract_hash = _fingerprint(contract)
    state_path = staging / "build_state.json"
    if not staging.exists():
        staging.mkdir(parents=True)
        (staging / "sources").mkdir()
        state = {
            "format_version": format_version,
            "contract": contract_hash,
            "completed_sources": [],
            "current_source": None,
            "resolved_sources": {},
            "disk_preflight": disk_report,
        }
        _atomic_json(state_path, state)
    else:
        if not state_path.is_file():
            raise ValueError(f"invalid staged build: {staging}; pass --restart to replace it")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("contract") != contract_hash:
            raise ValueError("staged build settings changed; pass --restart to replace it")
        state["disk_preflight"] = disk_report
        _atomic_json(state_path, state)

    document_iterators = document_iterators or {}
    unknown_iterators = set(document_iterators) - {source["id"] for source in settings["sources"]}
    if unknown_iterators:
        raise ValueError(f"document iterators have unknown sources: {', '.join(unknown_iterators)}")
    api = api or HfApi()
    for source in settings["sources"]:
        source_id = source["id"]
        if source_id in state["resolved_sources"]:
            continue
        if source_id in document_iterators:
            resolved = {
                "revision": "injected",
                "files": [],
                "file_list_sha256": _line_hash([]),
            }
        else:
            resolved = discover_parquet_files(source, seed, api)
        state["resolved_sources"][source_id] = resolved
        _atomic_json(state_path, state)

    dedup_path = staging / "dedup_hashes.bin"
    dedup_path.touch(exist_ok=True)
    source_ids = [source["id"] for source in settings["sources"]]
    completed = state["completed_sources"]
    if completed != source_ids[: len(completed)]:
        raise ValueError("staged completed sources are not a source-order prefix")
    summaries = {}
    journal_boundary = 0
    for source_id in completed:
        directory = staging / "sources" / source_id
        summary = _source_summary(directory)
        if summary.get("id") != source_id:
            raise ValueError("completed staged source ID is inconsistent")
        journal_boundary = _verify_source_integrity(
            directory,
            summary,
            dedup_path,
            journal_boundary,
        )
        summaries[source_id] = summary

    current = state.get("current_source")
    if current:
        source_id = current["id"]
        expected_source = source_ids[len(completed)] if len(completed) < len(source_ids) else None
        if source_id != expected_source or current["dedup_bytes_before"] != journal_boundary:
            raise ValueError("staged current source boundary is inconsistent")
        final_directory = staging / "sources" / source_id
        temporary_directory = staging / "sources" / f"{source_id}.building"
        if final_directory.is_dir():
            summary = _source_summary(final_directory)
            journal_boundary = _verify_source_integrity(
                final_directory,
                summary,
                dedup_path,
                journal_boundary,
                require_journal_end=True,
            )
            state["completed_sources"].append(source_id)
            state["current_source"] = None
            _atomic_json(state_path, state)
            (final_directory / "source_progress.json").unlink(missing_ok=True)
            _fsync_directory(final_directory)
            summaries[source_id] = summary
        elif (
            current.get("mode") == "files"
            and (temporary_directory / "source_progress.json").is_file()
        ):
            progress_path = temporary_directory / "source_progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("source_id") != source_id:
                raise ValueError("staged source progress has the wrong source ID")
            _recover_source_progress(
                temporary_directory,
                progress,
                state["resolved_sources"][source_id],
                dedup_path,
                journal_boundary,
            )
        else:
            if dedup_path.stat().st_size < journal_boundary:
                raise ValueError("staged dedup journal is shorter than completed sources")
            if dedup_path.stat().st_size > journal_boundary:
                _truncate(dedup_path, journal_boundary)
            shutil.rmtree(temporary_directory, ignore_errors=True)
            state["current_source"] = None
            _atomic_json(state_path, state)
    elif dedup_path.stat().st_size != journal_boundary:
        raise ValueError("staged dedup journal does not end at the completed-source boundary")

    accepted_hashes = _load_hashes(dedup_path)
    raw_directory = staging / ".raw"
    for source in settings["sources"]:
        source_id = source["id"]
        final_directory = staging / "sources" / source_id
        if source_id in state["completed_sources"]:
            continue
        temporary_directory = staging / "sources" / f"{source_id}.building"
        resuming = (state.get("current_source") or {}).get("id") == source_id
        if not resuming:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            temporary_directory.mkdir(parents=True)
            dedup_start = dedup_path.stat().st_size
            state["current_source"] = {
                "id": source_id,
                "dedup_bytes_before": dedup_start,
                "mode": "injected" if source_id in document_iterators else "files",
            }
            _atomic_json(state_path, state)
        else:
            dedup_start = state["current_source"]["dedup_bytes_before"]
        resolved = state["resolved_sources"][source_id]
        progress_path = temporary_directory / "source_progress.json"
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.is_file()
            else None
        )
        builder_settings = {
            "directory": temporary_directory,
            "source": source,
            "resolved": resolved,
            "tokenizer": tokenizer,
            "accepted_hashes": accepted_hashes,
            "dedup_start": dedup_start,
            "train_requested": settings["quotas"][source_id],
            "train_reserve": settings["train_reserve_tokens_per_source"],
            "validation_requested": settings["validation_tokens_per_source"],
            "validation_fraction": settings["validation_fraction"],
            "shard_tokens": settings["shards"]["tokens"],
            "filtering": settings["filtering"],
            "seed": seed,
        }
        with dedup_path.open("ab") as dedup_file:
            builder_settings["dedup_file"] = dedup_file
            if source_id in document_iterators:
                summary = _prepare_injected_source(
                    documents=iter(document_iterators[source_id]),
                    **builder_settings,
                )
            else:
                builder = SourceBuilder(progress=progress, **builder_settings)
                if progress is None:
                    progress = builder.progress(0)
                next_file = progress["next_file_index"]
                final_file = None
                for file_index in range(next_file, len(resolved["files"])):
                    filename = resolved["files"][file_index]
                    documents = iter_parquet_documents(
                        source=source,
                        revision=resolved["revision"],
                        filename=filename,
                        filtering=settings["filtering"],
                        cache_dir=raw_directory,
                        description=f"{source_id} {file_index + 1}/{len(resolved['files'])}",
                    )
                    try:
                        builder.consume(documents)
                    finally:
                        documents.close()
                    if builder.complete:
                        final_file = filename
                        break
                    progress = builder.progress(file_index + 1)
                summary = builder.finish(
                    files_completed=progress["next_file_index"],
                    final_file=final_file,
                )
        temporary_directory.replace(final_directory)
        _fsync_directory(final_directory.parent)
        (final_directory / "source_progress.json").unlink(missing_ok=True)
        _fsync_directory(final_directory)
        state["completed_sources"].append(source_id)
        state["current_source"] = None
        _atomic_json(state_path, state)
        summaries[source_id] = summary
        journal_boundary = summary["dedup_journal"]["end_byte"]

    if dedup_path.stat().st_size != journal_boundary:
        raise ValueError("dedup journal does not end at the final source boundary")
    ordered_summaries = [summaries[source["id"]] for source in settings["sources"]]
    dedup_checksum = _file_hash(dedup_path)
    aggregate_splits = {}
    for split in ("train", "val"):
        aggregate_splits[split] = {
            "requested_tokens": (
                requested_train_tokens
                if split == "train"
                else validation_tokens_per_source * len(ordered_summaries)
            ),
            "tokens": sum(source["splits"][split]["tokens"] for source in ordered_summaries),
            "documents": sum(source["splits"][split]["documents"] for source in ordered_summaries),
        }
    manifest = {
        "format": "speck_packed_tokens",
        "format_version": format_version,
        "dtype": "<u2",
        "requested_train_tokens": requested_train_tokens,
        "validation_tokens_per_source": validation_tokens_per_source,
        "mixture": {
            "phases": settings["phases"],
            "source_quotas": settings["quotas"],
        },
        "preparation": {
            "seed": seed,
            "validation_fraction": settings["validation_fraction"],
            "filtering": settings["filtering"],
            "shards": settings["shards"],
            "train_reserve_tokens_per_source": settings["train_reserve_tokens_per_source"],
            "reserve_basis": "(phase_count + 1) * maximum_loader_microbatch_tokens",
            "tokenizer_batch": {
                "maximum_documents": _MAX_TOKENIZER_DOCUMENTS,
                "maximum_characters": _MAX_TOKENIZER_CHARACTERS,
            },
            "disk_preflight": disk_report,
        },
        "dedup": {
            **settings["dedup"],
            "path": dedup_path.name,
            "accepted_hashes": dedup_path.stat().st_size // _DEDUP_BYTES,
            "sha256": dedup_checksum,
            "collision_policy": "128-bit collisions are treated as duplicates",
        },
        "tokenizer": tokenizer_manifest,
        "documents": sum(source["documents"] for source in ordered_summaries),
        "sources": ordered_summaries,
        "splits": aggregate_splits,
    }
    _atomic_json(staging / "manifest.json", manifest)
    shutil.rmtree(raw_directory, ignore_errors=True)
    staging.replace(output_dir)
    _fsync_directory(output_dir.parent)
    (output_dir / state_path.name).unlink()
    _fsync_directory(output_dir)
    print(
        f"prepared {manifest['splits']['train']['tokens']:,} train and "
        f"{manifest['splits']['val']['tokens']:,} validation tokens"
    )
    for source in ordered_summaries:
        train = source["splits"]["train"]
        print(
            f"{source['id']}: requested {train['requested_tokens']:,}, "
            f"reserve {train['reserve_tokens']:,}, actual {train['tokens']:,}"
        )
    print(f"manifest: {output_dir / 'manifest.json'}")
    return manifest


def _validate_manifest(manifest):
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
            for shard in split["shards"]:
                path = data_dir / shard["path"]
                expected_bytes = shard["tokens"] * np.dtype("<u2").itemsize
                if not path.is_file() or path.stat().st_size != expected_bytes:
                    raise ValueError(f"invalid packed token shard: {path}")
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
        if _slice_hash(dedup_path, journal["start_byte"], journal["end_byte"]) != journal["sha256"]:
            raise ValueError(f"packed dedup slice checksum mismatch: {source['id']}")
