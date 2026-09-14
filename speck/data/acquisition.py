"""Dataset acquisition components."""

import gzip
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

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
from speck.provenance.io import lines_sha256 as _line_hash


def _shuffle_seed(source_id, seed):
    payload = f"{source_id}\0{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def discover_source_files(source, seed, api=None):
    """Resolve one source revision and discover or verify its input files."""

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
    suffix = _SOURCE_FILE_SUFFIXES[source["file_format"]]
    available = {
        entry.path for entry in entries if getattr(entry, "path", "").lower().endswith(suffix)
    }
    files = source["files"]
    if files is not None:
        missing = set(files) - available
        if missing:
            raise RuntimeError(
                f"source {source['id']} repository tree is missing configured files: "
                f"{', '.join(sorted(missing))}"
            )
        files = list(files)
    else:
        files = sorted(available)
    if not files:
        raise RuntimeError(
            f"source {source['id']} repository tree contains no {source['file_format']} files"
        )
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


def _download_file(url, destination, description, attempts=20, repo=None):
    """Download one revision-pinned dataset file through the HF/Xet cache."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = unquote(urlparse(url).path)
    try:
        prefix, resolved = path.split("/resolve/", 1)
        revision, filename = resolved.split("/", 1)
        repo = repo or prefix.split("/datasets/", 1)[1]
    except (IndexError, ValueError) as error:
        raise ValueError(f"unexpected Hugging Face dataset URL: {url}") from error
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
        except (OSError, HfHubHTTPError, RuntimeError):
            destination.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                shutil.rmtree(cache_dir, ignore_errors=True)
                raise
            time.sleep(min(2**attempt, 60))


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


def _row_window(start_row, stop_row):
    if isinstance(start_row, bool) or not isinstance(start_row, int) or start_row < 0:
        raise ValueError("start_row must be a nonnegative integer")
    if stop_row is not None and (
        isinstance(stop_row, bool) or not isinstance(stop_row, int) or stop_row <= start_row
    ):
        raise ValueError("stop_row must be an integer greater than start_row")


def iter_parquet_documents(
    *,
    source,
    revision,
    filename,
    filtering,
    cache_dir=None,
    keep_raw=False,
    description=None,
    start_row=0,
    stop_row=None,
):
    """Yield filtered rows from one downloaded repository Parquet file."""

    _row_window(start_row, stop_row)
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
                if stop_row is not None and absolute_row >= stop_row:
                    return
                if absolute_row < start_row:
                    continue
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


def iter_jsonl_gzip_documents(
    *,
    source,
    revision,
    filename,
    filtering,
    cache_dir=None,
    keep_raw=False,
    description=None,
    start_row=0,
    stop_row=None,
):
    """Yield filtered rows from gzip or Zstandard JSONL, preserving physical row positions."""

    _row_window(start_row, stop_row)
    source = _validate_source(source)
    cache_dir = Path(cache_dir or default_data_dir / "raw")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(f"{source['repo']}\0{revision}\0{filename}".encode()).hexdigest()[
        :20
    ]
    zstd = source["file_format"] == "jsonl_zstd"
    local_path = cache_dir / (f"{cache_key}.zst" if zstd else f"{cache_key}.json.gz")
    if not local_path.exists():
        _download_file(
            _dataset_url(source["repo"], revision, filename),
            local_path,
            description or source["id"],
            repo=source["repo"],
        )
    required = {source["content_column"]}
    if source.get("language_column"):
        required.add(source["language_column"])
    if "min_score" in source["filters"]:
        required.add(source["score_column"])
    try:
        stream = (
            io.TextIOWrapper(pa.input_stream(str(local_path), compression="zstd"), encoding="utf-8")
            if zstd
            else gzip.open(local_path, "rt", encoding="utf-8")
        )
        with stream as handle:
            for row_number, line in enumerate(handle):
                if stop_row is not None and row_number >= stop_row:
                    return
                if row_number < start_row:
                    continue
                if not line.strip():
                    continue
                try:
                    values = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{filename} has invalid JSON at row {row_number}") from error
                if not isinstance(values, dict):
                    raise ValueError(f"{filename} row {row_number} must be an object")
                missing = required - values.keys()
                if missing:
                    raise ValueError(
                        f"{filename} is missing configured columns at row {row_number}: "
                        f"{sorted(missing)}"
                    )
                content = values[source["content_column"]]
                if not isinstance(content, str) or not content:
                    continue
                if not filtering["min_chars"] <= len(content) <= filtering["max_chars"]:
                    continue
                score = None
                score_column = source.get("score_column")
                if score_column and score_column in values:
                    raw_score = values[score_column]
                    try:
                        score = None if raw_score is None else float(raw_score)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"{filename} score at row {row_number} is not numeric"
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
                if language and language_column and values[language_column] != language:
                    continue
                detector = source.get("language_detector")
                if language and detector and _detect_language(content, detector) != language:
                    continue
                metadata = {}
                for alias, column in source["metadata_columns"].items():
                    value = values.get(column)
                    if zstd and column not in values:
                        value = values
                        for part in column.split("."):
                            value = value.get(part) if isinstance(value, dict) else None
                    if value is not None:
                        metadata[alias] = _metadata_value(value)
                yield {
                    "content": content,
                    "score": score,
                    "metadata": metadata,
                    "file": filename,
                    "row": row_number,
                }
    finally:
        if not keep_raw:
            local_path.unlink(missing_ok=True)


def iter_source_file_documents(**kwargs):
    """Dispatch one source file to its configured reader."""

    source = _validate_source(kwargs["source"])
    kwargs["source"] = source
    if source["file_format"] == "parquet":
        return iter_parquet_documents(**kwargs)
    if source["file_format"] in ("jsonl_gzip", "jsonl_zstd"):
        return iter_jsonl_gzip_documents(**kwargs)
    raise ValueError(f"unsupported source file format: {source['file_format']}")
