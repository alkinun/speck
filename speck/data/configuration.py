"""Dataset configuration components."""

import math
import shutil
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np

from speck.operations.runtime import base_dir

format_version = 3


default_data_dir = Path(base_dir()) / "data"


_SOURCE_FIELDS = {
    "id",
    "repo",
    "revision",
    "tree_path",
    "content_column",
    "file_format",
    "files",
    "language_detector",
    "score_column",
    "language_column",
    "metadata_columns",
    "filters",
    "packing",
    "passes",
    "record_format",
}


_DEDUP_SETTINGS = {
    "normalization": "NFKC+lower+whitespace",
    "hash": "blake2b-128",
    "scope": "global",
}


_DEDUP_BYTES = 16


_MAX_TOKENIZER_DOCUMENTS = 1024


_MAX_TOKENIZER_BATCH_CHARACTERS = 2_000_000


_MAX_DOCUMENT_CHARACTERS = 16_000_000


_RAW_SHARD_ALLOWANCE_BYTES = 20 * 1024**3


_MIN_INDEX_DEDUP_HEADROOM_BYTES = 5 * 1024**3


_LANGUAGE_DETECTORS = {"py3langid"}


_SOURCE_FILE_SUFFIXES = {
    "jsonl_gzip": ".json.gz",
    "jsonl_zstd": ".zst",
    "parquet": ".parquet",
}


def resolve_data_dir(output_dir=None, output_name=None):
    """Resolve an explicit path or an isolated name under the Speck data cache."""

    if output_dir is not None:
        return Path(output_dir).expanduser()
    if output_name is None:
        return default_data_dir / "packed"
    if not isinstance(output_name, str) or not output_name or Path(output_name).name != output_name:
        raise ValueError("data output_name must be a single non-empty path component")
    return default_data_dir / output_name


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
        raise ValueError("source ID must be a single non-empty path component")
    for key in ("repo", "tree_path", "content_column"):
        if not isinstance(source[key], str) or (key != "tree_path" and not source[key]):
            raise ValueError(f"source {source_id} {key} must be a string")
    for key in ("score_column", "language_column", "revision"):
        if source.get(key) is not None and not isinstance(source[key], str):
            raise ValueError(f"source {source_id} {key} must be null or a string")
    file_format = source.get("file_format", "parquet")
    if file_format not in _SOURCE_FILE_SUFFIXES:
        raise ValueError(f"source {source_id} has unsupported file_format")
    files = source.get("files")
    if files is not None:
        if not isinstance(files, list) or not files:
            raise ValueError(f"source {source_id} files must be a non-empty list")
        if any(
            not isinstance(filename, str)
            or not filename
            or PurePosixPath(filename).is_absolute()
            or ".." in PurePosixPath(filename).parts
            or not filename.lower().endswith(_SOURCE_FILE_SUFFIXES[file_format])
            for filename in files
        ) or len(files) != len(set(files)):
            raise ValueError(f"source {source_id} has invalid files")
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
        "min_tokens",
        "max_tokens",
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
    for key in ("min_tokens", "max_tokens"):
        if key in filters:
            _integer(filters[key], f"source {source_id} {key}", minimum=1)
    if filters.get("min_tokens", 1) > filters.get("max_tokens", math.inf):
        raise ValueError(f"source {source_id} min_tokens cannot exceed max_tokens")
    packing = source.get("packing")
    if packing is not None:
        if not isinstance(packing, dict) or set(packing) != {"row_tokens", "open_rows"}:
            raise ValueError(f"source {source_id} packing needs exactly row_tokens and open_rows")
        for key in ("row_tokens", "open_rows"):
            _integer(packing[key], f"source {source_id} packing {key}", minimum=1)
    record_format = source.get("record_format", "text")
    if record_format not in {"text", "messages"}:
        raise ValueError(f"source {source_id} record_format must be text or messages")
    if record_format == "messages" and packing is None:
        raise ValueError(f"source {source_id} messages records need row packing for their masks")
    if "passes" in source:
        # Declared repetition: the train pool shrinks to 1/passes of its exposure and repeats.
        _integer(source["passes"], f"source {source_id} passes", minimum=2)
        if packing is not None:
            raise ValueError(f"source {source_id} passes are not supported with row packing")
    return {
        **source,
        "revision": source.get("revision"),
        "file_format": file_format,
        "files": None if files is None else list(files),
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
    if max_chars > _MAX_DOCUMENT_CHARACTERS:
        raise ValueError(f"filtering.max_chars cannot exceed {_MAX_DOCUMENT_CHARACTERS:,}")
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
