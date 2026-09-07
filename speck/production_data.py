"""Disk-backed global deduplication, deny-ledger, cleanup, and resume tooling."""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from speck.code_near_duplicates import _jaccard, _shingles, _signature, _tokens

FORMAT = "speck_production_text_preprocess"
FORMAT_VERSION = 1
MANIFEST_FORMAT = "speck_production_text_preprocess_result"
LEDGER_FORMAT = "speck_removal_deny_ledger"
STATE_FORMAT = "speck_production_text_preprocess_state"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{name} must be a non-empty path component")
    return value


def _digest(value, name):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _path(value, name, config_dir):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return str((config_dir / path).resolve() if not path.is_absolute() else path.resolve())


def validate_preprocess_config(config, *, config_dir=None):
    """Validate immutable ordered inputs and bounded disk-backed index settings."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {
            "format",
            "format_version",
            "status",
            "sources",
            "deny_ledger",
            "policy",
            "checkpoint_records",
            "cleanup_files",
            "output_directory",
        },
        "production preprocess",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported production preprocess format")
    if config["status"] != "fixture_or_rehearsal_authorized_not_training_authority":
        raise ValueError("production preprocessing must remain non-authoritative")
    sources = config["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be non-empty")
    normalized_sources = []
    source_ids = []
    for index, source in enumerate(sources):
        _exact_keys(
            source,
            {
                "id",
                "precedence",
                "path",
                "sha256",
                "text_field",
                "content_sha256_field",
                "url_field",
                "domain_field",
                "blob_field",
            },
            f"source {index}",
        )
        source_id = _identifier(source["id"], f"source {index} id")
        source_ids.append(source_id)
        for field in ("text_field", "content_sha256_field"):
            if not isinstance(source[field], str) or not source[field]:
                raise ValueError(f"source {source_id} {field} must be non-empty")
        for field in ("url_field", "domain_field", "blob_field"):
            if source[field] is not None and (
                not isinstance(source[field], str) or not source[field]
            ):
                raise ValueError(f"source {source_id} {field} must be null or a string")
        normalized_sources.append(
            {
                **source,
                "precedence": _integer(source["precedence"], f"source {source_id} precedence", 1),
                "path": _path(source["path"], f"source {source_id} path", config_dir),
                "sha256": _digest(source["sha256"], f"source {source_id} sha256"),
            }
        )
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source IDs must be unique")
    if [source["precedence"] for source in normalized_sources] != list(range(1, len(sources) + 1)):
        raise ValueError("source precedence must be contiguous and ordered")
    ledger = config["deny_ledger"]
    _exact_keys(ledger, {"path", "sha256"}, "deny ledger")
    normalized_ledger = {
        "path": _path(ledger["path"], "deny ledger path", config_dir),
        "sha256": _digest(ledger["sha256"], "deny ledger sha256"),
    }
    policy = config["policy"]
    _exact_keys(
        policy,
        {
            "normalization",
            "token_pattern",
            "shingle_tokens",
            "minimum_document_tokens",
            "maximum_document_tokens",
            "num_perm",
            "minhash_seed",
            "bands",
            "verified_jaccard_threshold",
            "domain_match",
        },
        "policy",
    )
    if policy["normalization"] != "NFKC+lower+lexical-code-tokens":
        raise ValueError("unexpected production dedup normalization")
    if policy["domain_match"] != "exact_or_subdomain":
        raise ValueError("unexpected deny-ledger domain policy")
    try:
        re.compile(policy["token_pattern"])
    except (TypeError, re.error) as error:
        raise ValueError("invalid production token pattern") from error
    normalized_policy = dict(policy)
    for key, minimum in (
        ("shingle_tokens", 2),
        ("minimum_document_tokens", 1),
        ("maximum_document_tokens", 2),
        ("num_perm", 16),
        ("minhash_seed", 0),
        ("bands", 1),
    ):
        normalized_policy[key] = _integer(policy[key], f"policy.{key}", minimum)
    if normalized_policy["num_perm"] % normalized_policy["bands"]:
        raise ValueError("num_perm must divide evenly into bands")
    threshold = policy["verified_jaccard_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 < threshold <= 1
    ):
        raise ValueError("verified Jaccard threshold must be in (0, 1]")
    normalized_policy["verified_jaccard_threshold"] = float(threshold)
    cleanup = config["cleanup_files"]
    if not isinstance(cleanup, list):
        raise ValueError("cleanup_files must be a list")
    normalized_cleanup = []
    for index, item in enumerate(cleanup):
        _exact_keys(item, {"path", "sha256"}, f"cleanup file {index}")
        normalized_cleanup.append(
            {
                "path": _path(item["path"], f"cleanup file {index} path", config_dir),
                "sha256": _digest(item["sha256"], f"cleanup file {index} sha256"),
            }
        )
    protected = {
        *(source["path"] for source in normalized_sources),
        normalized_ledger["path"],
    }
    if any(item["path"] in protected for item in normalized_cleanup):
        raise ValueError("cleanup files cannot include source or deny-ledger inputs")
    output_directory = _path(config["output_directory"], "output directory", config_dir)
    output_path = Path(output_directory)
    staging_path = output_path.with_name(output_path.name + ".building")
    if any(
        Path(item["path"]).is_relative_to(output_path)
        or Path(item["path"]).is_relative_to(staging_path)
        for item in normalized_cleanup
    ):
        raise ValueError("cleanup files cannot be inside output or staging directories")
    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "sources": normalized_sources,
        "deny_ledger": normalized_ledger,
        "policy": normalized_policy,
        "checkpoint_records": _integer(config["checkpoint_records"], "checkpoint_records", 1),
        "cleanup_files": normalized_cleanup,
        "output_directory": output_directory,
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_preprocess_config(path):
    path = Path(path).resolve()
    return validate_preprocess_config(json.loads(path.read_text()), config_dir=path.parent)


def _load_ledger(config):
    path = Path(config["deny_ledger"]["path"])
    if not path.is_file() or _sha256(path) != config["deny_ledger"]["sha256"]:
        raise ValueError("deny ledger identity mismatch")
    ledger = json.loads(path.read_text())
    _exact_keys(ledger, {"format", "format_version", "status", "entries"}, "deny ledger")
    if (
        ledger["format"] != LEDGER_FORMAT
        or ledger["format_version"] != FORMAT_VERSION
        or ledger["status"] != "human_reviewed_deny_entries"
        or not isinstance(ledger["entries"], list)
    ):
        raise ValueError("invalid deny ledger")
    values = {kind: set() for kind in ("content_sha256", "url", "domain", "blob_id")}
    for index, entry in enumerate(ledger["entries"]):
        _exact_keys(
            entry, {"kind", "value", "reason", "authority", "recorded_at"}, f"deny entry {index}"
        )
        kind = entry["kind"]
        if kind not in values or any(
            not isinstance(entry[key], str) or not entry[key]
            for key in ("value", "reason", "authority", "recorded_at")
        ):
            raise ValueError(f"invalid deny entry {index}")
        if kind == "content_sha256":
            _digest(entry["value"], f"deny entry {index} content hash")
        values[kind].add(entry["value"].lower() if kind == "domain" else entry["value"])
    return ledger, values


def _database(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS docs (doc_seq INTEGER PRIMARY KEY, processed_index INTEGER NOT NULL, source_index INTEGER NOT NULL, source_id TEXT NOT NULL, line_number INTEGER NOT NULL, byte_offset INTEGER NOT NULL, content_sha256 TEXT NOT NULL, dedup_sha256 TEXT NOT NULL UNIQUE)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS bands (band INTEGER NOT NULL, band_hash BLOB NOT NULL, doc_seq INTEGER NOT NULL REFERENCES docs(doc_seq) ON DELETE CASCADE)"
    )
    connection.execute("CREATE INDEX IF NOT EXISTS band_lookup ON bands(band, band_hash)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS checkpoints (checkpoint_id INTEGER PRIMARY KEY, processed_records INTEGER NOT NULL, next_doc_seq INTEGER NOT NULL, index_chain TEXT NOT NULL)"
    )
    return connection


def _band_values(signature, bands):
    rows = len(signature.hashvalues) // bands
    return [
        hashlib.blake2b(
            signature.hashvalues[index * rows : (index + 1) * rows].tobytes(), digest_size=16
        ).digest()
        for index in range(bands)
    ]


def _record_domain(record, source):
    domain = record.get(source["domain_field"]) if source["domain_field"] else None
    if isinstance(domain, str) and domain:
        return domain.lower().rstrip(".")
    url = record.get(source["url_field"]) if source["url_field"] else None
    if not isinstance(url, str):
        return None
    try:
        return (urlparse(url).hostname or "").lower().rstrip(".") or None
    except ValueError:
        return None


def _deny_reason(record, source, content_sha256, values):
    if content_sha256 in values["content_sha256"]:
        return "content_sha256"
    url = record.get(source["url_field"]) if source["url_field"] else None
    if isinstance(url, str) and url in values["url"]:
        return "url"
    blob = record.get(source["blob_field"]) if source["blob_field"] else None
    if isinstance(blob, str) and blob in values["blob_id"]:
        return "blob_id"
    domain = _record_domain(record, source)
    if domain and any(
        domain == denied or domain.endswith("." + denied) for denied in values["domain"]
    ):
        return "domain"
    return None


def _candidate_text(connection, sources, doc_seq, handles):
    row = connection.execute(
        "SELECT source_index, byte_offset FROM docs WHERE doc_seq=?", (doc_seq,)
    ).fetchone()
    if row is None:
        raise ValueError("near-duplicate index references a missing document")
    source_index, offset = row
    handle = handles.setdefault(source_index, Path(sources[source_index]["path"]).open("rb"))
    handle.seek(offset)
    record = json.loads(handle.readline().decode("utf-8"))
    return record[sources[source_index]["text_field"]]


def _slice_sha256(path, start, end):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        handle.seek(start)
        remaining = end - start
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"checkpoint slice ended early: {path}")
            hasher.update(chunk)
            remaining -= len(chunk)
    return hasher.hexdigest()


def _verify_slices(path, slices, expected_size, description):
    boundary = 0
    for item in slices:
        if item["start"] != boundary or item["end"] < item["start"]:
            raise ValueError(f"{description} checkpoint slices are not contiguous")
        if _slice_sha256(path, item["start"], item["end"]) != item["sha256"]:
            raise ValueError(f"{description} checkpoint slice checksum mismatch")
        boundary = item["end"]
    if boundary != expected_size:
        raise ValueError(f"{description} checkpoint slices do not reach committed size")


def _checkpoint(connection, handles, removal, state_path, state):
    checkpoint_id = state["checkpoint_id"] + 1
    connection.execute(
        "INSERT INTO checkpoints VALUES (?, ?, ?, ?)",
        (
            checkpoint_id,
            state["processed_records"],
            state["next_doc_seq"],
            state["index_chain"],
        ),
    )
    connection.commit()
    for handle in handles.values():
        handle.flush()
        os.fsync(handle.fileno())
    removal.flush()
    os.fsync(removal.fileno())
    for index, handle in handles.items():
        key = str(index)
        start = state["output_sizes"].get(key, 0)
        end = handle.tell()
        if end > start:
            state["output_slices"].setdefault(key, []).append(
                {"start": start, "end": end, "sha256": _slice_sha256(handle.name, start, end)}
            )
        state["output_sizes"][key] = end
    removal_start = state["removal_size"]
    removal_end = removal.tell()
    if removal_end > removal_start:
        state["removal_slices"].append(
            {
                "start": removal_start,
                "end": removal_end,
                "sha256": _slice_sha256(removal.name, removal_start, removal_end),
            }
        )
    state["removal_size"] = removal_end
    state["checkpoint_id"] = checkpoint_id
    _write_json(state_path, state)


def _cleanup_published(config, output):
    receipt_path = output / "cleanup_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("format") != "speck_production_cleanup_receipt"
            or receipt.get("format_version") != FORMAT_VERSION
            or receipt.get("status") != "declared_cleanup_files_absent_after_published_manifest"
            or receipt.get("manifest_sha256") != _sha256(output / "manifest.json")
            or receipt.get("removed") != config["cleanup_files"]
            or any(Path(item["path"]).exists() for item in config["cleanup_files"])
        ):
            raise ValueError("published cleanup receipt is invalid")
        return receipt
    removed = []
    for item in config["cleanup_files"]:
        path = Path(item["path"])
        if path.exists():
            if not path.is_file() or _sha256(path) != item["sha256"]:
                raise ValueError(f"cleanup file identity mismatch: {path}")
            path.unlink()
        removed.append(item)
    receipt = {
        "format": "speck_production_cleanup_receipt",
        "format_version": FORMAT_VERSION,
        "status": "declared_cleanup_files_absent_after_published_manifest",
        "manifest_sha256": _sha256(output / "manifest.json"),
        "removed": removed,
    }
    _write_json(receipt_path, receipt)
    return receipt


def _verify_published(config, output, manifest):
    if (
        manifest.get("format") != MANIFEST_FORMAT
        or manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("plan_fingerprint") != config["plan_fingerprint"]
        or manifest.get("status")
        != "global_dedup_and_deny_complete_cleanup_receipt_required_not_training_authority"
    ):
        raise ValueError("published preprocess manifest identity is invalid")
    if set(manifest.get("outputs", {})) != {source["id"] for source in config["sources"]}:
        raise ValueError("published preprocess outputs do not cover configured sources")
    for entry in manifest["outputs"].values():
        path = output / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or _sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"published preprocess output identity mismatch: {path}")
    for key in ("removals", "index"):
        entry = manifest[key]
        path = output / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or _sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"published preprocess {key} identity mismatch")
    connection = sqlite3.connect(f"file:{output / manifest['index']['path']}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("published preprocess SQLite integrity check failed")
        documents = connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    finally:
        connection.close()
    if documents != manifest["index"]["documents"] or documents != manifest["counts"].get(
        "records_retained", 0
    ):
        raise ValueError("published preprocess SQLite document count mismatch")
    removed = sum(
        manifest["counts"].get(key, 0)
        for key in (
            "records_removed_deny_ledger",
            "records_removed_exact",
            "records_removed_near",
        )
    )
    if manifest["counts"].get("records_seen", 0) != documents + removed:
        raise ValueError("published preprocess aggregate record counts are inconsistent")


def preprocess_sources(config, *, restart=False, crash_after_records=None):
    """Run or resume the disk-backed global exact/near deduplication pass."""

    if "plan_fingerprint" not in config:
        config = validate_preprocess_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized production preprocess fingerprint mismatch")
    output = Path(config["output_directory"])
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text())
        _verify_published(config, output, manifest)
        result = {"manifest": manifest, "cleanup": _cleanup_published(config, output)}
        return result
    staging = output.with_name(output.name + ".building")
    if staging.exists() and restart:
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    for item in config["cleanup_files"]:
        cleanup_path = Path(item["path"])
        if not cleanup_path.is_file() or _sha256(cleanup_path) != item["sha256"]:
            raise ValueError(f"cleanup file identity mismatch before build: {cleanup_path}")
    for source in config["sources"]:
        path = Path(source["path"])
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ValueError(f"source identity mismatch: {source['id']}")
    ledger, denied = _load_ledger(config)
    state_path = staging / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "format": STATE_FORMAT,
            "format_version": FORMAT_VERSION,
            "contract": config["plan_fingerprint"],
            "source_index": 0,
            "line_number": 0,
            "byte_offset": 0,
            "processed_records": 0,
            "next_doc_seq": 0,
            "checkpoint_id": 0,
            "index_chain": hashlib.sha256(b"").hexdigest(),
            "output_sizes": {},
            "output_slices": {},
            "removal_size": 0,
            "removal_slices": [],
            "counts": {},
        }
    )
    if state.get("contract") != config["plan_fingerprint"]:
        raise ValueError("staged preprocess contract changed; use restart")
    output_paths = {
        index: staging / f"{source['id']}.jsonl" for index, source in enumerate(config["sources"])
    }
    output_handles = {index: path.open("ab+") for index, path in output_paths.items()}
    removal_path = staging / "removals.jsonl"
    removal = removal_path.open("ab+")
    for index, handle in output_handles.items():
        handle.truncate(state["output_sizes"].get(str(index), 0))
        handle.seek(0, os.SEEK_END)
        _verify_slices(
            handle.name,
            state["output_slices"].get(str(index), []),
            state["output_sizes"].get(str(index), 0),
            f"source {index} output",
        )
    removal.truncate(state["removal_size"])
    removal.seek(0, os.SEEK_END)
    _verify_slices(
        removal.name,
        state["removal_slices"],
        state["removal_size"],
        "removal output",
    )
    connection = _database(staging / "near_duplicates.sqlite3")
    if state["checkpoint_id"]:
        row = connection.execute(
            "SELECT processed_records, next_doc_seq, index_chain FROM checkpoints WHERE checkpoint_id=?",
            (state["checkpoint_id"],),
        ).fetchone()
        if row != (
            state["processed_records"],
            state["next_doc_seq"],
            state["index_chain"],
        ):
            raise ValueError("SQLite checkpoint does not match durable preprocess state")
    connection.execute("DELETE FROM docs WHERE processed_index>=?", (state["processed_records"],))
    connection.execute("DELETE FROM checkpoints WHERE checkpoint_id>?", (state["checkpoint_id"],))
    connection.commit()
    indexed = connection.execute(
        "SELECT dedup_sha256, content_sha256 FROM docs ORDER BY doc_seq"
    ).fetchall()
    chain = hashlib.sha256(b"").hexdigest()
    for dedup_sha256, content_sha256 in indexed:
        chain = hashlib.sha256(
            bytes.fromhex(chain) + bytes.fromhex(dedup_sha256) + bytes.fromhex(content_sha256)
        ).hexdigest()
    if len(indexed) != state["next_doc_seq"] or chain != state["index_chain"]:
        raise ValueError("SQLite accepted-document chain does not match durable state")
    counts = Counter(state["counts"])
    pattern = re.compile(config["policy"]["token_pattern"])
    candidate_handles = {}
    since_checkpoint = 0
    try:
        for source_index in range(state["source_index"], len(config["sources"])):
            source = config["sources"][source_index]
            start_line = state["line_number"] if source_index == state["source_index"] else 0
            start_offset = state["byte_offset"] if source_index == state["source_index"] else 0
            with Path(source["path"]).open("rb") as handle:
                handle.seek(start_offset)
                line_number = start_line
                while raw := handle.readline():
                    offset = handle.tell() - len(raw)
                    record = json.loads(raw.decode("utf-8"))
                    text = record.get(source["text_field"])
                    content_sha256 = record.get(source["content_sha256_field"])
                    if (
                        not isinstance(text, str)
                        or not isinstance(content_sha256, str)
                        or hashlib.sha256(text.encode()).hexdigest() != content_sha256
                    ):
                        raise ValueError(f"invalid source record: {source['id']}:{line_number}")
                    processed_index = state["processed_records"]
                    state["processed_records"] += 1
                    state["line_number"] = line_number + 1
                    state["byte_offset"] = handle.tell()
                    counts["records_seen"] += 1
                    reason = _deny_reason(record, source, content_sha256, denied)
                    kept = None
                    similarity = None
                    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
                    dedup = hashlib.sha256(normalized.encode()).hexdigest()
                    if reason is not None:
                        counts["records_removed_deny_ledger"] += 1
                    else:
                        exact = connection.execute(
                            "SELECT doc_seq, source_id, content_sha256 FROM docs WHERE dedup_sha256=?",
                            (dedup,),
                        ).fetchone()
                        if exact is not None:
                            reason = "exact_duplicate"
                            kept = {
                                "doc_seq": exact[0],
                                "source_id": exact[1],
                                "content_sha256": exact[2],
                            }
                            counts["records_removed_exact"] += 1
                    tokens = _tokens(text, pattern, config["policy"]["maximum_document_tokens"])
                    shingles = (
                        _shingles(tokens, config["policy"]["shingle_tokens"])
                        if len(tokens)
                        >= max(
                            config["policy"]["minimum_document_tokens"],
                            config["policy"]["shingle_tokens"],
                        )
                        else set()
                    )
                    signature = None
                    bands = []
                    if reason is None and shingles:
                        signature = _signature(
                            shingles,
                            config["policy"]["num_perm"],
                            config["policy"]["minhash_seed"],
                        )
                        bands = _band_values(signature, config["policy"]["bands"])
                        candidates = set()
                        for band, value in enumerate(bands):
                            candidates.update(
                                row[0]
                                for row in connection.execute(
                                    "SELECT doc_seq FROM bands WHERE band=? AND band_hash=?",
                                    (band, value),
                                )
                            )
                        for candidate in sorted(candidates):
                            candidate_text = _candidate_text(
                                connection, config["sources"], candidate, candidate_handles
                            )
                            candidate_tokens = _tokens(
                                candidate_text,
                                pattern,
                                config["policy"]["maximum_document_tokens"],
                            )
                            value = _jaccard(
                                shingles,
                                _shingles(candidate_tokens, config["policy"]["shingle_tokens"]),
                            )
                            if similarity is None or value > similarity:
                                similarity = value
                                owner = connection.execute(
                                    "SELECT source_id, content_sha256 FROM docs WHERE doc_seq=?",
                                    (candidate,),
                                ).fetchone()
                                kept = {
                                    "doc_seq": candidate,
                                    "source_id": owner[0],
                                    "content_sha256": owner[1],
                                }
                        if (
                            similarity is not None
                            and similarity >= config["policy"]["verified_jaccard_threshold"]
                        ):
                            reason = "near_duplicate"
                            counts["records_removed_near"] += 1
                    if reason is not None:
                        _write = {
                            "removed_source": source["id"],
                            "removed_line": line_number,
                            "removed_content_sha256": content_sha256,
                            "reason": reason,
                            "kept": kept,
                            "verified_shingle_jaccard": similarity,
                        }
                        removal.write(
                            (
                                json.dumps(_write, sort_keys=True, separators=(",", ":")) + "\n"
                            ).encode()
                        )
                    else:
                        output_handles[source_index].write(raw)
                        doc_seq = state["next_doc_seq"]
                        state["next_doc_seq"] += 1
                        connection.execute(
                            "INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                doc_seq,
                                processed_index,
                                source_index,
                                source["id"],
                                line_number,
                                offset,
                                content_sha256,
                                dedup,
                            ),
                        )
                        state["index_chain"] = hashlib.sha256(
                            bytes.fromhex(state["index_chain"])
                            + bytes.fromhex(dedup)
                            + bytes.fromhex(content_sha256)
                        ).hexdigest()
                        for band, value in enumerate(bands):
                            connection.execute(
                                "INSERT INTO bands VALUES (?, ?, ?)", (band, value, doc_seq)
                            )
                        counts["records_retained"] += 1
                    state["counts"] = dict(counts)
                    line_number += 1
                    since_checkpoint += 1
                    if (
                        crash_after_records is not None
                        and counts["records_seen"] == crash_after_records
                    ):
                        raise RuntimeError("injected production preprocess crash")
                    if since_checkpoint >= config["checkpoint_records"]:
                        _checkpoint(connection, output_handles, removal, state_path, state)
                        since_checkpoint = 0
            state["source_index"] = source_index + 1
            state["line_number"] = 0
            state["byte_offset"] = 0
            _checkpoint(connection, output_handles, removal, state_path, state)
            since_checkpoint = 0
        _checkpoint(connection, output_handles, removal, state_path, state)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        index_counts = {
            "documents": connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0],
            "band_entries": connection.execute("SELECT COUNT(*) FROM bands").fetchone()[0],
        }
    finally:
        connection.close()
        for handle in output_handles.values():
            handle.close()
        removal.close()
        for handle in candidate_handles.values():
            handle.close()
    outputs = {
        source["id"]: {
            "path": output_paths[index].name,
            "bytes": output_paths[index].stat().st_size,
            "sha256": _sha256(output_paths[index]),
        }
        for index, source in enumerate(config["sources"])
    }
    index_path = staging / "near_duplicates.sqlite3"
    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "global_dedup_and_deny_complete_cleanup_receipt_required_not_training_authority",
        "plan_fingerprint": config["plan_fingerprint"],
        "sources": config["sources"],
        "deny_ledger": {**config["deny_ledger"], "entries": len(ledger["entries"])},
        "policy": config["policy"],
        "counts": dict(sorted(counts.items())),
        "outputs": outputs,
        "removals": {
            "path": removal_path.name,
            "bytes": removal_path.stat().st_size,
            "sha256": _sha256(removal_path),
        },
        "index": {
            "path": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": _sha256(index_path),
            **index_counts,
        },
        "cleanup_files": config["cleanup_files"],
        "gates": {
            "source_and_ledger_identity": "pass",
            "global_exact_deduplication": "pass",
            "disk_backed_Minhash_candidates_and_verified_near_deduplication": "pass",
            "redacted_removal_records": "pass",
            "record_checkpoint_resume": "pass_by_contract_pending_rehearsal",
            "cleanup": "pending_external_receipt",
            "training_authority": "blocked",
        },
    }
    _write_json(staging / "manifest.json", manifest)
    state_path.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(output)
    descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    result = {"manifest": manifest, "cleanup": _cleanup_published(config, output)}
    _verify_published(config, output, manifest)
    return result
