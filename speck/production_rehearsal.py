"""Execute the rights-bound stages of the flagship 20B data rehearsal."""

import hashlib
import json
import os
import resource
import shutil
import sqlite3
import subprocess
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import speck.production_data as production_data
from speck.data_rehearsal import STAGE_FORMAT
from speck.dataset import (
    TokenShardWriter,
    _validate_source,
    discover_source_files,
    iter_source_file_documents,
)
from speck.io import atomic_json, file_sha256
from speck.production_data import preprocess_sources, validate_preprocess_config
from speck.text_contamination import (
    _exact_index,
    _load_tasks,
    _match_document,
    _ngram_index,
    validate_text_contamination_config,
)
from speck.tokenizer import Tokenizer
from speck.web_sample import _duplicate_line_ratio, _raw_pii

FORMAT = "speck_production_rehearsal_plan"
FORMAT_VERSION = 1
STAGES = (
    "source_identity",
    "acquisition",
    "global_dedup",
    "packing",
    "resume_cleanup",
    "firewall_disjointness",
)


def _exact_keys(value, expected, context):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{context} must contain exactly: {', '.join(sorted(expected))}")


def _identity(value, root, context):
    _exact_keys(value, {"path", "sha256"}, context)
    path = Path(value["path"]).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or file_sha256(path) != value["sha256"]:
        raise ValueError(f"{context} identity mismatch")
    return {"path": str(path), "sha256": value["sha256"]}


def _positive_integer(value, context):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_production_rehearsal_plan(value, *, config_dir=None):
    """Validate the exact source, quota, partition, storage, and packing plan."""

    root = Path(config_dir or ".").resolve()
    _exact_keys(
        value,
        {
            "format",
            "format_version",
            "status",
            "target_tokens",
            "seed",
            "source_registry",
            "rights_record",
            "deny_ledger",
            "tokenizer",
            "contamination_plan",
            "security",
            "filtering",
            "partition",
            "deduplication",
            "packing",
            "sources",
            "output_directory",
        },
        "production rehearsal plan",
    )
    if value["format"] != FORMAT or value["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported production rehearsal plan")
    if value["status"] != "frozen_20B_rehearsal_not_training_authority":
        raise ValueError("production rehearsal plan status is invalid")
    if value["target_tokens"] != 20_000_000_000:
        raise ValueError("production rehearsal target must be exactly 20B tokens")
    seed = _positive_integer(value["seed"], "seed")
    registry = _identity(value["source_registry"], root, "source registry")
    rights = _identity(value["rights_record"], root, "rights record")
    deny = _identity(value["deny_ledger"], root, "deny ledger")
    tokenizer = _identity(value["tokenizer"], root, "tokenizer")
    contamination = _identity(value["contamination_plan"], root, "contamination plan")
    rights_value = json.loads(Path(rights["path"]).read_text())
    if (
        rights_value.get("format") != "speck_human_source_rights_acceptance"
        or rights_value.get("status") != "all_sources_human_approved"
        or rights_value.get("automated_approval_made") is not False
    ):
        raise ValueError("production rehearsal rights record is not approved")
    approved = set(rights_value.get("approved_source_ids", ()))
    registry_value = json.loads(Path(registry["path"]).read_text())
    registry_sources = {source["id"] for source in registry_value.get("sources", ())}

    filtering = value["filtering"]
    _exact_keys(filtering, {"min_chars", "max_chars"}, "filtering")
    minimum = _positive_integer(filtering["min_chars"], "filtering.min_chars")
    maximum = _positive_integer(filtering["max_chars"], "filtering.max_chars")
    if minimum > maximum:
        raise ValueError("filtering minimum exceeds maximum")

    partition = value["partition"]
    _exact_keys(
        partition,
        {"holdout_modulus", "holdout_remainders", "normalization"},
        "partition",
    )
    modulus = _positive_integer(partition["holdout_modulus"], "partition.holdout_modulus")
    remainders = partition["holdout_remainders"]
    if (
        not isinstance(remainders, list)
        or not remainders
        or len(remainders) != len(set(remainders))
        or any(
            isinstance(remainder, bool)
            or not isinstance(remainder, int)
            or not 0 <= remainder < modulus
            for remainder in remainders
        )
        or partition["normalization"] != "NFKC+lower+whitespace"
    ):
        raise ValueError("partition settings are invalid")

    deduplication = value["deduplication"]
    _exact_keys(
        deduplication,
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
            "checkpoint_records",
        },
        "deduplication",
    )
    packing = value["packing"]
    _exact_keys(packing, {"shard_tokens", "acquisition_margin_percent"}, "packing")
    shard_tokens = _positive_integer(packing["shard_tokens"], "packing.shard_tokens")
    margin = _positive_integer(
        packing["acquisition_margin_percent"], "packing.acquisition_margin_percent"
    )
    security = value["security"]
    _exact_keys(
        security,
        {
            "gitleaks_binary",
            "gitleaks_version",
            "maximum_duplicate_line_ratio",
            "adult_host_terms",
            "allowed_emails",
            "allowed_ipv4",
        },
        "security",
    )
    gitleaks = _identity(security["gitleaks_binary"], root, "Gitleaks binary")
    duplicate_ratio = security["maximum_duplicate_line_ratio"]
    if (
        isinstance(duplicate_ratio, bool)
        or not isinstance(duplicate_ratio, (int, float))
        or not 0 <= duplicate_ratio <= 1
    ):
        raise ValueError("security duplicate-line ratio must be in [0, 1]")
    for field in ("adult_host_terms", "allowed_emails", "allowed_ipv4"):
        if not isinstance(security[field], list) or any(
            not isinstance(item, str) or not item for item in security[field]
        ):
            raise ValueError(f"security {field} must contain strings")

    sources = value["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("production rehearsal requires sources")
    ids = [source.get("id") for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("production rehearsal source IDs must be unique")
    normalized_sources = []
    total = 0
    for source in sources:
        if set(source) != {"id", "category", "target_tokens", "reader"}:
            raise ValueError("production rehearsal source fields are invalid")
        source_id = source["id"]
        if source_id not in approved or source_id not in registry_sources:
            raise ValueError(f"production rehearsal source is not approved: {source_id}")
        target = _positive_integer(source["target_tokens"], f"source {source_id} target_tokens")
        reader = _validate_source(source["reader"])
        if reader["id"] != source_id:
            raise ValueError(f"source reader ID mismatch: {source_id}")
        normalized_sources.append({**source, "target_tokens": target, "reader": reader})
        total += target
    if total != value["target_tokens"]:
        raise ValueError("production rehearsal source quotas do not sum to 20B")

    output = Path(value["output_directory"]).expanduser()
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    normalized = {
        **value,
        "seed": seed,
        "source_registry": registry,
        "rights_record": rights,
        "deny_ledger": deny,
        "tokenizer": tokenizer,
        "contamination_plan": contamination,
        "filtering": {"min_chars": minimum, "max_chars": maximum},
        "partition": {**partition, "holdout_modulus": modulus, "holdout_remainders": remainders},
        "deduplication": dict(deduplication),
        "packing": {
            "shard_tokens": shard_tokens,
            "acquisition_margin_percent": margin,
        },
        "security": {
            **security,
            "gitleaks_binary": gitleaks,
            "maximum_duplicate_line_ratio": float(duplicate_ratio),
        },
        "sources": normalized_sources,
        "output_directory": str(output),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_production_rehearsal_plan(path):
    path = Path(path).resolve()
    return validate_production_rehearsal_plan(json.loads(path.read_text()), config_dir=path.parent)


def _stage_result(path, stage_id, gates, metrics, details):
    value = {
        "format": STAGE_FORMAT,
        "format_version": 1,
        "status": "complete",
        "stage_id": stage_id,
        "gates": gates,
        "metrics": metrics,
        "details": details,
    }
    atomic_json(path, value)
    return value


def _load_bound_json(path, expected_sha256, context):
    path = Path(path)
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"{context} identity mismatch")
    return json.loads(path.read_text())


def _load_discovery(plan):
    path = Path(plan["output_directory"]) / "source-discovery.json"
    value = json.loads(path.read_text())
    if (
        value.get("format") != "speck_production_rehearsal_source_discovery"
        or value.get("plan_fingerprint") != plan["plan_fingerprint"]
    ):
        raise ValueError("source discovery does not match production rehearsal plan")
    return value


def _source_identity(plan, result_path):
    output = Path(plan["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    discovered = []
    for source in plan["sources"]:
        resolved = discover_source_files(source["reader"], plan["seed"])
        discovered.append(
            {
                "id": source["id"],
                "category": source["category"],
                "target_tokens": source["target_tokens"],
                "revision": resolved["revision"],
                "files": resolved["files"],
                "file_list_sha256": resolved["file_list_sha256"],
            }
        )
    discovery = {
        "format": "speck_production_rehearsal_source_discovery",
        "format_version": 1,
        "status": "revision_and_file_lists_frozen",
        "plan_fingerprint": plan["plan_fingerprint"],
        "sources": discovered,
    }
    discovery_path = output / "source-discovery.json"
    atomic_json(discovery_path, discovery)
    return _stage_result(
        result_path,
        "source_identity",
        {"source_identity": "pass"},
        {"records_seen": len(discovered)},
        {"discovery": {"path": str(discovery_path), "sha256": file_sha256(discovery_path)}},
    )


def _raw_local_path(cache_dir, source, revision, filename):
    suffix = ".parquet" if source["file_format"] == "parquet" else ".json.gz"
    key = hashlib.sha256(f"{source['repo']}\0{revision}\0{filename}".encode()).hexdigest()[:20]
    return Path(cache_dir) / f"{key}{suffix}"


def _partition(content, modulus, remainders):
    normalized = " ".join(unicodedata.normalize("NFKC", content).lower().split())
    digest = hashlib.sha256(normalized.encode()).digest()
    return "holdout" if int.from_bytes(digest[:8], "big") % modulus in remainders else "train"


def _record(content, metadata):
    url = metadata.get("url")
    host = None
    if isinstance(url, str):
        try:
            host = urlparse(url).hostname
        except ValueError:
            host = None
    return {
        "text": content,
        "released_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "url": url if isinstance(url, str) else None,
        "host": host,
        "content_id": str(metadata.get("blob") or metadata.get("id") or "") or None,
    }


def _contamination_indexes(plan):
    path = Path(plan["contamination_plan"]["path"])
    config = validate_text_contamination_config(
        json.loads(path.read_text()), config_dir=path.parent
    )
    pattern, tasks, benchmarks = _load_tasks(config)
    policy = config["policy"]
    return {
        "pattern": pattern,
        "policy": policy,
        "primary": _ngram_index(tasks, policy["primary_ngram"], policy),
        "sensitivity": _ngram_index(tasks, policy["sensitivity_ngram"], policy),
        "exact": _exact_index(tasks, policy)[0],
        "benchmarks": benchmarks,
        "tasks": len(tasks),
    }


def _document_rejection(document, security, contamination):
    content = document["content"]
    metadata = document.get("metadata") or {}
    emails, ips = _raw_pii(
        content,
        {value.lower() for value in security["allowed_emails"]},
        set(security["allowed_ipv4"]),
    )
    if emails or ips:
        return "raw_email_or_ipv4"
    if _duplicate_line_ratio(content) > security["maximum_duplicate_line_ratio"]:
        return "duplicate_lines"
    url = metadata.get("url")
    host = None
    if isinstance(url, str):
        try:
            host = (urlparse(url).hostname or "").lower()
        except ValueError:
            host = None
    if host and any(term in host for term in security["adult_host_terms"]):
        return "adult_host"
    critical = _match_document(
        content,
        contamination["pattern"],
        contamination["primary"],
        contamination["sensitivity"],
        contamination["exact"],
        contamination["policy"],
    )[0]
    return "benchmark_contamination" if critical else None


def _gitleaks_filter(path, binary, reports):
    report = reports / f"{path.parent.name}-{path.stem}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            binary,
            "detect",
            "--no-git",
            "--source",
            str(path),
            "--report-format",
            "json",
            "--report-path",
            str(report),
            "--redact=100",
            "--exit-code",
            "0",
            "--no-banner",
            "--log-level",
            "error",
        ],
        check=True,
    )
    findings = json.loads(report.read_text())
    lines = {finding["StartLine"] for finding in findings}
    if any(finding.get("Secret") != "REDACTED" for finding in findings):
        raise ValueError("Gitleaks rehearsal report is not fully redacted")
    if not lines:
        return 0, {"path": str(report), "sha256": file_sha256(report), "findings": 0}
    cleaned = path.with_suffix(path.suffix + ".security-clean")
    with path.open("rb") as source, cleaned.open("wb") as destination:
        for line_number, line in enumerate(source, 1):
            if line_number not in lines:
                destination.write(line)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(cleaned, path)
    return len(lines), {
        "path": str(report),
        "sha256": file_sha256(report),
        "findings": len(findings),
    }


def _truncate_outputs(paths, sizes):
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.truncate(sizes.get(key, 0))


def _acquisition(plan, result_path):
    discovery = _load_discovery(plan)
    output = Path(plan["output_directory"])
    final = output / "acquired"
    staging = output / "acquired.building"
    if (final / "manifest.json").is_file():
        acquisition_path, manifest = _acquisition_manifest(plan)
        elapsed = max(manifest["elapsed_seconds"], 1e-9)
        return _stage_result(
            result_path,
            "acquisition",
            {},
            {
                "download_bytes": manifest["download_bytes"],
                "download_bytes_per_second": manifest["download_bytes"] / elapsed,
                "filtered_bytes": manifest["filtered_bytes"],
            },
            {
                "manifest": {
                    "path": str(acquisition_path),
                    "sha256": file_sha256(acquisition_path),
                }
            },
        )
    state_path = staging / "state.json"
    raw_dir = staging / "raw"
    staging.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(plan["tokenizer"]["path"])
    contamination = _contamination_indexes(plan)
    security = plan["security"]
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "format": "speck_production_rehearsal_acquisition_state",
            "format_version": 1,
            "plan_fingerprint": plan["plan_fingerprint"],
            "source_index": 0,
            "next_file_index": 0,
            "output_sizes": {},
            "sources": {},
            "download_bytes": 0,
            "filtered_bytes": 0,
            "records": 0,
            "rejections": {},
            "started_at_monotonic": time.monotonic(),
        }
    )
    if state.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("acquisition state belongs to a different plan")
    paths = {
        f"{split}__{source['id']}": staging / split / f"{source['id']}.jsonl"
        for source in plan["sources"]
        for split in ("holdout", "train")
    }
    _truncate_outputs(paths, state["output_sizes"])
    margin = plan["packing"]["acquisition_margin_percent"]
    discovery_by_id = {source["id"]: source for source in discovery["sources"]}
    for source_index in range(state["source_index"], len(plan["sources"])):
        source = plan["sources"][source_index]
        source_id = source["id"]
        reader = source["reader"]
        resolved = discovery_by_id[source_id]
        target = source["target_tokens"] * (100 + margin) // 100
        counters = state["sources"].setdefault(
            source_id,
            {"train_tokens": 0, "holdout_tokens": 0, "records": 0, "files_completed": 0},
        )
        start_file = state["next_file_index"] if source_index == state["source_index"] else 0
        handles = {
            split: paths[f"{split}__{source_id}"].open("ab") for split in ("holdout", "train")
        }
        try:
            for file_index, filename in enumerate(resolved["files"][start_file:], start=start_file):
                if counters["train_tokens"] >= target:
                    break
                raw_path = _raw_local_path(raw_dir, reader, resolved["revision"], filename)
                batch = []
                characters = 0

                def flush():
                    nonlocal batch, characters
                    if not batch:
                        return
                    token_rows = tokenizer.encode_batch(
                        [document["content"] for document in batch], bos=True, eos=True
                    )
                    for document, tokens in zip(batch, token_rows, strict=True):
                        split = _partition(
                            document["content"],
                            plan["partition"]["holdout_modulus"],
                            plan["partition"]["holdout_remainders"],
                        )
                        line = (
                            json.dumps(
                                _record(document["content"], document.get("metadata") or {}),
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode()
                        handles[split].write(line)
                        counters[f"{split}_tokens"] += len(tokens)
                        counters["records"] += 1
                        state["records"] += 1
                        state["filtered_bytes"] += len(document["content"].encode())
                    batch = []
                    characters = 0

                documents = iter_source_file_documents(
                    source=reader,
                    revision=resolved["revision"],
                    filename=filename,
                    filtering=plan["filtering"],
                    cache_dir=raw_dir,
                    keep_raw=True,
                    description=f"20B rehearsal {source_id}",
                )
                try:
                    for document in documents:
                        reason = _document_rejection(document, security, contamination)
                        if reason is not None:
                            state["rejections"][reason] = state["rejections"].get(reason, 0) + 1
                            continue
                        batch.append(document)
                        characters += len(document["content"])
                        if len(batch) >= 512 or characters >= 1_000_000:
                            flush()
                        if counters["train_tokens"] >= target:
                            break
                    flush()
                finally:
                    close = getattr(documents, "close", None)
                    if close is not None:
                        close()
                if raw_path.is_file():
                    state["download_bytes"] += raw_path.stat().st_size
                    raw_path.unlink()
                counters["files_completed"] = file_index + 1
                for split, handle in handles.items():
                    handle.flush()
                    os.fsync(handle.fileno())
                    state["output_sizes"][f"{split}__{source_id}"] = handle.tell()
                state["source_index"] = source_index
                state["next_file_index"] = file_index + 1
                atomic_json(state_path, state)
            if counters["train_tokens"] < target:
                raise RuntimeError(
                    f"source {source_id} exhausted at {counters['train_tokens']:,}/{target:,} acquisition tokens"
                )
        finally:
            for handle in handles.values():
                handle.close()
        state["source_index"] = source_index + 1
        state["next_file_index"] = 0
        atomic_json(state_path, state)
    raw_files = [path for path in raw_dir.rglob("*") if path.is_file()] if raw_dir.exists() else []
    if raw_files:
        raise RuntimeError("acquisition left raw download files after completed source boundaries")
    reports = {}
    gitleaks_removed = 0
    for key, path in paths.items():
        removed, report = _gitleaks_filter(
            path, security["gitleaks_binary"]["path"], staging / "gitleaks-reports"
        )
        gitleaks_removed += removed
        reports[key] = report
    manifest_sources = {}
    for source in plan["sources"]:
        source_id = source["id"]
        manifest_sources[source_id] = {"counts": state["sources"][source_id], "outputs": {}}
        for split in ("holdout", "train"):
            path = paths[f"{split}__{source_id}"]
            manifest_sources[source_id]["outputs"][split] = {
                "path": str(path.relative_to(staging)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    elapsed = max(time.monotonic() - state["started_at_monotonic"], 1e-9)
    manifest = {
        "format": "speck_production_rehearsal_acquisition",
        "format_version": 1,
        "status": "rights_bound_sources_acquired_not_training_authority",
        "plan_fingerprint": plan["plan_fingerprint"],
        "discovery_sha256": file_sha256(output / "source-discovery.json"),
        "sources": manifest_sources,
        "download_bytes": state["download_bytes"],
        "filtered_bytes": state["filtered_bytes"],
        "records": state["records"],
        "rejections": {**state["rejections"], "gitleaks": gitleaks_removed},
        "contamination": {
            "plan_sha256": plan["contamination_plan"]["sha256"],
            "tasks": contamination["tasks"],
            "benchmarks": contamination["benchmarks"],
        },
        "gitleaks": {
            "binary_sha256": security["gitleaks_binary"]["sha256"],
            "version": security["gitleaks_version"],
            "reports": reports,
        },
        "elapsed_seconds": elapsed,
    }
    state_path.unlink()
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    atomic_json(staging / "manifest.json", manifest)
    staging.replace(final)
    return _stage_result(
        result_path,
        "acquisition",
        {},
        {
            "download_bytes": manifest["download_bytes"],
            "download_bytes_per_second": manifest["download_bytes"] / elapsed,
            "filtered_bytes": manifest["filtered_bytes"],
        },
        {
            "manifest": {
                "path": str(final / "manifest.json"),
                "sha256": file_sha256(final / "manifest.json"),
            }
        },
    )


def _acquisition_manifest(plan):
    path = Path(plan["output_directory"]) / "acquired/manifest.json"
    value = json.loads(path.read_text())
    if value.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("acquisition manifest belongs to a different plan")
    for source in value["sources"].values():
        for item in source["outputs"].values():
            file = path.parent / item["path"]
            if file.stat().st_size != item["bytes"] or file_sha256(file) != item["sha256"]:
                raise ValueError("acquisition output identity mismatch")
    return path, value


def _global_dedup(plan, result_path):
    acquisition_path, acquisition = _acquisition_manifest(plan)
    output = Path(plan["output_directory"])
    sources = []
    precedence = 0
    for split in ("holdout", "train"):
        for source in plan["sources"]:
            precedence += 1
            item = acquisition["sources"][source["id"]]["outputs"][split]
            sources.append(
                {
                    "id": f"{split}__{source['id']}",
                    "precedence": precedence,
                    "path": str(acquisition_path.parent / item["path"]),
                    "sha256": item["sha256"],
                    "text_field": "text",
                    "content_sha256_field": "released_content_sha256",
                    "url_field": "url",
                    "domain_field": "host",
                    "blob_field": "content_id",
                }
            )
    config = {
        "format": "speck_production_text_preprocess",
        "format_version": 1,
        "status": "fixture_or_rehearsal_authorized_not_training_authority",
        "sources": sources,
        "deny_ledger": plan["deny_ledger"],
        "policy": {
            key: value
            for key, value in plan["deduplication"].items()
            if key != "checkpoint_records"
        },
        "checkpoint_records": plan["deduplication"]["checkpoint_records"],
        "cleanup_files": [],
        "output_directory": str(output / "deduplicated"),
    }
    atomic_json(output / "global-dedup-config.json", config)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    original_signature = production_data._signature

    def batched_signature(shingles, num_perm, seed):
        from datasketch import MinHash

        value = MinHash(num_perm=num_perm, seed=seed)
        value.update_batch(shingles)
        return value

    production_data._signature = batched_signature
    try:
        result = preprocess_sources(validate_preprocess_config(config))
    finally:
        production_data._signature = original_signature
    manifest = result["manifest"]
    index = output / "deduplicated" / manifest["index"]["path"]
    peak = max(before, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return _stage_result(
        result_path,
        "global_dedup",
        {"global_exact_deduplication": "pass", "global_near_deduplication": "pass"},
        {
            "records_seen": manifest["counts"]["records_seen"],
            "records_retained": manifest["counts"]["records_retained"],
            "sqlite_index_bytes": index.stat().st_size,
            "peak_rss_bytes": peak,
        },
        {
            "config": {
                "path": str(output / "global-dedup-config.json"),
                "sha256": file_sha256(output / "global-dedup-config.json"),
            },
            "manifest": {
                "path": str(output / "deduplicated/manifest.json"),
                "sha256": file_sha256(output / "deduplicated/manifest.json"),
            },
            "signature_implementation": "datasketch MinHash.update_batch; hashvalues qualified identical to the frozen scalar update loop",
        },
    )


def _dedup_manifest(plan):
    path = Path(plan["output_directory"]) / "deduplicated/manifest.json"
    value = json.loads(path.read_text())
    if (
        value.get("status")
        != "global_dedup_and_deny_complete_cleanup_receipt_required_not_training_authority"
    ):
        raise ValueError("deduplication manifest is incomplete")
    return path, value


def _pack_source(tokenizer, input_path, directory, target_tokens, shard_tokens):
    if directory.exists():
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            for shard in manifest["shards"]:
                path = directory / shard["path"]
                if file_sha256(path) != shard["sha256"]:
                    raise ValueError("packed rehearsal shard identity mismatch")
            return manifest
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    writer = TokenShardWriter(directory, "train", shard_tokens)
    documents = 0
    batch = []
    characters = 0
    with Path(input_path).open(encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line)["text"]
            batch.append(text)
            characters += len(text)
            if len(batch) >= 512 or characters >= 1_000_000:
                for tokens in tokenizer.encode_batch(batch, bos=True, eos=True):
                    writer.write(tokens)
                    documents += 1
                    if writer.total_tokens >= target_tokens:
                        break
                batch = []
                characters = 0
                if writer.total_tokens >= target_tokens:
                    break
        if batch and writer.total_tokens < target_tokens:
            for tokens in tokenizer.encode_batch(batch, bos=True, eos=True):
                writer.write(tokens)
                documents += 1
                if writer.total_tokens >= target_tokens:
                    break
    if writer.total_tokens < target_tokens:
        raise RuntimeError(
            f"deduplicated source exhausted at {writer.total_tokens:,}/{target_tokens:,} packed tokens"
        )
    writer.finish()
    manifest = {
        "format": "speck_production_rehearsal_packed_source",
        "format_version": 1,
        "status": "packed_not_training_authority",
        "target_tokens": target_tokens,
        "tokens": writer.total_tokens,
        "documents": documents,
        "shards": writer.shards,
    }
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def _packing(plan, result_path):
    dedup_path, dedup = _dedup_manifest(plan)
    output = Path(plan["output_directory"]) / "packed"
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(plan["tokenizer"]["path"])
    started = time.perf_counter()
    sources = {}
    total_tokens = 0
    for source in plan["sources"]:
        source_id = source["id"]
        item = dedup["outputs"][f"train__{source_id}"]
        manifest = _pack_source(
            tokenizer,
            dedup_path.parent / item["path"],
            output / source_id,
            source["target_tokens"],
            plan["packing"]["shard_tokens"],
        )
        sources[source_id] = manifest
        total_tokens += manifest["tokens"]
    elapsed = max(time.perf_counter() - started, 1e-9)
    packed_bytes = sum(
        (output / source_id / shard["path"]).stat().st_size
        for source_id, source in sources.items()
        for shard in source["shards"]
    )
    manifest = {
        "format": "speck_production_rehearsal_packed",
        "format_version": 1,
        "status": "20B_packed_not_training_authority",
        "plan_fingerprint": plan["plan_fingerprint"],
        "tokenizer_sha256": plan["tokenizer"]["sha256"],
        "sources": sources,
        "packed_tokens": total_tokens,
        "packed_bytes": packed_bytes,
        "elapsed_seconds": elapsed,
    }
    atomic_json(output / "manifest.json", manifest)
    return _stage_result(
        result_path,
        "packing",
        {"packing_integrity": "pass"},
        {
            "packing_tokens_per_second": total_tokens / elapsed,
            "packed_tokens": total_tokens,
            "packed_bytes": packed_bytes,
            "unique_tokens": total_tokens,
        },
        {
            "manifest": {
                "path": str(output / "manifest.json"),
                "sha256": file_sha256(output / "manifest.json"),
            }
        },
    )


def _resume_cleanup(plan, result_path):
    output = Path(plan["output_directory"])
    acquisition_path, acquisition = _acquisition_manifest(plan)
    raw_files = (
        [str(path) for path in (output / "acquired.building/raw").rglob("*") if path.is_file()]
        if (output / "acquired.building/raw").exists()
        else []
    )
    packed = json.loads((output / "packed/manifest.json").read_text())
    if raw_files or packed.get("packed_tokens", 0) < plan["target_tokens"]:
        raise RuntimeError("rehearsal cleanup or packed resume boundary is incomplete")
    probe = output / "resume-probe"
    probe.mkdir(exist_ok=True)
    first_source = plan["sources"][0]["id"]
    source_item = acquisition["sources"][first_source]["outputs"]["train"]
    source_path = acquisition_path.parent / source_item["path"]
    probe_input = probe / "input.jsonl"
    if not probe_input.exists():
        with source_path.open("rb") as source, probe_input.open("wb") as destination:
            for _ in range(2_000):
                line = source.readline()
                if not line:
                    break
                destination.write(line)
    probe_source = {
        "id": "production_probe",
        "precedence": 1,
        "path": str(probe_input),
        "sha256": file_sha256(probe_input),
        "text_field": "text",
        "content_sha256_field": "released_content_sha256",
        "url_field": "url",
        "domain_field": "host",
        "blob_field": "content_id",
    }

    def probe_config(name):
        return validate_preprocess_config(
            {
                "format": "speck_production_text_preprocess",
                "format_version": 1,
                "status": "fixture_or_rehearsal_authorized_not_training_authority",
                "sources": [probe_source],
                "deny_ledger": plan["deny_ledger"],
                "policy": {
                    key: value
                    for key, value in plan["deduplication"].items()
                    if key != "checkpoint_records"
                },
                "checkpoint_records": 100,
                "cleanup_files": [],
                "output_directory": str(probe / name),
            }
        )

    interrupted_config = probe_config("interrupted")
    if not Path(interrupted_config["output_directory"]).exists():
        try:
            preprocess_sources(interrupted_config, crash_after_records=500)
        except RuntimeError as error:
            if "injected production preprocess crash" not in str(error):
                raise
        else:
            raise RuntimeError("production-data interruption probe did not interrupt")
    resumed = preprocess_sources(interrupted_config)["manifest"]
    clean = preprocess_sources(probe_config("clean"))["manifest"]
    resumed_output = resumed["outputs"]["production_probe"]
    clean_output = clean["outputs"]["production_probe"]
    resumed_logical = _logical_sqlite_identity(probe / "interrupted" / resumed["index"]["path"])
    clean_logical = _logical_sqlite_identity(probe / "clean" / clean["index"]["path"])
    if (
        resumed_output["sha256"] != clean_output["sha256"]
        or resumed["removals"]["sha256"] != clean["removals"]["sha256"]
        or resumed_logical != clean_logical
        or resumed["counts"] != clean["counts"]
    ):
        raise RuntimeError("resumed production-data probe differs from uninterrupted output")
    dedup_path, dedup = _dedup_manifest(plan)
    packed_source = packed["sources"][first_source]
    reopened = _pack_source(
        Tokenizer(plan["tokenizer"]["path"]),
        dedup_path.parent / dedup["outputs"][f"train__{first_source}"]["path"],
        output / "packed" / first_source,
        plan["sources"][0]["target_tokens"],
        plan["packing"]["shard_tokens"],
    )
    if reopened != packed_source:
        raise RuntimeError("reopened packed source differs from its completed manifest")
    return _stage_result(
        result_path,
        "resume_cleanup",
        {"acquisition_cleanup": "pass", "interruption_resume": "pass"},
        {},
        {
            "acquisition_manifest_sha256": file_sha256(acquisition_path),
            "acquired_records": acquisition["records"],
            "completed_source_boundaries": len(acquisition["sources"]),
            "resume_basis": "durable source-file acquisition checkpoints, a real-data injected record-level global-dedup interruption matching uninterrupted output, and verified source-level packing reopen",
            "probe_records_seen": resumed["counts"]["records_seen"],
            "probe_output_sha256": resumed_output["sha256"],
            "probe_logical_sqlite_identity": resumed_logical,
            "probe_physical_sqlite_sha256": {
                "resumed": resumed["index"]["sha256"],
                "uninterrupted": clean["index"]["sha256"],
                "equal": resumed["index"]["sha256"] == clean["index"]["sha256"],
                "authority": "diagnostic only; SQLite page layout is not the logical index identity",
            },
        },
    )


def _logical_sqlite_identity(path):
    path = Path(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("resume probe SQLite integrity check failed")
        tables = {
            "docs": (
                "SELECT doc_seq, processed_index, source_index, source_id, line_number, "
                "byte_offset, content_sha256, dedup_sha256 FROM docs ORDER BY doc_seq"
            ),
            "bands": "SELECT band, band_hash, doc_seq FROM bands ORDER BY band, band_hash, doc_seq",
            "checkpoints": (
                "SELECT checkpoint_id, processed_records, next_doc_seq, index_chain "
                "FROM checkpoints ORDER BY checkpoint_id"
            ),
        }
        result = {}
        for table, query in tables.items():
            digest = hashlib.sha256()
            rows = 0
            for row in connection.execute(query):
                values = [value.hex() if isinstance(value, bytes) else value for value in row]
                digest.update(
                    (json.dumps(values, sort_keys=True, separators=(",", ":")) + "\n").encode()
                )
                rows += 1
            result[table] = {"rows": rows, "sha256": digest.hexdigest()}
        return result
    finally:
        connection.close()


def _firewall_disjointness(plan, result_path):
    dedup_path, dedup = _dedup_manifest(plan)
    output = Path(plan["output_directory"])
    holdout = {
        source["id"]: dedup["outputs"][f"holdout__{source['id']}"] for source in plan["sources"]
    }
    training = {
        source["id"]: dedup["outputs"][f"train__{source['id']}"] for source in plan["sources"]
    }
    holdout_hashes = set()
    for item in holdout.values():
        with (dedup_path.parent / item["path"]).open(encoding="utf-8") as handle:
            holdout_hashes.update(json.loads(line)["released_content_sha256"] for line in handle)
    overlap = 0
    for item in training.values():
        with (dedup_path.parent / item["path"]).open(encoding="utf-8") as handle:
            overlap += sum(
                json.loads(line)["released_content_sha256"] in holdout_hashes for line in handle
            )
    if overlap:
        raise RuntimeError("training and heldout rehearsal outputs overlap by content hash")
    manifest = {
        "format": "speck_production_rehearsal_firewall_candidates",
        "format_version": 1,
        "status": "disjoint_candidate_pool_not_consumer_authority",
        "plan_fingerprint": plan["plan_fingerprint"],
        "dedup_manifest_sha256": file_sha256(dedup_path),
        "holdout": holdout,
        "training": training,
        "exact_content_hash_overlap": overlap,
        "near_duplicate_boundary": "holdout inputs precede training inputs in the global verified-Jaccard preprocessor, so matching later training records are removed before packing",
    }
    path = output / "firewall-candidates.json"
    atomic_json(path, manifest)
    return _stage_result(
        result_path,
        "firewall_disjointness",
        {"firewall_training_disjointness": "pass"},
        {},
        {"manifest": {"path": str(path), "sha256": file_sha256(path)}},
    )


def run_production_rehearsal_stage(plan, stage_id, result_path):
    """Execute one immutable orchestration stage and write its standard result."""

    if stage_id not in STAGES:
        raise ValueError(f"unknown production rehearsal stage: {stage_id}")
    functions = {
        "source_identity": _source_identity,
        "acquisition": _acquisition,
        "global_dedup": _global_dedup,
        "packing": _packing,
        "resume_cleanup": _resume_cleanup,
        "firewall_disjointness": _firewall_disjointness,
    }
    return functions[stage_id](plan, Path(result_path).resolve())
