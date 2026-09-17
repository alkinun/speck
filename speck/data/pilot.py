"""Assemble one finite pilot from retained stock using the maintained data pipeline."""

import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from speck.config import load_experiment
from speck.data.configuration import derive_source_quotas
from speck.data.dataset import prepare_dataset, resolve_data_dir, verify_shards
from speck.data.production_data import load_preprocess_config, preprocess_sources
from speck.evaluation.protocol import BenchmarkExclusion
from speck.provenance.io import atomic_json, file_sha256
from speck.provenance.repository import repository_root
from speck.tokenization.tokenizer import get_tokenizer


def checked_json(path, expected):
    path = Path(path)
    if file_sha256(path) != expected:
        raise ValueError(f"input identity mismatch: {path}")
    return json.loads(path.read_text())


def archived_json(identity, destination):
    """Recover exact historical records without depending on removed working-tree paths."""

    raw = subprocess.check_output(
        ["git", "show", f"{identity['revision']}:{identity['path']}"], cwd=repository_root()
    )
    if hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        raise ValueError("historical input digest mismatch")
    destination = Path(destination)
    if destination.exists() and destination.read_bytes() != raw:
        raise ValueError("historical record destination contains different bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return json.loads(raw)


def cached_documents(manifest_path, expected_sha256, tokenizer):
    """Verify source text and reuse its checked per-document token counts."""

    manifest_path = Path(manifest_path)
    stock = checked_json(manifest_path, expected_sha256)
    if stock.get("status") != "complete_document_token_cache_not_training_view":
        raise ValueError("pilot input requires a completed document token stock")
    plan = stock["plan"]
    if plan["tokenizer"]["sha256"] != tokenizer.fingerprint():
        raise ValueError("pilot stock uses a different tokenizer")
    checked_json(plan["parent_manifest"]["path"], plan["parent_manifest"]["sha256"])
    source = Path(plan["input"]["path"])
    index = manifest_path.parent / stock["documents"]["path"]
    if (
        file_sha256(source) != plan["input"]["sha256"]
        or file_sha256(index) != stock["documents"]["sha256"]
    ):
        raise ValueError("stock text or document index changed")
    with source.open() as text, index.open() as spans:
        for ordinal, (raw, span_raw) in enumerate(zip(text, spans, strict=True)):
            row, span = json.loads(raw), json.loads(span_raw)
            digest = hashlib.sha256(row["text"].encode()).hexdigest()
            if (
                span["ordinal"] != ordinal
                or digest != span["released_content_sha256"]
                or digest != row["released_content_sha256"]
                or row["content_id"] != span["content_id"]
            ):
                raise ValueError("stock document and span identity differ")
            yield row, span["token_count"]


def code_documents(receipt, tokenizer, language, opened):
    """Read only completed, hash-bound acquisition units for the requested language."""

    if receipt.get("status") != "content_acquisition_complete":
        raise ValueError("code acquisition is incomplete")
    for unit in receipt["units"]:
        manifest = unit["manifest"]
        if manifest["language"] != language:
            continue
        identity = unit["manifest_identity"]
        if checked_json(identity["path"], identity["sha256"]) != manifest:
            raise ValueError("code acquisition manifest differs from its receipt")
        path = Path(identity["path"]).parent / manifest["output"]["path"]
        if file_sha256(path) != manifest["output"]["sha256"]:
            raise ValueError("code acquisition payload changed")
        opened.append({"path": str(path), "sha256": manifest["output"]["sha256"]})
        with path.open() as handle:
            batch = [json.loads(line) for line in handle]
        encoded = tokenizer.encode_batch([row["text"] for row in batch], bos=True, eos=True)
        for row, tokens in zip(batch, encoded, strict=True):
            if hashlib.sha256(row["text"].encode()).hexdigest() != row["released_content_sha256"]:
                raise ValueError("code document hash mismatch")
            yield row, len(tokens)


def select_candidates(experiment, runtime_root, work, evaluation):
    """Select finite whole-document prefixes, then exclude pinned benchmark overlap."""

    experiment, runtime_root, work = Path(experiment), Path(runtime_root), Path(work)
    configs = load_experiment(experiment, "data", "tokenizer", "inputs")
    inputs, data = configs["inputs"], configs["data"]
    if inputs.get("format") != "speck_retained_pilot_inputs" or inputs.get("format_version") != 1:
        raise ValueError("unsupported retained pilot input configuration")
    sources = inputs["sources"]
    quotas, _ = derive_source_quotas(
        data["sources"], data["mixture"], data["requested_train_tokens"]
    )
    if [row["id"] for row in sources] != [row["id"] for row in data["sources"]]:
        raise ValueError("pilot stock and data source ordering differ")
    multiplier = inputs["candidate_multiplier"]
    if type(multiplier) is not int or multiplier < 2:
        raise ValueError("pilot candidate multiplier must be an integer >= 2")
    languages = inputs["code_languages_percent"]
    if sum(languages.values()) != 100 or any(
        type(v) is not int or v < 1 for v in languages.values()
    ):
        raise ValueError("code language percentages must be positive integers summing to 100")
    protocol = json.loads(Path(evaluation).read_text())
    if protocol["protocol_sha256"] != file_sha256(experiment / "evaluation.json"):
        raise ValueError("prepared evaluation differs from the pilot protocol")
    contract = {
        "configs": {name: file_sha256(experiment / f"{name}.json") for name in configs},
        "evaluation_sha256": file_sha256(evaluation),
    }
    selection_path = work / "selection.json"
    if selection_path.exists():
        result = json.loads(selection_path.read_text())
        if result["contract"] != contract:
            raise ValueError("pilot selection contract changed")
        for source in result["sources"]:
            if file_sha256(source["path"]) != source["sha256"]:
                raise ValueError("selected pilot payload changed")
        return result
    work.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer(**configs["tokenizer"])
    exclusion = BenchmarkExclusion(protocol)
    code_receipt = archived_json(inputs["archived"]["code_receipt"], work / "code-receipt.json")
    archived_json(inputs["archived"]["deny_ledger"], work / "deny-ledger.json")
    selected = []
    for source in sources:
        name = source["id"]
        target = quotas[name] * multiplier + 4 * data["validation_tokens_per_source"]
        final = work / f"{name}.jsonl"
        receipt_path = work / f"{name}.selection.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["contract"] != contract or file_sha256(final) != receipt["sha256"]:
                raise ValueError("existing source selection changed")
            selected.append(receipt)
            continue
        staging = final.with_suffix(".jsonl.building")
        # Incomplete selections are preserved; start a new work directory after a failure.
        if staging.exists() or final.exists():
            raise FileExistsError(f"incomplete source selection exists: {staging}")
        counts, rejections, opened = Counter(), Counter(), []
        if "stock_manifest" in source:
            streams = {
                "all": (
                    target,
                    cached_documents(
                        runtime_root / source["stock_manifest"], source["sha256"], tokenizer
                    ),
                )
            }
        else:
            streams = {
                language: (
                    math.ceil(target * weight / 100),
                    code_documents(code_receipt, tokenizer, language, opened),
                )
                for language, weight in languages.items()
            }
        language_counts = {}
        with staging.open("x") as handle:
            for language, (required, stream) in streams.items():
                retained = 0
                try:
                    for row, tokens in stream:
                        counts["seen_documents"] += 1
                        matches = exclusion.matches(row["text"])
                        if matches:
                            counts["benchmark_rejected_documents"] += 1
                            rejections.update(reference.split(":", 1)[0] for reference in matches)
                            continue
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        counts["documents"] += 1
                        counts["tokens"] += tokens
                        retained += tokens
                        if retained >= required:
                            break
                finally:
                    stream.close()
                if retained < required:
                    raise ValueError(
                        f"insufficient pilot candidates: {name}/{language}: {retained} < {required}"
                    )
                language_counts[language] = retained
        staging.replace(final)
        receipt = {
            "id": name,
            "path": str(final.resolve()),
            "sha256": file_sha256(final),
            "contract": contract,
            "counts": dict(counts),
            "benchmark_rejections": dict(rejections),
            "tokens_by_language": language_counts,
            "acquisition_payloads_opened": opened,
        }
        atomic_json(receipt_path, receipt)
        selected.append(receipt)
        print(f"Selected {name}: {counts['tokens']:,} candidate tokens", flush=True)
    first = next(source for source in sources if "stock_manifest" in source)
    stock = checked_json(runtime_root / first["stock_manifest"], first["sha256"])
    parent = checked_json(
        stock["plan"]["parent_manifest"]["path"], stock["plan"]["parent_manifest"]["sha256"]
    )
    references = parent["sources"][:12]
    if (
        any(not row["id"].startswith("firewall_reference__") for row in references)
        or len(references) != 12
    ):
        raise ValueError("retained reference inventory is incomplete")
    preprocess = {
        "format": "speck_production_text_preprocess",
        "format_version": 2,
        "status": "fixture_or_rehearsal_authorized_not_training_authority",
        "sources": [
            *references,
            *[
                {
                    "id": row["id"],
                    "precedence": index + 13,
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "text_field": "text",
                    "content_sha256_field": "released_content_sha256",
                    "url_field": "url",
                    "domain_field": "host",
                    "blob_field": "content_id",
                }
                for index, row in enumerate(selected)
            ],
        ],
        "deny_ledger": {
            "path": str((work / "deny-ledger.json").resolve()),
            "sha256": inputs["archived"]["deny_ledger"]["sha256"],
        },
        "policy": parent["policy"],
        "checkpoint_records": 1000,
        "cleanup_files": [],
        "output_directory": str((work / "excluded").resolve()),
        "sqlite": {
            "journal_mode": "WAL",
            "synchronous": "FULL",
            "wal_autocheckpoint_pages": 65536,
            "page_size": 4096,
            "cache_size_kib": 2000,
        },
    }
    atomic_json(work / "preprocess.json", preprocess)
    result = {
        "contract": contract,
        "sources": selected,
        "selection": "finite source-order whole-document prefixes; no repeated input units",
        "status": "selected_pending_joint_exclusion",
    }
    atomic_json(selection_path, result)
    return result


def exclude_candidates(work):
    work = Path(work)
    return preprocess_sources(
        load_preprocess_config(work / "preprocess.json"), batched_minhash=True
    )


def pack_candidates(experiment, work):
    """Pack only the jointly excluded sources with deterministic global split/dedup."""

    work = Path(work)
    configs = load_experiment(experiment, "data", "tokenizer")
    result = exclude_candidates(work)  # Reopens and verifies the completed exclusion.
    manifest = result["manifest"]
    iterators = {}

    def documents(path):
        with path.open() as handle:
            for ordinal, raw in enumerate(handle):
                row = json.loads(raw)
                yield {
                    "content": row["text"],
                    "row": ordinal,
                    "metadata": {"content_id": row["content_id"], "language": row.get("language")},
                }

    for source in configs["data"]["sources"]:
        entry = manifest["outputs"][source["id"]]
        path = work / "excluded" / entry["path"]
        iterators[source["id"]] = documents(path)
    try:
        packed = prepare_dataset(
            **configs["data"],
            tokenizer=get_tokenizer(**configs["tokenizer"]),
            document_iterators=iterators,
        )
    finally:
        for stream in iterators.values():
            stream.close()
    output = resolve_data_dir(configs["data"].get("output_dir"), configs["data"].get("output_name"))
    verify_shards(output, packed)
    atomic_json(
        work / "packing.json",
        {
            "manifest": {
                "path": str(output / "manifest.json"),
                "sha256": file_sha256(output / "manifest.json"),
            },
            "selection_sha256": file_sha256(work / "selection.json"),
            "exclusion_sha256": file_sha256(work / "excluded/manifest.json"),
            "status": "packed_and_reopened_hardware_qualification_pending",
        },
    )
    return packed
