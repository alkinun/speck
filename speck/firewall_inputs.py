"""Prepare deterministic source-balanced inputs for the real data firewall."""

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

import speck.production_data as production_data
from speck.io import atomic_json, file_sha256
from speck.production_data import preprocess_sources, validate_preprocess_config

FORMAT = "speck_firewall_input_preparation"
FORMAT_VERSION = 1
FALLBACK_STATUS = "project_owner_accepted_non_gpu_operations_fallback_through_150B"


def _identity(value, root, context):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{context} must contain path and sha256")
    path = Path(value["path"]).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or file_sha256(path) != value["sha256"]:
        raise ValueError(f"{context} identity mismatch")
    return {"path": str(path), "sha256": value["sha256"]}


def validate_firewall_input_plan(value, *, config_dir=None):
    root = Path(config_dir or ".").resolve()
    expected = {
        "format",
        "format_version",
        "status",
        "seed",
        "operations_fallback",
        "deny_ledger",
        "deduplication",
        "inputs",
        "output_directory",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("firewall input plan fields are invalid")
    if (
        value["format"] != FORMAT
        or value["format_version"] != FORMAT_VERSION
        or value["status"] != "preparation_authorized_not_consumer_or_training_authority"
    ):
        raise ValueError("unsupported firewall input plan")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
        raise ValueError("firewall input seed must be an integer")
    fallback = _identity(value["operations_fallback"], root, "operations fallback")
    deny = _identity(value["deny_ledger"], root, "deny ledger")
    deduplication = value["deduplication"]
    if not isinstance(deduplication, dict) or "checkpoint_records" not in deduplication:
        raise ValueError("firewall input plan requires deduplication settings")
    inputs = value["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("firewall input plan requires inputs")
    ids = []
    normalized = []
    categories = set()
    roles = {}
    for item in inputs:
        if set(item) != {
            "id",
            "source_id",
            "category",
            "role",
            "input",
            "view_target_bytes",
            "firewall_group",
        }:
            raise ValueError("firewall input declaration fields are invalid")
        if item["role"] not in {"primary", "unseen"}:
            raise ValueError("firewall input role must be primary or unseen")
        target = item["view_target_bytes"]
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            raise ValueError("firewall view target must be positive")
        ids.append(item["id"])
        categories.add(item["category"])
        roles.setdefault(item["category"], set()).add(item["role"])
        normalized.append({**item, "input": _identity(item["input"], root, f"input {item['id']}")})
    if len(ids) != len(set(ids)):
        raise ValueError("firewall input IDs must be unique")
    expected_categories = {"web", "code", "math", "synthetic", "science", "reference"}
    if categories != expected_categories or any(
        roles[category] != {"primary", "unseen"} for category in categories
    ):
        raise ValueError("every firewall category requires primary and unseen inputs")
    output = Path(value["output_directory"]).expanduser()
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    normalized_plan = {
        **value,
        "operations_fallback": fallback,
        "deny_ledger": deny,
        "inputs": normalized,
        "output_directory": str(output),
    }
    normalized_plan["plan_fingerprint"] = hashlib.sha256(
        json.dumps(normalized_plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return normalized_plan


def load_firewall_input_plan(path):
    path = Path(path).resolve()
    return validate_firewall_input_plan(json.loads(path.read_text()), config_dir=path.parent)


def _validated_plan(plan):
    if "plan_fingerprint" not in plan:
        return validate_firewall_input_plan(plan)
    payload = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if plan["plan_fingerprint"] != fingerprint:
        raise ValueError("normalized firewall input plan fingerprint mismatch")
    return plan


def _verify_operations_fallback(plan):
    identity = plan["operations_fallback"]
    path = Path(identity["path"])
    if not path.is_file() or file_sha256(path) != identity["sha256"]:
        raise ValueError("firewall input operations fallback identity mismatch")
    fallback = json.loads(path.read_text())
    if (
        fallback.get("format") != "speck_production_data_calibration_fallback"
        or fallback.get("status") != FALLBACK_STATUS
        or fallback.get("training_authority") is not False
    ):
        raise ValueError("firewall input operations fallback is invalid")


def _select_view(item, destination, seed):
    source_path = Path(item["input"]["path"])
    index_path = destination.with_suffix(".selection.sqlite3")
    connection = sqlite3.connect(index_path)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS candidates (score BLOB NOT NULL, byte_offset INTEGER PRIMARY KEY, line_bytes INTEGER NOT NULL, text_bytes INTEGER NOT NULL)"
        )
        connection.execute("CREATE INDEX IF NOT EXISTS candidates_score ON candidates(score)")
        if connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0:
            with source_path.open("rb") as source:
                while raw := source.readline():
                    offset = source.tell() - len(raw)
                    record = json.loads(raw)
                    text = record.get("text")
                    digest = record.get("released_content_sha256")
                    if (
                        not isinstance(text, str)
                        or not isinstance(digest, str)
                        or hashlib.sha256(text.encode()).hexdigest() != digest
                    ):
                        raise ValueError(f"invalid firewall source record: {item['id']}")
                    score = hashlib.sha256(f"{seed}\0{item['id']}\0{digest}".encode()).digest()
                    connection.execute(
                        "INSERT INTO candidates VALUES (?, ?, ?, ?)",
                        (score, offset, len(raw), len(text.encode())),
                    )
            connection.commit()
        selected = []
        total = 0
        for score, offset, line_bytes, text_bytes in connection.execute(
            "SELECT score, byte_offset, line_bytes, text_bytes FROM candidates ORDER BY score, byte_offset"
        ):
            selected.append((score, offset, line_bytes))
            total += text_bytes
            if total >= item["view_target_bytes"]:
                break
        if total < item["view_target_bytes"]:
            raise RuntimeError(f"firewall source {item['id']} cannot meet its view target")
        with source_path.open("rb") as source, destination.open("w", encoding="utf-8") as output:
            for _, offset, line_bytes in selected:
                source.seek(offset)
                raw = source.read(line_bytes)
                record = json.loads(raw)
                record["firewall_group"] = item["firewall_group"]
                record["firewall_parent_source_id"] = item["source_id"]
                output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()
            os.fsync(output.fileno())
    finally:
        connection.close()
    index_path.unlink()
    return {
        "path": destination.name,
        "sha256": file_sha256(destination),
        "records": len(selected),
        "text_bytes": total,
        "target_text_bytes": item["view_target_bytes"],
    }


def _verify_published_inputs(plan, output, manifest):
    if manifest.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("published firewall inputs belong to a different plan")
    for item in manifest.get("views", {}).values():
        path = output / item["path"]
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ValueError("published firewall source view identity mismatch")
    deduplication = manifest.get("deduplication", {})
    dedup_manifest = deduplication.get("manifest", {})
    path = output / dedup_manifest.get("path", "")
    if not path.is_file() or file_sha256(path) != dedup_manifest.get("sha256"):
        raise ValueError("published firewall dedup manifest identity mismatch")
    for item in deduplication.get("outputs", {}).values():
        path = output / "deduplicated" / item["path"]
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ValueError("published firewall dedup output identity mismatch")


def prepare_firewall_inputs(plan, *, restart=False):
    """Build deterministic views and globally near-deduplicate them before partitioning."""

    plan = _validated_plan(plan)
    _verify_operations_fallback(plan)
    output = Path(plan["output_directory"])
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        _verify_published_inputs(plan, output, manifest)
        return manifest
    if output.exists() and restart:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    view_state_path = output / "view-state.json"
    if view_state_path.is_file():
        view_state = json.loads(view_state_path.read_text())
        if view_state.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise ValueError("staged firewall views belong to a different plan; use restart")
    else:
        view_state = {"plan_fingerprint": plan["plan_fingerprint"], "views": {}}
    views = view_state["views"]
    for item in plan["inputs"]:
        destination = output / f"{item['id']}.jsonl"
        existing = views.get(item["id"])
        if (
            existing is not None
            and destination.is_file()
            and file_sha256(destination) == existing.get("sha256")
        ):
            continue
        if destination.exists():
            destination.unlink()
        views[item["id"]] = _select_view(item, destination, plan["seed"])
        atomic_json(view_state_path, view_state)
    ordered = sorted(plan["inputs"], key=lambda item: (item["role"] != "unseen", item["id"]))
    sources = [
        {
            "id": item["id"],
            "precedence": index + 1,
            "path": str(output / views[item["id"]]["path"]),
            "sha256": views[item["id"]]["sha256"],
            "text_field": "text",
            "content_sha256_field": "released_content_sha256",
            "url_field": "url",
            "domain_field": "host",
            "blob_field": "content_id",
        }
        for index, item in enumerate(ordered)
    ]
    dedup_config = {
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
    atomic_json(output / "dedup-config.json", dedup_config)
    original_signature = production_data._signature

    def batched_signature(shingles, num_perm, seed):
        from datasketch import MinHash

        value = MinHash(num_perm=num_perm, seed=seed)
        value.update_batch(shingles)
        return value

    production_data._signature = batched_signature
    try:
        dedup = preprocess_sources(validate_preprocess_config(dedup_config))["manifest"]
    finally:
        production_data._signature = original_signature
    manifest = {
        "format": "speck_firewall_prepared_inputs",
        "format_version": 1,
        "status": "views_and_global_near_dedup_complete_not_consumer_authority",
        "plan_fingerprint": plan["plan_fingerprint"],
        "operations_fallback": plan["operations_fallback"],
        "views": views,
        "deduplication": {
            "manifest": {
                "path": "deduplicated/manifest.json",
                "sha256": file_sha256(output / "deduplicated/manifest.json"),
            },
            "counts": dedup["counts"],
            "outputs": dedup["outputs"],
        },
        "inputs": [
            {
                "id": item["id"],
                "source_id": item["source_id"],
                "category": item["category"],
                "role": item["role"],
                "firewall_group": item["firewall_group"],
            }
            for item in plan["inputs"]
        ],
        "consumer_authority": False,
        "training_authority": False,
    }
    atomic_json(output / "manifest.json", manifest)
    view_state_path.unlink()
    return manifest


def freeze_calibrated_firewall_config(firewall_plan, prepared_manifest, destination):
    """Freeze final construction inputs and quotas from a verified preparation result."""

    plan_path = Path(firewall_plan).resolve()
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("format") != "speck_flagship_firewall_plan"
        or plan.get("format_version") != 2
        or plan.get("status")
        != "calibrated_production_input_plan_frozen_not_consumer_or_training_authority"
        or plan.get("training_authority") is not False
    ):
        raise ValueError("unsupported flagship firewall plan")
    prepared_path = Path(prepared_manifest).resolve()
    prepared = json.loads(prepared_path.read_text())
    input_identity = plan.get("input_preparation", {})
    input_plan_path = Path(input_identity.get("path", ""))
    if not input_plan_path.is_absolute():
        input_plan_path = Path(__file__).parents[1] / input_plan_path
    if not input_plan_path.is_file() or file_sha256(input_plan_path) != input_identity.get(
        "sha256"
    ):
        raise ValueError("firewall plan input-preparation identity mismatch")
    input_plan = load_firewall_input_plan(input_plan_path)
    output = Path(input_plan["output_directory"])
    if prepared_path != output / "manifest.json":
        raise ValueError("prepared firewall manifest path differs from its frozen plan")
    _verify_published_inputs(input_plan, output, prepared)
    if prepared.get("status") != "views_and_global_near_dedup_complete_not_consumer_authority":
        raise ValueError("firewall input preparation is incomplete")
    targets = plan["targets"]
    primary_allocations = {
        "tokenizer_train": targets["tokenizer_train_bytes_per_category"],
        "tokenizer_eval": targets["tokenizer_eval_bytes_per_category"],
        "selection_heldout": (
            targets["selection_bytes_per_category"]
            - targets["minimum_unseen_selection_bytes_per_category"]
        ),
        "D5_tokenizer": targets["D5_tokenizer_bytes_per_category"],
        "E2_mixture": targets["E2_mixture_bytes_per_category"],
    }
    unseen_allocations = {
        "tokenizer_train": 0,
        "tokenizer_eval": 0,
        "selection_heldout": targets["minimum_unseen_selection_bytes_per_category"],
        "D5_tokenizer": 0,
        "E2_mixture": 0,
    }
    declarations = {item["id"]: item for item in prepared["inputs"]}
    authority = {}
    for key, identity in plan["authority"].items():
        path = Path(identity["path"])
        if not path.is_absolute():
            path = Path(__file__).parents[1] / path
        path = path.resolve()
        if not path.is_file() or file_sha256(path) != identity["sha256"]:
            raise ValueError(f"firewall {key} authority identity mismatch")
        authority[key] = {"path": str(path), "sha256": identity["sha256"]}
    categories = []
    for category in ("web", "code", "math", "synthetic", "science", "reference"):
        inputs = []
        for role, allocations in (
            ("primary", primary_allocations),
            ("unseen", unseen_allocations),
        ):
            input_id = f"{category}_{role}"
            declaration = declarations[input_id]
            result = prepared["deduplication"]["outputs"][input_id]
            path = output / "deduplicated" / result["path"]
            if not path.is_file() or file_sha256(path) != result["sha256"]:
                raise ValueError(f"prepared firewall output identity mismatch: {input_id}")
            inputs.append(
                {
                    "id": input_id,
                    "source_id": declaration["source_id"],
                    "path": str(path),
                    "sha256": result["sha256"],
                    "format": "jsonl",
                    "text_field": "text",
                    "content_sha256_field": "released_content_sha256",
                    "domain_field": "firewall_group",
                    "training_mixture_eligible": role == "primary",
                    "allocations": dict(allocations),
                }
            )
        categories.append({"id": category, "inputs": inputs})
    config = {
        "format": "speck_calibrated_data_firewall",
        "format_version": 1,
        "status": "construction_authorized_not_training_authority",
        "authority": authority,
        "policy": {
            "global_dedup_normalization": "NFKC+lower+whitespace",
            "tokenizer_train_bytes_per_category": targets["tokenizer_train_bytes_per_category"],
            "tokenizer_eval_bytes_per_category": targets["tokenizer_eval_bytes_per_category"],
            "selection_bytes_per_category": targets["selection_bytes_per_category"],
            "minimum_unseen_selection_bytes_per_category": targets[
                "minimum_unseen_selection_bytes_per_category"
            ],
            "minimum_selection_domains_per_category": 2,
            "minimum_heldout_domains_per_category": 1,
            "partition_seeds": plan["partition_seeds"],
            "sealed_audits": [
                {
                    **audit,
                    "bytes_per_category": targets[f"{audit['id']}_bytes_per_category"],
                }
                for audit in plan["sealed_audits"]
            ],
        },
        "categories": categories,
        "output_directory": plan["output_directory"],
    }
    destination = Path(destination)
    from speck.data_firewall_calibrated import validate_calibrated_firewall_config

    validate_calibrated_firewall_config(config, config_dir=destination.parent)
    atomic_json(destination, config)
    return config
