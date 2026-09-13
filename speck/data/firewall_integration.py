"""Bind complete firewall references to raw acquisition outputs and engineering banks."""

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

import speck.data.production_data as production_data
from speck.data.acquisition_units import _bound_identity, _digest, _unit_config
from speck.data.sources.code_near_duplicates import _jaccard, _shingles, _tokens
from speck.provenance.io import durable_json, file_sha256

CATEGORIES = ("web", "code", "math", "synthetic", "science", "reference")


def reference_sources(identity, root):
    """Read commitment metadata and derive all twelve input views, never audit payloads."""

    identity = _bound_identity(identity, Path(root))
    plan = json.loads(Path(identity["path"]).read_text())
    if plan.get("format") != "speck_flagship_firewall_plan" or plan.get("format_version") != 4:
        raise ValueError("integration requires the complete v4 firewall contract")
    firewall_identity = _bound_identity(plan["construction"]["runtime_manifest"], Path(root))
    prepared_identity = _bound_identity(plan["input_preparation"]["runtime_manifest"], Path(root))
    firewall = json.loads(Path(firewall_identity["path"]).read_text())
    prepared_path = Path(prepared_identity["path"])
    prepared = json.loads(prepared_path.read_text())
    if (
        firewall.get("status") != "production_firewall_complete_rights_and_operations_bound"
        or prepared.get("status") != "views_and_global_near_dedup_complete_not_consumer_authority"
    ):
        raise ValueError("firewall reference construction is incomplete")
    dedup_identity = _bound_identity(prepared["deduplication"]["manifest"], prepared_path.parent)
    dedup = json.loads(Path(dedup_identity["path"]).read_text())
    declared = {
        item["id"]: item for category in firewall["categories"] for item in category["inputs"]
    }
    expected = {f"{category}_{role}" for category in CATEGORIES for role in ("unseen", "primary")}
    outputs = prepared["deduplication"]["outputs"]
    if set(declared) != expected or set(outputs) != expected or dedup["outputs"] != outputs:
        raise ValueError("firewall does not bind exactly the twelve prepared reference views")
    sources = []
    for role in ("unseen", "primary"):
        for category in CATEGORIES:
            key = f"{category}_{role}"
            entry = outputs[key]
            path = (Path(dedup_identity["path"]).parent / entry["path"]).resolve()
            if (
                path != Path(declared[key]["path"]).resolve()
                or entry["sha256"] != declared[key]["sha256"]
            ):
                raise ValueError("reference source differs from the final firewall commitment")
            sources.append(
                {
                    "id": f"firewall_reference__{key}",
                    "precedence": len(sources) + 1,
                    "path": str(path),
                    "sha256": entry["sha256"],
                    "text_field": "text",
                    "content_sha256_field": "released_content_sha256",
                    "url_field": "url",
                    "domain_field": "host",
                    "blob_field": "content_id",
                }
            )
    return {
        "firewall_plan": identity,
        "firewall_manifest": firewall_identity,
        "prepared_manifest": prepared_identity,
        "reference_dedup_manifest": dedup_identity,
        "sources": sources,
        "policy": dedup["policy"],
        "records": prepared["deduplication"]["counts"]["records_retained"],
    }


def verify_excluded_parent(parent, references):
    """Check exact reference identities/precedence and preservation in a dedup result."""

    if (
        parent.get("sources", [])[:12] != references["sources"]
        or parent.get("policy") != references["policy"]
        or parent.get("gates", {}).get("global_exact_deduplication") != "pass"
        or parent.get("gates", {}).get(
            "disk_backed_Minhash_candidates_and_verified_near_deduplication"
        )
        != "pass"
    ):
        raise ValueError("excluded parent does not bind the complete reference pass")
    for source in references["sources"]:
        if parent.get("outputs", {}).get(source["id"], {}).get("sha256") != source["sha256"]:
            raise ValueError("a firewall reference was removed or changed during the pass")


def group_acquisition_units(plan, acquired, output):
    """Concatenate verified disjoint units per category, preserving each original record."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    inputs = []
    for unit in plan["units"]:
        directory = Path(acquired) / unit["id"]
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest.get("config_sha256") != _digest(_unit_config(plan, unit))
            or manifest.get("status") != "complete_not_training_data"
        ):
            raise ValueError("group input differs from its completed acquisition unit")
        item = manifest["output"]
        path = directory / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("group input acquisition identity mismatch")
        inputs.append(
            {
                "unit_id": unit["id"],
                "category": unit["category"],
                "path": str(path),
                "sha256": item["sha256"],
            }
        )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["inputs"] != inputs:
            raise ValueError("grouped acquisition inputs changed")
        for entry in manifest["outputs"].values():
            if file_sha256(output / entry["path"]) != entry["sha256"]:
                raise ValueError("grouped acquisition output changed")
        return manifest
    entries = {}
    for category in CATEGORIES:
        path = output / f"{category}.jsonl"
        if path.exists():
            raise FileExistsError(f"unpublished grouped output exists: {path}")
        records = utf8_bytes = 0
        with path.open("xb") as handle:
            for item in inputs:
                if item["category"] != category:
                    continue
                with Path(item["path"]).open("rb") as source:
                    for raw in source:
                        handle.write(raw)
                        records += 1
                        utf8_bytes += len(json.loads(raw)["text"].encode())
        entries[category] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "records": records,
            "utf8_bytes": utf8_bytes,
        }
    manifest = {"format": "speck_acquisition_category_groups", "inputs": inputs, "outputs": entries}
    durable_json(manifest_path, manifest)
    return manifest


def _batched_signature(shingles, num_perm, seed):
    from datasketch import MinHash

    value = MinHash(num_perm=num_perm, seed=seed)
    value.update_batch(shingles)
    return value


def make_reference_controls(references, output):
    """Freeze an exact and a matched near replay of one eligible reference record."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policy = references["policy"]
    pattern = re.compile(policy["token_pattern"])
    source = references["sources"][0]
    if file_sha256(source["path"]) != source["sha256"]:
        raise ValueError("reference control input identity mismatch")
    with Path(source["path"]).open() as handle:
        for line_number, raw in enumerate(handle):
            record = json.loads(raw)
            tokens = _tokens(record["text"], pattern, policy["maximum_document_tokens"])
            if not 100 <= len(tokens) < policy["maximum_document_tokens"] - 10:
                continue
            near = {**record, "text": record["text"] + " speck_engineering_reference_probe"}
            near["released_content_sha256"] = hashlib.sha256(near["text"].encode()).hexdigest()
            left = _shingles(tokens, policy["shingle_tokens"])
            right = _shingles(
                _tokens(near["text"], pattern, policy["maximum_document_tokens"]),
                policy["shingle_tokens"],
            )
            similarity = _jaccard(left, right)
            signatures = [
                _batched_signature(value, policy["num_perm"], policy["minhash_seed"])
                for value in (left, right)
            ]
            bands = [production_data._band_values(value, policy["bands"]) for value in signatures]
            if similarity >= policy["verified_jaccard_threshold"] and any(
                a == b for a, b in zip(*bands, strict=True)
            ):
                break
        else:
            raise RuntimeError("no eligible reference control record")
    result = {
        "reference_source": source["id"],
        "reference_line": line_number,
        "verified_jaccard": similarity,
        "outputs": {},
    }
    for name, value in (("firewall_exact_control", record), ("firewall_near_control", near)):
        path = output / f"{name}.jsonl"
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("reference control changed")
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(payload)
        result["outputs"][name] = {"path": str(path), "sha256": file_sha256(path)}
    durable_json(output / "manifest.json", result)
    return result


def exclusion_config(plan, references, grouped_directory, controls, output):
    grouped_directory = Path(grouped_directory)
    groups = json.loads((grouped_directory / "manifest.json").read_text())
    sources = list(references["sources"])
    if {
        k: v for k, v in plan["base"]["deduplication"].items() if k != "checkpoint_records"
    } != references["policy"]:
        raise ValueError("acquisition and reference dedup policies differ")
    for category in CATEGORIES:
        entry = groups["outputs"][category]
        sources.append(
            {
                "id": f"acquired_train__{category}",
                "precedence": len(sources) + 1,
                "path": str(grouped_directory / entry["path"]),
                "sha256": entry["sha256"],
                "text_field": "text",
                "content_sha256_field": "released_content_sha256",
                "url_field": "url",
                "domain_field": "host",
                "blob_field": "content_id",
            }
        )
    for name, entry in controls["outputs"].items():
        sources.append({**sources[-1], "id": name, "precedence": len(sources) + 1, **entry})
    return {
        "format": "speck_production_text_preprocess",
        "format_version": 1,
        "status": "fixture_or_rehearsal_authorized_not_training_authority",
        "sources": sources,
        "deny_ledger": plan["base"]["deny_ledger"],
        "policy": references["policy"],
        "checkpoint_records": 10000,
        "cleanup_files": [],
        "output_directory": str(output),
    }


def run_exclusion(config, *, crash_after_records=None):
    original = production_data._signature
    started = time.perf_counter()
    production_data._signature = _batched_signature
    try:
        result = production_data.preprocess_sources(config, crash_after_records=crash_after_records)
    finally:
        production_data._signature = original
    return {"result": result, "elapsed_seconds": time.perf_counter() - started}


def analyze_exclusion(directory, references):
    directory = Path(directory)
    parent = json.loads((directory / "manifest.json").read_text())
    verify_excluded_parent(parent, references)
    connection = sqlite3.connect(
        (directory / parent["index"]["path"]).as_uri() + "?mode=ro", uri=True
    )
    try:
        overlap = connection.execute(
            "SELECT COUNT(*) FROM (SELECT content_sha256 FROM docs WHERE source_index<12 "
            "INTERSECT SELECT content_sha256 FROM docs WHERE source_index>=12)"
        ).fetchone()[0]
    finally:
        connection.close()
    if overlap:
        raise ValueError("retained candidate/reference exact overlap")
    removal_counts = {}
    controls = {}
    with (directory / parent["removals"]["path"]).open() as handle:
        for line in handle:
            row = json.loads(line)
            source = row["removed_source"]
            if source.startswith("acquired_train__"):
                key = (
                    "reference"
                    if (row.get("kept") or {})
                    .get("source_id", "")
                    .startswith("firewall_reference__")
                    else "other"
                )
                counts = removal_counts.setdefault(source, Counter())
                counts[f"{key}:{row['reason']}"] += 1
            elif source in ("firewall_exact_control", "firewall_near_control"):
                controls[source] = {
                    "reason": row["reason"],
                    "kept_source": row["kept"]["source_id"],
                    "verified_jaccard": row["verified_shingle_jaccard"],
                }
    for name, reason in (
        ("firewall_exact_control", "exact_duplicate"),
        ("firewall_near_control", "near_duplicate"),
    ):
        if (
            controls.get(name, {}).get("reason") != reason
            or not controls[name]["kept_source"].startswith("firewall_reference__")
            or parent["outputs"][name]["bytes"] != 0
        ):
            raise ValueError("firewall exact/near reference control failed")
    retained = {}
    for category in CATEGORIES:
        entry = parent["outputs"][f"acquired_train__{category}"]
        path = directory / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("excluded candidate output identity mismatch")
        records = utf8_bytes = 0
        with path.open() as handle:
            for line in handle:
                records += 1
                utf8_bytes += len(json.loads(line)["text"].encode())
        retained[category] = {"records": records, "utf8_bytes": utf8_bytes, **entry}
    return {
        "references": references,
        "retained": retained,
        "candidate_removals": removal_counts,
        "controls": controls,
        "exact_reference_overlap": overlap,
        "reference_outputs_preserved": True,
        "parent_manifest": {
            "path": str(directory / "manifest.json"),
            "sha256": file_sha256(directory / "manifest.json"),
        },
        "training_authority": False,
    }
