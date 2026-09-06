"""Validate the append-only HELMET archive-local rights audit offline."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

EXPECTED_PATH_FAMILIES = {
    "alce": 2,
    "json_kv": 5,
    "kilt": 24,
    "msmarco": 6,
    "ruler": 15,
}
EXPECTED_AUDIT_FAMILIES = {
    "alce_citation": 2,
    "json_kv": 5,
    "kilt_rag": 24,
    "ms_marco_rerank": 6,
    "ruler_niah": 15,
}
EXPECTED_SOURCES = {
    "alce_root_license_and_readme",
    "asqa_root",
    "hotpotqa_data_terms",
    "kilt_root_license_and_readme",
    "ms_marco_terms",
    "natural_questions_data_page",
    "popqa_root",
    "qampari_root",
    "ruler_root_license",
    "triviaqa_data_page",
}
CONFIG_BINDINGS = {
    "recall": ("configs/recall.yaml", "configs/recall_short.yaml"),
    "rag": ("configs/rag.yaml", "configs/rag_short.yaml"),
    "rerank": ("configs/rerank.yaml", "configs/rerank_short.yaml"),
    "cite": ("configs/cite.yaml", "configs/cite_short.yaml"),
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path_family_counts(paths):
    families = Counter()
    for value in paths:
        path = PurePosixPath(value)
        _require(
            len(path.parts) >= 3 and path.parts[0] == "data",
            f"invalid HELMET archive-local path: {value}",
        )
        families[path.parts[1]] += 1
    return dict(sorted(families.items()))


def validate_audit(audit, inspection, contract, inspection_sha256):
    _require(
        audit.get("format") == "speck_helmet_archive_local_rights_audit"
        and audit.get("format_version") == 1
        and audit.get("status") == "five_families_audited_zero_extraction_qualified",
        "invalid HELMET archive-local rights audit identity",
    )
    _require(
        audit.get("scope", "").startswith("metadata-only operational rights"),
        "HELMET archive-local rights scope changed",
    )
    method = audit.get("method", {})
    _require(
        method.get("payload_content_read") is False
        and method.get("payload_extracted") is False
        and method.get("archive_member_names_used") is True,
        "HELMET archive-local audit crossed its metadata-only boundary",
    )

    archive = audit.get("archive", {})
    inspected_archive = inspection.get("archive", {})
    inspected = inspection.get("inspection", {})
    _require(
        archive.get("inspection_sha256") == inspection_sha256
        and archive.get("sha256") == inspected_archive.get("sha256")
        and archive.get("bytes") == inspected_archive.get("bytes")
        and archive.get("member_identity_sha256") == inspected.get("member_identity_sha256"),
        "HELMET archive-local audit does not match the inventory pin",
    )
    declared_paths = inspected.get("declared_local_paths", ())
    _require(
        archive.get("config_declared_local_paths") == 52
        and inspected.get("declared_local_paths_found") == 52
        and inspected.get("declared_local_paths_missing") == []
        and inspected.get("license_and_metadata_files") == []
        and _path_family_counts(declared_paths) == EXPECTED_PATH_FAMILIES,
        "HELMET archive-local path-family accounting changed",
    )

    _require(
        contract.get("format") == "speck_external_evaluation_suite"
        and contract.get("suite_id") == "helmet"
        and contract.get("upstream", {}).get("revision")
        == audit.get("helmet_source", {}).get("revision"),
        "HELMET source contract does not match the rights audit",
    )
    required_files = {
        entry["path"]: entry["sha256"]
        for entry in contract.get("upstream", {}).get("required_files", ())
    }
    source = audit.get("helmet_source", {})
    _require(
        source.get("license_sha256") == required_files.get("LICENSE")
        and source.get("readme_sha256") == required_files.get("README.md")
        and source.get("data_loader_sha256") == required_files.get("data.py"),
        "HELMET source-file pins changed",
    )
    for category, (full_path, short_path) in CONFIG_BINDINGS.items():
        pins = audit.get("config_pins", {}).get(category, {})
        _require(
            pins.get("full_sha256") == required_files.get(full_path)
            and pins.get("short_sha256") == required_files.get(short_path),
            f"HELMET {category} config pins changed",
        )

    families = audit.get("families", ())
    family_counts = {entry.get("id"): entry.get("paths") for entry in families}
    _require(
        family_counts == EXPECTED_AUDIT_FAMILIES
        and sum(family_counts.values()) == 52
        and all(
            entry.get("exact_payload_derivation_qualified") is False
            and entry.get("attribution_chain_qualified") is False
            and entry.get("extraction_authorized") is False
            for entry in families
        ),
        "HELMET archive-local family disposition changed",
    )
    source_ids = {entry.get("id") for entry in audit.get("authoritative_sources", ())}
    _require(
        source_ids == EXPECTED_SOURCES,
        "HELMET archive-local authority-source inventory changed",
    )

    decision = audit.get("decision", {})
    _require(
        decision.get("families_audited") == 5
        and decision.get("paths_accounted") == 52
        and decision.get("extraction_qualified_families") == []
        and decision.get("extraction_authorized") is False
        and decision.get("evaluation_use_authorized") is False
        and decision.get("helmet_execution_authorized") is False
        and decision.get("archive_remains_unextracted") is True
        and decision.get("contact_authorized") is False,
        "HELMET archive-local fail-closed decision changed",
    )
    return {
        "status": "valid",
        "families": 5,
        "paths": 52,
        "extraction_qualified_families": 0,
        "archive_member_identity_sha256": archive["member_identity_sha256"],
    }


def validate_files(audit_path, inspection_path, contract_path):
    audit_path = Path(audit_path)
    inspection_path = Path(inspection_path)
    contract_path = Path(contract_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return validate_audit(audit, inspection, contract, file_sha256(inspection_path))


def main(argv=None):
    args = arguments(argv)
    report = validate_files(args.audit, args.inspection, args.contract)
    print(
        "HELMET archive-local rights audit: "
        f"{report['status']} ({report['paths']} paths, "
        f"{report['extraction_qualified_families']} extraction-qualified families)"
    )


if __name__ == "__main__":
    main()
