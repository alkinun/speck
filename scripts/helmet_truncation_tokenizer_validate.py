"""Validate the fail-closed HELMET truncation-tokenizer readiness decision."""

import argparse
import hashlib
import json
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--runtime-protocol", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--synthetic-readiness", type=Path, required=True)
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


def validate_readiness(readiness, runtime_protocol, runtime_audit, synthetic_readiness, root):
    _require(
        readiness.get("format") == "speck_helmet_truncation_tokenizer_readiness"
        and readiness.get("format_version") == 1
        and readiness.get("status") == "exact_identity_acceptance_and_replacement_oracle_blocked",
        "invalid HELMET truncation-tokenizer identity",
    )
    scope = readiness.get("scope", "")
    _require(
        scope.startswith("metadata-only decision")
        and all(
            boundary in scope
            for boundary in (
                "no license was accepted",
                "no gated access was attempted",
                "no tokenizer payload was acquired",
                "no source dataset was accessed",
                "no prompt was materialized",
                "no model was executed",
            )
        ),
        "HELMET truncation-tokenizer scope changed",
    )

    inputs = readiness.get("inputs", {})
    for key, loaded in (
        ("runtime_protocol", runtime_protocol),
        ("runtime_audit", runtime_audit),
        ("synthetic_reconstruction_audit", synthetic_readiness),
    ):
        pin = inputs.get(key, {})
        path = root / pin.get("path", "")
        _require(
            path.is_file() and file_sha256(path) == pin.get("sha256"),
            f"HELMET truncation-tokenizer {key} input changed",
        )
        _require(loaded == json.loads(path.read_text(encoding="utf-8")), f"{key} load mismatch")

    dependency = runtime_audit.get("gated_dependencies", ())
    _require(
        len(dependency) == 1
        and dependency[0].get("id") == "llama2_tokenizer"
        and dependency[0].get("status") == "authorized_acceptance_and_exact_identity_blocked",
        "HELMET runtime tokenizer dependency changed",
    )
    source = inputs.get("helmet_source", {})
    _require(
        source.get("revision") == "af609c4d51b97fc35012099380aa889da961c42d"
        and source.get("data_loader_sha256")
        == "559977d8f97357c77f2bc1a554e7d02dffebce99f450ba034864ca3e07c09348"
        and source.get("transformers_version") == "5.1.0",
        "HELMET truncation-tokenizer source pin changed",
    )
    llama = inputs.get("llama2", {})
    _require(
        llama.get("model_repository_ref_in_helmet") == "unversioned"
        and llama.get("anonymous_identity_resolution") == "rejected_requires_credentials"
        and llama.get("local_tokenizer_payload_found") is False
        and llama.get("license_acceptance_recorded_for_entity") is False
        and llama.get("license_scope_reviewed_for_architecture_research") is False
        and llama.get("official_license_sha256")
        == "e30d9339da533205f88140009e32d62c1d73911b8a0bef0d0b252f549ed8e9c4",
        "HELMET Llama-2 authority boundary changed",
    )

    code = readiness.get("code_analysis", {})
    _require(
        code.get("auto_tokenizer_literal") == "meta-llama/Llama-2-7b-hf"
        and code.get("literal_call_sites") == 3
        and code.get("revision_argument_present") is False
        and code.get("suffix") == " ... [the rest of the text is omitted]"
        and {entry.get("id") for entry in code.get("operations", ())}
        == {"minimum_length_filter", "narrativeqa_preselection", "context_truncation"}
        and len(code.get("implementation_sensitive_details", ())) == 8,
        "HELMET tokenizer code-semantics evidence changed",
    )
    affected = readiness.get("affected_surface", {})
    _require(
        affected.get("runtime_config_entries") == 25
        and affected.get("runtime_categories") == {"longqa": 15, "summ": 10}
        and sum(affected.get("runtime_datasets", {}).values()) == 25
        and affected.get("separate_generation_surface", {}).get("archive_local_task_length_cells")
        == 15
        and affected.get("separate_generation_surface", {}).get("same_exact_identity_required")
        is True,
        "HELMET tokenizer affected surface changed",
    )

    replacement = readiness.get("replacement_gate", {})
    _require(
        replacement.get("candidate_replacements_evaluated") == []
        and len(replacement.get("minimum_future_protocol", ())) == 7
        and replacement.get("fixture_only_equivalence_sufficient") is False
        and replacement.get("real_data_equivalence_required") is True
        and replacement.get("exact_oracle_required") is True,
        "HELMET tokenizer replacement gate changed",
    )
    license_boundary = readiness.get("license_boundary", {})
    _require(
        license_boundary.get("tokenizer_access_requires_acceptance") is True
        and license_boundary.get("organizational_authority_to_accept_recorded") is False
        and license_boundary.get("architecture_research_scope_decided") is False
        and license_boundary.get("legal_conclusion_made") is False,
        "HELMET tokenizer license boundary changed",
    )
    decision = readiness.get("decision", {})
    false_fields = (
        "exact_tokenizer_identity_qualified",
        "license_acceptance_authorized",
        "tokenizer_payload_acquisition_authorized",
        "anonymous_access_retry_authorized",
        "replacement_oracle_available",
        "replacement_protocol_ready_to_execute",
        "fixture_only_replacement_authorized",
        "runtime_prompt_materialization_authorized",
        "ruler_regeneration_authorized",
        "candidate_execution_authorized",
        "evaluation_manifest_changed",
    )
    _require(
        all(decision.get(field) is False for field in false_fields),
        "HELMET tokenizer fail-closed decision changed",
    )
    return {
        "status": "valid",
        "runtime_entries": 25,
        "ruler_generation_cells": 15,
        "replacement_oracle_available": False,
        "evaluation_manifest_changed": False,
    }


def validate_files(
    readiness_path,
    runtime_protocol_path,
    runtime_audit_path,
    synthetic_readiness_path,
    root=None,
):
    readiness_path = Path(readiness_path).resolve()
    root = Path(root).resolve() if root else readiness_path.parents[2]
    paths = (
        readiness_path,
        Path(runtime_protocol_path),
        Path(runtime_audit_path),
        Path(synthetic_readiness_path),
    )
    values = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return validate_readiness(*values, root)


def main(argv=None):
    args = arguments(argv)
    report = validate_files(
        args.readiness,
        args.runtime_protocol,
        args.runtime_audit,
        args.synthetic_readiness,
    )
    print(
        "HELMET truncation tokenizer: "
        f"{report['status']} ({report['runtime_entries']} runtime entries, "
        f"oracle={str(report['replacement_oracle_available']).lower()})"
    )


if __name__ == "__main__":
    main()
