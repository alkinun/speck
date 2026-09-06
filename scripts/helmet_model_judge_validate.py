"""Validate the fail-closed HELMET model-judge readiness audit offline."""

import argparse
import hashlib
import json
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--runtime-protocol", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--helmet-contract", type=Path, required=True)
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


def validate_readiness(readiness, runtime_protocol, runtime_audit, helmet_contract, root):
    _require(
        readiness.get("format") == "speck_helmet_model_judge_readiness"
        and readiness.get("format_version") == 1
        and readiness.get("status")
        == "exact_judge_audited_reproducibility_denominator_data_handling_and_replacement_blocked",
        "invalid HELMET model-judge readiness identity",
    )
    scope = readiness.get("scope", "")
    _require(
        scope.startswith("metadata-only audit")
        and all(
            boundary in scope
            for boundary in (
                "no API credential was inspected",
                "no API request was sent",
                "no batch or file was created",
                "no evaluation data was read",
                "no judge output was generated",
                "no model score was computed",
            )
        ),
        "HELMET model-judge scope changed",
    )

    inputs = readiness.get("inputs", {})
    runtime_pin = inputs.get("runtime_audit", {})
    runtime_path = root / runtime_pin.get("path", "")
    _require(
        runtime_path.is_file()
        and file_sha256(runtime_path) == runtime_pin.get("sha256")
        and runtime_audit == json.loads(runtime_path.read_text(encoding="utf-8")),
        "HELMET model-judge runtime input changed",
    )
    protocol_pin = inputs.get("runtime_protocol", {})
    protocol_path = root / protocol_pin.get("path", "")
    _require(
        protocol_path.is_file()
        and file_sha256(protocol_path) == protocol_pin.get("sha256")
        and runtime_protocol == json.loads(protocol_path.read_text(encoding="utf-8")),
        "HELMET model-judge runtime protocol changed",
    )
    source = inputs.get("helmet_source", {})
    required_files = {
        entry["path"]: entry["sha256"]
        for entry in helmet_contract.get("upstream", {}).get("required_files", ())
    }
    _require(
        source.get("revision") == helmet_contract.get("upstream", {}).get("revision")
        and source.get("model_utils_sha256") == required_files.get("model_utils.py")
        and source.get("requirements_sha256") == required_files.get("requirements.txt"),
        "HELMET model-judge source pins changed",
    )
    audit_files = {
        entry["path"]: entry["sha256"]
        for entry in runtime_protocol.get("suite", {}).get("code_files", ())
    }
    _require(
        source.get("longqa_judge_sha256") == audit_files["scripts/eval_gpt4_longqa.py"]
        and source.get("summarization_judge_sha256") == audit_files["scripts/eval_gpt4_summ.py"],
        "HELMET model-judge script pins changed",
    )

    official = inputs.get("official_openai_documentation", {})
    _require(
        official.get("accessed_on") == "2026-09-06"
        and official.get("model_page", {}).get("observation")
        == "gpt-4o-2024-05-13 is listed as a deprecated snapshot"
        and len(official.get("chat_completions_reference", {}).get("observations", ())) == 3
        and len(official.get("data_controls", {}).get("observations", ())) == 4
        and all(
            len(official[key].get("markdown_sha256", "")) == 64
            for key in ("model_page", "chat_completions_reference", "data_controls")
        ),
        "HELMET official OpenAI documentation evidence changed",
    )

    request = readiness.get("request_contract", {})
    _require(
        request.get("endpoint") == "/v1/chat/completions through the Batch API"
        and request.get("model") == "gpt-4o-2024-05-13"
        and request.get("model_status_on_audit_date") == "deprecated"
        and request.get("organization_availability_verified") is False
        and request.get("temperature") == 0.1
        and request.get("top_p") == 0.9
        and request.get("seed") == 42
        and request.get("seed_source") == "OpenAIModel default"
        and request.get("generation_max_length") == {"longqa": 2048, "summarization": 4096}
        and request.get("openai_sdk_version_pinned") is False
        and request.get("requests_per_example") == {"narrativeqa": 1, "summarization": 3}
        and request.get("system_fingerprint_returned_by_adapter") is True
        and request.get("system_fingerprint_preserved_in_scored_json") is False
        and request.get("system_fingerprint_consistency_gate") is False
        and request.get("repeatability_trials") == 1,
        "HELMET model-judge request contract changed",
    )

    surfaces = {entry.get("id"): entry for entry in readiness.get("judge_surfaces", ())}
    _require(
        set(surfaces) == {"narrativeqa", "summarization"}
        and surfaces["narrativeqa"].get("runtime_entries") == 5
        and surfaces["summarization"].get("runtime_entries") == 10
        and all(entry.get("schema_or_range_validation") is False for entry in surfaces.values())
        and all(
            entry.get("system_fingerprint_preserved_in_scored_json") is False
            for entry in surfaces.values()
        )
        and all(entry.get("fixed_denominator_required") is False for entry in surfaces.values()),
        "HELMET model-judge scoring surface changed",
    )
    data_path = readiness.get("batch_data_path", {})
    _require(
        data_path.get("full_dataset_text_transmitted") is True
        and data_path.get("candidate_outputs_transmitted") is True
        and data_path.get("batch_input_file_uploaded") is True
        and data_path.get("batch_output_file_created") is True
        and data_path.get("remote_file_or_batch_deleted_by_helmet") is False
        and data_path.get("data_rights_qualified_for_transmission") is False
        and data_path.get("organization_retention_controls_verified") is False
        and data_path.get("data_residency_verified") is False
        and data_path.get("cost_budget_frozen") is False,
        "HELMET model-judge data-handling boundary changed",
    )
    _require(
        len(readiness.get("reproducibility_failures", ())) == 10,
        "HELMET model-judge reproducibility inventory changed",
    )
    replacement = readiness.get("replacement_gate", {})
    _require(
        replacement.get("replacement_selected") is False
        and len(replacement.get("minimum_future_protocol", ())) == 8
        and replacement.get("newer_model_substitution_authorized") is False
        and replacement.get("local_model_substitution_authorized") is False
        and replacement.get("successful_only_denominator_authorized") is False,
        "HELMET model-judge replacement gate changed",
    )
    decision = readiness.get("decision", {})
    _require(
        decision.get("historical_judge_source_identity_qualified") is True
        and decision.get("earlier_seed_record_corrected") is True
        and decision.get("exact_model_currently_deprecated") is True
        and all(
            decision.get(field) is False
            for field in (
                "current_organization_availability_qualified",
                "reproducibility_qualified",
                "fixed_sample_denominator_qualified",
                "data_transmission_authorized",
                "data_retention_and_deletion_qualified",
                "cost_qualified",
                "replacement_qualified",
                "api_execution_authorized",
                "candidate_scoring_authorized",
                "evaluation_manifest_changed",
            )
        ),
        "HELMET model-judge fail-closed decision changed",
    )
    return {
        "status": "valid",
        "runtime_entries": 15,
        "seed": 42,
        "repeatability_trials": 1,
        "candidate_scoring_authorized": False,
        "evaluation_manifest_changed": False,
    }


def validate_files(
    readiness_path,
    runtime_protocol_path,
    runtime_audit_path,
    helmet_contract_path,
    root=None,
):
    readiness_path = Path(readiness_path).resolve()
    root = Path(root).resolve() if root else readiness_path.parents[2]
    return validate_readiness(
        json.loads(readiness_path.read_text(encoding="utf-8")),
        json.loads(Path(runtime_protocol_path).read_text(encoding="utf-8")),
        json.loads(Path(runtime_audit_path).read_text(encoding="utf-8")),
        json.loads(Path(helmet_contract_path).read_text(encoding="utf-8")),
        root,
    )


def main(argv=None):
    args = arguments(argv)
    report = validate_files(
        args.readiness,
        args.runtime_protocol,
        args.runtime_audit,
        args.helmet_contract,
    )
    print(
        "HELMET model judge: "
        f"{report['status']} ({report['runtime_entries']} entries, "
        f"scoring={str(report['candidate_scoring_authorized']).lower()})"
    )


if __name__ == "__main__":
    main()
