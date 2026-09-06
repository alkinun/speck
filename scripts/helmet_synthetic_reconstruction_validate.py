"""Validate HELMET synthetic-recall reconstruction readiness without payload access."""

import argparse
import hashlib
import json
from pathlib import Path

LENGTHS = (8192, 16384, 32768, 65536, 131072)
TASKS = ("niah_multikey_2", "niah_multikey_3", "niah_multivalue")
CONFIG_PATHS = {
    "recall_config_sha256": "configs/recall.yaml",
    "recall_short_config_sha256": "configs/recall_short.yaml",
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--rights-audit", type=Path, required=True)
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


def validate_readiness(readiness, rights_audit, helmet_contract, root):
    _require(
        readiness.get("format") == "speck_helmet_synthetic_reconstruction_readiness"
        and readiness.get("format_version") == 1
        and readiness.get("status")
        == "source_paths_audited_exact_reconstruction_and_helmet_substitution_blocked",
        "invalid HELMET synthetic reconstruction identity",
    )
    _require(
        readiness.get("scope", "").startswith("metadata-only reconstruction-readiness")
        and "no archive payload read" in readiness.get("scope", "")
        and "no dataset generation" in readiness.get("scope", "")
        and "no model execution" in readiness.get("scope", ""),
        "HELMET synthetic reconstruction crossed its scope boundary",
    )

    inputs = readiness.get("inputs", {})
    rights_pin = inputs.get("archive_local_rights_audit", {})
    _require(
        rights_audit.get("status") == "five_families_audited_zero_extraction_qualified"
        and rights_pin.get("sha256") == file_sha256(root / rights_pin.get("path", ""))
        and rights_pin.get("sha256")
        == "c7fd9daacafcf726809d6cead6649be4b21527129135c25b13cd500f65a9acf2",
        "HELMET archive-local rights input changed",
    )

    source = inputs.get("helmet_source", {})
    required_files = {
        entry["path"]: entry["sha256"]
        for entry in helmet_contract.get("upstream", {}).get("required_files", ())
    }
    _require(
        helmet_contract.get("suite_id") == "helmet"
        and source.get("revision") == helmet_contract.get("upstream", {}).get("revision")
        and source.get("readme_sha256") == required_files.get("README.md")
        and source.get("data_loader_sha256") == required_files.get("data.py"),
        "HELMET synthetic reconstruction source pins changed",
    )
    for key, path in CONFIG_PATHS.items():
        _require(
            source.get(key) == required_files.get(path),
            f"HELMET synthetic reconstruction {path} pin changed",
        )

    paper = inputs.get("helmet_paper", {})
    _require(
        paper.get("id") == "arXiv:2410.02694v3"
        and len(paper.get("authoritative_statements", ())) == 5
        and all(
            len(paper.get(key, "")) == 64
            for key in ("eprint_tar_sha256", "dataset_tex_sha256", "appendix_tex_sha256")
        ),
        "HELMET paper-source evidence changed",
    )

    ruler_source = inputs.get("ruler_generator", {})
    _require(
        ruler_source.get("revision") == "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
        and ruler_source.get("license") == "Apache-2.0",
        "HELMET RULER source evidence changed",
    )
    existing = readiness.get("existing_ruler_evidence", {})
    qualifications = existing.get("qualifications", ())
    _require(
        tuple(entry.get("length") for entry in qualifications) == LENGTHS
        and existing.get("task_length_cells") == 15
        and existing.get("cases") == 1500
        and existing.get("samples_per_task_length") == 100,
        "HELMET RULER qualification inventory changed",
    )
    for entry in qualifications:
        report_path = root / entry.get("report", "")
        _require(
            report_path.is_file() and file_sha256(report_path) == entry.get("report_sha256"),
            f"HELMET RULER report pin changed at {entry.get('length')}",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cases = {case.get("task"): case.get("sha256") for case in report.get("cases", ())}
        _require(
            report.get("status") == "qualified_offline_deterministic_case_stream"
            and report.get("generator", {}).get("revision") == ruler_source.get("revision")
            and report.get("tokenizer", {}).get("identity_sha256")
            == existing.get("tokenizer_identity_sha256")
            and report.get("samples_per_task") == 100
            and entry.get("task_sha256") == {task: cases.get(task) for task in TASKS},
            f"HELMET RULER task evidence changed at {entry.get('length')}",
        )
    ruler_decision = existing.get("decision", {})
    _require(
        ruler_decision.get("source_and_current_case_generation_qualified") is True
        and ruler_decision.get("helmet_generation_tokenizer_matches") is False
        and ruler_decision.get("helmet_generator_revision_proven") is False
        and ruler_decision.get("helmet_archive_byte_parity_proven") is False
        and ruler_decision.get("may_be_reported_as_official_helmet") is False,
        "HELMET RULER non-substitution decision changed",
    )

    lost = inputs.get("lost_in_the_middle", {})
    json_kv = readiness.get("json_kv_analysis", {})
    lost_generator = json_kv.get("lost_in_the_middle_generator", {})
    json_decision = json_kv.get("decision", {})
    _require(
        lost.get("revision") == "29b8a6d042ce29abccee3db1a73171a107d7e6af"
        and lost.get("license") == "MIT"
        and lost_generator.get("released_key_counts") == [75, 140, 300]
        and lost_generator.get("released_examples_per_key_count") == 500
        and lost_generator.get("queries_per_dictionary") == 1
        and lost_generator.get("uses_seeded_uuid_generation") is False
        and lost_generator.get("uses_seeded_gold_selection") is False
        and len(json_kv.get("missing_exact_specification", ())) == 8
        and json_decision.get("concept_reconstructable") is True
        and json_decision.get("exact_helmet_semantics_reconstructable") is False
        and json_decision.get("exact_helmet_bytes_reconstructable") is False
        and json_decision.get("may_be_reported_as_official_helmet") is False,
        "HELMET JSON-KV reconstruction decision changed",
    )

    decision = readiness.get("decision", {})
    _require(
        decision.get("archive_extraction_authorized") is False
        and decision.get("official_helmet_recall_reconstruction_qualified") is False
        and decision.get("existing_ruler_cases_substitute_for_helmet") is False
        and decision.get("json_kv_implementation_authorized_as_helmet") is False
        and decision.get("candidate_execution_authorized") is False
        and decision.get("evaluation_manifest_changed") is False,
        "HELMET synthetic reconstruction fail-closed decision changed",
    )
    return {
        "status": "valid",
        "ruler_task_length_cells": 15,
        "ruler_cases": 1500,
        "official_helmet_reconstruction_qualified": False,
        "evaluation_manifest_changed": False,
    }


def validate_files(readiness_path, rights_audit_path, helmet_contract_path, root=None):
    readiness_path = Path(readiness_path).resolve()
    root = Path(root).resolve() if root else readiness_path.parents[2]
    return validate_readiness(
        json.loads(readiness_path.read_text(encoding="utf-8")),
        json.loads(Path(rights_audit_path).read_text(encoding="utf-8")),
        json.loads(Path(helmet_contract_path).read_text(encoding="utf-8")),
        root,
    )


def main(argv=None):
    args = arguments(argv)
    report = validate_files(args.readiness, args.rights_audit, args.helmet_contract)
    print(
        "HELMET synthetic reconstruction: "
        f"{report['status']} ({report['ruler_task_length_cells']} RULER cells, "
        f"official={str(report['official_helmet_reconstruction_qualified']).lower()})"
    )


if __name__ == "__main__":
    main()
