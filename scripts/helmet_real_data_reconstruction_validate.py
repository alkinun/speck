"""Validate HELMET archive-local real-data reconstruction readiness offline."""

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_FAMILY_PATHS = {
    "kilt_rag": 24,
    "ms_marco_rerank": 6,
    "alce_citation": 2,
}
CONFIG_PATHS = {
    "rag": "configs/rag.yaml",
    "rag_short": "configs/rag_short.yaml",
    "rerank": "configs/rerank.yaml",
    "rerank_short": "configs/rerank_short.yaml",
    "cite": "configs/cite.yaml",
    "cite_short": "configs/cite_short.yaml",
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
        readiness.get("format") == "speck_helmet_real_data_reconstruction_readiness"
        and readiness.get("format_version") == 1
        and readiness.get("status")
        == "three_families_32_paths_audited_exact_reconstruction_rights_and_substitution_blocked",
        "invalid HELMET real-data reconstruction identity",
    )
    scope = readiness.get("scope", "")
    _require(
        scope.startswith("metadata-only technical reconstruction decision")
        and all(
            boundary in scope
            for boundary in (
                "no archive payload content was read",
                "no source dataset was acquired",
                "no retriever was executed",
                "no archive was extracted",
                "no model was evaluated",
            )
        ),
        "HELMET real-data reconstruction crossed its scope boundary",
    )

    inputs = readiness.get("inputs", {})
    rights_pin = inputs.get("archive_local_rights_audit", {})
    rights_path = root / rights_pin.get("path", "")
    _require(
        rights_path.is_file()
        and file_sha256(rights_path) == rights_pin.get("sha256")
        and rights_audit.get("status") == "five_families_audited_zero_extraction_qualified",
        "HELMET real-data rights input changed",
    )

    helmet = inputs.get("helmet_source", {})
    required_files = {
        entry["path"]: entry["sha256"]
        for entry in helmet_contract.get("upstream", {}).get("required_files", ())
    }
    _require(
        helmet_contract.get("suite_id") == "helmet"
        and helmet.get("revision") == helmet_contract.get("upstream", {}).get("revision")
        and helmet.get("data_loader_sha256") == required_files.get("data.py"),
        "HELMET real-data source pins changed",
    )
    for key, path in CONFIG_PATHS.items():
        _require(
            helmet.get("configs", {}).get(key) == required_files.get(path),
            f"HELMET real-data {path} pin changed",
        )

    paper = inputs.get("helmet_paper", {})
    _require(
        paper.get("id") == "arXiv:2410.02694v3"
        and paper.get("eprint_tar_sha256")
        == "13f6b2e82f5619a581f05f8a7508a9b7f836887130a0a17f242b5586ffe9dbd9"
        and paper.get("dataset_tex_sha256")
        == "e8a595f04f41302b8b3cfd7ee538ff05ed302ad1b8b64a624de38790f2ea07b2"
        and paper.get("appendix_tex_sha256")
        == "5dcb27037b68251c2fe29b763147be57ce03808f9da469f36e9c6531ebe23b3b",
        "HELMET real-data paper evidence changed",
    )

    sources = inputs.get("source_discovery", {})
    _require(
        set(sources)
        == {
            "kilt",
            "helmet_rag_retriever",
            "popqa",
            "alce",
            "alce_gtr_retriever",
            "ms_marco_terms",
        }
        and sources["kilt"].get("knowledge_source_payload_acquired") is False
        and sources["helmet_rag_retriever"].get("paper_revision_pinned") is False
        and sources["helmet_rag_retriever"].get("observed_head_is_used_identity") is False
        and sources["popqa"].get("payload_acquired") is False
        and sources["alce"].get("released_retrieval_depth") == 100
        and sources["alce"].get("released_data_ref") == "unversioned Hugging Face main"
        and sources["alce_gtr_retriever"].get("alce_revision_pinned") is False
        and sources["alce_gtr_retriever"].get("observed_head_is_used_identity") is False,
        "HELMET real-data source-discovery boundary changed",
    )

    families = readiness.get("families", ())
    family_map = {entry.get("id"): entry for entry in families}
    _require(
        {key: value.get("archive_paths") for key, value in family_map.items()}
        == EXPECTED_FAMILY_PATHS
        and sum(EXPECTED_FAMILY_PATHS.values()) == 32
        and all(
            entry.get("rights_qualified") is False
            and entry.get("exact_semantics_reconstructable") is False
            and entry.get("exact_bytes_reconstructable") is False
            and entry.get("may_substitute_for_official_helmet") is False
            and entry.get("runtime", {}).get(
                "sample_and_demo_selection_deterministic_given_exact_payload"
            )
            is True
            for entry in families
        ),
        "HELMET real-data family disposition changed",
    )
    rag = family_map["kilt_rag"]
    rerank = family_map["ms_marco_rerank"]
    cite = family_map["alce_citation"]
    _require(
        rag.get("test_files") == 20
        and rag.get("demo_files") == 4
        and rag.get("passages_by_length") == [50, 105, 220, 440, 1000]
        and rag.get("depth_variants")
        == {
            "Natural Questions": 6,
            "TriviaQA": 6,
            "HotpotQA": 3,
            "PopQA": 6,
        }
        and len(rag.get("missing_exact_reconstruction_inputs", ())) == 10,
        "HELMET RAG reconstruction evidence changed",
    )
    _require(
        rerank.get("test_files") == 5
        and rerank.get("demo_files") == 1
        and rerank.get("passages_by_length") == [50, 130, 285, 600, 1000]
        and rerank.get("permutations") == 3
        and len(rerank.get("missing_exact_reconstruction_inputs", ())) == 9,
        "HELMET reranking reconstruction evidence changed",
    )
    _require(
        cite.get("test_files") == 2
        and cite.get("documents_by_length") == [30, 75, 165, 345, 700]
        and cite.get("archive_retrieval_depth") == 2000
        and cite.get("source_mismatch", {}).get("alce_released_gtr_depth") == 100
        and cite.get("source_mismatch", {}).get("helmet_archive_gtr_depth") == 2000
        and cite.get("source_mismatch", {}).get("helmet_top2000_generation_code_released") is False
        and len(cite.get("missing_exact_reconstruction_inputs", ())) == 7,
        "HELMET citation reconstruction evidence changed",
    )

    decision = readiness.get("decision", {})
    _require(
        decision.get("families_audited") == 3
        and decision.get("archive_paths_accounted") == 32
        and decision.get("runtime_deterministic_given_exact_payload") is True
        and decision.get("exact_payload_construction_released") is False
        and decision.get("rights_qualified") is False
        and decision.get("exact_official_reconstruction_qualified") is False
        and decision.get("archive_extraction_authorized") is False
        and decision.get("source_payload_acquisition_authorized") is False
        and decision.get("retriever_execution_authorized") is False
        and decision.get("candidate_execution_authorized") is False
        and decision.get("evaluation_manifest_changed") is False
        and decision.get("clean_room_substitute_authorized_as_helmet") is False,
        "HELMET real-data fail-closed decision changed",
    )
    return {
        "status": "valid",
        "families": 3,
        "archive_paths": 32,
        "exact_official_reconstruction_qualified": False,
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
        "HELMET real-data reconstruction: "
        f"{report['status']} ({report['archive_paths']} paths, "
        f"official={str(report['exact_official_reconstruction_qualified']).lower()})"
    )


if __name__ == "__main__":
    main()
