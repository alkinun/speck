"""Validate the preregistered Speck Paper 1 research program."""

import hashlib
import json
import math
from pathlib import Path

PROGRAM_FILES = (
    "README.md",
    "claims.json",
    "baseline_matrix.json",
    "baseline_analysis.json",
    "baseline_collection_v2.json",
    "baseline_automation_v1.json",
    "proxy_disposition_v1.json",
    "sequence_cache_representation_v1.json",
    "hca_readiness_v1.json",
    "csa_readiness_v1.json",
    "raw_local_readiness_v1.json",
    "ratio_placement_readiness_v1.json",
    "attnres_readiness_v1.json",
    "stable_latentmoe_readiness_v1.json",
    "interaction_readiness_v1.json",
    "scaling_readiness_v1.json",
    "systems_cost_readiness_v1.json",
    "novelty_landscape_v1.json",
    "novelty_code_availability_v1.json",
    "adakv_code_audit_v1.json",
    "adaptive_cache_budget_v1.json",
    "adaptive_cache_gqa_v1.json",
    "proxy_launch_v1.json",
    "contamination_v1.json",
    "contamination_disposition_v1.json",
    "experiment_program.json",
    "paper_outline.md",
    "reference_audit.md",
    "reporting_checklist.md",
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load paper program file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"paper program file must contain an object: {path}")
    return value


def _require(value, keys, context):
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")


def _unique_ids(values, context):
    identifiers = [value.get("id") for value in values]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError(f"every {context} requires an id")
    duplicates = sorted(
        {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
    )
    if duplicates:
        raise ValueError(f"duplicate {context} ids: {', '.join(duplicates)}")
    return set(identifiers)


def _identity_without_status_and_result(value):
    payload = {key: value[key] for key in value if key not in {"status", "result"}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_evaluation_evidence(evidence, repository_root):
    _require(
        evidence,
        {
            "status",
            "contamination_protocol",
            "contamination_protocol_sha256",
            "contamination_result",
            "contamination_result_sha256",
            "contamination_result_status",
            "contamination_disposition_protocol",
            "contamination_disposition_protocol_sha256",
            "contamination_disposition",
            "contamination_disposition_sha256",
            "contamination_disposition_status",
            "active_evaluation_manifest",
            "active_evaluation_manifest_sha256",
            "active_evaluation_manifest_id",
            "helmet_runtime_dependency_protocol",
            "helmet_runtime_dependency_protocol_sha256",
            "helmet_runtime_dependency_audit",
            "helmet_runtime_dependency_audit_sha256",
            "helmet_runtime_dependency_status",
            "helmet_materializer_preflight_protocol",
            "helmet_materializer_preflight_protocol_sha256",
            "helmet_materializer_preflight",
            "helmet_materializer_preflight_sha256",
            "helmet_materializer_preflight_status",
            "helmet_clinc_source_protocol",
            "helmet_clinc_source_protocol_sha256",
            "helmet_clinc_source_qualification",
            "helmet_clinc_source_qualification_sha256",
            "helmet_clinc_source_status",
            "helmet_trec_rights_protocol",
            "helmet_trec_rights_protocol_sha256",
            "helmet_trec_rights_decision",
            "helmet_trec_rights_decision_sha256",
            "helmet_trec_rights_status",
            "helmet_multilexsum_decision_protocol",
            "helmet_multilexsum_decision_protocol_sha256",
            "helmet_multilexsum_decision",
            "helmet_multilexsum_decision_sha256",
            "helmet_multilexsum_status",
            "helmet_narrativeqa_decision_protocol",
            "helmet_narrativeqa_decision_protocol_sha256",
            "helmet_narrativeqa_decision",
            "helmet_narrativeqa_decision_sha256",
            "helmet_narrativeqa_status",
            "helmet_infinitebench_decision_protocol",
            "helmet_infinitebench_decision_protocol_sha256",
            "helmet_infinitebench_decision",
            "helmet_infinitebench_decision_sha256",
            "helmet_infinitebench_status",
            "helmet_seeded_demo_repair_protocol",
            "helmet_seeded_demo_repair_protocol_sha256",
            "helmet_seeded_demo_repair_qualification",
            "helmet_seeded_demo_repair_qualification_sha256",
            "helmet_seeded_demo_repair_status",
            "ruler_v1_decision",
            "ruler_v2_decision",
        },
        "paper evaluation evidence",
    )
    if (
        evidence["status"]
        != "ruler_v1_failed_v2_frozen_helmet_sources_dispositioned_seeded_demo_repair_qualified_not_activated"
    ):
        raise ValueError("paper evaluation evidence must preserve v1, RULER v2, and HELMET")
    for path_key, hash_key in (
        ("contamination_protocol", "contamination_protocol_sha256"),
        ("contamination_result", "contamination_result_sha256"),
        (
            "contamination_disposition_protocol",
            "contamination_disposition_protocol_sha256",
        ),
        ("contamination_disposition", "contamination_disposition_sha256"),
        ("active_evaluation_manifest", "active_evaluation_manifest_sha256"),
        (
            "helmet_runtime_dependency_protocol",
            "helmet_runtime_dependency_protocol_sha256",
        ),
        ("helmet_runtime_dependency_audit", "helmet_runtime_dependency_audit_sha256"),
        (
            "helmet_materializer_preflight_protocol",
            "helmet_materializer_preflight_protocol_sha256",
        ),
        ("helmet_materializer_preflight", "helmet_materializer_preflight_sha256"),
        ("helmet_clinc_source_protocol", "helmet_clinc_source_protocol_sha256"),
        (
            "helmet_clinc_source_qualification",
            "helmet_clinc_source_qualification_sha256",
        ),
        ("helmet_trec_rights_protocol", "helmet_trec_rights_protocol_sha256"),
        ("helmet_trec_rights_decision", "helmet_trec_rights_decision_sha256"),
        (
            "helmet_multilexsum_decision_protocol",
            "helmet_multilexsum_decision_protocol_sha256",
        ),
        ("helmet_multilexsum_decision", "helmet_multilexsum_decision_sha256"),
        (
            "helmet_narrativeqa_decision_protocol",
            "helmet_narrativeqa_decision_protocol_sha256",
        ),
        ("helmet_narrativeqa_decision", "helmet_narrativeqa_decision_sha256"),
        (
            "helmet_infinitebench_decision_protocol",
            "helmet_infinitebench_decision_protocol_sha256",
        ),
        ("helmet_infinitebench_decision", "helmet_infinitebench_decision_sha256"),
        (
            "helmet_seeded_demo_repair_protocol",
            "helmet_seeded_demo_repair_protocol_sha256",
        ),
        (
            "helmet_seeded_demo_repair_qualification",
            "helmet_seeded_demo_repair_qualification_sha256",
        ),
    ):
        path = repository_root / evidence[path_key]
        if not path.is_file() or _file_sha256(path) != evidence[hash_key]:
            raise ValueError(f"paper evaluation {path_key} does not match its pin")

    protocol = _load_json(repository_root / evidence["contamination_protocol"])
    result = _load_json(repository_root / evidence["contamination_result"])
    disposition_protocol = _load_json(
        repository_root / evidence["contamination_disposition_protocol"]
    )
    disposition = _load_json(repository_root / evidence["contamination_disposition"])
    active_manifest = _load_json(repository_root / evidence["active_evaluation_manifest"])
    helmet_runtime_protocol = _load_json(
        repository_root / evidence["helmet_runtime_dependency_protocol"]
    )
    helmet_runtime_audit = _load_json(
        repository_root / evidence["helmet_runtime_dependency_audit"]
    )
    helmet_materializer_protocol = _load_json(
        repository_root / evidence["helmet_materializer_preflight_protocol"]
    )
    helmet_materializer = _load_json(
        repository_root / evidence["helmet_materializer_preflight"]
    )
    helmet_clinc_protocol = _load_json(
        repository_root / evidence["helmet_clinc_source_protocol"]
    )
    helmet_clinc = _load_json(repository_root / evidence["helmet_clinc_source_qualification"])
    helmet_trec_protocol = _load_json(repository_root / evidence["helmet_trec_rights_protocol"])
    helmet_trec = _load_json(repository_root / evidence["helmet_trec_rights_decision"])
    helmet_multilexsum_protocol = _load_json(
        repository_root / evidence["helmet_multilexsum_decision_protocol"]
    )
    helmet_multilexsum = _load_json(
        repository_root / evidence["helmet_multilexsum_decision"]
    )
    helmet_narrativeqa_protocol = _load_json(
        repository_root / evidence["helmet_narrativeqa_decision_protocol"]
    )
    helmet_narrativeqa = _load_json(
        repository_root / evidence["helmet_narrativeqa_decision"]
    )
    helmet_infinitebench_protocol = _load_json(
        repository_root / evidence["helmet_infinitebench_decision_protocol"]
    )
    helmet_infinitebench = _load_json(
        repository_root / evidence["helmet_infinitebench_decision"]
    )
    helmet_seeded_protocol = _load_json(
        repository_root / evidence["helmet_seeded_demo_repair_protocol"]
    )
    helmet_seeded = _load_json(
        repository_root / evidence["helmet_seeded_demo_repair_qualification"]
    )
    expected_unaffected = {
        "cwe",
        "fwe",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multiquery",
        "niah_multivalue",
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "vt",
    }
    if (
        protocol.get("format") != "speck_contamination_protocol"
        or protocol.get("status") != "executed_failed_critical_overlap_detected"
        or protocol.get("result", {}).get("path") != evidence["contamination_result"]
        or protocol.get("result", {}).get("sha256") != evidence["contamination_result_sha256"]
        or result.get("format") != "speck_contamination_audit"
        or result.get("status") != evidence["contamination_result_status"]
        or result.get("protocol", {}).get("spec_identity_sha256")
        != _identity_without_status_and_result(protocol)
        or result.get("training", {}).get("total_tokens_scanned") != 393_216_000
        or result.get("evaluation", {}).get("total_cases") != 7_800
        or result.get("evaluation", {}).get("native_transformers_parity_samples") != 78
        or result.get("decisions")
        != {"full_prompt": True, "answer_anchored": False, "context": "descriptive_only"}
        or result.get("matches", {}).get("unique_patterns_by_kind", {}).get(
            "answer_anchored"
        )
        != 28
        or result.get("matches", {}).get("unique_patterns_by_kind", {}).get("context") != 42
        or result.get("matches", {}).get("unique_patterns_by_kind", {}).get(
            "full_prompt", 0
        )
        != 0
    ):
        raise ValueError("paper contamination audit evidence is invalid")
    if (
        disposition_protocol.get("format")
        != "speck_contamination_disposition_protocol"
        or disposition_protocol.get("status") != "executed_failed"
        or disposition_protocol.get("audit", {}).get("sha256")
        != evidence["contamination_result_sha256"]
        or disposition_protocol.get("result", {}).get("path")
        != evidence["contamination_disposition"]
        or disposition_protocol.get("result", {}).get("sha256")
        != evidence["contamination_disposition_sha256"]
        or disposition.get("format") != "speck_contamination_disposition"
        or disposition.get("status") != evidence["contamination_disposition_status"]
        or disposition.get("protocol", {}).get("spec_identity_sha256")
        != _identity_without_status_and_result(disposition_protocol)
        or disposition.get("audit", {}).get("sha256")
        != evidence["contamination_result_sha256"]
        or disposition.get("derived", {}).get("all_matched_references_reconstructed") is not True
        or set(disposition.get("derived", {}).get("critical_quarantine_tasks", ()))
        != {"qa_1", "qa_2"}
        or set(disposition.get("derived", {}).get("context_overlap_tasks", ()))
        != {"qa_1", "qa_2"}
        or set(disposition.get("derived", {}).get("no_detected_critical_match_tasks", ()))
        != expected_unaffected
        or len(disposition.get("derived", {}).get("critical_affected_cells", ())) != 12
        or disposition.get("decision", {}).get("ruler_v1") != "failed"
        or disposition.get("decision", {}).get("threshold_changed") is not False
        or disposition.get("decision", {}).get("candidate_execution_authorized") is not False
        or evidence["ruler_v1_decision"] != "failed_critical_overlap_preserved"
    ):
        raise ValueError("paper contamination disposition evidence is invalid")
    ruler = next(
        (suite for suite in active_manifest.get("external_suites", ()) if suite.get("id") == "ruler"),
        {},
    )
    if (
        active_manifest.get("manifest_id") != evidence["active_evaluation_manifest_id"]
        or active_manifest.get("manifest_id") != "architecture-evaluation-v2"
        or active_manifest.get("supersedes", {}).get("sha256")
        != "b5439be0ffb37c8b8b46aec9791068b8e145dd5292419e90a335a447969b21ab"
        or set(ruler.get("primary_tasks", ())) != expected_unaffected
        or set(ruler.get("quarantined_tasks", ())) != {"qa_1", "qa_2"}
        or ruler.get("primary_cases") != 6_600
        or ruler.get("source_document_qa_guardrail", {}).get("suite") != "helmet"
        or evidence["ruler_v2_decision"]
        != "eleven_synthetic_tasks_primary_qa_quarantined_helmet_qa_guardrail_pending"
    ):
        raise ValueError("paper RULER v2 evaluation manifest is invalid")
    if (
        helmet_runtime_protocol.get("format")
        != "speck_helmet_runtime_dependency_protocol"
        or helmet_runtime_protocol.get("status") != "executed_blocked"
        or helmet_runtime_protocol.get("result", {}).get("sha256")
        != evidence["helmet_runtime_dependency_audit_sha256"]
        or helmet_runtime_audit.get("format") != "speck_helmet_runtime_dependency_audit"
        or helmet_runtime_audit.get("status")
        != evidence["helmet_runtime_dependency_status"]
        or helmet_runtime_audit.get("inventory", {}).get("by_mode")
        != {"archive_local": 55, "runtime_loaded": 50}
        or helmet_runtime_audit.get("decision", {}).get("execution_authorized") is not False
    ):
        raise ValueError("paper HELMET runtime dependency evidence is invalid")
    if (
        helmet_materializer_protocol.get("format")
        != "speck_helmet_materializer_preflight_protocol"
        or helmet_materializer_protocol.get("status") != "executed_qualified"
        or helmet_materializer_protocol.get("result", {}).get("sha256")
        != evidence["helmet_materializer_preflight_sha256"]
        or helmet_materializer.get("format") != "speck_helmet_materializer_preflight"
        or helmet_materializer.get("status")
        != evidence["helmet_materializer_preflight_status"]
        or set(
            helmet_materializer.get("decision", {}).get("strategy_qualified_for", ())
        )
        != {"banking77", "nlu_evaluation_data"}
        or helmet_materializer.get("decision", {}).get("helmet_execution_authorized")
        is not False
    ):
        raise ValueError("paper HELMET materializer preflight evidence is invalid")
    if (
        helmet_clinc_protocol.get("format") != "speck_helmet_data_only_source_protocol"
        or helmet_clinc_protocol.get("status") != "executed_qualified"
        or helmet_clinc_protocol.get("result", {}).get("sha256")
        != evidence["helmet_clinc_source_qualification_sha256"]
        or helmet_clinc.get("format")
        != "speck_helmet_data_only_source_qualification"
        or helmet_clinc.get("status") != evidence["helmet_clinc_source_status"]
        or set(helmet_clinc.get("source", {}).get("splits", {}))
        != {"train", "validation"}
        or helmet_clinc.get("decision", {}).get("source_snapshot_qualified") is not True
        or helmet_clinc.get("decision", {}).get("helmet_execution_authorized") is not False
    ):
        raise ValueError("paper HELMET CLINC source evidence is invalid")
    if (
        helmet_trec_protocol.get("format") != "speck_helmet_rights_decision_protocol"
        or helmet_trec_protocol.get("format_version") != 2
        or helmet_trec_protocol.get("status") != "executed_blocked"
        or helmet_trec_protocol.get("result", {}).get("sha256")
        != evidence["helmet_trec_rights_decision_sha256"]
        or helmet_trec.get("format") != "speck_helmet_rights_decision"
        or helmet_trec.get("status") != evidence["helmet_trec_rights_status"]
        or helmet_trec.get("payload_files_acquired") != 0
        or helmet_trec.get("decision", {}).get("payload_acquisition_authorized") is not False
        or helmet_trec.get("decision", {}).get("evaluation_use_authorized") is not False
    ):
        raise ValueError("paper HELMET TREC rights evidence is invalid")
    if (
        helmet_multilexsum_protocol.get("format")
        != "speck_helmet_multilexsum_decision_protocol"
        or helmet_multilexsum_protocol.get("status") != "executed_blocked"
        or helmet_multilexsum_protocol.get("result", {}).get("sha256")
        != evidence["helmet_multilexsum_decision_sha256"]
        or helmet_multilexsum.get("format") != "speck_helmet_multilexsum_decision"
        or helmet_multilexsum.get("status") != evidence["helmet_multilexsum_status"]
        or helmet_multilexsum.get("payload_files_acquired") != 0
        or helmet_multilexsum.get("helmet_analysis", {}).get("prompt_deterministic")
        is not False
        or helmet_multilexsum.get("decision", {}).get("evaluation_use_authorized")
        is not False
    ):
        raise ValueError("paper HELMET Multi-LexSum evidence is invalid")
    if (
        helmet_narrativeqa_protocol.get("format")
        != "speck_helmet_narrativeqa_decision_protocol"
        or helmet_narrativeqa_protocol.get("status") != "executed_blocked"
        or helmet_narrativeqa_protocol.get("result", {}).get("sha256")
        != evidence["helmet_narrativeqa_decision_sha256"]
        or helmet_narrativeqa.get("format") != "speck_helmet_narrativeqa_decision"
        or helmet_narrativeqa.get("status") != evidence["helmet_narrativeqa_status"]
        or helmet_narrativeqa.get("payload_files_acquired") != 0
        or helmet_narrativeqa.get("helmet_analysis", {}).get("prompt_deterministic")
        is not False
        or helmet_narrativeqa.get("decision", {}).get("embedded_work_rights_qualified")
        is not False
    ):
        raise ValueError("paper HELMET NarrativeQA evidence is invalid")
    if (
        helmet_infinitebench_protocol.get("format")
        != "speck_helmet_infinitebench_decision_protocol"
        or helmet_infinitebench_protocol.get("status") != "executed_blocked"
        or helmet_infinitebench_protocol.get("result", {}).get("sha256")
        != evidence["helmet_infinitebench_decision_sha256"]
        or helmet_infinitebench.get("format") != "speck_helmet_infinitebench_decision"
        or helmet_infinitebench.get("status") != evidence["helmet_infinitebench_status"]
        or helmet_infinitebench.get("payload_files_acquired") != 0
        or helmet_infinitebench.get("helmet_analysis", {}).get(
            "prompt_selection_deterministic"
        )
        is not True
        or helmet_infinitebench.get("decision", {}).get("embedded_work_rights_qualified")
        is not False
    ):
        raise ValueError("paper HELMET InfiniteBench evidence is invalid")
    if (
        helmet_seeded_protocol.get("format") != "speck_helmet_seeded_demos_protocol"
        or helmet_seeded_protocol.get("status") != "executed_qualified"
        or helmet_seeded_protocol.get("result", {}).get("sha256")
        != evidence["helmet_seeded_demo_repair_qualification_sha256"]
        or helmet_seeded.get("format") != "speck_helmet_seeded_demos_qualification"
        or helmet_seeded.get("status") != evidence["helmet_seeded_demo_repair_status"]
        or helmet_seeded.get("decision", {}).get("patch_qualified") is not True
        or helmet_seeded.get("decision", {}).get("real_dataset_prompts_qualified")
        is not False
        or helmet_seeded.get("decision", {}).get("rights_blockers_changed") is not False
    ):
        raise ValueError("paper HELMET seeded-demo repair evidence is invalid")


def _validate_proxy_launch_evidence(program, repository_root):
    evidence = program["proxy_launch_evidence"]
    _require(
        evidence,
        {
            "status",
            "contract",
            "contract_sha256",
            "qualification",
            "qualification_sha256",
            "qualification_status",
            "runner_revision",
        },
        "paper proxy launch evidence",
    )
    if evidence["status"] != "proxy_training_authorized_release_claims_blocked":
        raise ValueError("paper proxy launch evidence has an invalid decision boundary")
    for path_key, hash_key in (
        ("contract", "contract_sha256"),
        ("qualification", "qualification_sha256"),
    ):
        path = repository_root / evidence[path_key]
        if not path.is_file() or _file_sha256(path) != evidence[hash_key]:
            raise ValueError(f"paper proxy launch {path_key} does not match its pin")

    contract = _load_json(repository_root / evidence["contract"])
    result = _load_json(repository_root / evidence["qualification"])
    prerequisites = {entry.get("id"): entry for entry in contract.get("prerequisites", ())}
    if (
        contract.get("format") != "speck_paper_proxy_launch_contract"
        or contract.get("format_version") != 1
        or contract.get("paper_id") != program["paper_id"]
        or contract.get("policy_id") != program["policy_id"]
        or contract.get("status") != "frozen_before_first_proxy_result"
        or len(prerequisites) != 6
    ):
        raise ValueError("paper proxy launch contract is invalid")
    for prerequisite in prerequisites.values():
        path = repository_root / prerequisite.get("path", "")
        if not path.is_file() or _file_sha256(path) != prerequisite.get("sha256"):
            raise ValueError("paper proxy launch prerequisite no longer matches its pin")
    baseline = program["baseline_evidence"]
    evaluation = program["evaluation_evidence"]
    if (
        prerequisites.get("baseline_matrix", {}).get("sha256")
        != baseline["matrix_sha256"]
        or prerequisites.get("analysis_plan", {}).get("sha256")
        != baseline["analysis_plan_sha256"]
        or prerequisites.get("materialization", {}).get("sha256")
        != baseline["materialization_sha256"]
        or prerequisites.get("hardware_preflight", {}).get("sha256")
        != baseline["preflight_v2_sha256"]
        or prerequisites.get("storage_qualification", {}).get("sha256")
        != baseline["storage_qualification_sha256"]
        or prerequisites.get("evaluation_manifest", {}).get("sha256")
        != evaluation["active_evaluation_manifest_sha256"]
        or contract.get("evaluation_boundary", {})
        .get("release_and_capability_claims", {})
        .get("status")
        != "blocked"
        or contract.get("execution", {}).get("control_first") is not True
        or contract.get("execution", {}).get("interim_quality_decisions") != 0
    ):
        raise ValueError("paper proxy launch contract does not match the research program")

    decision = result.get("decision", {})
    live = result.get("live_gate", {})
    controls = contract["execution"]["control_runs"]
    if (
        result.get("format") != "speck_paper_proxy_launch_qualification"
        or result.get("format_version") != 1
        or result.get("status") != evidence["qualification_status"]
        or result.get("status") != evidence["status"]
        or result.get("contract", {}).get("sha256") != evidence["contract_sha256"]
        or result.get("repository", {}).get("clean") is not True
        or result.get("runner_revision") != evidence["runner_revision"]
        or result.get("runner_sha256")
        != _file_sha256(repository_root / "scripts/paper_proxy_launch_qualify.py")
        or live.get("passed") is not True
        or live.get("existing_checkpoint_targets")
        or live.get("existing_result_records")
        or result.get("next_run", {}).get("run") != controls[0]
        or decision.get("proxy_training_authorized") is not True
        or decision.get("release_claims_authorized") is not False
        or decision.get("long_context_capability_claims_authorized") is not False
        or decision.get("architecture_promotion_authorized") is not False
        or decision.get("paper_scale_pretraining_authorized") is not False
        or decision.get("external_missing_suites_preserved_as_failed_release_gates")
        is not True
    ):
        raise ValueError("paper proxy launch qualification is invalid")


def _validate_sequence_cache_design(reference, repository_root, paper_id, policy_id):
    _require(reference, {"contract", "sha256", "status"}, "sequence cache design reference")
    path = repository_root / reference["contract"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("sequence cache representation design does not match its pin")
    design = _load_json(path)
    if (
        design.get("format") != "speck_sequence_cache_representation_design"
        or design.get("format_version") != 1
        or design.get("paper_id") != paper_id
        or design.get("policy_id") != policy_id
        or design.get("status") != reference["status"]
        or len(design.get("launch_blockers", ())) < 5
    ):
        raise ValueError("sequence cache representation design identity is invalid")
    arms = {arm.get("id"): arm for arm in design.get("arms", ())}
    if set(arms) != {"gqa3", "mqa1", "nope_mla128"}:
        raise ValueError("sequence cache representation arms are incomplete")
    control = arms["gqa3"]
    mqa = arms["mqa1"]
    mla = arms["nope_mla128"]
    if (
        control.get("parameters") != 153_958_938
        or control.get("bf16_cache_bytes_per_token_all_five_memories") != 3_840
        or mqa.get("raw_attention_parameter_reduction") != 983_040
        or mqa.get("uniform_ffn_parameter_reallocation") != 967_680
        or mqa.get("parameters") != 153_943_578
        or mqa.get("analytic_flops_per_token_at_4096") != 1_021_509_120
        or mqa.get("bf16_cache_bytes_per_token_all_five_memories") != 1_280
        or mqa.get("state_reduction_from_gqa3") != 2 / 3
        or mla.get("kv_latent_dim") != 128
        or mla.get("query_compression") is not False
        or mla.get("parameters_before_implementation_audit") != 153_959_258
        or mla.get("bf16_cache_bytes_per_token_all_five_memories") != 1_280
        or mla.get("implementation") != "not implemented"
    ):
        raise ValueError("sequence cache representation geometry is invalid")
    systems = design.get("evidence_stages", {}).get("systems", {})
    decision = design.get("decision", {})
    if (
        systems.get("minimum_state_reduction") != 0.25
        or systems.get("minimum_primary_improvement_mqa1") != 0.1
        or systems.get("minimum_primary_improvement_mla128") != 0.2
        or systems.get("repeats") != 5
        or systems.get("interleave_arms") is not True
        or decision.get("no_quality_cost_trade") is not True
        or decision.get("analytic_savings_are_not_sufficient") is not True
        or decision.get("promotion_authority_now") is not False
        or decision.get("training_authorized_now") is not False
    ):
        raise ValueError("sequence cache representation decision gate is invalid")


def _validate_hca_readiness(reference, repository_root, paper_id, policy_id):
    _require(reference, {"contract", "sha256", "status"}, "HCA readiness reference")
    path = repository_root / reference["contract"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("HCA readiness gate does not match its pin")
    gate = _load_json(path)
    if (
        gate.get("format") != "speck_hca_readiness_gate"
        or gate.get("format_version") != 1
        or gate.get("paper_id") != paper_id
        or gate.get("policy_id") != policy_id
        or gate.get("status") != reference["status"]
        or len(gate.get("current_blockers", ())) < 7
    ):
        raise ValueError("HCA readiness gate identity is invalid")
    causal = gate.get("causal_semantics_required", {})
    compression = gate.get("compressor_isolation_before_rate_selection", {})
    compressor_arms = compression.get("reference_arms", ())
    rate = gate.get("conditional_rate_grid", {})
    systems = gate.get("systems_gate", {})
    decision = gate.get("decision", {})
    if (
        len(compressor_arms) != 3
        or {arm.get("id") for arm in compressor_arms}
        != {"block_mean", "scalar_position_softmax", "channel_position_softmax"}
        or rate.get("rates_tokens_per_summary") != [32, 64, 128, 256]
        or rate.get("completed_summaries_at_4096") != [128, 64, 32, 16]
        or rate.get("completed_summaries_at_131072") != [4096, 2048, 1024, 512]
        or rate.get("completed_summaries_at_1048576")
        != [32768, 16384, 8192, 4096]
        or "O(L^2/m)" not in gate.get("accounting_required", {}).get(
            "complexity_statement", ""
        )
        or len(causal) < 8
        or systems.get("custom_runtime_minimum_primary_improvement") != 0.2
        or systems.get("minimum_state_reduction") != 0.25
        or decision.get("compressor_selected") is not False
        or decision.get("compression_rate_selected") is not False
        or decision.get("implementation_authorized") is not False
        or decision.get("training_authorized") is not False
        or decision.get("promotion_authority") is not False
    ):
        raise ValueError("HCA readiness design is incomplete")


def _validate_csa_readiness(reference, repository_root, paper_id, policy_id):
    _require(reference, {"contract", "sha256", "status"}, "CSA readiness reference")
    path = repository_root / reference["contract"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("CSA readiness gate does not match its pin")
    gate = _load_json(path)
    selectors = gate.get("selector_feasibility_sequence", ())
    grid = gate.get("conditional_geometry_grid", {})
    complexity = gate.get("complexity_and_state_accounting", {})
    systems = gate.get("systems_gate", {})
    decision = gate.get("decision", {})
    if (
        gate.get("format") != "speck_csa_readiness_gate"
        or gate.get("format_version") != 1
        or gate.get("paper_id") != paper_id
        or gate.get("policy_id") != policy_id
        or gate.get("status") != reference["status"]
        or len(gate.get("ordered_prerequisites", ())) < 5
        or [selector.get("id") for selector in selectors]
        != ["oracle_block_mass", "mean_key", "learned_block_indexer"]
        or grid.get("high_resolution_compression_tokens") != [2, 4, 8]
        or grid.get("selection_block_raw_tokens") != [32, 64, 128]
        or grid.get("attended_raw_token_budgets") != [512, 2048, 8192]
        or "O(L^2/m)" not in complexity.get("dense_index_scan", "")
        or "O(Lk)" not in complexity.get("selected_attention", "")
        or systems.get("custom_runtime_minimum_primary_improvement") != 0.2
        or systems.get("minimum_state_reduction") != 0.25
        or systems.get("repeats") != 5
        or systems.get("interleave_arms") is not True
        or any(
            decision.get(key) is not False
            for key in (
                "block_granularity_selected",
                "compression_selected",
                "budget_selected",
                "selector_selected",
                "implementation_authorized",
                "training_authorized",
                "token_level_selection_authorized",
                "promotion_authority",
            )
        )
    ):
        raise ValueError("CSA readiness design is incomplete")


def _validate_raw_local_readiness(reference, repository_root, paper_id, policy_id):
    _require(reference, {"contract", "sha256", "status"}, "raw-local readiness reference")
    path = repository_root / reference["contract"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("raw-local readiness gate does not match its pin")
    gate = _load_json(path)
    fixed = gate.get("fixed_parent", {})
    formulation = gate.get("first_formulation", {})
    causal = gate.get("causal_and_state_semantics_required", {})
    windows = gate.get("conditional_window_grid", {})
    systems = gate.get("systems_and_accounting", {})
    decision = gate.get("decision", {})
    if (
        gate.get("format") != "speck_raw_local_readiness_gate"
        or gate.get("format_version") != 1
        or gate.get("paper_id") != paper_id
        or gate.get("policy_id") != policy_id
        or gate.get("status") != reference["status"]
        or len(gate.get("ordered_prerequisites", ())) < 5
        or fixed.get("global_slots") != [3, 7, 11, 15, 19]
        or formulation.get("placement")
        != "one local ring at each of the same five global integration slots; do not add attention to the fifteen KDA-only layers"
        or "one causal softmax" not in formulation.get("normalization", "")
        or len(causal) < 8
        or windows.get("control") != 0
        or windows.get("windows") != [64, 128, 256, 512]
        or systems.get("simple_component_minimum_primary_improvement") != 0.1
        or systems.get("repeats") != 5
        or systems.get("interleave_arms") is not True
        or any(
            decision.get(key) is not False
            for key in (
                "fusion_qualified",
                "window_selected",
                "placement_selected",
                "implementation_authorized",
                "training_authorized",
                "promotion_authority",
            )
        )
    ):
        raise ValueError("raw-local readiness design is incomplete")


def _validate_ratio_placement_readiness(reference, repository_root, paper_id, policy_id):
    _require(reference, {"contract", "sha256", "status"}, "ratio readiness reference")
    path = repository_root / reference["contract"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("ratio/placement readiness gate does not match its pin")
    gate = _load_json(path)
    correction = gate.get("ratio_correction", {})
    isolation = gate.get("ratio_isolation", {})
    arms = isolation.get("arms", ())
    successor = gate.get("placement_successor_after_ratio", {})
    decision = gate.get("decision", {})
    expected = [
        ("ratio_1_to_1", 10, 10, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]),
        ("ratio_3_to_1", 15, 5, [3, 7, 11, 15, 19]),
        ("ratio_9_to_1", 18, 2, [9, 19]),
    ]
    observed = [
        (
            arm.get("id"),
            arm.get("recurrent_layers"),
            arm.get("global_layers"),
            arm.get("global_indices_zero_based"),
        )
        for arm in arms
    ]
    if (
        gate.get("format") != "speck_ratio_placement_readiness_gate"
        or gate.get("format_version") != 1
        or gate.get("paper_id") != paper_id
        or gate.get("policy_id") != policy_id
        or gate.get("status") != reference["status"]
        or len(gate.get("ordered_prerequisites", ())) < 5
        or correction.get("previous_labels") != ["1:1", "3:1", "7:1"]
        or correction.get("replacement_labels") != ["1:1", "3:1", "9:1"]
        or correction.get("changed_before_ratio_results") is not True
        or observed != expected
        or successor.get("status") != "not_frozen_until_one_count_is_selected"
        or successor.get("required_shared_count") is not True
        or len(gate.get("mechanistic_endpoints", ())) < 7
        or any(
            decision.get(key) is not False
            for key in (
                "ratio_selected",
                "placement_selected",
                "exact_placement_successor_frozen",
                "implementation_authorized",
                "training_authorized",
                "promotion_authority",
            )
        )
    ):
        raise ValueError("ratio/placement readiness design is incomplete")


def _validate_attnres_readiness(reference, repository_root, paper_id, policy_id):
    _require(reference, {"contract", "sha256", "status"}, "AttnRes readiness reference")
    path = repository_root / reference["contract"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("AttnRes readiness gate does not match its pin")
    gate = _load_json(path)
    graph = gate.get("module_graph", {})
    arms = gate.get("initial_isolation_arms", ())
    choices = gate.get("fixed_attnres_choices", {})
    block = gate.get("block_count_successor", {})
    geometry = gate.get("depth_width_successor", {})
    decision = gate.get("decision", {})
    if (
        gate.get("format") != "speck_attnres_readiness_gate"
        or gate.get("format_version") != 1
        or gate.get("paper_id") != paper_id
        or gate.get("policy_id") != policy_id
        or gate.get("status") != reference["status"]
        or graph.get("transformer_blocks") != 20
        or graph.get("residual_modules_per_block") != 2
        or graph.get("residual_modules") != 40
        or [arm.get("id") for arm in arms]
        != ["standard_prenorm", "static_depth_weights", "full_attnres", "block_attnres_8"]
        or arms[2].get("maximum_sources") != 40
        or arms[3].get("target_completed_blocks") != 8
        or arms[3].get("modules_per_block") != 5
        or arms[3].get("maximum_sources_in_final_block") != 9
        or choices.get("aggregation") != "softmax"
        or choices.get("depth_heads") != 1
        or choices.get("key_normalization") != "RMSNorm"
        or choices.get("pseudo_query_initialization") != 0
        or block.get("target_completed_block_counts") != [4, 8, 12]
        or block.get("residual_modules") != 40
        or geometry.get("status") != "required_before_any_depth-routing_promotion"
        or len(gate.get("mechanistic_diagnostics", ())) < 8
        or any(
            decision.get(key) is not False
            for key in (
                "residual_selected",
                "block_count_selected",
                "depth_width_geometry_selected",
                "implementation_authorized",
                "training_authorized",
                "promotion_authority",
            )
        )
    ):
        raise ValueError("AttnRes readiness design is incomplete")


def _validate_stable_latentmoe_readiness(reference, repository_root, paper_id, policy_id):
    _require(reference, {"contract", "sha256", "status"}, "LatentMoE readiness reference")
    path = repository_root / reference["contract"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("Stable LatentMoE readiness gate does not match its pin")
    gate = _load_json(path)
    stages = gate.get("required_decomposition_order", ())
    semantics = gate.get("initial_conventional_moe_semantics_to_freeze", {})
    intervention = gate.get("intervention_policy_required", {})
    systems = gate.get("systems_gate", {})
    decision = gate.get("decision", {})
    if (
        gate.get("format") != "speck_stable_latentmoe_readiness_gate"
        or gate.get("format_version") != 1
        or gate.get("paper_id") != paper_id
        or gate.get("policy_id") != policy_id
        or gate.get("status") != reference["status"]
        or len(gate.get("hard_blockers", ())) < 8
        or [stage.get("id") for stage in stages]
        != [
            "dense_vs_conventional_moe",
            "latent_projection",
            "latent_normalization",
            "bounded_activation",
            "balancing",
            "expert_geometry",
        ]
        or len(semantics) < 9
        or len(gate.get("correctness_before_training", ())) < 10
        or len(gate.get("stability_diagnostics", {})) < 4
        or len(intervention.get("freeze_before_training", ())) < 5
        or systems.get("single_device_required") is not True
        or systems.get("expert_parallel_required_before_promotion") is not True
        or systems.get("minimum_primary_improvement") != 0.2
        or systems.get("repeats") != 5
        or systems.get("interleave_arms") is not True
        or any(
            decision.get(key) is not False
            for key in (
                "primary_spec_qualified",
                "sequence_parent_selected",
                "depth_parent_selected",
                "hardware_envelope_selected",
                "conventional_moe_qualified",
                "latent_projection_qualified",
                "bounded_activation_qualified",
                "balancing_qualified",
                "expert_geometry_selected",
                "implementation_authorized",
                "training_authorized",
                "promotion_authority",
            )
        )
    ):
        raise ValueError("Stable LatentMoE readiness design is incomplete")


def _validate_interaction_readiness(reference, repository_root, paper_id, policy_id):
    _require(reference, {"contract", "sha256", "status"}, "interaction readiness reference")
    path = repository_root / reference["contract"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("interaction readiness gate does not match its pin")
    gate = _load_json(path)
    axes = gate.get("axis_units", {})
    cube = gate.get("discovery_cube", {})
    contrasts = gate.get("contrast_contract", {})
    thresholds = gate.get("decision_thresholds", {})
    removals = gate.get("final_removal_program", {})
    decision = gate.get("decision", {})
    if (
        gate.get("format") != "speck_interaction_readiness_gate"
        or gate.get("format_version") != 1
        or gate.get("paper_id") != paper_id
        or gate.get("policy_id") != policy_id
        or gate.get("status") != reference["status"]
        or len(gate.get("entry_gate", ())) < 5
        or set(axes) != {"sequence", "depth", "width"}
        or cube.get("factors") != ["S", "D", "W"]
        or cube.get("cells")
        != ["000", "100", "010", "001", "110", "101", "011", "111"]
        or cube.get("paired_cells") != 3
        or cube.get("model_runs") != 24
        or cube.get("promotion_authority") is not False
        or set(contrasts.get("pairwise_interactions", {})) != {"SD", "SW", "DW"}
        or set(contrasts.get("conditional_removal_effects", {}))
        != {"sequence_from_full", "depth_from_full", "width_from_full"}
        or thresholds.get("harmful_language_interaction_margin_nats") != 0.01
        or thresholds.get("harmful_source_interaction_margin_nats") != 0.02
        or len(gate.get("matching_views", ())) < 7
        or set(removals.get("subcomponent_removals", {}))
        != {"sequence", "depth", "width"}
        or len(gate.get("scale_transfer", {})) < 4
        or any(
            decision.get(key) is not False
            for key in (
                "sequence_bundle_selected",
                "depth_bundle_selected",
                "width_bundle_selected",
                "cube_materialized",
                "cube_training_authorized",
                "combined_architecture_selected",
                "promotion_authority",
            )
        )
    ):
        raise ValueError("interaction readiness design is incomplete")


def _validate_scaling_readiness(reference, repository_root, paper_id, policy_id):
    _require(reference, {"contract", "sha256", "status"}, "scaling readiness reference")
    path = repository_root / reference["contract"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("scaling readiness gate does not match its pin")
    gate = _load_json(path)
    scales = gate.get("scale_points", {})
    allocation = gate.get("compute_optimal_allocation", {})
    fit = gate.get("confirmatory_fit", {})
    uncertainty = gate.get("uncertainty", {})
    held_out = gate.get("held_out_prediction", {})
    horizon = gate.get("training_horizon_interaction", {})
    decision = gate.get("decision", {})
    if (
        gate.get("format") != "speck_scaling_readiness_gate"
        or gate.get("format_version") != 1
        or gate.get("paper_id") != paper_id
        or gate.get("policy_id") != policy_id
        or gate.get("status") != reference["status"]
        or len(gate.get("entry_gate", ())) < 6
        or scales.get("fit_active_parameters")
        != [30_000_000, 60_000_000, 150_000_000, 350_000_000, 600_000_000]
        or scales.get("held_out_active_parameters") != 1_200_000_000
        or scales.get("held_out_training_tokens") != 20_000_000_000
        or scales.get("fit_points_per_architecture") != 5
        or scales.get("minimum_paired_seed_data_cells_per_fit_point") != 3
        or allocation.get("pilot_parameter_points")
        != [30_000_000, 60_000_000, 150_000_000]
        or allocation.get("pilot_tokens_per_active_parameter") != [10, 20, 40]
        or allocation.get("joint_fit") != "L(N,D)=E+A*N^(-alpha)+B*D^(-beta)"
        or fit.get("analytic_compute_family") != "L(C)=E_C+A_C*C^(-gamma_C)"
        or fit.get("shared_family") is not True
        or uncertainty.get("method")
        != "paired hierarchical bootstrap that resamples seed/data cells within scale and refits allocation plus both frontier models"
        or uncertainty.get("resamples") != 10_000
        or uncertainty.get("seed") != 42
        or uncertainty.get("full_pipeline_refit") is not True
        or held_out.get("scale_active_parameters") != 1_200_000_000
        or held_out.get("training_tokens") != 20_000_000_000
        or horizon.get("checkpoints_tokens_per_active_parameter") != [2, 5, 10, 20]
        or any(
            decision.get(key) is not False
            for key in (
                "candidate_architecture_selected",
                "control_architecture_selected",
                "geometry_rules_selected",
                "hardware_envelope_selected",
                "allocation_pilots_authorized",
                "fit_runs_authorized",
                "held_out_run_authorized",
                "scaling_claim_authorized",
                "paper_scale_authorized",
            )
        )
    ):
        raise ValueError("scaling readiness design is incomplete")


def _validate_systems_cost_readiness(reference, repository_root, paper_id, policy_id):
    _require(reference, {"contract", "sha256", "status"}, "systems-cost reference")
    path = repository_root / reference["contract"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("systems-cost readiness gate does not match its pin")
    gate = _load_json(path)
    source = gate.get("input", {})
    source_path = repository_root / source.get("cost_envelopes", "")
    audit = gate.get("proxy_control_envelope_audit", {})
    runs = audit.get("runs", ())
    hierarchy = gate.get("evidence_hierarchy", ())
    training = gate.get("training_measurement", {})
    energy = gate.get("energy_measurement", {})
    serving = gate.get("serving_matrix", {})
    monetary = gate.get("monetary_accounting", {})
    decision = gate.get("decision", {})
    if (
        gate.get("format") != "speck_systems_cost_readiness_gate"
        or gate.get("format_version") != 1
        or gate.get("paper_id") != paper_id
        or gate.get("policy_id") != policy_id
        or gate.get("status") != reference["status"]
        or not source_path.is_file()
        or _file_sha256(source_path) != source.get("cost_envelopes_sha256")
        or source.get("candidate_result_records_present") != 0
        or len(runs) != 3
        or [run.get("pair") for run in runs] != [0, 1, 2]
        or any(run.get("gpu_hours_pass") is not False for run in runs)
        or any(run.get("complete_envelope_pass") is not False for run in runs)
        or any(run.get("throughput_pass") is not True for run in runs)
        or [entry.get("id") for entry in hierarchy]
        != ["analytic", "operator", "model_runtime", "serving_system", "monetary"]
        or len(training.get("time_categories", ())) < 7
        or energy.get("sampling_hz") != 1
        or energy.get("report_both") is not True
        or serving.get("prompt_lengths") != [512, 4096, 32768, 131072]
        or serving.get("output_tokens") != 128
        or serving.get("minimum_requests_for_p99") != 1000
        or monetary.get("current_status")
        != "blocked because v1 contains neither hardware amortization nor electricity price"
        or len(gate.get("datacenter_v2_requirements", ())) < 8
        or decision.get("proxy_control_hard_envelope_pass") is not False
        or decision.get("proxy_quality_decision_changed") is not False
        or decision.get("candidate_cost_result_available") is not False
        or decision.get("consumer_serving_claim_authorized") is not False
        or decision.get("datacenter_profile_qualified") is not False
        or decision.get("monetary_claim_authorized") is not False
        or decision.get("architecture_cost_promotion_authorized") is not False
        or decision.get("paper_scale_authorized") is not False
    ):
        raise ValueError("systems-cost readiness design is incomplete")
    control_results = sorted(
        (repository_root / "results/Speck-Paper1/runs").glob("*dense_global_param_match.json")
    )
    if len(control_results) != 3:
        raise ValueError("systems-cost audit requires exactly three dense controls")
    for result_path, audited in zip(control_results, runs):
        result = _load_json(result_path)
        active = result.get("timing", {}).get("active_seconds")
        steady = result.get("final_validation", {}).get("steady_training_seconds")
        if (
            _file_sha256(result_path) != audited.get("result_sha256")
            or not math.isclose(active, audited.get("active_seconds"), rel_tol=0, abs_tol=1e-9)
            or not math.isclose(active / 3600, audited.get("gpu_hours"), rel_tol=0, abs_tol=1e-12)
            or not math.isclose(
                result.get("training_tokens") / steady,
                audited.get("steady_tokens_per_second"),
                rel_tol=0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                result.get("peak_allocated_bytes") / 2**30,
                audited.get("peak_allocated_gib"),
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("systems-cost dense-control audit does not match result evidence")


def _validate_novelty_landscape(reference, repository_root, paper_id):
    _require(reference, {"audit", "sha256", "status"}, "novelty landscape reference")
    path = repository_root / reference["audit"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("novelty landscape audit does not match its pin")
    audit = _load_json(path)
    sources = audit.get("sources", ())
    overlaps = audit.get("overlap_decisions", ())
    hypotheses = audit.get("surviving_hypotheses", ())
    claim_gate = audit.get("claim_gate", {})
    decision = audit.get("decision", {})
    if (
        audit.get("format") != "speck_novelty_landscape_audit"
        or audit.get("format_version") != 1
        or audit.get("paper_id") != paper_id
        or audit.get("status") != reference["status"]
        or [source.get("id") for source in sources]
        != [
            "arxiv_2606_30562v1",
            "arxiv_2605_05219v1",
            "arxiv_2407_11550v5",
            "arxiv_2404_04793v2",
            "arxiv_2605_05697v1",
            "arxiv_2511_00819v1",
        ]
        or [source.get("review_scope") for source in sources]
        != [
            "full_text",
            "full_text",
            "full_text",
            "full_text",
            "full_text",
            "full_text",
        ]
        or any(
            not (repository_root / source.get("local_note", "")).is_file()
            for source in sources
        )
        or len(overlaps) < 6
        or any(overlap.get("decision") == "novel" for overlap in overlaps)
        or [hypothesis.get("id") for hypothesis in hypotheses]
        != ["N1_role_grounded_placement_law", "N2_all_required_source_recall_law"]
        or any(
            hypothesis.get("novelty_status")
            != "plausibly_distinct_full_landscape_and_evidence_pending"
            for hypothesis in hypotheses
        )
        or len(claim_gate.get("full_landscape_before_claim", ())) < 5
        or len(claim_gate.get("evidence_before_claim", ())) < 6
        or decision.get("novel_mechanism_established") is not False
        or decision.get("novel_composition_rule_established") is not False
        or decision.get("novel_generalizable_law_established") is not False
        or decision.get("novel_inseparable_systems_method_established") is not False
        or decision.get("paper_novelty_gate_pass") is not False
        or decision.get("architecture_freeze_authorized") is not False
    ):
        raise ValueError("novelty landscape audit is incomplete")


def _validate_novelty_code_availability(reference, repository_root, paper_id):
    _require(reference, {"audit", "sha256", "status"}, "novelty code reference")
    path = repository_root / reference["audit"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("novelty code availability audit does not match its pin")
    audit = _load_json(path)
    sources = {source.get("id"): source for source in audit.get("sources", ())}
    summary = audit.get("summary", {})
    decision = audit.get("decision", {})
    if (
        audit.get("format") != "speck_novelty_code_availability_audit"
        or audit.get("format_version") != 1
        or audit.get("paper_id") != paper_id
        or audit.get("status") != reference["status"]
        or set(sources)
        != {
            "flashmorph",
            "sparse_prefix_caching",
            "adakv",
            "squeezeattention",
            "budgeted_attention_allocation",
            "alternating_sparse_attention",
        }
        or sources["flashmorph"].get("revision")
        != "b9b635379380fc4774d3e54eefd2fc6f10c89ee2"
        or sources["flashmorph"].get("code_like_files") != 0
        or sources["adakv"].get("revision")
        != "04497abac4c1a58426f3daf1014578990e225cc5"
        or sources["adakv"].get("code_like_files") != 15
        or len(sources["adakv"].get("license_files", ())) != 2
        or sources["squeezeattention"].get("revision")
        != "a1933d18b09c668312cbabdfd7b1fa73888c0f6f"
        or sources["squeezeattention"].get("root_license_files")
        or summary.get("sources") != 6
        or summary.get("immutable_repository_snapshots") != 3
        or summary.get("repositories_with_code") != 2
        or summary.get("repositories_with_root_license_covering_code") != 1
        or summary.get("third_party_code_executed") is not False
        or summary.get("code_payloads_cloned") is not False
        or decision.get("adakv_deeper_audit_authorized") is not True
        or decision.get("adakv_execution_authorized") is not False
        or decision.get("novelty_gate_changed") is not False
        or decision.get("architecture_freeze_authorized") is not False
    ):
        raise ValueError("novelty code availability audit is incomplete")


def _validate_adakv_code_audit(reference, repository_root, paper_id):
    _require(reference, {"audit", "sha256", "status"}, "Ada-KV code reference")
    path = repository_root / reference["audit"]
    if not path.is_file() or _file_sha256(path) != reference["sha256"]:
        raise ValueError("Ada-KV code audit does not match its pin")
    audit = _load_json(path)
    repository = audit.get("repository", {})
    licenses = audit.get("licenses", ())
    files = {entry.get("path"): entry for entry in audit.get("pinned_files", ())}
    decision = audit.get("decision", {})
    if (
        audit.get("format") != "speck_adakv_code_audit"
        or audit.get("format_version") != 1
        or audit.get("paper_id") != paper_id
        or audit.get("status") != reference["status"]
        or repository.get("revision") != "04497abac4c1a58426f3daf1014578990e225cc5"
        or repository.get("files") != 38
        or repository.get("code_like_files") != 15
        or repository.get("code_payload_cloned") is not False
        or repository.get("third_party_code_executed") is not False
        or [license_entry.get("license") for license_entry in licenses] != ["MIT", "MIT"]
        or set(files)
        != {
            "pyproject.toml",
            "freeze_requirements.txt",
            "makefile",
            "csrc/makefile",
            "adaptive_snapkv/monkeypatch/monkeypatch.py",
            "adaptive_snapkv/monkeypatch/snapkv_utils.py",
            "experiments/LongBench/pred.py",
        }
        or len(audit.get("environment_findings", ())) < 5
        or len(audit.get("runtime_findings", ())) < 4
        or len(audit.get("evaluation_findings", ())) < 5
        or len(audit.get("clean_room_requirements", ())) < 7
        or decision.get("source_identity_qualified") is not True
        or decision.get("license_identity_qualified") is not True
        or decision.get("portable_environment_qualified") is not False
        or decision.get("dependency_identity_qualified") is not False
        or decision.get("fixture_behavior_qualified") is not False
        or decision.get("native_extension_qualified") is not False
        or decision.get("upstream_execution_authorized") is not False
        or decision.get("reuse_in_speck_authorized") is not False
        or decision.get("novelty_gate_changed") is not False
    ):
        raise ValueError("Ada-KV code audit is incomplete")


def _validate_adaptive_cache_budget(reference, repository_root, paper_id):
    _require(
        reference,
        {"protocol", "protocol_sha256", "qualification", "qualification_sha256", "status"},
        "adaptive cache budget reference",
    )
    protocol_path = repository_root / reference["protocol"]
    result_path = repository_root / reference["qualification"]
    if not protocol_path.is_file() or _file_sha256(protocol_path) != reference["protocol_sha256"]:
        raise ValueError("adaptive cache budget protocol does not match its pin")
    if not result_path.is_file() or _file_sha256(result_path) != reference["qualification_sha256"]:
        raise ValueError("adaptive cache budget qualification does not match its pin")
    protocol = _load_json(protocol_path)
    result = _load_json(result_path)
    cases = result.get("cases", {})
    decision = result.get("decision", {})
    implementation = result.get("implementation", {})
    tests = result.get("tests", {})
    sources = protocol.get("sources", ())
    source_audit = sources[1] if len(sources) > 1 else {}
    if (
        protocol.get("format") != "speck_adaptive_cache_budget_protocol"
        or protocol.get("format_version") != 1
        or protocol.get("paper_id") != paper_id
        or protocol.get("status") != "frozen_before_local_implementation"
        or source_audit.get("audit") != "research/paper-1/adakv_code_audit_v1.json"
        or not (repository_root / source_audit.get("audit", "")).is_file()
        or _file_sha256(repository_root / source_audit["audit"]) != source_audit.get("sha256")
        or protocol.get("decision", {}).get("model_integration_authorized") is not False
        or protocol.get("decision", {}).get("architecture_claim_authorized") is not False
        or result.get("format") != "speck_adaptive_cache_budget_qualification"
        or result.get("format_version") != 1
        or result.get("status") != reference["status"]
        or result.get("protocol", {}).get("path") != reference["protocol"]
        or result.get("protocol", {}).get("sha256") != reference["protocol_sha256"]
        or cases.get("random_matrix_cases") != 200
        or cases.get("random_budget_cases") != 1700
        or cases.get("adversarial_budget_cases") != 84
        or cases.get("oracle_allocation_comparisons") != 28240
        or abs(cases.get("maximum_optimality_gap", math.inf)) > 1e-12
        or cases.get("minimum_adaptive_minus_uniform_mass", -math.inf) < -1e-12
        or not all(
            cases.get(key) is True
            for key in (
                "budget_conservation_pass",
                "per_head_identity_pass",
                "deterministic_ties_pass",
                "uniform_control_dominance_pass",
                "bound_monotonicity_pass",
            )
        )
        or decision.get("reference_qualified") is not True
        or decision.get("upstream_code_used") is not False
        or decision.get("upstream_code_executed") is not False
        or decision.get("model_integration_authorized") is not False
        or decision.get("novelty_gate_changed") is not False
        or implementation.get("path") != "speck/cache_budget.py"
        or _file_sha256(repository_root / implementation["path"]) != implementation.get("sha256")
        or tests.get("path") != "tests/test_cache_budget.py"
        or _file_sha256(repository_root / tests["path"]) != tests.get("sha256")
        or result.get("runner_sha256")
        != _file_sha256(repository_root / "scripts/adaptive_cache_budget_qualify.py")
    ):
        raise ValueError("adaptive cache budget qualification is incomplete")


def _validate_adaptive_cache_gqa(reference, repository_root, paper_id):
    _require(
        reference,
        {"protocol", "protocol_sha256", "qualification", "qualification_sha256", "status"},
        "adaptive cache GQA reference",
    )
    protocol_path = repository_root / reference["protocol"]
    result_path = repository_root / reference["qualification"]
    if not protocol_path.is_file() or _file_sha256(protocol_path) != reference["protocol_sha256"]:
        raise ValueError("adaptive cache GQA protocol does not match its pin")
    if not result_path.is_file() or _file_sha256(result_path) != reference["qualification_sha256"]:
        raise ValueError("adaptive cache GQA qualification does not match its pin")
    protocol = _load_json(protocol_path)
    result = _load_json(result_path)
    cases = result.get("cases", {})
    maximum = result.get("max_negative_control", {})
    safeguard = result.get("safeguard_negative_witness", {})
    decision = result.get("decision", {})
    implementation = result.get("implementation", {})
    tests = result.get("tests", {})
    if (
        protocol.get("format") != "speck_adaptive_cache_gqa_protocol"
        or protocol.get("format_version") != 1
        or protocol.get("paper_id") != paper_id
        or protocol.get("status") != "frozen_before_local_implementation"
        or protocol.get("decision", {}).get("model_integration_authorized") is not False
        or protocol.get("decision", {}).get("training_authorized") is not False
        or protocol.get("decision", {}).get("safeguard_implementation_authorized") is not False
        or result.get("format") != "speck_adaptive_cache_gqa_qualification"
        or result.get("format_version") != 1
        or result.get("status") != reference["status"]
        or result.get("protocol", {}).get("path") != reference["protocol"]
        or result.get("protocol", {}).get("sha256") != reference["protocol_sha256"]
        or cases.get("random_tensor_cases") != 600
        or cases.get("random_budget_cases") != 4200
        or cases.get("speck_gqa3_budget_cases") != 500
        or cases.get("oracle_allocation_comparisons") != 22000
        or abs(cases.get("maximum_mean_times_group_minus_sum_gap", math.inf)) > 1e-12
        or abs(cases.get("maximum_optimality_gap", math.inf)) > 1e-12
        or abs(cases.get("maximum_direct_minus_scaled_bound_gap", math.inf)) > 1e-12
        or not all(
            cases.get(key) is True
            for key in (
                "mean_sum_identity_pass",
                "physical_capacity_conservation_pass",
                "query_mass_optimality_pass",
                "bound_scaling_and_monotonicity_pass",
            )
        )
        or maximum.get("mean_selected") != [[1, 0]]
        or maximum.get("max_selected") != [[0, 1]]
        or maximum.get("strict_counterexample_pass") is not True
        or not maximum.get("mean_query_retained_mass", -math.inf)
        > maximum.get("max_query_retained_mass", math.inf)
        or safeguard.get("adaptive_budgets") != [4, 1, 1]
        or safeguard.get("code_like_rounded_budgets") != [3, 1, 1]
        or safeguard.get("capacity_deficit") != 1
        or safeguard.get("conservation_failure_reproduced") is not True
        or decision.get("equal_group_mean_reference_qualified") is not True
        or decision.get("max_aggregation_primary_authorized") is not False
        or decision.get("safeguard_authorized") is not False
        or decision.get("upstream_code_used") is not False
        or decision.get("upstream_code_executed") is not False
        or decision.get("model_integration_authorized") is not False
        or decision.get("training_authorized") is not False
        or decision.get("novelty_gate_changed") is not False
        or implementation.get("path") != "speck/cache_budget_gqa.py"
        or _file_sha256(repository_root / implementation["path"]) != implementation.get("sha256")
        or tests.get("path") != "tests/test_cache_budget_gqa.py"
        or _file_sha256(repository_root / tests["path"]) != tests.get("sha256")
        or result.get("runner_sha256")
        != _file_sha256(repository_root / "scripts/adaptive_cache_gqa_qualify.py")
    ):
        raise ValueError("adaptive cache GQA qualification is incomplete")
    for source in protocol.get("sources", ()):
        for path_key, hash_key in (
            ("local_note", "local_note_sha256"),
            ("audit", "sha256"),
            ("protocol", "protocol_sha256"),
            ("qualification", "qualification_sha256"),
        ):
            if path_key in source:
                path = repository_root / source[path_key]
                if not path.is_file() or _file_sha256(path) != source[hash_key]:
                    raise ValueError(f"adaptive cache GQA {path_key} does not match its source pin")


def _validate_claims(claims):
    _require(
        claims,
        {
            "format",
            "format_version",
            "paper_id",
            "status",
            "central_claim_id",
            "claims",
            "non_claims",
        },
        "paper claims",
    )
    if claims["format"] != "speck_paper_claims" or claims["format_version"] != 1:
        raise ValueError("paper claims must use format version 1")
    if claims["status"] != "hypotheses_only":
        raise ValueError("Paper 1 claims must remain hypotheses before confirmatory experiments")
    claim_ids = _unique_ids(claims["claims"], "paper claim")
    if claims["central_claim_id"] not in claim_ids:
        raise ValueError("central paper claim does not exist")
    required = {
        "id",
        "type",
        "statement",
        "status",
        "required_evidence",
        "falsifiers",
        "paper_sections",
    }
    for claim in claims["claims"]:
        _require(claim, required, f"paper claim {claim.get('id')}")
        if claim["status"] != "hypothesis":
            raise ValueError(f"unconfirmed paper claim {claim['id']} cannot be promoted")
        if not claim["required_evidence"] or not claim["falsifiers"] or not claim["paper_sections"]:
            raise ValueError(f"paper claim {claim['id']} lacks evidence or falsification criteria")
    central = next(claim for claim in claims["claims"] if claim["id"] == claims["central_claim_id"])
    if central["type"] != "central":
        raise ValueError("central paper claim must have type central")
    if len(claims["non_claims"]) < 5:
        raise ValueError("paper program requires explicit non-claims")
    return claim_ids


def _validate_program(program, paper_id, claim_ids, repository_root):
    _require(
        program,
        {
            "format",
            "format_version",
            "paper_id",
            "policy_id",
            "status",
            "baseline_evidence",
            "proxy_launch_evidence",
            "evaluation_evidence",
            "controls",
            "matching_views",
            "scales",
            "axes",
            "interaction_program",
            "scaling_program",
            "evaluation_program",
            "analysis_program",
            "large_pretraining_gate",
            "sequence_cache_representation",
            "hca_readiness",
            "csa_readiness",
            "raw_local_readiness",
            "ratio_placement_readiness",
            "attnres_readiness",
            "stable_latentmoe_readiness",
            "interaction_readiness",
            "scaling_readiness",
            "systems_cost_readiness",
            "novelty_landscape",
            "novelty_code_availability",
            "adakv_code_audit",
            "adaptive_cache_budget",
            "adaptive_cache_gqa",
        },
        "paper experiment program",
    )
    if program["format"] != "speck_paper_experiment_program" or program["format_version"] != 1:
        raise ValueError("paper experiment program must use format version 1")
    if program["paper_id"] != paper_id:
        raise ValueError("claims and experiment program use different paper ids")
    _validate_novelty_landscape(
        program["novelty_landscape"],
        repository_root,
        paper_id,
    )
    _validate_novelty_code_availability(
        program["novelty_code_availability"],
        repository_root,
        paper_id,
    )
    _validate_adakv_code_audit(
        program["adakv_code_audit"],
        repository_root,
        paper_id,
    )
    _validate_adaptive_cache_budget(
        program["adaptive_cache_budget"],
        repository_root,
        paper_id,
    )
    _validate_adaptive_cache_gqa(
        program["adaptive_cache_gqa"],
        repository_root,
        paper_id,
    )
    _validate_sequence_cache_design(
        program["sequence_cache_representation"],
        repository_root,
        paper_id,
        program["policy_id"],
    )
    _validate_hca_readiness(
        program["hca_readiness"],
        repository_root,
        paper_id,
        program["policy_id"],
    )
    _validate_csa_readiness(
        program["csa_readiness"],
        repository_root,
        paper_id,
        program["policy_id"],
    )
    _validate_raw_local_readiness(
        program["raw_local_readiness"],
        repository_root,
        paper_id,
        program["policy_id"],
    )
    _validate_ratio_placement_readiness(
        program["ratio_placement_readiness"],
        repository_root,
        paper_id,
        program["policy_id"],
    )
    _validate_attnres_readiness(
        program["attnres_readiness"],
        repository_root,
        paper_id,
        program["policy_id"],
    )
    _validate_stable_latentmoe_readiness(
        program["stable_latentmoe_readiness"],
        repository_root,
        paper_id,
        program["policy_id"],
    )
    _validate_interaction_readiness(
        program["interaction_readiness"],
        repository_root,
        paper_id,
        program["policy_id"],
    )
    _validate_scaling_readiness(
        program["scaling_readiness"],
        repository_root,
        paper_id,
        program["policy_id"],
    )
    _validate_systems_cost_readiness(
        program["systems_cost_readiness"],
        repository_root,
        paper_id,
        program["policy_id"],
    )
    _validate_evaluation_evidence(program["evaluation_evidence"], repository_root)
    evidence = program["baseline_evidence"]
    _require(
        evidence,
        {
            "status",
            "matrix",
            "matrix_sha256",
            "analysis_plan",
            "analysis_plan_sha256",
            "preflight",
            "preflight_sha256",
            "preflight_status",
            "cuda_decode_contract",
            "cuda_decode_contract_sha256",
            "cuda_decode_result",
            "cuda_decode_result_sha256",
            "trained_decode_contract",
            "trained_decode_contract_sha256",
            "trained_decode_result",
            "trained_decode_result_sha256",
            "runtime_preflight_classification",
            "cache_equivalence_v2_contract",
            "cache_equivalence_v2_contract_sha256",
            "cache_equivalence_v2_control_lock",
            "cache_equivalence_v2_control_lock_sha256",
            "cache_equivalence_v2_analysis",
            "cache_equivalence_v2_analysis_sha256",
            "cache_equivalence_v2_status",
            "cache_equivalence_v3_contract",
            "cache_equivalence_v3_contract_sha256",
            "cache_equivalence_v3_analysis",
            "cache_equivalence_v3_analysis_sha256",
            "cache_equivalence_v3_status",
            "preflight_v2",
            "preflight_v2_sha256",
            "preflight_v2_status",
            "materialization",
            "materialization_sha256",
            "audit",
            "audit_sha256",
            "storage_qualification",
            "storage_qualification_sha256",
            "storage_qualification_status",
            "collection_correction",
            "collection_correction_sha256",
            "collection_correction_status",
            "automation_contract",
            "automation_contract_sha256",
            "automation_contract_status",
            "proxy_disposition_contract",
            "proxy_disposition_contract_sha256",
            "proxy_disposition_contract_status",
            "dense_control_results",
            "runner_revision",
        },
        "paper baseline evidence",
    )
    if evidence["status"] != "historical_evidence_qualified_proxy_launch_blocked":
        raise ValueError("paper baseline evidence cannot claim an unexecuted launch")
    for path_key, hash_key in (
        ("matrix", "matrix_sha256"),
        ("analysis_plan", "analysis_plan_sha256"),
        ("preflight", "preflight_sha256"),
        ("cuda_decode_contract", "cuda_decode_contract_sha256"),
        ("cuda_decode_result", "cuda_decode_result_sha256"),
        ("trained_decode_contract", "trained_decode_contract_sha256"),
        ("trained_decode_result", "trained_decode_result_sha256"),
        ("cache_equivalence_v2_contract", "cache_equivalence_v2_contract_sha256"),
        (
            "cache_equivalence_v2_control_lock",
            "cache_equivalence_v2_control_lock_sha256",
        ),
        ("cache_equivalence_v2_analysis", "cache_equivalence_v2_analysis_sha256"),
        ("cache_equivalence_v3_contract", "cache_equivalence_v3_contract_sha256"),
        ("cache_equivalence_v3_analysis", "cache_equivalence_v3_analysis_sha256"),
        ("preflight_v2", "preflight_v2_sha256"),
        ("materialization", "materialization_sha256"),
        ("audit", "audit_sha256"),
        ("storage_qualification", "storage_qualification_sha256"),
        ("collection_correction", "collection_correction_sha256"),
        ("automation_contract", "automation_contract_sha256"),
        ("proxy_disposition_contract", "proxy_disposition_contract_sha256"),
    ):
        path = repository_root / evidence[path_key]
        if not path.is_file() or _file_sha256(path) != evidence[hash_key]:
            raise ValueError(f"paper baseline {path_key} does not match its pin")
    audit = _load_json(repository_root / evidence["audit"])
    if (
        audit.get("format") != "speck_paper_baseline_audit"
        or audit.get("status") != evidence["status"]
        or audit.get("paper_id") != paper_id
        or audit.get("runner_revision") != evidence["runner_revision"]
        or audit.get("contract_sha256") != evidence["matrix_sha256"]
    ):
        raise ValueError("paper baseline audit is invalid")
    storage_qualification = _load_json(repository_root / evidence["storage_qualification"])
    baseline_matrix = _load_json(repository_root / evidence["matrix"])
    storage_runs = storage_qualification.get("operational_binding", {}).get("runs", ())
    checkpoint_directories = [run.get("checkpoint_directory") for run in storage_runs]
    if (
        storage_qualification.get("format")
        != "speck_paper_baseline_storage_qualification"
        or storage_qualification.get("status") != evidence["storage_qualification_status"]
        or storage_qualification.get("matrix", {}).get("path")
        != str((repository_root / evidence["matrix"]).resolve())
        or storage_qualification.get("capacity", {}).get("proxy_floor_passed") is not True
        or storage_qualification.get("capacity", {}).get("finalist_floor_passed") is not True
        or storage_qualification.get("provenance", {}).get(
            "existing_checkpoint_or_optimizer_artifacts_moved"
        )
        != 0
        or storage_qualification.get("provenance", {}).get(
            "existing_checkpoint_or_optimizer_artifacts_deleted"
        )
        != 0
        or storage_qualification.get("provenance", {}).get("cleanup_counted_as_capacity")
        is not False
        or storage_qualification.get("operational_binding", {}).get(
            "scientific_config_changed"
        )
        is not False
        or len(storage_runs)
        != baseline_matrix["storage_contract"]["proxy_confirmation_model_runs"]
        or len(set(checkpoint_directories)) != len(checkpoint_directories)
    ):
        raise ValueError("paper baseline storage qualification is invalid")
    preflight = _load_json(repository_root / evidence["preflight"])
    if (
        preflight.get("format") != "speck_paper_baseline_preflight"
        or preflight.get("format_version") != 1
        or preflight.get("status") != evidence["preflight_status"]
        or preflight.get("matrix_sha256") != evidence["matrix_sha256"]
    ):
        raise ValueError("paper baseline preflight is invalid")
    decode_contract = _load_json(repository_root / evidence["cuda_decode_contract"])
    decode_result = _load_json(repository_root / evidence["cuda_decode_result"])
    trained_contract = _load_json(repository_root / evidence["trained_decode_contract"])
    trained_result = _load_json(repository_root / evidence["trained_decode_result"])
    if (
        decode_contract.get("format") != "speck_cuda_decode_diagnostic_contract"
        or decode_contract.get("trigger_result_sha256") != evidence["preflight_sha256"]
        or decode_result.get("format") != "speck_cuda_decode_diagnostic"
        or decode_result.get("status") != "complete_failure_classification"
        or decode_result.get("contract_sha256") != evidence["cuda_decode_contract_sha256"]
        or trained_contract.get("format") != "speck_cuda_decode_trained_sentinel_contract"
        or trained_contract.get("trigger_result_sha256") != evidence["cuda_decode_result_sha256"]
        or trained_result.get("format") != "speck_cuda_decode_trained_sentinel"
        or trained_result.get("status") != "complete"
        or trained_result.get("contract_sha256") != evidence["trained_decode_contract_sha256"]
    ):
        raise ValueError("paper baseline CUDA decode diagnosis is invalid")
    cache_contract = _load_json(repository_root / evidence["cache_equivalence_v2_contract"])
    cache_lock = _load_json(repository_root / evidence["cache_equivalence_v2_control_lock"])
    cache_analysis = _load_json(repository_root / evidence["cache_equivalence_v2_analysis"])
    if (
        cache_contract.get("format") != "speck_cache_equivalence_contract"
        or cache_contract.get("format_version") != 2
        or cache_lock.get("format") != "speck_cache_equivalence_control_lock"
        or cache_lock.get("contract_sha256") != evidence["cache_equivalence_v2_contract_sha256"]
        or cache_analysis.get("format") != "speck_cache_equivalence_analysis"
        or cache_analysis.get("status") != evidence["cache_equivalence_v2_status"]
        or cache_analysis.get("contract_sha256") != evidence["cache_equivalence_v2_contract_sha256"]
        or cache_analysis.get("control_lock", {}).get("sha256")
        != evidence["cache_equivalence_v2_control_lock_sha256"]
    ):
        raise ValueError("paper baseline cache-equivalence v2 evidence is invalid")
    cache_v3_contract = _load_json(repository_root / evidence["cache_equivalence_v3_contract"])
    cache_v3_analysis = _load_json(repository_root / evidence["cache_equivalence_v3_analysis"])
    preflight_v2 = _load_json(repository_root / evidence["preflight_v2"])
    if (
        cache_v3_contract.get("format") != "speck_cache_equivalence_contract"
        or cache_v3_contract.get("format_version") != 3
        or cache_v3_analysis.get("format") != "speck_cache_equivalence_analysis"
        or cache_v3_analysis.get("format_version") != 3
        or cache_v3_analysis.get("status") != evidence["cache_equivalence_v3_status"]
        or cache_v3_analysis.get("contract_sha256")
        != evidence["cache_equivalence_v3_contract_sha256"]
        or not all(decision.get("passed") for decision in cache_v3_analysis.get("decisions", ()))
        or preflight_v2.get("format") != "speck_paper_baseline_preflight"
        or preflight_v2.get("format_version") != 2
        or preflight_v2.get("status") != evidence["preflight_v2_status"]
        or preflight_v2.get("prerequisites", {}).get("cache_equivalence_v3", {}).get("sha256")
        != evidence["cache_equivalence_v3_analysis_sha256"]
    ):
        raise ValueError("paper baseline cache-equivalence v3/preflight evidence is invalid")
    collection = _load_json(repository_root / evidence["collection_correction"])
    correction = collection.get("correction", {})
    trigger_checkpoint = collection.get("trigger", {}).get("checkpoint", {})
    if (
        collection.get("format") != "speck_paper_baseline_collection_correction"
        or collection.get("status") != evidence["collection_correction_status"]
        or collection.get("predecessor", {}).get("analysis_plan_sha256")
        != evidence["analysis_plan_sha256"]
        or correction.get("tokens_per_validation_step") != 16_384
        or correction.get("intermediate", {}).get("requested_tokens") != 5_000_000
        or correction.get("intermediate", {}).get("evaluated_tokens") != 4_997_120
        or correction.get("final", {}).get("requested_tokens") != 20_000_000
        or correction.get("final", {}).get("evaluated_tokens") != 19_988_480
        or collection.get("decision", {}).get("retain_completed_control") is not True
        or collection.get("decision", {}).get("rerun_control") is not False
        or collection.get("decision", {}).get("candidate_access_changed") is not False
        or collection.get("implementation", {}).get("collector_sha256")
        != _file_sha256(repository_root / "speck/paper_baseline_analysis.py")
        or collection.get("implementation", {}).get("tests_sha256")
        != _file_sha256(repository_root / "tests/test_paper_baseline_analysis.py")
    ):
        raise ValueError("paper baseline collection correction evidence is invalid")
    control_entries = evidence["dense_control_results"]
    if (
        not 1 <= len(control_entries) <= 3
        or [entry.get("pair") for entry in control_entries] != list(range(len(control_entries)))
    ):
        raise ValueError("paper dense-control result sequence is invalid")
    controls = []
    for entry in control_entries:
        path = repository_root / entry.get("path", "")
        if not path.is_file() or _file_sha256(path) != entry.get("sha256"):
            raise ValueError("paper dense-control result does not match its pin")
        control = _load_json(path)
        expected_pair = baseline_matrix["planned_primary_baselines"][
            "proxy_confirmation_pairs"
        ][entry["pair"]]
        if (
            control.get("format") != "speck_paper_baseline_run_result"
            or control.get("status") != entry.get("status")
            or control.get("arm_id") != "dense_global_param_match"
            or control.get("pair") != expected_pair
            or control.get("training_tokens") != 131_072_000
            or control.get("final_validation", {}).get("validation_tokens") != 19_988_480
            or control.get("non_finite_steps") != 0
        ):
            raise ValueError("paper dense-control result evidence is invalid")
        controls.append(control)
    automation = _load_json(repository_root / evidence["automation_contract"])
    automation_inputs = automation.get("inputs", {})
    if (
        automation.get("format") != "speck_paper_baseline_automation_contract"
        or automation.get("status") != evidence["automation_contract_status"]
        or automation_inputs.get("analysis_plan_sha256") != evidence["analysis_plan_sha256"]
        or automation_inputs.get("storage_qualification_sha256")
        != evidence["storage_qualification_sha256"]
        or automation_inputs.get("collection_correction_sha256")
        != evidence["collection_correction_sha256"]
        or automation_inputs.get("qualified_dense_controls")
        != [
            {"pair": entry["pair"], "sha256": entry["sha256"]}
            for entry in control_entries[:2]
        ]
        or automation_inputs.get("candidate_checkpoint_directories_present") != 0
        or automation_inputs.get("candidate_result_records_present") != 0
        or automation.get("event_contract", {}).get("polling") is not False
        or automation.get("decision_contract", {}).get("quality_dependent_branching")
        is not False
        or automation.get("decision_contract", {}).get(
            "interim_efficacy_or_futility_looks"
        )
        != 0
        or automation.get("implementation", {}).get("runner_sha256")
        != _file_sha256(repository_root / "scripts/paper_baseline_continue.py")
        or automation.get("implementation", {}).get("tests_sha256")
        != _file_sha256(repository_root / "tests/test_paper_baseline_continue.py")
    ):
        raise ValueError("paper baseline automation contract is invalid")
    disposition = _load_json(repository_root / evidence["proxy_disposition_contract"])
    disposition_inputs = disposition.get("inputs", {})
    disposition_branches = {
        branch.get("id"): branch for branch in disposition.get("branches", ())
    }
    if (
        disposition.get("format") != "speck_paper_proxy_disposition_contract"
        or disposition.get("status") != evidence["proxy_disposition_contract_status"]
        or disposition_inputs.get("analysis_plan_sha256") != evidence["analysis_plan_sha256"]
        or disposition_inputs.get("automation_contract_sha256")
        != evidence["automation_contract_sha256"]
        or disposition_inputs.get("control_result_sha256")
        != [entry["sha256"] for entry in control_entries]
        or disposition_inputs.get("candidate_checkpoint_directories_present") != 0
        or disposition_inputs.get("candidate_result_records_present") != 0
        or set(disposition_branches)
        != {
            "integrity_incomplete",
            "aggregate_noninferiority_failed",
            "source_guardrail_failed",
            "quality_screen_passed",
        }
        or disposition_branches["quality_screen_passed"].get(
            "finalist_materialization_authorized"
        )
        is not True
        or any(
            branch.get("finalist_materialization_authorized") is not False
            for identifier, branch in disposition_branches.items()
            if identifier != "quality_screen_passed"
        )
        or len(disposition.get("forbidden_after_any_branch", ())) < 5
        or len(disposition.get("finalist_preconditions_if_eligible", ())) < 6
    ):
        raise ValueError("paper proxy disposition contract is invalid")
    if (
        controls[0].get("checkpoint", {}).get("model_sha256")
        != trigger_checkpoint.get("model_sha256")
        or controls[0].get("checkpoint", {}).get("optimizer_sha256")
        != trigger_checkpoint.get("optimizer_sha256")
        or controls[0].get("checkpoint", {}).get("metadata_sha256")
        != trigger_checkpoint.get("metadata_sha256")
    ):
        raise ValueError("paper dense-control zero does not match the correction trigger")
    target_reference = evidence.get("time_to_quality_target")
    candidate_entries = evidence.get("candidate_results", [])
    analysis_reference = evidence.get("proxy_analysis_result")
    if len(controls) < 3:
        if target_reference is not None or candidate_entries or analysis_reference is not None:
            raise ValueError("paper target lock requires all three dense controls")
    else:
        if not isinstance(target_reference, dict):
            raise ValueError("paper target lock is missing after three dense controls")
        target_path = repository_root / target_reference.get("path", "")
        if not target_path.is_file() or _file_sha256(target_path) != target_reference.get(
            "sha256"
        ):
            raise ValueError("paper target lock does not match its pin")
        target = _load_json(target_path)
        expected_target = math.ceil(
            max(control["final_validation"]["validation_loss"] for control in controls)
            * 1_000_000
        ) / 1_000_000
        if (
            target.get("format") != "speck_paper_baseline_time_to_quality_lock"
            or target.get("status") != target_reference.get("status")
            or target.get("analysis_plan_sha256") != evidence["analysis_plan_sha256"]
            or target.get("baseline_matrix_sha256") != evidence["matrix_sha256"]
            or target.get("validation_loss_target") != expected_target
            or [entry.get("sha256") for entry in target.get("control_results", ())]
            != [entry["sha256"] for entry in control_entries]
        ):
            raise ValueError("paper time-to-quality target evidence is invalid")
        if (
            disposition_inputs.get("target_lock_sha256") != target_reference["sha256"]
            or disposition_inputs.get("target_validation_loss")
            != target["validation_loss_target"]
        ):
            raise ValueError("paper proxy disposition does not match the target lock")

        if (
            len(candidate_entries) > 3
            or [entry.get("pair") for entry in candidate_entries]
            != list(range(len(candidate_entries)))
        ):
            raise ValueError("paper candidate result sequence is invalid")
        candidates = []
        for entry in candidate_entries:
            path = repository_root / entry.get("path", "")
            if not path.is_file() or _file_sha256(path) != entry.get("sha256"):
                raise ValueError("paper candidate result does not match its pin")
            candidate = _load_json(path)
            expected_pair = baseline_matrix["planned_primary_baselines"][
                "proxy_confirmation_pairs"
            ][entry["pair"]]
            if (
                candidate.get("format") != "speck_paper_baseline_run_result"
                or candidate.get("status") != entry.get("status")
                or candidate.get("arm_id") != "five_cache_kda_gqa"
                or candidate.get("pair") != expected_pair
                or candidate.get("training_tokens") != 131_072_000
                or candidate.get("final_validation", {}).get("validation_tokens")
                != 19_988_480
                or candidate.get("non_finite_steps") != 0
            ):
                raise ValueError("paper candidate result evidence is invalid")
            candidates.append(candidate)
        if len(candidates) < 3:
            if analysis_reference is not None:
                raise ValueError("paper proxy analysis requires all three candidates")
        else:
            if not isinstance(analysis_reference, dict):
                raise ValueError("paper proxy analysis is missing after all candidate runs")
            analysis_path = repository_root / analysis_reference.get("path", "")
            if not analysis_path.is_file() or _file_sha256(
                analysis_path
            ) != analysis_reference.get("sha256"):
                raise ValueError("paper proxy analysis does not match its pin")
            proxy_analysis = _load_json(analysis_path)
            if (
                proxy_analysis.get("format") != "speck_paper_baseline_analysis"
                or proxy_analysis.get("status") != analysis_reference.get("status")
                or proxy_analysis.get("analysis_plan_sha256")
                != evidence["analysis_plan_sha256"]
                or proxy_analysis.get("baseline_matrix_sha256") != evidence["matrix_sha256"]
                or proxy_analysis.get("time_to_quality_lock", {}).get("sha256")
                != target_reference["sha256"]
                or len(proxy_analysis.get("paired_results", ())) != 3
            ):
                raise ValueError("paper proxy analysis evidence is invalid")
    _validate_proxy_launch_evidence(program, repository_root)
    policy = repository_root / "research" / program["policy_id"] / "policy.json"
    if not policy.is_file():
        raise ValueError("paper experiment program references a missing promotion policy")
    controls = program["controls"]
    _require(
        controls, {"primary_conventional", "primary_hybrid", "mechanism_controls"}, "paper controls"
    )
    if len(program["matching_views"]) < 5:
        raise ValueError("paper program does not preserve enough matching views")

    scale_ids = _unique_ids(program["scales"], "paper scale")
    required_scales = {
        "mechanism",
        "proxy_discovery",
        "proxy_finalist",
        "medium_transfer",
        "target_sentinel",
        "paper_scale",
    }
    if scale_ids != required_scales:
        raise ValueError("paper program scale ladder is incomplete")
    paper_scale = next(scale for scale in program["scales"] if scale["id"] == "paper_scale")
    if (
        paper_scale.get("active_parameters") is not None
        or paper_scale.get("training_tokens") is not None
    ):
        raise ValueError("paper-scale geometry cannot be selected before the launch gate")

    axes = program["axes"]
    axis_ids = _unique_ids(axes, "paper axis")
    if axis_ids != {"sequence", "depth", "width"}:
        raise ValueError("paper program must isolate sequence, depth, and width")
    referenced_claims = set()
    for axis in axes:
        axis_claims = set(axis.get("claim_ids", ()))
        if not axis_claims or not axis_claims <= claim_ids:
            raise ValueError(f"paper axis {axis['id']} references invalid claims")
        referenced_claims |= axis_claims
        families = axis.get("ordered_families", ())
        _unique_ids(families, f"{axis['id']} experiment family")
        if len(families) < 3 or len(axis.get("required_diagnostics", ())) < 5:
            raise ValueError(f"paper axis {axis['id']} lacks experiments or diagnostics")
        for family in families:
            _require(
                family, {"id", "fixed", "arms", "decision"}, f"experiment family {family.get('id')}"
            )
            if len(family["arms"]) < 2:
                raise ValueError(f"experiment family {family['id']} needs at least two arms")
    if not {"C1", "C2", "C3", "C4", "C6"} <= referenced_claims:
        raise ValueError("paper axes do not cover all component and systems claims")

    interaction = program["interaction_program"]
    if len(interaction.get("pairwise_cells", ())) != 3 or "2^3" not in interaction.get(
        "full_factorial_requirement", ""
    ):
        raise ValueError("paper program lacks the complete tri-axis interaction design")
    scaling = program["scaling_program"]
    if scaling.get("minimum_points_per_architecture", 0) < 5 or "C5" not in scaling.get(
        "claim_ids", ()
    ):
        raise ValueError("paper scaling claim lacks five-point evidence")
    if len(program["analysis_program"]) < 4:
        raise ValueError("paper program lacks mechanistic analysis coverage")

    gate = program["large_pretraining_gate"]
    if gate.get("status") != "blocked" or len(gate.get("required", ())) < 10:
        raise ValueError("paper-scale pretraining must remain blocked on explicit prerequisites")
    return {"scales": len(scale_ids), "axes": len(axis_ids)}


def _validate_baseline_matrix(matrix, program, paper_id, repository_root):
    _require(
        matrix,
        {
            "format",
            "format_version",
            "paper_id",
            "policy_id",
            "status",
            "historical_evidence",
            "planned_primary_baselines",
            "future_finalist_design",
            "analysis_contract",
            "storage_contract",
            "launch_gates",
        },
        "paper baseline matrix",
    )
    if matrix["format"] != "speck_paper_baseline_matrix" or matrix["format_version"] != 1:
        raise ValueError("paper baseline matrix must use format version 1")
    if matrix["paper_id"] != paper_id or matrix["policy_id"] != program["policy_id"]:
        raise ValueError("paper baseline matrix references a different paper or policy")
    if matrix["status"] != "historical_evidence_audited_new_launch_blocked":
        raise ValueError("paper baseline launch must remain blocked before paired preflight")

    historical = matrix["historical_evidence"]
    _require(
        historical,
        {"authority", "shared_expected", "source_artifacts", "arms", "known_limits"},
        "historical baseline evidence",
    )
    historical_ids = _unique_ids(historical["arms"], "historical baseline arm")
    if historical_ids != {
        "dense_global",
        "swa_2048",
        "gdn_global_silu_rope",
        "gdn_global_sigmoid_nope",
        "kda_global_sigmoid_nope",
    }:
        raise ValueError("historical baseline inventory is incomplete")
    required_arm = {
        "id",
        "role",
        "experiment",
        "checkpoint_run",
        "parameters",
        "flops_per_token_at_4096",
        "validation_loss",
        "mixer_counts",
    }
    for arm in historical["arms"]:
        _require(arm, required_arm, f"historical baseline {arm.get('id')}")
        experiment = repository_root / arm["experiment"]
        if not experiment.is_dir() or any(
            not (experiment / f"{name}.json").is_file()
            for name in ("data", "model", "tokenizer", "train")
        ):
            raise ValueError(f"historical baseline {arm['id']} experiment is missing")
        if arm["parameters"] < 130_000_000 or arm["parameters"] > 170_000_000:
            raise ValueError(f"historical baseline {arm['id']} is outside the proxy scale")
    for artifact in historical["source_artifacts"]:
        _require(artifact, {"path", "sha256"}, "historical baseline source artifact")
        path = repository_root / artifact["path"]
        if not path.is_file() or _file_sha256(path) != artifact["sha256"]:
            raise ValueError("historical baseline source artifact does not match its pin")

    planned = matrix["planned_primary_baselines"]
    _require(
        planned,
        {
            "family_id",
            "output_root",
            "scientific_scope",
            "arms",
            "parameter_matching",
            "shared_training",
            "proxy_confirmation_pairs",
            "matching_views",
            "decision_rule",
        },
        "planned primary baselines",
    )
    planned_ids = _unique_ids(planned["arms"], "planned baseline arm")
    if planned_ids != {"dense_global_param_match", "five_cache_kda_gqa"}:
        raise ValueError("planned primary baseline arms are incomplete")
    control_ids = {arm["control_id"] for arm in planned["arms"]}
    if control_ids != {
        program["controls"]["primary_conventional"]["id"],
        program["controls"]["primary_hybrid"]["id"],
    }:
        raise ValueError("planned baseline arms do not implement the paper controls")
    for arm in planned["arms"]:
        _require(
            arm,
            {
                "id",
                "control_id",
                "template_experiment",
                "transform",
                "parameters",
                "flops_per_token_at_4096",
                "device_batch_size",
            },
            f"planned baseline {arm.get('id')}",
        )
        if not (repository_root / arm["template_experiment"]).is_dir():
            raise ValueError(f"planned baseline {arm['id']} template is missing")
    parameters = [arm["parameters"] for arm in planned["arms"]]
    relative = (max(parameters) - min(parameters)) / min(parameters)
    matching = planned["parameter_matching"]
    if (
        relative > matching.get("maximum_relative_difference", 0)
        or abs(relative - matching.get("actual_relative_difference", -1)) > 1e-15
        or max(parameters) - min(parameters) != matching.get("actual_parameter_difference")
    ):
        raise ValueError("planned primary baseline parameter match is invalid")
    training = planned["shared_training"]
    batch_tokens = training.get("batch_tokens")
    training_tokens = training.get("training_tokens")
    if (
        batch_tokens != 65_536
        or training_tokens != 131_072_000
        or training_tokens % batch_tokens
        or training.get("sequence_length") != 4_096
    ):
        raise ValueError("planned baseline training geometry is invalid")
    pairs = planned["proxy_confirmation_pairs"]
    if [pair.get("seed") for pair in pairs] != [42, 43, 44]:
        raise ValueError("planned baseline confirmation requires seeds 42, 43, and 44")
    offsets = [pair.get("data_token_offset") for pair in pairs]
    if (
        len(set(offsets)) != 3
        or any(
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or offset % batch_tokens
            for offset in offsets
        )
        or any(left + training_tokens > right for left, right in zip(offsets, offsets[1:]))
    ):
        raise ValueError("planned baseline packed-data windows must be aligned and disjoint")
    compute = planned["matching_views"].get("compute_matched", {})
    if (
        compute.get("reference_tokens") != training_tokens
        or compute.get("dense_global_tokens", 0) % batch_tokens
        or not 0 < compute.get("dense_global_tokens", 0) < training_tokens
    ):
        raise ValueError("planned baseline compute-matched view is invalid")

    analysis_contract = matrix["analysis_contract"]
    analysis_path = repository_root / analysis_contract.get("path", "")
    evidence = program["baseline_evidence"]
    if (
        analysis_contract.get("status") != "frozen_before_results"
        or analysis_contract.get("path") != evidence.get("analysis_plan")
        or not analysis_path.is_file()
        or _file_sha256(analysis_path) != evidence.get("analysis_plan_sha256")
    ):
        raise ValueError("paper baseline analysis contract does not match its pin")
    analysis = _load_json(analysis_path)
    if (
        analysis.get("format") != "speck_paper_baseline_analysis_plan"
        or analysis.get("format_version") != 1
        or analysis.get("status") != "frozen_before_results"
        or analysis.get("paper_id") != paper_id
        or analysis.get("policy_id") != program["policy_id"]
        or analysis.get("baseline_matrix_sha256") != evidence.get("matrix_sha256")
        or analysis.get("pairs") != pairs
        or analysis.get("stopping_rule", {}).get("required_complete_model_runs") != 2 * len(pairs)
        or analysis.get("stopping_rule", {}).get("interim_efficacy_looks") != 0
        or analysis.get("stopping_rule", {}).get("interim_futility_looks") != 0
    ):
        raise ValueError("paper baseline analysis contract is invalid")

    finalist = matrix["future_finalist_design"]
    if (
        finalist.get("status") != "not_materialized_until_proxy_confirmation"
        or finalist.get("seeds") != [42, 43, 44]
        or len(finalist.get("data_token_offsets", ())) != 2
        or finalist.get("paired_runs") != 6
        or finalist.get("total_model_runs") != 12
        or finalist.get("training_tokens_per_run", 0) < max(parameters) * 10
    ):
        raise ValueError("future finalist baseline design is invalid")
    storage = matrix["storage_contract"]
    if (
        storage.get("proxy_confirmation_model_runs") != 2 * len(pairs)
        or storage.get("future_finalist_model_runs") != finalist["total_model_runs"]
        or storage.get("minimum_free_bytes_before_proxy_launch", 0)
        < storage.get("estimated_proxy_checkpoint_bytes", 0)
        or storage.get("minimum_free_bytes_before_finalist_launch", 0)
        < storage.get("estimated_finalist_checkpoint_bytes", 0)
    ):
        raise ValueError("paper baseline storage contract is insufficient")
    if len(matrix["launch_gates"]) < 6:
        raise ValueError("paper baseline launch gates are incomplete")
    return {
        "historical_arms": len(historical_ids),
        "planned_arms": len(planned_ids),
        "proxy_pairs": len(pairs),
    }


def _validate_markdown(directory):
    required_headings = {
        "README.md": ("## Working thesis", "## Novelty gate", "## Current state"),
        "paper_outline.md": (
            "## 3. Speck architecture",
            "## 11. Mechanistic analysis",
            "## 14. Limitations",
        ),
        "reference_audit.md": (
            "## Structural comparison",
            "## Lessons incorporated into Speck",
            "## Standard Speck intentionally raises",
        ),
        "reporting_checklist.md": (
            "## Claim integrity",
            "## Correctness and systems",
            "## Reproducibility and release",
        ),
    }
    for name, headings in required_headings.items():
        content = (directory / name).read_text(encoding="utf-8")
        missing = [heading for heading in headings if heading not in content]
        if missing:
            raise ValueError(f"{name} is missing required sections: {', '.join(missing)}")


def validate_paper_program(directory, repository_root=None):
    """Validate Paper 1 claims, experiment coverage, launch gate, and manuscript structure."""

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"paper program directory does not exist: {directory}")
    missing = [name for name in PROGRAM_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"paper program is missing files: {', '.join(missing)}")
    claims = _load_json(directory / "claims.json")
    claim_ids = _validate_claims(claims)
    program = _load_json(directory / "experiment_program.json")
    repository_root = (
        directory.parents[1]
        if repository_root is None
        else Path(repository_root).expanduser().resolve()
    )
    inventory = _validate_program(program, claims["paper_id"], claim_ids, repository_root)
    baseline = _validate_baseline_matrix(
        _load_json(directory / "baseline_matrix.json"),
        program,
        claims["paper_id"],
        repository_root,
    )
    _validate_markdown(directory)
    return {
        "paper_id": claims["paper_id"],
        "status": "valid_hypotheses_only",
        "program_files": list(PROGRAM_FILES),
        "claims": len(claim_ids),
        "non_claims": len(claims["non_claims"]),
        "scales": inventory["scales"],
        "axes": inventory["axes"],
        "historical_baseline_arms": baseline["historical_arms"],
        "planned_primary_baseline_arms": baseline["planned_arms"],
        "proxy_confirmation_pairs": baseline["proxy_pairs"],
        "proxy_training": "authorized",
        "paper_scale_pretraining": "blocked",
    }
