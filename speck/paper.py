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
        },
        "paper experiment program",
    )
    if program["format"] != "speck_paper_experiment_program" or program["format_version"] != 1:
        raise ValueError("paper experiment program must use format version 1")
    if program["paper_id"] != paper_id:
        raise ValueError("claims and experiment program use different paper ids")
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
