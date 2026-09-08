import hashlib
import json
from pathlib import Path

import torch

from speck.architecture import AttentionSpec, KimiDeltaAttentionSpec, SwiGLUSpec
from speck.config import load_experiment
from speck.model import build_model

ROOT = Path(__file__).parents[1]
FLAGSHIP = ROOT / "research" / "flagship"
SHAPE_A = FLAGSHIP / "targets" / "shape-a"


def test_execution_plan_balances_the_grant_and_keeps_reserve_conditional():
    plan = json.loads((FLAGSHIP / "plan.json").read_text())
    phases = plan["phases"]

    assert plan["format"] == "speck_flagship_execution_plan"
    assert plan["status"] == "pregrant"
    assert sum(phase["gpu_hours"] for phase in phases) == 5_000
    assert sum(phase["gpu_hours"] for phase in phases if not phase.get("conditional")) == 4_111
    reserve = next(phase for phase in phases if phase["id"] == "P7")
    assert reserve["gpu_hours"] == plan["budget"]["reserve_gpu_hours"] == 889
    assert reserve["conditional"] is True
    assert "mixture of experts" in plan["scope"]["excluded"]
    assert "dense-width flagship" in plan["flexibility"]["binding"]


def test_data_plan_has_a_complete_english_mixture_and_matches_execution_budget():
    data_plan = json.loads((FLAGSHIP / "data_plan.json").read_text())
    execution_plan = json.loads((FLAGSHIP / "plan.json").read_text())
    categories = data_plan["categories"]
    experiments = data_plan["experiments"]

    assert data_plan["format"] == "speck_flagship_data_plan"
    assert data_plan["language_scope"]["natural_language"] == ["en"]
    assert [category["id"] for category in categories] == [
        "web",
        "code",
        "math",
        "synthetic",
        "science",
        "reference",
    ]
    assert sum(category["prior_percent"] for category in categories) == 100
    assert all(
        category["min_percent"] <= category["prior_percent"] <= category["max_percent"]
        for category in categories
    )
    assert sum(experiment["new_runs"] for experiment in experiments) == 80
    assert sum(experiment["gpu_hours"] for experiment in experiments) == 593
    assert data_plan["totals"] == {"new_runs": 80, "gpu_hours": 593}

    phase_hours = {phase["id"]: phase["gpu_hours"] for phase in execution_plan["phases"]}
    assert phase_hours["P1"] == 230
    assert phase_hours["P2"] == 566
    assert execution_plan["budget"]["mandatory_gpu_hours"] == 4_111
    assert execution_plan["budget"]["reserve_gpu_hours"] == 889


def test_data_plan_preserves_selection_firewall_and_replication():
    data_plan = json.loads((FLAGSHIP / "data_plan.json").read_text())
    experiments = {experiment["id"]: experiment for experiment in data_plan["experiments"]}
    selection = data_plan["selection"]

    assert data_plan["partitions"] == [
        "tokenizer_sample",
        "selection_heldout",
        "sealed_audit",
    ]
    assert selection["primary_unit"] == "bits_per_utf8_byte"
    assert selection["e2a_advance"] == experiments["E2b"]["new_runs"] == 6
    assert (
        selection["e2b_advance"] * selection["final_confirmation_seeds"]
        == experiments["E2c"]["new_runs"]
    )
    assert experiments["E5"]["reused_control_runs"] == 2
    assert data_plan["firewall_contract"] == {
        "plan": "research/flagship/firewall_plan.json",
        "status": "fixture_tooling_ready_real_targets_and_authority_pending",
        "real_materialization": "blocked",
    }


def test_architecture_plan_closes_causal_gaps_and_matches_execution_budget():
    architecture_plan = json.loads((FLAGSHIP / "architecture_plan.json").read_text())
    execution_plan = json.loads((FLAGSHIP / "plan.json").read_text())
    decisions = {decision["id"]: decision for decision in architecture_plan["decisions"]}
    scale_program = architecture_plan["scale_program"]
    totals = architecture_plan["totals"]

    assert architecture_plan["format"] == "speck_flagship_architecture_plan"
    assert set(decisions) == {"C0", "D2", "D3", "D4", "D6", "D7", "D8"}
    assert sum(decision["new_runs"] for decision in decisions.values()) == 21
    assert sum(decision["gpu_hours"] for decision in decisions.values()) == 353
    assert sum(stage["new_runs"] for stage in scale_program) == 14
    assert sum(stage["gpu_hours"] for stage in scale_program) == 290
    assert totals["architecture_new_runs"] == 35
    assert totals["architecture_gpu_hours"] == 643
    assert decisions["D7"]["reuses_control"] == "C0"
    assert decisions["D8"]["reuses_control"] == "C0"
    assert decisions["D8"]["requires_pregrant_implementation"] is True

    phases = {phase["id"]: phase["gpu_hours"] for phase in execution_plan["phases"]}
    assert phases["P2"] == 566
    assert execution_plan["budget"]["mandatory_gpu_hours"] == 4_111
    assert execution_plan["budget"]["reserve_gpu_hours"] == 889
    assert sum(phases[phase] for phase in ("P1", "P2", "P3")) == (
        593 + totals["architecture_gpu_hours"]
    )


def test_architecture_plan_inherits_promotion_margins_and_separates_systems_outcomes():
    architecture_plan = json.loads((FLAGSHIP / "architecture_plan.json").read_text())
    promotion_policy = json.loads(
        (ROOT / "research" / "architecture-promotion-v1" / "policy.json").read_text()
    )
    promotion = architecture_plan["promotion"]
    statistical_policy = promotion_policy["statistical_contract"]
    language_policy = statistical_policy["language_loss"]
    systems_policy = statistical_policy["systems"]

    assert (
        promotion["aggregate_non_inferiority_margin_nats"]
        == language_policy["default_non_inferiority_margin_nats"]
    )
    assert promotion["source_guardrail_nats"] == language_policy["source_guardrail_nats"]
    assert (
        promotion["simple_component_cost_improvement_percent"]
        == 100 * systems_policy["simple_component_minimum_primary_improvement"]
    )
    assert (
        promotion["custom_runtime_cost_improvement_percent"]
        == 100 * systems_policy["custom_runtime_component_minimum_primary_improvement"]
    )
    assert (
        promotion["state_reduction_threshold_percent"]
        == 100 * systems_policy["minimum_state_reduction_for_memory_claim"]
    )
    assert promotion["paired_seeds_for_launch_decisions"] == 3
    assert promotion["tie_rule"] == "keep_default"
    assert architecture_plan["systems"]["minimum_interleaved_blocks"] >= 5
    assert set(architecture_plan["systems"]["separate_outcomes"]) == {
        "analytic_flops_and_state",
        "wall_clock",
        "energy",
        "peak_memory",
    }


def test_source_registry_is_pinned_and_tokenizer_allocations_cover_every_category():
    registry = json.loads((FLAGSHIP / "source_registry.json").read_text())
    tokenizer_plan = json.loads((FLAGSHIP / "tokenizer_plan.json").read_text())
    sources = {source["id"]: source for source in registry["sources"]}

    assert registry["format"] == "speck_flagship_source_registry"
    assert registry["status"] == "candidate_registry_not_training_authority"
    assert len(sources) == len(registry["sources"]) == 38
    assert all(
        len(source["revision"]) == 40
        and all(character in "0123456789abcdef" for character in source["revision"])
        for source in sources.values()
    )
    assert all(source["official_url"].startswith("https://") for source in sources.values())

    stack_v3 = sources["stack_v3_train_permissive"]
    stack_edu = sources["stack_edu"]
    common_pile_code = sources["common_pile_stackv2_edu"]
    python_edu = sources["python_edu"]
    python_peps = sources["common_pile_python_peps"]
    assert stack_v3["repo"] == "HuggingFaceCode/stack-v3-train"
    assert stack_v3["priority"] == "primary_screen"
    assert "license_type=permissive" in stack_v3["subset"]
    assert (
        stack_v3["pipeline"]
        == "security_language_license_contamination_partition_and_five_source_overlap_pass_training_blocked"
    )
    assert (
        stack_edu["pipeline"]
        == "bounded_swh_security_contamination_partition_and_five_source_overlap_pass_training_blocked"
    )
    assert (
        common_pile_code["pipeline"]
        == "bounded_inline_security_contamination_partition_and_five_source_overlap_pass_training_blocked"
    )
    assert "five_source_overlap_pass" in python_edu["pipeline"]
    assert python_edu["pipeline"].endswith("rights_blocked")
    assert "five_source_overlap_pass" in python_peps["pipeline"]
    for source_id in (
        "ultrafineweb_en_v1_4",
        "fineweb_edu",
        "dclm_baseline",
        "fineweb_base",
    ):
        assert sources[source_id]["pipeline"].endswith("contamination_pass_rights_blocked")
    for source_id in (
        "finemath_4plus",
        "infiwebmath_4plus",
        "megamath_web_pro",
        "openwebmath",
        "proof_pile_2_algebraic_stack",
        "megamath_code",
    ):
        assert sources[source_id]["pipeline"].endswith("rights_blocked")
    for source_id in (
        "cosmopedia_v2",
        "ultrafineweb_l3_multi_style",
        "ultrafineweb_l3_qa",
        "megamath_qa",
    ):
        assert sources[source_id]["pipeline"].endswith("rights_blocked")
    for source_id in (
        "pes2o_v3",
        "finepdfs_edu_en",
        "common_pile_arxiv",
        "common_pile_pubmed",
        "proof_pile_2_arxiv",
    ):
        assert sources[source_id]["pipeline"].endswith("rights_blocked")
    for source_id in (
        "finewiki_en",
        "common_pile_stackexchange",
        "common_pile_gutenberg",
        "common_pile_libretexts",
        "common_pile_oercommons",
        "common_pile_pressbooks",
    ):
        assert sources[source_id]["pipeline"].endswith("rights_blocked")

    totals = {
        category: {"training_bytes": 0, "evaluation_bytes": 0}
        for category in registry["categories"]
    }
    for allocation in registry["tokenizer_sample_allocations"]:
        source = sources[allocation["source_id"]]
        assert source["priority"] != "hold_terms"
        totals[source["category"]]["training_bytes"] += allocation["training_bytes"]
        totals[source["category"]]["evaluation_bytes"] += allocation["evaluation_bytes"]

    assert set(totals) == set(tokenizer_plan["categories"])
    assert all(
        value["training_bytes"] == tokenizer_plan["sample"]["training_bytes_per_category"]
        and value["evaluation_bytes"] == tokenizer_plan["sample"]["evaluation_bytes_per_category"]
        for value in totals.values()
    )


def test_execution_plan_dependencies_are_acyclic_and_days_fit_the_grant():
    plan = json.loads((FLAGSHIP / "plan.json").read_text())
    positions = {phase["id"]: index for index, phase in enumerate(plan["phases"])}

    assert len(positions) == len(plan["phases"])
    for phase in plan["phases"]:
        assert 0 <= phase["days"][0] <= phase["days"][1] <= plan["calendar_days"]
        assert all(positions[parent] < positions[phase["id"]] for parent in phase["depends_on"])


def test_shape_a_is_current_bounded_and_deliberately_not_launchable():
    configs = load_experiment(SHAPE_A, "long_context", "model")
    target = json.loads((SHAPE_A / "target.json").read_text())
    with torch.device("meta"):
        model = build_model(configs["model"], vocab_size=32_000)

    mixers = [
        stage.branches[0]
        for invocation in model.execution_plan
        for stage in invocation.block.stages[:1]
    ]
    feed_forwards = [
        branch
        for invocation in model.execution_plan
        for stage in invocation.block.stages
        for branch in stage.branches
        if isinstance(branch, SwiGLUSpec)
    ]
    attention = [branch for branch in mixers if isinstance(branch, AttentionSpec)]

    assert model.parameter_count() == target["parameters"] == 1_195_878_432
    assert sum(isinstance(branch, KimiDeltaAttentionSpec) for branch in mixers) == 18
    assert len(attention) == 6
    assert all((branch.num_key_value_heads, branch.rope_dim) == (4, 0) for branch in attention)
    assert len(feed_forwards) == 24
    assert all(branch.intermediate_size == 5_120 for branch in feed_forwards)
    assert model.config.max_position_embeddings == 131_072
    assert configs["long_context"]["lengths"][-1] == 131_072
    assert target["status"] == "planning_default_not_launchable"
    assert not (SHAPE_A / "data.json").exists()
    assert not (SHAPE_A / "train.json").exists()


def test_storage_readiness_result_clears_the_pregrant_threshold():
    result = json.loads(
        (ROOT / "results" / "storage" / "checkpoint-relocation-20260906.json").read_text()
    )

    assert result["format"] == "speck_checkpoint_storage_relocation"
    assert result["relocated"]["bytes"] > 80_000_000_000
    assert result["verification"]["checksum_dry_run_changes_per_family"] == 0
    assert result["verification"]["latest_checkpoint_metadata_loaded_through_symlink"] is True
    assert result["after"]["root_use_percent"] < 80


def test_web_firewall_result_is_hash_bound_and_remains_non_authoritative():
    result = json.loads((ROOT / "results" / "data" / "web-contamination-20260907.json").read_text())

    assert result["status"] == (
        "bounded_web_benchmark_decontamination_pass_training_authority_blocked"
    )
    for key, identity in result["implementation"].items():
        if key == "git_pre_scan_freeze":
            continue
        path, digest = identity
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["frozen_benchmark_policy"]["payloads"] == 20
    assert result["frozen_benchmark_policy"]["tasks"] == 63_652
    assert result["runtime"]["records_removed_critical"] == 92
    assert result["runtime"]["records_retained"] == 31_000
    assert result["verification"]["residual_critical_records"] == 0
    assert all(
        source["training_partition_bytes"] >= source["training_target_bytes"]
        and source["evaluation_partition_bytes"] >= source["evaluation_target_bytes"]
        for source in result["sources"].values()
    )

    rights = json.loads((ROOT / "results" / "data" / "web-rights-review-20260907.json").read_text())
    assert rights["status"].endswith("training_authority_blocked")
    assert "no source is approved" in rights["decision"]


def test_math_qualification_result_is_hash_bound_and_remains_non_authoritative():
    result = json.loads(
        (ROOT / "results" / "data" / "math-tokenizer-sources-20260907.json").read_text()
    )

    assert (
        result["status"] == "bounded_math_technical_qualification_pass_training_authority_blocked"
    )
    for path, digest in result["implementation"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in result["configs"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["security"]["affected_records_removed"] == 1
    assert result["overlap"]["exact_cross_source_matches"] == 3
    assert result["overlap"]["verified_near_cross_source_matches"] == 7
    assert result["contamination"]["records_removed_critical"] == 980
    assert result["contamination"]["residual_critical_records_after_full_rescan"] == 0
    assert result["aggregate_final_partition"]["training_bytes"] >= 100_000_000
    assert result["aggregate_final_partition"]["evaluation_bytes"] >= 10_000_000
    assert all(
        source["training_partition_bytes"] >= source["training_target_bytes"]
        and source["evaluation_partition_bytes"] >= source["evaluation_target_bytes"]
        for source in result["sources"].values()
    )

    rights = json.loads(
        (ROOT / "results" / "data" / "math-rights-review-20260907.json").read_text()
    )
    assert rights["status"].endswith("training_authority_blocked")
    assert "no math source is approved" in rights["decision"]


def test_synthetic_qualification_is_corrected_hash_bound_and_non_authoritative():
    result = json.loads(
        (ROOT / "results" / "data" / "synthetic-tokenizer-sources-20260907.json").read_text()
    )

    assert result["status"] == (
        "bounded_synthetic_technical_qualification_pass_training_authority_blocked"
    )
    assert result["lineage_correction"]["superseded_outputs_forbidden"] is True
    assert result["lineage_correction"]["maximum_observed_successor_seed_prompt_jaccard"] < 0.8
    for path, digest in result["implementation"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in result["configs"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["security"]["affected_records_removed"] == 6
    assert result["overlap"]["exact_cross_source_matches"] == 0
    assert result["overlap"]["verified_near_cross_source_matches"] == 0
    assert result["contamination"]["records_removed_critical"] == 353
    assert result["contamination"]["residual_critical_records_after_full_rescan"] == 0
    assert result["aggregate_final_partition"]["training_bytes"] >= 100_000_000
    assert result["aggregate_final_partition"]["evaluation_bytes"] >= 10_000_000
    assert all(
        source["training_partition_bytes"] >= source["training_target_bytes"]
        and source["evaluation_partition_bytes"] >= source["evaluation_target_bytes"]
        for source in result["sources"].values()
    )

    rights = json.loads(
        (ROOT / "results" / "data" / "synthetic-rights-review-20260907.json").read_text()
    )
    assert rights["status"].endswith("training_authority_blocked")
    assert "no synthetic source is approved" in rights["decision"]


def test_science_qualification_preserves_failed_gate_and_is_non_authoritative():
    result = json.loads(
        (ROOT / "results" / "data" / "science-tokenizer-sources-20260907.json").read_text()
    )

    assert result["status"] == (
        "bounded_science_technical_qualification_pass_training_authority_blocked"
    )
    for path, digest in result["implementation"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in result["configs"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["initial_firewall_failure"]["policy_changed"] is False
    assert set(result["initial_firewall_failure"]["failed_sources"]) == {
        "common_pile_arxiv",
        "proof_pile_2_arxiv",
    }
    assert result["security"]["affected_records_removed"] == 10
    assert result["overlap"]["exact_cross_source_matches"] == 0
    assert result["overlap"]["verified_near_cross_source_matches"] == 0
    assert result["contamination"]["records_removed_critical"] == 137
    assert result["contamination"]["residual_critical_records_after_full_rescan"] == 0
    assert result["aggregate_final_partition"]["training_bytes"] >= 100_000_000
    assert result["aggregate_final_partition"]["evaluation_bytes"] >= 10_000_000
    assert all(
        source["training_partition_bytes"] >= source["training_target_bytes"]
        and source["evaluation_partition_bytes"] >= source["evaluation_target_bytes"]
        for source in result["sources"].values()
    )

    rights = json.loads(
        (ROOT / "results" / "data" / "science-rights-review-20260907.json").read_text()
    )
    assert rights["status"].endswith("training_authority_blocked")
    assert "no science source is approved" in rights["decision"]


def test_reference_qualification_preserves_failures_and_is_non_authoritative():
    result = json.loads(
        (ROOT / "results" / "data" / "reference-tokenizer-sources-20260907.json").read_text()
    )
    assert result["status"] == (
        "bounded_reference_technical_qualification_pass_training_authority_blocked"
    )
    for path, digest in result["implementation"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in result["configs"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert set(result["initial_source_failures"]) == {
        "common_pile_gutenberg",
        "common_pile_oercommons",
    }
    assert result["security"]["affected_records_removed"] == 1
    assert result["overlap"]["exact_cross_source_matches"] == 0
    assert result["overlap"]["verified_near_cross_source_matches"] == 2
    assert result["contamination"]["records_removed_critical"] == 331
    assert result["contamination"]["residual_critical_records_after_full_rescan"] == 0
    assert result["aggregate_final_partition"]["training_bytes"] >= 100_000_000
    assert result["aggregate_final_partition"]["evaluation_bytes"] >= 10_000_000
    rights = json.loads(
        (ROOT / "results" / "data" / "reference-rights-review-20260907.json").read_text()
    )
    assert rights["status"].endswith("training_authority_blocked")
    assert "no reference source is approved" in rights["decision"]


def test_three_partition_firewall_tooling_is_hash_bound_and_fixture_only():
    result = json.loads(
        (ROOT / "results" / "data" / "data-firewall-tooling-20260907.json").read_text()
    )
    assert result["status"] == (
        "fixture_qualified_real_materialization_and_training_authority_blocked"
    )
    for path, digest in result["implementation"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["fixture_validation"]["focused_tests"] == 8
    assert result["fixture_validation"]["real_qualified_source_records_read"] == 0
    assert result["fixture_validation"]["persistent_fixture_partitions_created"] == 0
    assert any("no real selection-heldout" in value for value in result["limitations"])
    assert any("no tokenizer or model training" in value for value in result["limitations"])


def test_production_data_tooling_is_hash_bound_and_does_not_claim_rehearsal():
    result = json.loads(
        (ROOT / "results" / "data" / "production-data-tooling-20260907.json").read_text()
    )
    assert result["status"] == "fixture_qualified_20B_rehearsal_and_training_authority_blocked"
    assert result["implementation"]["preprocessor"][1] == (
        "45d2ccce8b8ef1e2564116de8c5cc1f427fb93a1feb1af4fbdfcc4c4dff51431"
    )
    assert result["implementation"]["preprocessor_tests"][1] == (
        "983bd78678fe4c269905aff420131339e21650ab3d23b4cf6cc9fcb2d24f6a10"
    )
    for name, (path, digest) in result["implementation"].items():
        if name not in {"preprocessor", "preprocessor_tests"}:
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["fixture_validation"]["new_preprocessor_tests"] == 7
    assert result["fixture_validation"]["combined_preprocessor_and_packer_tests"] == 38
    assert result["fixture_validation"]["real_qualified_source_records_read"] == 0
    assert any("20B rehearsal has not run" in value for value in result["limitations"])
    assert any("no full corpus" in value for value in result["limitations"])


def test_runtime_cleanup_successor_is_hash_bound_and_non_authoritative():
    result = json.loads((ROOT / "results" / "runtime-cleanup-successor-20260908.json").read_text())
    assert result["status"] == (
        "checkpoint_recovery_and_handle_reuse_fixture_pass_gpu_and_20B_pending"
    )
    assert result["implementation"]["checkpoint"][1] == (
        "0233da9740ac7291efabb37207fad23b9bc2f867607f3cb961857db7e7564c8b"
    )
    for name, (path, digest) in result["implementation"].items():
        if name != "checkpoint":
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for lineage in result["supersedes"].values():
        assert (
            hashlib.sha256((ROOT / lineage["historical_evidence"]).read_bytes()).hexdigest()
            == (lineage["historical_evidence_sha256"])
        )
        assert lineage["historical_evidence_modified"] is False
    assert all(result["checkpoint_qualification"].values())
    assert result["production_data_qualification"]["training_authority"] == "blocked"
    assert result["formatting_qualification"]["pyproject_historical_hash_preserved"] is True
    assert any("20B" in value for value in result["limitations"])
    assert any(
        "no production operations or training authority" in value for value in result["limitations"]
    )


def test_source_rights_decision_contract_is_hash_bound_and_all_pending():
    result = json.loads(
        (ROOT / "results" / "data" / "source-rights-decision-readiness-20260907.json").read_text()
    )
    assert result["status"] == "all_selected_sources_covered_human_decisions_pending"
    for path, digest in result["implementation"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["coverage"]["selected_sources"] == 30
    assert result["coverage"]["evidence_packets"] == 6
    assert result["current_assessment"] == {
        "pending": 30,
        "approved": 0,
        "rejected": 0,
        "scope_fields_pending": 6,
        "authority_fields_pending": 4,
        "signed_at_pending": True,
        "automated_approval_made": False,
    }


def test_data_launch_gate_is_hash_bound_and_has_no_real_receipt():
    result = json.loads((ROOT / "results" / "data" / "data-launch-gate-20260907.json").read_text())
    assert result["status"] == "fixture_qualified_real_receipt_and_training_authority_blocked"
    historical = {
        "gate": "0be9672a1347ca7a452e744156a31c970731e2800534b824acb692c0b809334e",
        "training_entrypoint": "ffc9455a292c321551f3db3b526f97334de0fe62ce38802de8a458b547e4f300",
        "gate_tests": "491559ac3dccf11f78baa02c957d0b6c7e592cfd91f954404a2d3a7cdc864025",
        "training_tests": "2184ff28a2250c52dfecaa97a7dcd0a0e04095c53ff3b229f3262db08e41c8ba",
    }
    for name, digest in historical.items():
        assert result["implementation"][name][1] == digest
    for name, (path, digest) in result["implementation"].items():
        if name not in historical:
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["training_enforcement"]["marker"] == ("train.requires_data_launch_authority=true")
    assert result["training_enforcement"]["verification_point"].endswith(
        "before model construction"
    )
    assert result["validation"]["real_receipts_issued"] == 0
    assert result["validation"]["models_constructed_by_gate_tests"] == 0


def test_launch_risk_hardening_successor_preserves_prior_results_and_binds_current_code():
    result = json.loads(
        (ROOT / "results" / "launch-risk-hardening-successor-20260908.json").read_text()
    )
    assert result["status"] == "cpu_failure_injection_pass_cluster_execution_pending"
    for predecessor in result["supersedes"].values():
        assert (
            hashlib.sha256((ROOT / predecessor["path"]).read_bytes()).hexdigest()
            == (predecessor["sha256"])
        )
        assert predecessor["modified"] is False
    assert result["implementation"]["pregrant_record"][1] == (
        "e2f890af3994cae104f094aef6ab83f0dca594304b58e26335eeef482cba06ed"
    )
    for name, (path, digest) in result["implementation"].items():
        if name != "pregrant_record":
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["validation"]["gpu_or_slurm_commands_run"] == 0
    assert result["authority"] == "engineering_evidence_only_not_training_authority"


def test_preaccess_engineering_integration_is_hash_bound_and_not_launch_authority():
    result = json.loads(
        (ROOT / "results" / "preaccess-engineering-integration-20260908.json").read_text()
    )
    assert result["status"] == (
        "cpu_preaccess_integration_pass_hardware_and_real_data_gates_pending"
    )
    for section in ("implementation", "contracts"):
        for path, digest in result[section].values():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["completed_preaccess_work"]["embedding_head"]["tie_word_embeddings"] is True
    assert result["completed_preaccess_work"]["slurm"]["scientific_promotion_automated"] is False
    assert result["completed_preaccess_work"]["slurm"]["reserve_spend_automated"] is False
    assert result["completed_preaccess_work"]["slurm"]["live_scheduler_commands_run"] == 0
    assert result["remaining_hardware_gates"]
    assert result["remaining_data_gates"]
    assert result["authority"] == (
        "CPU_engineering_integration_only_not_paid_launch_or_training_authority"
    )


def test_data_rehearsal_orchestration_is_hash_bound_but_has_not_run_20B():
    result = json.loads(
        (ROOT / "results" / "data" / "data-rehearsal-tooling-20260907.json").read_text()
    )
    assert result["status"] == "fixture_orchestration_qualified_real_20B_rehearsal_blocked"
    for path, digest in result["implementation"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["stage_order"] == [
        "source_identity",
        "acquisition",
        "global_dedup",
        "packing",
        "resume_cleanup",
        "firewall_disjointness",
    ]
    assert result["validation"]["real_stage_commands_run"] == 0
    assert result["validation"]["real_source_records_read"] == 0
    assert result["validation"]["persistent_operations_records_issued"] == 0


def test_fullsize_tokenizer_fixture_has_no_real_selection_authority():
    result = json.loads(
        (ROOT / "results" / "data" / "tokenizer-fullsize-fixture-20260907.json").read_text()
    )
    assert result["status"] == "fullsize_pipeline_fixture_pass_no_scientific_selection_authority"
    assert result["implementation"]["tokenizer_pipeline"][1] == (
        "c1ccea27a60132da4785b598b36c81d252117518f06b0b0df928e94c1bb8ab40"
    )
    for name, (path, digest) in result["implementation"].items():
        if name != "tokenizer_pipeline":
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["real_inputs"]["sources"] == 30
    assert result["real_inputs"]["materialized"] is False
    assert result["fixture"]["selection_authority"] is False
    assert result["fixture"]["category_output_hashes_path_independent"] is True
    assert all(
        tokenizer["path_independent_repeat_equal"] and all(tokenizer["static_hard_gates"].values())
        for tokenizer in result["tokenizers"].values()
    )
    assert result["training_authority"] == "blocked"


def test_tokenizer_static_nomination_policy_is_hash_bound_and_pre_results():
    result = json.loads(
        (
            ROOT / "results" / "data" / "tokenizer-static-nomination-fixture-20260907.json"
        ).read_text()
    )
    assert result["status"] == "policy_fixture_qualified_before_real_tokenizer_outputs"
    assert result["implementation"]["module"][1] == (
        "50b9f82819cb22a43238c195527c4534802c1a8a25c76d86ba396070cbfb99c1"
    )
    assert result["implementation"]["tests"][1] == (
        "ee812f6550c1c1daba1fd87d2ae6646a090f1e39849413a50f06ed8654a3e95f"
    )
    for name, (path, digest) in result["implementation"].items():
        if name not in {"module", "tests"}:
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["policy"]["selection_authority"] is False
    assert result["fixture"]["advancement_authority"] is False
    assert result["real_static_comparison"].startswith("blocked")
    assert result["final_tokenizer_decision"].startswith("blocked")


def test_tokenizer_v2_successor_is_hash_bound_and_formal_runs_remain_blocked():
    result = json.loads(
        (ROOT / "results" / "data" / "tokenizer-v2-contract-20260908.json").read_text()
    )
    assert result["status"] == ("corrected_v2_contract_fixture_pass_formal_training_and_D5_blocked")
    for path, digest in result["implementation"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for lineage in result["supersedes"].values():
        assert (
            hashlib.sha256((ROOT / lineage["path"]).read_bytes()).hexdigest() == lineage["sha256"]
        )
        assert lineage["modified"] is False
    assert result["v2_contract"]["allow_whitespace_only_pieces"] is True
    assert result["v2_contract"]["training_authority"] == "blocked"
    assert result["local_policy_fixture"]["advancement_authority"] is False
    assert result["local_policy_fixture"]["selection_authority"] is False
    assert result["validation"]["formal_tokenizer_models_trained"] == 0
    assert result["validation"]["language_model_pilot_runs"] == 0
    assert result["validation"]["D5_tokenizer_openings"] == 0


def test_tokenizer_pilot_analysis_is_hash_bound_and_D5_stays_unopened():
    result = json.loads(
        (ROOT / "results" / "data" / "tokenizer-pilot-analysis-fixture-20260907.json").read_text()
    )
    assert result["status"] == "seven_run_analysis_fixture_pass_real_pilot_and_D5_audit_blocked"
    for path, digest in result["implementation"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert result["frozen_run_matrix"]["total_runs"] == 7
    assert result["statistics"]["eligibility_requires_both_fixed_document_and_fixed_flop"] is True
    assert result["ranking"]["audit_opening_authorized"] is False
    assert result["validation"]["real_LM_runs"] == 0
    assert result["validation"]["D5_audit_openings"] == 0


def test_local_tokenizer_study_is_hash_bound_but_has_no_D5_authority():
    result = json.loads(
        (ROOT / "results" / "data" / "tokenizer-local-study-20260907.json").read_text()
    )
    assert result["status"] == (
        "exploratory_whitespace_piece_treatment_pass_no_D5_or_launch_authority"
    )
    for path, digest in result["implementation"].values():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    same_cost = result["findings"]["same_cost_custom_32000"]
    assert same_cost["same_embedding_and_head_parameters_as_mistral"] is True
    assert same_cost["relative_delta_vs_mistral"] < -0.07
    assert same_cost["sources_improved"] == same_cost["sources_compared"] == 30
    assert same_cost["all_hard_gates_pass"] is True
    reproducibility = result["findings"]["reproducibility"]
    assert reproducibility["model_hashes_equal"] is True
    assert reproducibility["training_stream_hashes_equal"] is True
    assert result["scope"]["gpu_runs"] == 0
    assert result["scope"]["language_model_runs"] == 0
    assert result["scope"]["sealed_audit_openings"] == 0
    assert result["scope"]["selection_authority"] is False
    assert result["scope"]["launch_authority"] is False
