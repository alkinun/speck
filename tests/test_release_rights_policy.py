import hashlib
import json
from pathlib import Path

from speck.source_rights import assess_rights_template, validate_rights_template

ROOT = Path(__file__).parents[1]
FLAGSHIP = ROOT / "research/flagship"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_release_policy_freezes_apache_weights_mit_code_and_no_corpus_release():
    policy = json.loads((FLAGSHIP / "release_and_data_use_policy_v1.json").read_text())

    assert policy["status"] == "human_project_policy_approved"
    assert policy["authority"] == {
        "name": "Alkin Unlu",
        "role": "Project owner",
        "organization": "SpeckLabs",
    }
    assert policy["release_licenses"]["source_code"] == "MIT"
    assert policy["release_licenses"]["model_weights"] == "Apache-2.0"
    assert policy["intended_scope"] == {
        "research_training": True,
        "paper_publication": True,
        "public_model_weight_release": True,
        "commercial_use": True,
        "source_data_redistribution": False,
        "derived_packed_shard_redistribution": False,
    }
    assert policy["publication"]["corpus_bytes"] == "not published"
    assert policy["publication"]["packed_shards"] == "not published"


def test_source_registry_successor_preserves_allocations_and_binds_policy():
    predecessor = json.loads((FLAGSHIP / "source_registry.json").read_text())
    registry = json.loads((FLAGSHIP / "source_registry_v2.json").read_text())

    assert registry["format_version"] == 2
    assert _sha256(ROOT / registry["supersedes"]["path"]) == registry["supersedes"]["sha256"]
    assert (
        _sha256(ROOT / registry["release_and_data_use_policy"]["path"])
        == registry["release_and_data_use_policy"]["sha256"]
    )
    assert registry["tokenizer_sample_allocations"] == predecessor["tokenizer_sample_allocations"]
    selected = {item["source_id"] for item in registry["tokenizer_sample_allocations"]}
    assert all(
        source.get("human_use_decision")
        == "approved_guarded_internal_training_no_corpus_redistribution"
        for source in registry["sources"]
        if source["id"] in selected
    )


def test_human_acceptance_covers_all_selected_sources_and_is_portable():
    input_path = FLAGSHIP / "source_rights_acceptance_input_v1.json"
    input_record = json.loads(input_path.read_text())
    normalized = validate_rights_template(
        input_record, config_dir=input_path.parent, verify_evidence=True
    )
    assessment = assess_rights_template(
        input_record, config_dir=input_path.parent, verify_evidence=True
    )
    acceptance = json.loads((FLAGSHIP / "source_rights_acceptance_v1.json").read_text())

    assert assessment["counts"] == {"pending": 0, "approve": 30, "reject": 0}
    assert assessment["incomplete_approved_sources"] == []
    assert assessment["scope_fields_pending"] == []
    assert assessment["authority_fields_pending"] == []
    assert assessment["signed_at_pending"] is False
    assert acceptance["format"] == "speck_human_source_rights_acceptance"
    assert acceptance["status"] == "all_sources_human_approved"
    assert acceptance["automated_approval_made"] is False
    assert acceptance["approved_source_ids"] == normalized["required_source_ids"]
    assert acceptance["source_registry"] == "source_registry_v2.json"
    assert all(not Path(packet["path"]).is_absolute() for packet in acceptance["evidence_packets"])
    assert _sha256(FLAGSHIP / acceptance["source_registry"]) == acceptance["source_registry_sha256"]
    assert all(
        _sha256(FLAGSHIP / packet["path"]) == packet["sha256"]
        for packet in acceptance["evidence_packets"]
    )


def test_active_successor_plans_remove_only_the_completed_rights_blocker():
    launch = json.loads((FLAGSHIP / "data_launch_plan_v2.json").read_text())
    rehearsal = json.loads((FLAGSHIP / "data_rehearsal_plan_v2.json").read_text())
    tokenizer = json.loads((FLAGSHIP / "tokenizer_plan_v5.json").read_text())
    tokenizer_v4 = json.loads((FLAGSHIP / "tokenizer_plan_v4.json").read_text())

    for value in (launch, rehearsal, tokenizer):
        assert _sha256(ROOT / value["supersedes"]["path"]) == value["supersedes"]["sha256"]
    for value in (launch, rehearsal):
        assert _sha256(ROOT / value["release_policy"]["path"]) == value["release_policy"]["sha256"]
    assert (
        _sha256(ROOT / tokenizer["release_and_data_use_policy"]["path"])
        == tokenizer["release_and_data_use_policy"]["sha256"]
    )
    assert launch["status"] == "rights_approved_remaining_real_authorities_pending"
    assert (
        _sha256(ROOT / launch["source_rights_acceptance"]["path"])
        == launch["source_rights_acceptance"]["sha256"]
    )
    assert "unsigned 30-source human rights template" not in launch["blockers"]
    assert len(launch["blockers"]) == 5
    assert rehearsal["real_manifest"]["status"] == "pending_materialization"
    assert rehearsal["training_authority"].startswith("blocked_pending_real_20B")
    assert tokenizer["format_version"] == 5
    assert tokenizer["candidates"] == tokenizer_v4["candidates"]
    assert tokenizer["trainer"] == tokenizer_v4["trainer"]
    assert tokenizer["model_accounting"] == tokenizer_v4["model_accounting"]
    assert tokenizer["language_model_pilot"] == tokenizer_v4["language_model_pilot"]
    assert (
        _sha256(ROOT / tokenizer["source_rights_acceptance"]["path"])
        == tokenizer["source_rights_acceptance"]["sha256"]
    )


def test_real_20b_rehearsal_successor_binds_frozen_production_and_orchestration_plans():
    plan = json.loads((FLAGSHIP / "data_rehearsal_plan_v3.json").read_text())

    assert plan["status"] == "real_20B_manifest_frozen_launch_pending"
    assert _sha256(ROOT / plan["supersedes"]["path"]) == plan["supersedes"]["sha256"]
    assert plan["real_manifest"]["target_tokens"] == 20_000_000_000
    for key in ("production_plan", "orchestration"):
        identity = plan["real_manifest"][key]
        assert _sha256(ROOT / identity["path"]) == identity["sha256"]


def test_time_bounded_calibration_successor_preserves_paused_20b_fallback():
    plan = json.loads((FLAGSHIP / "data_rehearsal_plan_v4.json").read_text())

    assert plan["status"] == "time_bounded_2B_calibration_frozen_storage_mount_required"
    assert _sha256(ROOT / plan["supersedes"]["path"]) == plan["supersedes"]["sha256"]
    assert plan["real_manifest"]["target_tokens"] == 2_000_000_000
    identity = plan["real_manifest"]["calibration_plan"]
    assert _sha256(ROOT / identity["path"]) == identity["sha256"]
    assert plan["real_manifest"]["operations_authority"] is False
    assert plan["real_manifest"]["training_authority"] is False
    assert plan["real_manifest"]["paused_20B_attempt"].endswith("data-rehearsal-20b-v1")
