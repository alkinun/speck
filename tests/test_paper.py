import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from speck.paper import validate_paper_program

root = Path(__file__).parents[1]
program = root / "research" / "paper-1"


def test_checked_paper_program_authorizes_proxy_but_blocks_paper_scale():
    assert validate_paper_program(program) == {
        "paper_id": "speck-paper-1",
        "status": "valid_hypotheses_only",
        "program_files": [
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
            "proxy_launch_v1.json",
            "contamination_v1.json",
            "contamination_disposition_v1.json",
            "experiment_program.json",
            "paper_outline.md",
            "reference_audit.md",
            "reporting_checklist.md",
        ],
        "claims": 7,
        "non_claims": 8,
        "scales": 6,
        "axes": 3,
        "historical_baseline_arms": 5,
        "planned_primary_baseline_arms": 2,
        "proxy_confirmation_pairs": 3,
        "proxy_training": "authorized",
        "paper_scale_pretraining": "blocked",
    }


def test_paper_program_rejects_an_unknown_axis_claim(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["axes"][0]["claim_ids"].append("C99")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid claims"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_baseline_audit_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["baseline_evidence"]["audit_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline audit"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_proxy_launch_qualification_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["proxy_launch_evidence"]["qualification_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="proxy launch qualification"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_dense_control_result_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["baseline_evidence"]["dense_control_results"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="dense-control result"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_dense_control_rollback_after_disposition_freeze(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["baseline_evidence"]["dense_control_results"] = value["baseline_evidence"][
        "dense_control_results"
    ][:2]
    value["baseline_evidence"]["time_to_quality_target"] = {
        "path": "not-allowed.json",
        "sha256": "0" * 64,
        "status": "locked_from_controls_before_candidate_analysis",
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="proxy disposition contract"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_proxy_disposition_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["baseline_evidence"]["proxy_disposition_contract_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="proxy_disposition_contract"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_sequence_cache_design_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["sequence_cache_representation"]["sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="sequence cache representation"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_hca_readiness_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["hca_readiness"]["sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="HCA readiness"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_csa_readiness_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["csa_readiness"]["sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="CSA readiness"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_raw_local_readiness_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["raw_local_readiness"]["sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="raw-local readiness"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_contamination_disposition_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["evaluation_evidence"]["contamination_disposition_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="contamination_disposition"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_helmet_runtime_audit_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["evaluation_evidence"]["helmet_runtime_dependency_audit_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="helmet_runtime_dependency_audit"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_helmet_materializer_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["evaluation_evidence"]["helmet_materializer_preflight_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="helmet_materializer_preflight"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_helmet_clinc_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["evaluation_evidence"]["helmet_clinc_source_qualification_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="helmet_clinc_source_qualification"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_helmet_trec_rights_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["evaluation_evidence"]["helmet_trec_rights_decision_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="helmet_trec_rights_decision"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_multilexsum_decision_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["evaluation_evidence"]["helmet_multilexsum_decision_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="helmet_multilexsum_decision"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_narrativeqa_decision_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["evaluation_evidence"]["helmet_narrativeqa_decision_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="helmet_narrativeqa_decision"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_infinitebench_decision_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["evaluation_evidence"]["helmet_infinitebench_decision_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="helmet_infinitebench_decision"):
        validate_paper_program(copied, repository_root=root)


def test_paper_program_rejects_seeded_demo_repair_pin_drift(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["evaluation_evidence"]["helmet_seeded_demo_repair_qualification_sha256"] = (
        "0" * 64
    )
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="helmet_seeded_demo_repair_qualification"):
        validate_paper_program(copied, repository_root=root)
