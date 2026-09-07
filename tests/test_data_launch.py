import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from speck.data_launch import issue_launch_receipt, validate_launch_request, verify_launch_receipt
from speck.dataloader import manifest_fingerprint

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": _sha256(path)}


def _request(tmp_path):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    files = {
        name: _json(experiment / f"{name}.json", {"fixture": name})
        for name in ("data", "tokenizer", "model", "train")
    }
    rights = _json(
        tmp_path / "rights.json",
        {
            "format": "speck_human_source_rights_acceptance",
            "status": "all_sources_human_approved",
            "approved_source_ids": ["source"],
            "automated_approval_made": False,
        },
    )
    deny = _json(
        tmp_path / "deny.json",
        {
            "format": "speck_removal_deny_ledger",
            "format_version": 1,
            "status": "human_reviewed_deny_entries",
            "entries": [],
        },
    )
    operations = _json(
        tmp_path / "operations.json",
        {
            "format": "speck_production_data_operations_qualification",
            "status": "production_data_operations_qualified",
            "deny_ledger_sha256": deny["sha256"],
            "gates": {
                "global_exact_deduplication": "pass",
                "global_near_deduplication": "pass",
                "acquisition_cleanup": "pass",
                "interruption_resume": "pass",
                "firewall_training_disjointness": "pass",
            },
        },
    )
    firewall = _json(
        tmp_path / "firewall.json",
        {
            "format": "speck_data_firewall_manifest",
            "status": "production_firewall_complete_rights_and_operations_bound",
            "authority": {
                "rights_record_sha256": rights["sha256"],
                "production_record_sha256": operations["sha256"],
            },
            "gates": {
                "global_content_disjointness": "pass",
                "sealed_audits_unopened": "pass",
            },
        },
    )
    tokenizer = _json(
        tmp_path / "tokenizer-decision.json",
        {
            "format": "speck_tokenizer_decision",
            "status": "tokenizer_selected_and_frozen",
            "tokenizer_fingerprint": "fixture-tokenizer",
        },
    )
    packed_value = {
        "format": "speck_packed_tokens",
        "tokenizer": {"fingerprint": "fixture-tokenizer"},
        "dedup": {"scope": "global"},
    }
    packed = _json(tmp_path / "packed-manifest.json", packed_value)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return (
        {
            "format": "speck_flagship_data_launch_request",
            "format_version": 1,
            "status": "preflight_requested_not_authority",
            "repository": str(ROOT),
            "git_commit": commit,
            "experiment_directory": str(experiment),
            "experiment_files": files,
            "rights_record": rights,
            "production_operations_record": operations,
            "deny_ledger": deny,
            "firewall_manifest": firewall,
            "tokenizer_decision": tokenizer,
            "packed_manifest": packed,
            "receipt_path": str(tmp_path / "receipt.json"),
        },
        packed_value,
    )


def test_launch_receipt_binds_every_data_authority_and_exact_launch(tmp_path):
    raw, packed = _request(tmp_path)
    request = validate_launch_request(raw)
    receipt = issue_launch_receipt(request)

    assert receipt["status"] == "flagship_data_launch_preflight_passed"
    assert (
        verify_launch_receipt(
            raw["receipt_path"],
            repository=ROOT,
            experiment_directory=raw["experiment_directory"],
            packed_manifest_fingerprint=manifest_fingerprint(packed),
            tokenizer_fingerprint="fixture-tokenizer",
        )
        == receipt
    )
    with pytest.raises(FileExistsError):
        issue_launch_receipt(request)
    Path(raw["rights_record"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="receipt rights record identity mismatch"):
        verify_launch_receipt(
            raw["receipt_path"],
            repository=ROOT,
            experiment_directory=raw["experiment_directory"],
            packed_manifest_fingerprint=manifest_fingerprint(packed),
            tokenizer_fingerprint="fixture-tokenizer",
        )


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ("rights_record", "human rights acceptance"),
        ("production_operations_record", "production operations authority"),
        ("firewall_manifest", "production firewall authority"),
        ("tokenizer_decision", "tokenizer decision"),
        ("packed_manifest", "packed dataset"),
    ],
)
def test_launch_preflight_fails_each_missing_authority(tmp_path, record, message):
    raw, _ = _request(tmp_path)
    path = Path(raw[record]["path"])
    value = json.loads(path.read_text())
    if record == "packed_manifest":
        value["format"] = "pending"
    else:
        value["status"] = "pending"
    path.write_text(json.dumps(value))
    raw[record]["sha256"] = _sha256(path)

    with pytest.raises(ValueError, match=message):
        issue_launch_receipt(raw)


def test_flagship_launch_plan_requires_marked_pre_model_verification():
    plan = json.loads((ROOT / "research/flagship/data_launch_plan.json").read_text())

    assert plan["status"] == "fixture_preflight_ready_real_authorities_pending"
    assert plan["training_integration"]["marker"] == (
        "train.requires_data_launch_authority=true"
    )
    assert plan["training_integration"]["verification_point"].endswith(
        "before model construction"
    )
    assert plan["training_integration"]["historical_configs_default"] is False
    assert plan["training_integration"]["flagship_configs_must_set_marker"] is True
    assert plan["receipt"]["real_receipt_status"] == "blocked"
    assert plan["training_authority"] == "blocked"
