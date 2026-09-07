import hashlib
import json
from pathlib import Path

import pytest

from speck.data_firewall import (
    authorize_consumer,
    construct_firewall,
    open_sealed_audit,
    validate_firewall_config,
)

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_input(path, source, domain, count=200):
    records = []
    for index in range(count):
        text = f"{source} document {index} from {domain}. Distinct fixture content for firewall validation."
        records.append(
            {
                "text": text,
                "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "host": domain,
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _config(tmp_path, mode="fixture_only"):
    categories = []
    for category in ("web", "code"):
        primary = tmp_path / f"{category}-primary.jsonl"
        unseen = tmp_path / f"{category}-unseen.jsonl"
        _write_input(primary, f"{category}-primary", f"{category}.primary.example")
        _write_input(unseen, f"{category}-unseen", f"{category}.unseen.example")
        zero = {
            key: 0
            for key in (
                "tokenizer_train",
                "tokenizer_eval",
                "selection_heldout",
                "D5_tokenizer",
                "E2_mixture",
            )
        }
        categories.append(
            {
                "id": category,
                "inputs": [
                    {
                        "id": f"{category}-primary",
                        "source_id": f"{category}_primary",
                        "path": str(primary),
                        "sha256": _sha256(primary),
                        "format": "jsonl",
                        "text_field": "text",
                        "content_sha256_field": "released_content_sha256",
                        "domain_field": "host",
                        "training_mixture_eligible": True,
                        "allocations": {
                            **zero,
                            "tokenizer_train": 500,
                            "tokenizer_eval": 200,
                            "selection_heldout": 200,
                            "D5_tokenizer": 200,
                            "E2_mixture": 200,
                        },
                    },
                    {
                        "id": f"{category}-unseen",
                        "source_id": f"{category}_unseen",
                        "path": str(unseen),
                        "sha256": _sha256(unseen),
                        "format": "jsonl",
                        "text_field": "text",
                        "content_sha256_field": "released_content_sha256",
                        "domain_field": "host",
                        "training_mixture_eligible": False,
                        "allocations": {**zero, "selection_heldout": 200},
                    },
                ],
            }
        )
    authority = {
        "mode": "fixture_only",
        "rights_record": None,
        "rights_record_sha256": None,
        "rights_status": None,
        "production_record": None,
        "production_record_sha256": None,
        "production_status": None,
    }
    if mode == "production":
        rights = tmp_path / "rights.json"
        production = tmp_path / "production.json"
        rights.write_text(
            json.dumps(
                {
                    "format": "speck_human_source_rights_acceptance",
                    "status": "all_sources_human_approved",
                    "authority": {
                        "name": "Fixture Approver",
                        "role": "test authority",
                        "organization": "fixture organization",
                    },
                    "signed_at": "2026-09-07T00:00:00Z",
                    "scope": "fixture data only",
                    "approved_source_ids": [
                        "web_primary",
                        "web_unseen",
                        "code_primary",
                        "code_unseen",
                    ],
                }
            )
        )
        production.write_text(
            json.dumps(
                {
                    "format": "speck_production_data_operations_qualification",
                    "status": "production_data_operations_qualified",
                    "qualified_at": "2026-09-07T00:00:00Z",
                    "gates": {
                        "global_exact_deduplication": "pass",
                        "global_near_deduplication": "pass",
                        "acquisition_cleanup": "pass",
                        "interruption_resume": "pass",
                    },
                }
            )
        )
        authority = {
            "mode": "production",
            "rights_record": str(rights),
            "rights_record_sha256": _sha256(rights),
            "rights_status": "all_sources_human_approved",
            "production_record": str(production),
            "production_record_sha256": _sha256(production),
            "production_status": "production_data_operations_qualified",
        }
    return {
        "format": "speck_data_firewall",
        "format_version": 1,
        "status": "construction_authorized_not_training_authority",
        "authority": authority,
        "policy": {
            "global_dedup_normalization": "NFKC+lower+whitespace",
            "tokenizer_train_bytes_per_category": 500,
            "tokenizer_eval_bytes_per_category": 200,
            "selection_bytes_per_category": 400,
            "minimum_unseen_selection_bytes_per_category": 200,
            "minimum_selection_domains_per_category": 2,
            "minimum_heldout_domains_per_category": 1,
            "partition_seeds": {
                "tokenizer_train": 41,
                "tokenizer_eval": 42,
                "selection_heldout": 43,
                "D5_tokenizer": 44,
                "E2_mixture": 45,
            },
            "sealed_audits": [
                {
                    "id": "D5_tokenizer",
                    "bytes_per_category": 200,
                    "required_finalists": 2,
                    "fallback": "mistral-32k",
                },
                {
                    "id": "E2_mixture",
                    "bytes_per_category": 200,
                    "required_finalists": 3,
                    "fallback": "balanced_prior",
                },
            ],
        },
        "categories": categories,
        "output_directory": str(tmp_path / f"firewall-{mode}"),
    }


def _hashes(manifest, root, destination):
    values = set()
    for category in manifest["categories"]:
        path = root / category["outputs"][destination]["path"]
        if destination in {"D5_tokenizer", "E2_mixture"}:
            path.chmod(0o400)
        values.update(json.loads(line)["dedup_sha256"] for line in path.read_text().splitlines())
        if destination in {"D5_tokenizer", "E2_mixture"}:
            path.chmod(0)
    return values


def test_fixture_firewall_is_globally_disjoint_balanced_and_non_authoritative(tmp_path):
    config = validate_firewall_config(_config(tmp_path))
    manifest = construct_firewall(config)
    root = Path(config["output_directory"])
    sets = {
        destination: _hashes(manifest, root, destination)
        for destination in (
            "tokenizer_train",
            "tokenizer_eval",
            "selection_heldout",
            "D5_tokenizer",
            "E2_mixture",
        )
    }

    assert manifest["status"] == "fixture_firewall_complete_not_training_authority"
    for index, left in enumerate(sets):
        assert all(sets[left].isdisjoint(sets[right]) for right in list(sets)[index + 1 :])
    assert all(category["selection"]["domains"] >= 2 for category in manifest["categories"])
    assert all(category["selection"]["heldout_domains"] >= 1 for category in manifest["categories"])
    train_paths = [
        root / category["outputs"]["tokenizer_train"]["path"] for category in manifest["categories"]
    ]
    assert authorize_consumer(root / "manifest.json", "fixture_validation", train_paths)[
        "authorized"
    ]
    with pytest.raises(PermissionError, match="fixture firewall"):
        authorize_consumer(root / "manifest.json", "tokenizer_training", train_paths)


def test_production_consumers_are_partition_specific_and_model_training_is_denied(tmp_path):
    config = validate_firewall_config(_config(tmp_path, "production"))
    manifest = construct_firewall(config)
    root = Path(config["output_directory"])
    train = [
        root / category["outputs"]["tokenizer_train"]["path"] for category in manifest["categories"]
    ]
    selection = [
        root / category["outputs"]["selection_heldout"]["path"]
        for category in manifest["categories"]
    ]

    assert authorize_consumer(root / "manifest.json", "tokenizer_training", train)[
        "training_authority"
    ]
    assert authorize_consumer(root / "manifest.json", "data_selection", selection)["authorized"]
    with pytest.raises(PermissionError, match="model training"):
        authorize_consumer(root / "manifest.json", "model_training", selection)
    with pytest.raises(PermissionError, match="cannot access"):
        authorize_consumer(root / "manifest.json", "tokenizer_training", selection)


def test_sealed_audit_requires_frozen_ranking_and_opens_only_once(tmp_path):
    config = validate_firewall_config(_config(tmp_path))
    construct_firewall(config)
    root = Path(config["output_directory"])
    ranking = tmp_path / "ranking.json"
    ranking.write_text(
        json.dumps(
            {
                "status": "ranking_frozen_before_audit",
                "audit_identity": "E2_mixture",
                "ranked_finalists": ["first", "second", "third"],
            }
        )
    )
    request = {
        "format": "speck_sealed_audit_open_request",
        "format_version": 1,
        "audit_identity": "E2_mixture",
        "manifest_sha256": _sha256(root / "manifest.json"),
        "ranking_report": str(ranking),
        "ranking_report_sha256": _sha256(ranking),
        "ranked_finalists": ["first", "second", "third"],
    }
    opened = open_sealed_audit(root / "manifest.json", request, tmp_path / "receipts")

    assert len(opened["paths"]) == 2
    assert Path(opened["receipt"]).is_file()
    sealed = [
        root / category["outputs"]["D5_tokenizer"]["path"]
        for category in json.loads((root / "manifest.json").read_text())["categories"]
    ]
    with pytest.raises(PermissionError, match="one-opening"):
        authorize_consumer(root / "manifest.json", "fixture_validation", sealed)
    with pytest.raises(FileExistsError):
        open_sealed_audit(root / "manifest.json", request, tmp_path / "receipts")


def test_production_config_rejects_unapproved_authority(tmp_path):
    raw = _config(tmp_path, "production")
    raw["authority"]["rights_status"] = "pending"
    with pytest.raises(ValueError, match="human rights approval"):
        validate_firewall_config(raw)


def test_production_build_requires_human_source_coverage(tmp_path):
    raw = _config(tmp_path, "production")
    rights_path = Path(raw["authority"]["rights_record"])
    rights = json.loads(rights_path.read_text())
    rights["approved_source_ids"].remove("code_unseen")
    rights_path.write_text(json.dumps(rights))
    raw["authority"]["rights_record_sha256"] = _sha256(rights_path)
    config = validate_firewall_config(raw)

    with pytest.raises(ValueError, match="does not approve every firewall source"):
        construct_firewall(config)


def test_flagship_firewall_plan_defers_real_sizes_and_separates_audit_identities():
    plan = json.loads((ROOT / "research/flagship/firewall_plan.json").read_text())

    assert plan["status"] == "fixture_tooling_ready_real_targets_and_authority_pending"
    assert plan["top_level_partitions"] == [
        "tokenizer_sample",
        "selection_heldout",
        "sealed_audit",
    ]
    assert [audit["id"] for audit in plan["sealed_audit"]["identities"]] == [
        "D5_tokenizer",
        "E2_mixture",
    ]
    assert [audit["required_finalists"] for audit in plan["sealed_audit"]["identities"]] == [
        2,
        3,
    ]
    assert plan["sealed_audit"]["adversarial_security_claim"] is False
    assert plan["real_materialization"]["status"] == "blocked"
    assert plan["consumer_enforcement"]["model_training_may_consume_firewall_files"] is False


def test_firewall_fails_when_selection_has_no_truly_heldout_domain(tmp_path):
    raw = _config(tmp_path)
    raw["policy"]["minimum_heldout_domains_per_category"] = 2
    config = validate_firewall_config(raw)

    with pytest.raises(RuntimeError, match="held-out domain"):
        construct_firewall(config)


def test_audit_claim_survives_tampered_payload_and_prevents_retry(tmp_path):
    config = validate_firewall_config(_config(tmp_path))
    construct_firewall(config)
    root = Path(config["output_directory"])
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / manifest["categories"][0]["outputs"]["D5_tokenizer"]["path"]
    path.chmod(0o600)
    path.write_text(path.read_text() + "tampered\n")
    path.chmod(0)
    ranking = tmp_path / "ranking-d5.json"
    ranking.write_text(
        json.dumps(
            {
                "status": "ranking_frozen_before_audit",
                "audit_identity": "D5_tokenizer",
                "ranked_finalists": ["custom", "mistral"],
            }
        )
    )
    request = {
        "format": "speck_sealed_audit_open_request",
        "format_version": 1,
        "audit_identity": "D5_tokenizer",
        "manifest_sha256": _sha256(root / "manifest.json"),
        "ranking_report": str(ranking),
        "ranking_report_sha256": _sha256(ranking),
        "ranked_finalists": ["custom", "mistral"],
    }
    receipts = tmp_path / "receipts-tampered"

    with pytest.raises(ValueError, match="identity mismatch after opening"):
        open_sealed_audit(root / "manifest.json", request, receipts)
    assert (receipts / "D5_tokenizer.json").is_file()
    with pytest.raises(FileExistsError):
        open_sealed_audit(root / "manifest.json", request, receipts)
