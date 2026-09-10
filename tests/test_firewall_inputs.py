import hashlib
import json
from pathlib import Path

from speck.firewall_inputs import (
    freeze_calibrated_firewall_config,
    prepare_firewall_inputs,
    validate_firewall_input_plan,
)

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(source, index):
    text = f"{source} document {index} contains sufficiently distinct words for global duplicate analysis."
    return {
        "text": text,
        "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "host": f"{source}.example",
        "content_id": f"{source}-{index}",
    }


def test_real_firewall_plan_is_hash_bound_and_keeps_model_training_blocked():
    plan = json.loads((ROOT / "research/flagship/firewall_plan_v3.json").read_text())

    assert (
        plan["status"]
        == "calibrated_inputs_prepared_construction_config_frozen_not_training_authority"
    )
    assert plan["training_authority"] is False
    assert plan["final_corpus_exclusion"]["status"] == "required_before_data_launch"
    assert "verified-near" in plan["final_corpus_exclusion"]["rule"]
    assert plan["targets"] == {
        "tokenizer_train_bytes_per_category": 100_000_000,
        "tokenizer_eval_bytes_per_category": 10_000_000,
        "selection_bytes_per_category": 20_000_000,
        "minimum_unseen_selection_bytes_per_category": 10_000_000,
        "D5_tokenizer_bytes_per_category": 10_000_000,
        "E2_mixture_bytes_per_category": 20_000_000,
    }
    for identity in plan["authority"].values():
        assert _sha256(ROOT / identity["path"]) == identity["sha256"]
    input_identity = plan["input_preparation"]
    assert _sha256(ROOT / input_identity["path"]) == input_identity["sha256"]
    assert _sha256(ROOT / input_identity["result"]["path"]) == input_identity["result"]["sha256"]
    construction = plan["construction"]["config"]
    assert _sha256(ROOT / construction["path"]) == construction["sha256"]
    for path, digest in plan["implementation"].values():
        assert _sha256(ROOT / path) == digest


def test_firewall_input_views_are_deterministic_and_globally_near_deduplicated(tmp_path):
    fallback = tmp_path / "fallback.json"
    fallback.write_text(
        json.dumps(
            {
                "format": "speck_production_data_calibration_fallback",
                "status": "project_owner_accepted_non_gpu_operations_fallback_through_150B",
                "training_authority": False,
            }
        )
    )
    deny = tmp_path / "deny.json"
    deny.write_text(
        json.dumps(
            {
                "format": "speck_removal_deny_ledger",
                "format_version": 1,
                "status": "human_reviewed_deny_entries",
                "entries": [],
            }
        )
    )
    inputs = []
    for category in ("web", "code", "math", "synthetic", "science", "reference"):
        for role in ("primary", "unseen"):
            source = f"{category}_{role}"
            path = tmp_path / f"{source}.jsonl"
            path.write_text(
                "".join(json.dumps(_record(source, index)) + "\n" for index in range(20))
            )
            inputs.append(
                {
                    "id": source,
                    "source_id": source,
                    "category": category,
                    "role": role,
                    "input": {"path": str(path), "sha256": _sha256(path)},
                    "view_target_bytes": 100,
                    "firewall_group": f"{role}:{source}",
                }
            )
    raw_plan = {
        "format": "speck_firewall_input_preparation",
        "format_version": 1,
        "status": "preparation_authorized_not_consumer_or_training_authority",
        "seed": 42,
        "operations_fallback": {"path": str(fallback), "sha256": _sha256(fallback)},
        "deny_ledger": {"path": str(deny), "sha256": _sha256(deny)},
        "deduplication": {
            "normalization": "NFKC+lower+lexical-code-tokens",
            "token_pattern": "[A-Za-z]+|[^\\s]",
            "shingle_tokens": 3,
            "minimum_document_tokens": 3,
            "maximum_document_tokens": 1000,
            "num_perm": 32,
            "minhash_seed": 42,
            "bands": 8,
            "verified_jaccard_threshold": 0.8,
            "domain_match": "exact_or_subdomain",
            "checkpoint_records": 10,
        },
        "inputs": inputs,
        "output_directory": str(tmp_path / "prepared"),
    }
    input_plan_path = tmp_path / "input-plan.json"
    input_plan_path.write_text(json.dumps(raw_plan))
    plan = validate_firewall_input_plan(raw_plan)
    first = prepare_firewall_inputs(plan)
    second = prepare_firewall_inputs(plan)

    assert first == second
    assert len(first["views"]) == 12
    assert first["deduplication"]["counts"]["records_retained"] < 24
    assert first["deduplication"]["counts"]["records_removed_near"] > 0
    assert first["consumer_authority"] is False
    assert first["training_authority"] is False

    firewall_plan_path = tmp_path / "firewall-plan.json"
    firewall_plan_path.write_text(
        json.dumps(
            {
                "format": "speck_flagship_firewall_plan",
                "format_version": 2,
                "status": "calibrated_production_input_plan_frozen_not_consumer_or_training_authority",
                "input_preparation": {
                    "path": str(input_plan_path),
                    "sha256": _sha256(input_plan_path),
                },
                "authority": {
                    "rights_record": {"path": str(deny), "sha256": _sha256(deny)},
                    "operations_fallback": {
                        "path": str(fallback),
                        "sha256": _sha256(fallback),
                    },
                },
                "targets": {
                    "tokenizer_train_bytes_per_category": 10,
                    "tokenizer_eval_bytes_per_category": 10,
                    "selection_bytes_per_category": 20,
                    "minimum_unseen_selection_bytes_per_category": 10,
                    "D5_tokenizer_bytes_per_category": 10,
                    "E2_mixture_bytes_per_category": 10,
                },
                "partition_seeds": {
                    "tokenizer_train": 1,
                    "tokenizer_eval": 2,
                    "selection_heldout": 3,
                    "D5_tokenizer": 4,
                    "E2_mixture": 5,
                },
                "sealed_audits": [
                    {"id": "D5_tokenizer", "required_finalists": 2, "fallback": "mistral-32k"},
                    {"id": "E2_mixture", "required_finalists": 3, "fallback": "balanced_prior"},
                ],
                "output_directory": str(tmp_path / "firewall"),
                "training_authority": False,
            }
        )
    )
    config_path = tmp_path / "firewall-config.json"
    config = freeze_calibrated_firewall_config(
        firewall_plan_path, tmp_path / "prepared/manifest.json", config_path
    )

    assert config_path.is_file()
    assert len(config["categories"]) == 6
    assert all(len(category["inputs"]) == 2 for category in config["categories"])
    assert config["categories"][0]["inputs"][1]["allocations"]["tokenizer_train"] == 0
