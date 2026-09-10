import hashlib
import json
from pathlib import Path

from speck.data_firewall import authorize_consumer
from speck.data_firewall_calibrated import (
    construct_calibrated_firewall,
    validate_calibrated_firewall_config,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _input(path, source, group):
    rows = []
    for index in range(100):
        text = f"{source} record {index} has distinct content for calibrated firewall construction."
        rows.append(
            {
                "text": text,
                "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "firewall_group": group,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_calibration_fallback_can_authorize_real_non_model_firewall_consumers(tmp_path):
    primary = tmp_path / "primary.jsonl"
    unseen = tmp_path / "unseen.jsonl"
    _input(primary, "primary", "primary")
    _input(unseen, "unseen", "unseen")
    rights = tmp_path / "rights.json"
    rights.write_text(
        json.dumps(
            {
                "format": "speck_human_source_rights_acceptance",
                "status": "all_sources_human_approved",
                "approved_source_ids": ["primary_source", "unseen_source"],
                "automated_approval_made": False,
            }
        )
    )
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "gates": {
                    "source_identity": "pass",
                    "global_exact_deduplication": "pass",
                    "global_near_deduplication": "pass",
                    "packing_integrity": "pass",
                    "acquisition_cleanup": "pass",
                    "interruption_resume": "pass",
                    "firewall_training_disjointness": "pass",
                }
            }
        )
    )
    fallback = tmp_path / "fallback.json"
    fallback.write_text(
        json.dumps(
            {
                "format": "speck_production_data_calibration_fallback",
                "status": "project_owner_accepted_non_gpu_operations_fallback_through_150B",
                "training_authority": False,
                "calibration_analysis": {"path": str(analysis), "sha256": _sha256(analysis)},
            }
        )
    )
    zero = {
        "tokenizer_train": 0,
        "tokenizer_eval": 0,
        "selection_heldout": 0,
        "D5_tokenizer": 0,
        "E2_mixture": 0,
    }
    config = validate_calibrated_firewall_config(
        {
            "format": "speck_calibrated_data_firewall",
            "format_version": 1,
            "status": "construction_authorized_not_training_authority",
            "authority": {
                "rights_record": {"path": str(rights), "sha256": _sha256(rights)},
                "operations_fallback": {"path": str(fallback), "sha256": _sha256(fallback)},
            },
            "policy": {
                "global_dedup_normalization": "NFKC+lower+whitespace",
                "tokenizer_train_bytes_per_category": 100,
                "tokenizer_eval_bytes_per_category": 100,
                "selection_bytes_per_category": 200,
                "minimum_unseen_selection_bytes_per_category": 100,
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
                        "bytes_per_category": 100,
                        "required_finalists": 2,
                        "fallback": "mistral-32k",
                    },
                    {
                        "id": "E2_mixture",
                        "bytes_per_category": 100,
                        "required_finalists": 3,
                        "fallback": "balanced_prior",
                    },
                ],
            },
            "categories": [
                {
                    "id": "web",
                    "inputs": [
                        {
                            "id": "primary",
                            "source_id": "primary_source",
                            "path": str(primary),
                            "sha256": _sha256(primary),
                            "format": "jsonl",
                            "text_field": "text",
                            "content_sha256_field": "released_content_sha256",
                            "domain_field": "firewall_group",
                            "training_mixture_eligible": True,
                            "allocations": {
                                **zero,
                                "tokenizer_train": 100,
                                "tokenizer_eval": 100,
                                "selection_heldout": 100,
                                "D5_tokenizer": 100,
                                "E2_mixture": 100,
                            },
                        },
                        {
                            "id": "unseen",
                            "source_id": "unseen_source",
                            "path": str(unseen),
                            "sha256": _sha256(unseen),
                            "format": "jsonl",
                            "text_field": "text",
                            "content_sha256_field": "released_content_sha256",
                            "domain_field": "firewall_group",
                            "training_mixture_eligible": False,
                            "allocations": {**zero, "selection_heldout": 100},
                        },
                    ],
                }
            ],
            "output_directory": str(tmp_path / "firewall"),
        }
    )
    manifest = construct_calibrated_firewall(config)
    train = tmp_path / "firewall/tokenizer-train-web.jsonl"

    assert manifest["status"] == "production_firewall_complete_rights_and_operations_bound"
    assert authorize_consumer(tmp_path / "firewall/manifest.json", "tokenizer_training", [train])[
        "training_authority"
    ]
