import json
from pathlib import Path

import speck.production_calibration as production_calibration
from speck.data_rehearsal import STAGE_FORMAT
from speck.io import atomic_json
from speck.production_calibration import run_production_calibration


def test_six_stage_2b_calibration_projects_without_issuing_authority(tmp_path, monkeypatch):
    gates = {
        "source_identity": {"source_identity": "pass"},
        "acquisition": {},
        "global_dedup": {
            "global_exact_deduplication": "pass",
            "global_near_deduplication": "pass",
        },
        "packing": {"packing_integrity": "pass"},
        "resume_cleanup": {"acquisition_cleanup": "pass", "interruption_resume": "pass"},
        "firewall_disjointness": {"firewall_training_disjointness": "pass"},
    }
    metrics = {
        "source_identity": {"records_seen": 100},
        "acquisition": {
            "download_bytes": 10,
            "download_bytes_per_second": 2,
            "filtered_bytes": 20,
        },
        "global_dedup": {
            "records_retained": 90,
            "sqlite_index_bytes": 30,
            "peak_rss_bytes": 40,
        },
        "packing": {
            "packing_tokens_per_second": 50,
            "packed_tokens": 2_000_000_000,
            "packed_bytes": 4_000_000_000,
            "unique_tokens": 2_000_000_000,
        },
        "resume_cleanup": {},
        "firewall_disjointness": {},
    }

    def fake_stage(plan, stage_id, result_path):
        result = {
            "format": STAGE_FORMAT,
            "format_version": 1,
            "status": "complete",
            "stage_id": stage_id,
            "gates": gates[stage_id],
            "metrics": metrics[stage_id],
            "details": {},
        }
        atomic_json(result_path, result)
        return result

    monkeypatch.setattr(production_calibration, "run_production_rehearsal_stage", fake_stage)
    plan = {
        "output_directory": str(tmp_path / "work"),
        "target_tokens": 2_000_000_000,
        "plan_fingerprint": "a" * 64,
        "rights_record": {"path": "rights", "sha256": "b" * 64},
        "deny_ledger": {"path": "deny", "sha256": "c" * 64},
        "sources": [
            {"id": category, "category": category}
            for category in ("web", "code", "math", "synthetic", "science", "reference")
        ],
    }
    result = run_production_calibration(plan)

    assert result["status"] == (
        "production_2B_calibration_complete_not_20B_operations_or_training_authority"
    )
    assert result["operations_authority"] is False
    assert result["training_authority"] is False
    assert result["projection_20B"]["linear_scale_factor"] == 10
    assert result["projection_20B"]["projected_packed_bytes"] == 40_000_000_000
    assert json.loads((tmp_path / "calibration-manifest.json").read_text()) == result


def test_checked_2b_plan_preserves_six_categories_and_exact_quota():
    root = Path(__file__).parents[1]
    plan = json.loads(
        (root / "research/flagship/data_calibration_2b_v1/production_plan.json").read_text()
    )

    assert plan["format"] == "speck_production_calibration_plan"
    assert plan["target_tokens"] == 2_000_000_000
    assert sum(source["target_tokens"] for source in plan["sources"]) == 2_000_000_000
    assert [source["category"] for source in plan["sources"]] == [
        "web",
        "code",
        "math",
        "synthetic",
        "science",
        "reference",
    ]


def test_calibration_loader_preserves_normalized_reader_defaults(tmp_path, monkeypatch):
    raw = {
        "format": "speck_production_calibration_plan",
        "format_version": 1,
        "status": "frozen_2B_calibration_not_operations_or_training_authority",
        "target_tokens": 2_000_000_000,
        "sources": [
            {
                "id": "source",
                "category": "web",
                "target_tokens": 2_000_000_000,
                "reader": {"id": "source"},
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw))

    def fake_validate(candidate, **kwargs):
        assert candidate["target_tokens"] == 20_000_000_000
        assert candidate["sources"][0]["target_tokens"] == 20_000_000_000
        return {
            **candidate,
            "sources": [
                {
                    **candidate["sources"][0],
                    "reader": {"id": "source", "file_format": "parquet"},
                }
            ],
            "plan_fingerprint": "old",
        }

    monkeypatch.setattr(production_calibration, "validate_production_rehearsal_plan", fake_validate)
    normalized = production_calibration.load_production_calibration_plan(path)

    assert normalized["target_tokens"] == 2_000_000_000
    assert normalized["sources"][0]["target_tokens"] == 2_000_000_000
    assert normalized["sources"][0]["reader"]["file_format"] == "parquet"
