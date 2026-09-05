from copy import deepcopy

from scripts.base_train import arguments
from scripts.paper_baseline_storage_qualify import storage_contract_identity


def test_checkpoint_output_override_is_operational_cli_state(tmp_path):
    path = tmp_path / "checkpoints" / "run"
    parsed = arguments(["experiment", "--output-dir", str(path)])

    assert parsed.output_dir == path


def test_storage_identity_ignores_evidence_registration():
    matrix = {
        "paper_id": "paper",
        "planned_primary_baselines": {
            "family_id": "family",
            "output_root": "experiments/family",
            "arms": [],
            "shared_training": {},
            "proxy_confirmation_pairs": [],
        },
        "storage_contract": {
            "checkpoint_retention": "all",
            "estimated_max_bytes_per_run": 1,
            "proxy_confirmation_model_runs": 2,
            "estimated_proxy_checkpoint_bytes": 2,
            "future_finalist_model_runs": 4,
            "estimated_finalist_checkpoint_bytes": 4,
            "minimum_free_bytes_before_proxy_launch": 8,
            "minimum_free_bytes_before_finalist_launch": 16,
            "deletion_policy": "none",
        },
    }
    copied = deepcopy(matrix)
    copied["storage_contract"]["qualification"] = "report.json"

    assert storage_contract_identity(copied) == storage_contract_identity(matrix)
