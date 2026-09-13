import copy

import pytest
import torch

from speck.provenance.io import file_sha256
from speck.tokenization.pilot import _validate_run
from speck.tokenization.pilot_full_train import _write_evaluation
from speck.tokenization.pilot_orchestration import (
    build_completed_run_record,
    orchestration_boundaries,
)
from speck.tokenization.pilot_recovery import recover_terminal_report
from speck.training.checkpoint import save
from tests.tokenization.test_tokenizer_pilot_orchestration import categories, run


def fixture(tmp_path):
    value = run(tmp_path)
    value.update(run_id="tokenizer-pilot-custom-seed-42", run_fingerprint="a" * 64)
    value["stops"]["run_stop_aligned_tokens"] = 96
    value["flop_correction"]["sha256"] = file_sha256(value["flop_correction"]["path"])
    output = tmp_path / "output"
    (output / "evaluations").mkdir(parents=True)
    boundaries = orchestration_boundaries(value)
    evaluations = {step: categories(1 - step / 100) for step in boundaries["evaluation_steps"]}
    timing = {step: float(step * 2) for step in evaluations}
    identities = [
        _write_evaluation(output, step, evaluations[step], timing[step]) for step in evaluations
    ]
    metadata = {
        "step": 12,
        "run_fingerprint": value["run_fingerprint"],
        "data_state": {"token_offset": 96},
        "evaluation_files": identities,
        "scientific_run": True,
        "D5_opening": False,
    }
    save(output / "checkpoints", 12, {"weight": torch.ones(2)}, {}, metadata)
    files = [
        output / "checkpoints" / name
        for name in (
            "model_000012.pt",
            "optimizer_000012.pt",
            "metadata_000012.json",
            "complete_000012",
        )
    ]
    interruption = {
        "format": "speck_tokenizer_pilot_finalization_interruption",
        "status": "terminal_checkpoint_present_report_recovery_required",
        "run_fingerprint": value["run_fingerprint"],
        "final_step": 12,
        "evaluation_boundaries": identities,
        "checkpoint_files": [
            {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    execution = {
        "run": value,
        "output_directory": str(output),
        "execution_record": {"path": "fixture", "sha256": "b" * 64},
    }
    expected = build_completed_run_record(value, evaluations, timing, 1024)
    return execution, interruption, expected


def test_recovery_preserves_every_quality_and_timing_value_and_discloses_memory_gap(tmp_path):
    execution, interruption, expected = fixture(tmp_path)
    result, summary = recover_terminal_report(execution, interruption)
    assert result["format_version"] == summary["format_version"] == 2
    assert summary["peak_memory_bytes"] is None
    assert result["recovery"]["finite_model_state"]
    comparable = {
        key: value
        for key, value in result.items()
        if key not in ("missing_measurements", "recovery")
    }
    comparable["format_version"] = 1
    for view in ("fixed_document", "fixed_flop"):
        comparable[view] = {**comparable[view], "peak_memory_bytes": 1024}
    assert comparable == expected


@pytest.mark.parametrize("damage", ["checkpoint", "evaluation", "stop", "accounting"])
def test_recovery_rejects_changed_or_nonterminal_artifacts(tmp_path, damage):
    execution, interruption, _ = fixture(tmp_path)
    if damage == "checkpoint":
        path = interruption["checkpoint_files"][0]["path"]
    elif damage == "evaluation":
        path = str(tmp_path / "output" / interruption["evaluation_boundaries"][0]["path"])
    elif damage == "accounting":
        path = execution["run"]["flop_correction"]["path"]
    else:
        interruption["final_step"] -= 1
        path = None
    if path:
        from pathlib import Path

        Path(path).write_text("changed")
    with pytest.raises(ValueError):
        recover_terminal_report(execution, interruption)


def test_missing_memory_schema_cannot_impute_zero_or_hide_other_missing_measurements(tmp_path):
    execution, interruption, _ = fixture(tmp_path)
    result, _ = recover_terminal_report(execution, interruption)
    plan = {
        "confirmation": {"seeds": [42, 43, 44]},
        "stopping": {"fixed_document_mistral_tokens": 80},
    }
    _validate_run(result, plan)
    changed = copy.deepcopy(result)
    changed["fixed_document"]["peak_memory_bytes"] = 0
    with pytest.raises(ValueError, match="null"):
        _validate_run(changed, plan)
    changed = copy.deepcopy(result)
    changed["fixed_document"]["active_seconds"] = None
    with pytest.raises(ValueError, match="active_seconds"):
        _validate_run(changed, plan)
    result["format_version"] = 1
    with pytest.raises(ValueError, match="peak_memory"):
        _validate_run(result, plan)
