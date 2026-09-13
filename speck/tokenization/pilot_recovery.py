"""Recover a completed tokenizer run's report without replaying training or evaluation."""

import json
from pathlib import Path

import torch

from speck.provenance.io import file_sha256
from speck.tokenization.pilot import RECOVERED_MEMORY_GAP, _validate_run
from speck.tokenization.pilot_full_train import _load_evaluations
from speck.tokenization.pilot_orchestration import (
    build_completed_run_record,
    orchestration_boundaries,
)


def recover_terminal_report(execution, interruption):
    """Verify the retained terminal artifacts, then rebuild only the report payloads."""

    run = execution["run"]
    output = Path(execution["output_directory"])
    final = orchestration_boundaries(run)["final_step"]
    if (
        interruption.get("format") != "speck_tokenizer_pilot_finalization_interruption"
        or interruption.get("status") != "terminal_checkpoint_present_report_recovery_required"
        or interruption.get("run_fingerprint") != run["run_fingerprint"]
        or interruption.get("final_step") != final
    ):
        raise ValueError("terminal recovery does not match the recorded interrupted publication")
    expected = {
        output / "checkpoints" / name
        for name in (
            f"model_{final:06d}.pt",
            f"optimizer_{final:06d}.pt",
            f"metadata_{final:06d}.json",
            f"complete_{final:06d}",
        )
    }
    identities = interruption["checkpoint_files"]
    if {Path(item["path"]) for item in identities} != expected or len(identities) != 4:
        raise ValueError("terminal recovery requires the exact four checkpoint identities")
    for item in identities:
        path = Path(item["path"])
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise ValueError("terminal checkpoint identity mismatch")
    checkpoint = output / "checkpoints"
    if (checkpoint / f"complete_{final:06d}").read_text() != "complete\n":
        raise ValueError("terminal checkpoint completion marker is invalid")
    metadata = json.loads((checkpoint / f"metadata_{final:06d}.json").read_text())
    if (
        metadata.get("step") != final
        or metadata.get("run_fingerprint") != run["run_fingerprint"]
        or metadata.get("scientific_run") is not True
        or metadata.get("D5_opening") is not False
        or metadata.get("data_state", {}).get("token_offset")
        != run["stops"]["run_stop_aligned_tokens"]
        or metadata.get("evaluation_files") != interruption["evaluation_boundaries"]
    ):
        raise ValueError("terminal checkpoint state/evaluation lineage is inconsistent")
    correction = run["flop_correction"]
    if file_sha256(correction["path"]) != correction["sha256"]:
        raise ValueError("terminal recovery accounting input changed")
    weights = torch.load(
        checkpoint / f"model_{final:06d}.pt", map_location="cpu", weights_only=True
    )
    if not weights or any(
        not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all().item()
        for tensor in weights.values()
    ):
        raise ValueError("terminal model state contains invalid or non-finite tensors")
    del weights
    evaluations, timing, evaluation_files = _load_evaluations(output, metadata["evaluation_files"])
    result = build_completed_run_record(run, evaluations, timing, None)
    result["format_version"] = 2
    result["missing_measurements"] = dict(RECOVERED_MEMORY_GAP)
    result["recovery"] = {
        "method": "terminal checkpoint and bound evaluation records; no training/evaluation replay",
        "run_fingerprint": run["run_fingerprint"],
        "checkpoint_files": identities,
        "evaluation_files": evaluation_files,
        "accounting_input": correction,
        "finite_model_state": True,
    }
    horizon = json.loads(Path(correction["path"]).read_text())["reference"]["fixed_document_tokens"]
    _validate_run(
        result,
        {
            "confirmation": {"seeds": [42, 43, 44]},
            "stopping": {"fixed_document_mistral_tokens": horizon},
        },
    )
    summary = {
        "format": "speck_tokenizer_pilot_run_summary",
        "format_version": 2,
        "status": "complete",
        "run_id": run["run_id"],
        "run_fingerprint": run["run_fingerprint"],
        "execution_record": execution["execution_record"],
        "evaluation_files": evaluation_files,
        "final_checkpoint_step": final,
        "active_seconds": timing[final],
        "peak_memory_bytes": None,
        "missing_measurements": dict(RECOVERED_MEMORY_GAP),
        "D5_opening": False,
    }
    return result, summary
