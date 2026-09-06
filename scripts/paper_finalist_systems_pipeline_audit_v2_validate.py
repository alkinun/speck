"""Validate the converged static/live boundary for finalist systems evidence."""

import argparse
import hashlib
import json
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audit(audit, root):
    if (
        audit.get("format") != "speck_paper_finalist_systems_pipeline_interface_audit"
        or audit.get("format_version") != 2
        or audit.get("status") != "static_interfaces_composed_live_activation_gates_blocked"
    ):
        raise ValueError("invalid v2 systems pipeline audit identity")
    for name, reference in {
        "predecessor": audit.get("predecessor", {}),
        **audit.get("successors", {}),
    }.items():
        path = root / reference.get("path", "")
        if not path.is_file() or file_sha256(path) != reference.get("sha256"):
            raise ValueError(f"v2 systems pipeline {name} evidence changed")
    dispositions = audit.get("v1_blocker_dispositions", [])
    if (
        len(dispositions) != 7
        or [value.get("id") for value in dispositions]
        != [
            "sample_series_to_trace_envelope",
            "NVML_peak_used_memory",
            "thermal_idle_and_sampler_engine_orchestration",
            "compiled_runtime_attestation",
            "paired_batch_fingerprint_consumption",
            "trial_block_and_failure_assembly",
            "runtime_software_identity_chain",
        ]
        or dispositions[4].get("static_state") != "closed"
        or dispositions[4].get("remaining") is not None
    ):
        raise ValueError("v2 systems pipeline blocker dispositions changed")
    gates = audit.get("remaining_activation_gates", [])
    if [gate.get("id") for gate in gates] != [
        "language_sequence",
        "actual_path_GPU_sandbox_preflight",
        "live_sampler_environment",
        "runtime_probe_producer",
        "live_adapter_integration",
    ] or any(gate.get("current_pass") is not False for gate in gates):
        raise ValueError("v2 systems pipeline activation gates changed")
    activation = audit.get("activation", {})
    if any(
        activation.get(field) is not False
        for field in (
            "artifact_present",
            "may_be_created_before_all_five_gates_pass",
            "individual_or_static_composition_implies_execution",
            "systems_execution_authorized",
            "systems_claim_authorized",
        )
    ):
        raise ValueError("v2 systems pipeline activation boundary changed")
    decision = audit.get("decision", {})
    if (
        decision.get("v1_audit_preserved") is not True
        or decision.get("static_producer_consumer_contracts_materially_advanced") is not True
        or any(
            decision.get(field) is not False
            for field in (
                "end_to_end_live_pipeline_qualified",
                "systems_execution_authorized",
                "further_live_work_during_language_sequence_authorized",
                "active_language_sequence_or_program_changed",
            )
        )
    ):
        raise ValueError("v2 systems pipeline decision changed")
    return {
        "status": "valid_live_blocked",
        "historical_blockers": len(dispositions),
        "remaining_activation_gates": len(gates),
        "activation_present": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_audit(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).audit)
    print(
        "Finalist systems pipeline v2 audit: "
        f"{report['status']} ({report['historical_blockers']} dispositions, "
        f"{report['remaining_activation_gates']} live gates)"
    )


if __name__ == "__main__":
    main()
