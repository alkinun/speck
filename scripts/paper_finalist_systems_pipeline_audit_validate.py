"""Validate the blocked finalist systems producer-consumer interface audit."""

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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_audit(audit, root):
    require(
        audit.get("format") == "speck_paper_finalist_systems_pipeline_interface_audit"
        and audit.get("format_version") == 1
        and audit.get("status") == "components_qualified_end_to_end_pipeline_blocked",
        "invalid systems pipeline audit identity",
    )
    for name, reference in audit.get("inputs", {}).items():
        path = root / reference.get("path", "")
        require(
            path.is_file() and file_sha256(path) == reference.get("sha256"),
            f"systems pipeline {name} implementation changed",
        )
        qualification = reference.get("qualification")
        if qualification:
            qualification_path = root / qualification
            require(
                qualification_path.is_file()
                and file_sha256(qualification_path) == reference.get("qualification_sha256"),
                f"systems pipeline {name} qualification changed",
            )
    require(len(audit.get("inputs", {})) == 5, "systems pipeline input set changed")
    bridges = audit.get("qualified_bridges", [])
    require(
        [bridge.get("id") for bridge in bridges]
        == [
            "integrator_to_analyzer_energy",
            "engine_to_analyzer_primary_time",
            "engine_to_analyzer_torch_memory",
        ]
        and all(bridge.get("consumer_match") is True for bridge in bridges),
        "systems pipeline qualified bridges changed",
    )
    blockers = audit.get("blocking_interfaces", [])
    require(
        [blocker.get("id") for blocker in blockers]
        == [
            "sample_series_to_trace_envelope",
            "NVML_peak_used_memory",
            "thermal_idle_and_sampler_engine_orchestration",
            "compiled_runtime_attestation",
            "paired_batch_fingerprint_consumption",
            "trial_block_and_failure_assembly",
            "runtime_software_identity_chain",
        ],
        "systems pipeline blocker inventory changed",
    )
    nvml = blockers[1]
    runtime = blockers[3]
    require(
        nvml.get("sampler_missing_field") == "memory.used"
        and nvml.get("analyzer_required_field") == "peak_nvml_used_bytes"
        and runtime.get("engine_value") == {"kernel_fallback": None, "CUDA_phase_timing": False}
        and runtime.get("analyzer_requires_kernel_fallback_false") is True,
        "systems pipeline reproduced mismatch changed",
    )
    activation = audit.get("activation_boundary", {})
    require(
        activation.get("activation_artifact_present") is False
        and activation.get("component_qualification_implies_pipeline_qualification") is False
        and activation.get("component_qualification_implies_execution_authority") is False
        and activation.get("blocking_interfaces") == 7,
        "systems pipeline activation boundary changed",
    )
    decision = audit.get("decision", {})
    require(
        decision.get("individual_CPU_mock_components_remain_qualified") is True
        and all(
            decision.get(field) is False
            for field in (
                "end_to_end_systems_pipeline_qualified",
                "systems_execution_authorized",
                "systems_claim_authorized",
                "protocol_or_component_modified_by_audit",
                "active_language_sequence_or_program_changed",
            )
        ),
        "systems pipeline fail-closed decision changed",
    )
    return {
        "status": "valid_blocked",
        "qualified_bridges": len(bridges),
        "blocking_interfaces": len(blockers),
        "execution_authorized": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_audit(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).audit)
    print(
        "Finalist systems pipeline audit: "
        f"{report['status']} ({report['qualified_bridges']} bridges, "
        f"{report['blocking_interfaces']} blockers)"
    )


if __name__ == "__main__":
    main()
