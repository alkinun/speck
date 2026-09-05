"""Run a no-output CUDA preflight on the materialized Paper 1 finalist arms."""

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.paper_baseline_preflight import (
    atomic_json,
    command_output,
    preflight_arm,
    repository_revision,
)
from scripts.paper_baseline_preflight_v2 import load_prerequisites, qualify_arm
from speck.paper_finalist import file_sha256, materialize_finalist

EXPECTED = {
    "contract": {
        "path": "research/paper-1/finalist_materialization_v1.json",
        "sha256": "e48468774ed4c83e203fb5eecdb94ed9200837e12802cf406671545cb995c5e3",
    },
    "materialization": {
        "path": "experiments/Speck-Paper1-Finalist-131M/finalist_materialization.json",
        "sha256": "0ad729458cd105e7f2bbe1faecd1351c3e3079c3759551ccf46269f7a02d1792",
    },
    "qualification": {
        "path": "results/Speck-Paper1/finalist-qualification-v1.json",
        "sha256": "e6f74cac020fc63f9476f3c7038b47685c90dee0997efa5299cc18dd6e85c605",
    },
    "analysis_qualification": {
        "path": "results/Speck-Paper1/finalist-analysis-qualified-v1.json",
        "sha256": "a464004a51bdf45fb7e427e2e7915925687e641b5c065b823b9909cc666dd506",
    },
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def load_inputs(repository_root, contract_path):
    contract_path = Path(contract_path).expanduser().resolve()
    expected_contract = repository_root / EXPECTED["contract"]["path"]
    if (
        contract_path != expected_contract
        or not contract_path.is_file()
        or file_sha256(contract_path) != EXPECTED["contract"]["sha256"]
    ):
        raise ValueError("finalist preflight materialization contract changed")
    artifacts = {}
    for name, expected in EXPECTED.items():
        path = repository_root / expected["path"]
        if not path.is_file() or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"finalist preflight input changed: {name}")
        artifacts[name] = json.loads(path.read_text(encoding="utf-8"))
    materialization = materialize_finalist(contract_path, check=True)
    contract = artifacts["contract"]
    qualification = artifacts["qualification"]
    analysis = artifacts["analysis_qualification"]
    if (
        materialization != artifacts["materialization"]
        or qualification.get("decision", {}).get("materialization_qualified") is not True
        or qualification.get("decision", {}).get("data_windows_qualified") is not True
        or qualification.get("decision", {}).get("storage_qualified") is not True
        or qualification.get("decision", {}).get("output_absence_qualified") is not True
        or analysis.get("decision", {}).get("collector_implementation_qualified") is not True
        or analysis.get("decision", {}).get("analysis_implementation_qualified") is not True
        or contract.get("decision", {}).get("training_authorized") is not False
    ):
        raise ValueError("finalist preflight inputs are not qualified")
    external = [
        Path(contract["identity"]["checkpoint_root"]),
        repository_root / contract["identity"]["result_root"],
        repository_root / contract["identity"]["analysis_result"],
        repository_root / contract["identity"]["time_to_quality_lock"],
    ]
    if any(path.exists() for path in external):
        raise FileExistsError("finalist preflight requires every checkpoint/result path to be absent")
    return contract, artifacts


def arm_specs(contract):
    return [
        {
            "id": arm_id,
            "parameters": arm["parameters"],
            "flops_per_token_at_4096": arm["flops_per_token_at_4096"],
            "device_batch_size": 4,
        }
        for arm_id, arm in contract["parent_arms"].items()
    ]


def run(contract_path, device, runner_revision):
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("finalist preflight requires CUDA")
    repository_root = Path(__file__).parents[1]
    contract, artifacts = load_inputs(repository_root, contract_path)
    prerequisites = load_prerequisites(repository_root)
    cache_reference = prerequisites["cache_equivalence_v3"]
    output_root = repository_root / contract["identity"]["output_root"]
    arms = [
        qualify_arm(preflight_arm(arm, output_root, torch.device(device)), cache_reference)
        for arm in arm_specs(contract)
    ]
    return {
        "format": "speck_paper_finalist_preflight",
        "format_version": 1,
        "status": "qualified" if all(arm["passed"] for arm in arms) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_id": contract["paper_id"],
        "runner_revision": runner_revision,
        "runner_sha256": file_sha256(__file__),
        "inputs": {
            name: {**expected, "status": artifacts[name].get("status")}
            for name, expected in EXPECTED.items()
        },
        "prerequisites": prerequisites,
        "hardware": {
            "device": torch.cuda.get_device_name(torch.device(device)),
            "capability": list(torch.cuda.get_device_capability(torch.device(device))),
            "total_memory_bytes": torch.cuda.get_device_properties(
                torch.device(device)
            ).total_memory,
            "driver": command_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ).splitlines()[0],
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        },
        "arms": arms,
        "output_absence_after_preflight": {
            "checkpoint_root_absent": not Path(contract["identity"]["checkpoint_root"]).exists(),
            "result_root_absent": not (
                repository_root / contract["identity"]["result_root"]
            ).exists(),
            "analysis_result_absent": not (
                repository_root / contract["identity"]["analysis_result"]
            ).exists(),
            "time_to_quality_lock_absent": not (
                repository_root / contract["identity"]["time_to_quality_lock"]
            ).exists(),
        },
        "decision": {
            "exact_config_compiled_training_step_qualified": all(
                arm["compiled_training_step"]["within_peak_envelope"] for arm in arms
            ),
            "transformers_export_qualified": all(
                arm["transformers_export"]["passed"] for arm in arms
            ),
            "trained_topology_behavioral_cache_reference_qualified": all(
                arm["behavioral_cache_equivalence"]["passed"] for arm in arms
            ),
            "runtime_preflight_qualified": all(arm["passed"] for arm in arms),
            "release_dependencies_qualified": False,
            "training_authorized": False,
            "automatic_launch_authorized": False,
            "architecture_promotion_authorized": False,
            "paper_scale_authorized": False,
            "next_action": "register the preflight result, preserve release-suite blockers, and reassess the remaining launch boundary without starting training",
        },
        "limitations": [
            "one random-weight full-size compiled step per exact arm is feasibility evidence, not a training benchmark",
            "temporary Transformers exports are hashed but not retained",
            "trained-topology cache behavior is inherited from the pinned powered v3 qualification",
            "free-running cache divergence remains a reported risk",
            "RULER v2, NoLiMa, and HELMET are not qualified by this preflight",
            "the preflight has no training, attribution, promotion, novelty, or paper-scale authority",
        ],
    }


def main(argv=None):
    args = arguments(argv)
    report = run(args.contract, args.device, repository_revision())
    atomic_json(args.output, report)
    if report["status"] != "qualified":
        raise SystemExit("Paper 1 finalist preflight failed")
    print(f"qualified {len(report['arms'])} exact Paper 1 finalist arms")


if __name__ == "__main__":
    main()
