"""Run Paper 1 hardware preflight v2 under the qualified cache-equivalence contract."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.paper_baseline_preflight import (
    atomic_json,
    command_output,
    preflight_arm,
    repository_revision,
)
from speck.paper_baseline import file_sha256, load_matrix

PREREQUISITES = {
    "failed_v1_preflight": {
        "path": "results/Speck-Paper1/baseline-preflight.json",
        "sha256": "164cd1409afd01a25c776dd4471ade27c79eb0d28028042864abb58ed39f5f60",
        "format": "speck_paper_baseline_preflight",
        "format_version": 1,
        "status": "failed",
    },
    "cache_equivalence_v3": {
        "path": "results/Speck-Paper1/cache-equivalence-v3-analysis.json",
        "sha256": "43d2ca48287f1425a207f80d57cf130fab89413ff173eee12d997f6d1ff0d4ac",
        "format": "speck_cache_equivalence_analysis",
        "format_version": 3,
        "status": "qualified",
    },
    "kda_kernel": {
        "path": "results/KimiLinearTransfer/kda_kernel_qualification.json",
        "sha256": "f767c4f7d682cab03b08d7f10f78b360efff0ec12a8d0c292c0324ff04a86c1c",
        "format": "speck_kda_kernel_qualification",
        "format_version": 1,
        "passed": True,
    },
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def load_prerequisites(repository_root):
    results = {}
    for name, expected in PREREQUISITES.items():
        path = repository_root / expected["path"]
        if not path.is_file() or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Paper 1 preflight v2 prerequisite changed: {name}")
        value = json.loads(path.read_text(encoding="utf-8"))
        for key in ("format", "format_version", "status", "passed"):
            if key in expected and value.get(key) != expected[key]:
                raise ValueError(f"Paper 1 preflight v2 prerequisite is invalid: {name}")
        results[name] = {
            "path": expected["path"],
            "sha256": expected["sha256"],
            "status": value.get("status"),
            "passed": value.get("passed"),
        }
    cache = json.loads(
        (repository_root / PREREQUISITES["cache_equivalence_v3"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if not all(decision.get("passed") for decision in cache.get("decisions", ())):
        raise ValueError("Paper 1 cache-equivalence v3 candidate decisions are incomplete")
    return results


def qualify_arm(arm, cache_reference):
    v1_passed = arm["native_incremental"]["passed"]
    arm["native_incremental"]["authority"] = "retained v1 diagnostic without v2 pass/fail authority"
    arm["native_incremental"]["v1_elementwise_passed"] = v1_passed
    arm["behavioral_cache_equivalence"] = {
        **cache_reference,
        "scope": "trained topology, common-history behavior, 88 disjoint cases per length",
        "passed": True,
    }
    arm["passed"] = (
        arm["compiled_training_step"]["within_peak_envelope"]
        and arm["transformers_export"]["passed"]
        and arm["behavioral_cache_equivalence"]["passed"]
    )
    return arm


def run(matrix_path, device, runner_revision):
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Paper 1 baseline preflight v2 requires CUDA")
    matrix_path, matrix = load_matrix(matrix_path)
    repository_root = matrix_path.parents[2]
    prerequisites = load_prerequisites(repository_root)
    output_root = repository_root / matrix["planned_primary_baselines"]["output_root"]
    cache_reference = prerequisites["cache_equivalence_v3"]
    arms = [
        qualify_arm(preflight_arm(arm, output_root, torch.device(device)), cache_reference)
        for arm in matrix["planned_primary_baselines"]["arms"]
    ]
    return {
        "format": "speck_paper_baseline_preflight",
        "format_version": 2,
        "status": "qualified" if all(arm["passed"] for arm in arms) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner_revision": runner_revision,
        "matrix": str(matrix_path.relative_to(repository_root)),
        "matrix_sha256": file_sha256(matrix_path),
        "prerequisites": prerequisites,
        "decision_change": {
            "previous_result_preserved": PREREQUISITES["failed_v1_preflight"]["path"],
            "previous_status": "failed",
            "new_behavioral_authority": PREREQUISITES["cache_equivalence_v3"]["path"],
            "reason": "v1 full-model elementwise tolerance failed the conventional dense BF16 control; v3 prospectively qualified common-history distribution, ranking, and margin-conditioned behavior on a powered disjoint case stream",
            "unchanged": "isolated operator/state correctness, exact-shape compiled training, memory, export, and hard high-margin behavior remain mandatory",
        },
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
        "arms": arms,
        "limitations": [
            "random-weight exact-shape training/export feasibility plus separately qualified trained-topology cache behavior",
            "no language-quality, throughput, architecture-promotion, or paper-scale authorization",
            "free-running divergence remains a mandatory reported risk without v3 decision authority",
            "temporary CPU Transformers exports are hashed but not retained",
        ],
    }


def main(argv=None):
    args = arguments(argv)
    report = run(args.matrix, args.device, repository_revision())
    atomic_json(args.output, report)
    if report["status"] != "qualified":
        raise SystemExit("Paper 1 baseline preflight v2 failed")
    print(f"qualified {len(report['arms'])} Paper 1 baseline arms under preflight v2")


if __name__ == "__main__":
    main()
