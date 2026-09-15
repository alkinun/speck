"""Compile source assignments and additive approvals into a preparation-only first wave."""

import argparse
from pathlib import Path

from speck.experiments.first_wave_preparation import load_preparation
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = load_preparation(args.proposal)
    root = repository_root(__file__)
    result["implementation"] = [
        {"path": name, "sha256": file_sha256(root / name)}
        for name in (
            "speck/experiments/first_wave.py",
            "speck/experiments/first_wave_preparation.py",
            "speck/data/rights.py",
            "scripts/first_wave_preparation.py",
        )
    ]
    if "code_preparation_decision" in result["inputs"]:
        name = "speck/experiments/code_preparation_decision.py"
        result["implementation"].append({"path": name, "sha256": file_sha256(root / name)})
    durable_json(args.output, result)
    print(
        {
            "slots": len(result["logical_slots"]),
            "gpu_hours": sum(result["budget_gpu_hours"].values()),
            "source_capacity_tokens": result["source_capacity_total_tokens"],
        }
    )


if __name__ == "__main__":
    main()
