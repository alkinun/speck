"""Compile the first-wave proposal into logical comparisons and source-capacity requirements."""

import argparse
import json
from pathlib import Path

from speck.experiments.first_wave import load_first_wave
from speck.provenance.io import file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = load_first_wave(args.proposal)
    root = repository_root(__file__)
    result["implementation"] = [
        {"path": name, "sha256": file_sha256(root / name)}
        for name in ("speck/experiments/first_wave.py", "scripts/first_wave_plan.py")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "logical_slots": len(result["logical_slots"]),
                "gpu_hours": sum(result["budget_gpu_hours"].values()),
                "source_capacity_tokens": result["source_capacity_total_tokens"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
