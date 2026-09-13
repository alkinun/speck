"""Verify the retained reference-token bank and calculate conditional E1/E3 capacity gaps."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.screen_capacity import screen_capacity
from speck.provenance.io import file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("capacity review requires a clean identified checkout")
    if args.output.exists():
        raise FileExistsError(args.output)
    path = args.plan.resolve()
    plan = json.loads(path.read_text())
    if plan.get("format") != "speck_screen_capacity_review_plan" or plan.get("format_version") != 1:
        raise ValueError("unsupported screen capacity review")
    identities = {
        name: _bound_identity(plan[name], path.parent)
        for name in ("data_plan", "data_protocol", "source_registry", "capacity_result")
    }
    data_plan = json.loads(Path(identities["data_plan"]["path"]).read_text())
    registry = json.loads(Path(identities["source_registry"]["path"]).read_text())
    capacity = json.loads(Path(identities["capacity_result"]["path"]).read_text())
    if capacity.get("status") != "firewall_superset_exact_near_exclusion_and_1_2B_capacity_pass":
        raise ValueError("capacity record is not the checked excluded pilot corpus")
    runtime_identity = _bound_identity(capacity["runtime_manifest"], root)
    runtime = json.loads(Path(runtime_identity["path"]).read_text())
    inputs = []
    for source_id in capacity["pilot_training_capacity"]["per_source"]:
        if not source_id.startswith("pilot_train__"):
            raise ValueError("capacity inventory may only read retained training outputs")
        entry = runtime["outputs"][source_id]
        identity = {"path": entry["path"], "sha256": entry["sha256"]}
        inputs.append(
            {
                "source_id": source_id,
                **_bound_identity(identity, Path(runtime_identity["path"]).parent),
            }
        )
    tokenizer = _bound_identity(capacity["pilot_training_capacity"]["tokenizer"], root)
    result = screen_capacity(data_plan, registry, capacity)
    result.update(
        {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "repository_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "review_plan": {"path": str(args.plan), "sha256": file_sha256(path)},
            "inputs": {
                **identities,
                "runtime_manifest": runtime_identity,
                "training_files": inputs,
                "tokenizer": tokenizer,
            },
            "assumptions": plan["assumptions"],
            "verification": "Retained input hashes reverified; token counts are the archived measurements, not a new tokenization run.",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "available_reference_tokens": result["total_available_reference_tokens"],
                "balanced_pool_upper_bound": result["balanced_prior_pool_upper_bound_tokens"],
                "limiting_category": result["limiting_category"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
