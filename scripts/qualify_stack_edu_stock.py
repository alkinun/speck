"""Exercise real Stack-Edu fetch/filter/checkpoint replay on the first 512 C metadata rows."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from speck.data.production_rehearsal import _contamination_indexes
from speck.data.stack_edu_stock import count_code_tokens, load_stack_edu_preparation
from speck.data.stack_edu_units import acquire_stack_edu_unit
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root
from speck.tokenization.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("qualification requires a clean frozen implementation")
    if args.report.exists() or args.output.exists():
        raise FileExistsError("qualification output already exists; preserve any earlier attempt")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    plan = load_stack_edu_preparation(args.plan)
    unit = {**next(row for row in plan["units"] if row["language"] == "C"), "stop_row": 512}
    args.output.mkdir(parents=True)
    execution = {
        "repository_revision": revision,
        "plan": {"path": str(args.plan.resolve()), "sha256": file_sha256(args.plan)},
        "unit": unit,
    }
    durable_json(args.output / "execution.json", execution)
    contamination = _contamination_indexes(plan["base"])
    tokenizer = Tokenizer(plan["reference_tokenizer"]["path"])
    with ThreadPoolExecutor(
        max_workers=plan["base"]["stack_edu_policy"]["fetch"]["workers"]
    ) as executor:
        try:
            acquire_stack_edu_unit(
                plan,
                unit,
                args.output / "resumed",
                contamination,
                tokenizer,
                executor,
                interrupt_after_rows=256,
            )
        except RuntimeError as error:
            if str(error) != "injected Stack-Edu acquisition interruption":
                raise
            durable_json(args.output / "interruption.json", {"error": str(error), "expected": True})
        else:
            raise RuntimeError("expected checkpoint interruption did not occur")
        resumed = acquire_stack_edu_unit(
            plan, unit, args.output / "resumed", contamination, tokenizer, executor
        )
        clean = acquire_stack_edu_unit(
            plan, unit, args.output / "clean", contamination, tokenizer, executor
        )
        reopened = acquire_stack_edu_unit(
            plan, unit, args.output / "resumed", contamination, tokenizer, executor
        )
    if resumed["manifest"]["output"] != clean["manifest"]["output"] or not reopened["reused"]:
        raise RuntimeError("real-prefix replay or reopen parity failed")
    manifest = resumed["manifest"]
    capacity = count_code_tokens(
        args.output / "resumed" / unit["id"] / manifest["output"]["path"],
        plan["reference_tokenizer"],
    )
    if capacity["documents"] != manifest["retained_records"] or capacity["documents"] == 0:
        raise RuntimeError("real-prefix output/counter qualification failed")
    result = {
        "format": "speck_stack_edu_stock_prefix_qualification",
        "format_version": 1,
        "status": "bounded_real_prefix_replay_pass",
        **execution,
        "resumed": resumed,
        "clean": clean,
        "counter": capacity,
        "output_directory": str(args.output.resolve()),
        "training_authority": False,
        "boundary": "First 512 C metadata rows only, with real SWH payloads, original filters, Gitleaks, selected tokenizer, checkpoint interruption/replay, and complete-unit reopen. No full reference/candidate exclusion; these counts are diagnostics, not source capacity. Clean/replay fetches share verified cached inputs, so no independent network throughput comparison.",
    }
    durable_json(args.report, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "documents": capacity["documents"],
                "tokens": capacity["tokens"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
