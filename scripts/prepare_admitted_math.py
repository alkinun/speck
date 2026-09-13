"""Prepare admitted Math L2 text, exclude all firewall references, and measure usable capacity."""

import argparse
import json
import resource
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from speck.data.acquisition_units import _bound_identity, prepare_units
from speck.data.admitted_math import count_reference_tokens, load_math_preparation
from speck.data.checkpoint_replay import restore_reference_checkpoint
from speck.data.firewall_integration import (
    analyze_exclusion,
    group_acquisition_units,
    run_exclusion,
)
from speck.data.preparation_policy import load_preparation_policy
from speck.data.sqlite_wal import wal_policy
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("admitted source preparation requires a clean implementation")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if args.report.exists():
        raise FileExistsError(args.report)
    plan = load_math_preparation(args.plan)
    output = Path(plan["output_directory"])
    execution = {
        "repository_revision": revision,
        "plan": {"path": str(args.plan.resolve()), "sha256": file_sha256(args.plan)},
    }
    if output.exists():
        if not args.resume or json.loads((output / "execution.json").read_text()) != execution:
            raise ValueError("preparation resume requires the same frozen execution")
    else:
        if args.resume:
            raise ValueError("cannot resume an absent preparation")
        output.mkdir(parents=True)
        durable_json(output / "execution.json", execution)
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"events": []}

    def event(stage, details):
        progress["events"].append(
            {
                "stage": stage,
                "at": datetime.now(timezone.utc).isoformat(),
                "details": details,
                "free_bytes": shutil.disk_usage(output).free,
                "main_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            }
        )
        durable_json(progress_path, progress)
        print(f"{stage}: {json.dumps(details, sort_keys=True)}", flush=True)

    event(
        "started",
        {
            "source_id": plan["source_id"],
            "source_use": plan["source_use"]["identity"],
            "resume": args.resume,
        },
    )
    try:
        acquired_path = output / "acquisition-report.json"
        if acquired_path.exists():
            acquired = json.loads(acquired_path.read_text())
        else:
            acquired = prepare_units(plan, output / "acquired")
            durable_json(acquired_path, acquired)
        event(
            "acquired",
            {
                "retained_records": sum(
                    item["manifest"]["retained_records"] for item in acquired["units"]
                ),
                "elapsed_seconds": acquired["elapsed_seconds"],
            },
        )
        groups = group_acquisition_units(plan, output / "acquired", output / "groups")
        parent_identity = _bound_identity(plan["reference_parent"], args.plan.resolve().parent)
        parent = json.loads(Path(parent_identity["path"]).read_text())
        policy_identity = _bound_identity(plan["sqlite_policy"], args.plan.resolve().parent)
        policy = load_preparation_policy(policy_identity["path"])
        candidate_inputs = {
            f"acquired_train__{category}": {
                "path": str(output / "groups" / item["path"]),
                "sha256": item["sha256"],
            }
            for category, item in groups["outputs"].items()
        }
        config_path = output / "preprocess.json"
        restore_path = output / "restoration.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            restoration = json.loads(restore_path.read_text())
        else:
            config, restoration = restore_reference_checkpoint(
                parent,
                output / "excluded",
                sqlite_settings=policy["sqlite"],
                candidate_inputs=candidate_inputs,
            )
            durable_json(config_path, config)
            durable_json(restore_path, restoration)
        event("reference_prefix_ready", {"records": restoration["reference_records"]})
        excluded_path = output / "exclusion-report.json"
        if excluded_path.exists():
            excluded = json.loads(excluded_path.read_text())
        else:
            timing, storage = {}, {}
            with wal_policy(policy["sqlite"]["wal_autocheckpoint_pages"], storage):
                excluded = run_exclusion(config, timing=timing)
            excluded.update({"timing": timing, "storage": storage})
            durable_json(excluded_path, excluded)
        event(
            "excluded",
            {
                "elapsed_seconds": excluded["elapsed_seconds"],
                "observed_wal_bytes": excluded["storage"]["observed_peak_wal_bytes"],
            },
        )
        analysis = analyze_exclusion(output / "excluded", parent["analysis"]["references"])
        math = analysis["retained"]["math"]
        if any(
            value["records"]
            for category, value in analysis["retained"].items()
            if category != "math"
        ):
            raise RuntimeError("single-source preparation unexpectedly retained another category")
        count_path = output / "reference-count.json"
        if count_path.exists():
            capacity = json.loads(count_path.read_text())
        else:
            capacity = count_reference_tokens(
                output / "excluded" / math["path"], plan["reference_tokenizer"]
            )
            durable_json(count_path, capacity)
        peak = excluded["storage"]["observed_peak_wal_bytes"]
        capacity_pass = capacity["tokens"] >= plan["target_reference_tokens"]
        storage_pass = peak <= plan["maximum_observed_wal_bytes"]
        event(
            "capacity",
            {
                "reference_tokens": capacity["tokens"],
                "target": plan["target_reference_tokens"],
                "capacity_pass": capacity_pass,
                "storage_pass": storage_pass,
            },
        )
        result = {
            "format": "speck_admitted_math_preparation_result",
            "format_version": 1,
            "status": "reference_capacity_target_met"
            if capacity_pass and storage_pass
            else "prepared_text_requires_capacity_or_storage_followup",
            **execution,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "source_use_extension": plan["source_use"]["identity"],
            "acquisition": acquired,
            "groups": groups,
            "reference_parent": parent_identity,
            "restoration": restoration,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "exclusion": excluded,
            "analysis": analysis,
            "reference_capacity": capacity,
            "capacity_target_pass": capacity_pass,
            "storage_gate_pass": storage_pass,
            "within_prior_512MiB_wal_observation": peak <= 536870912,
            "progress": {"path": str(progress_path), "sha256": file_sha256(progress_path)},
            "boundary": "Admitted natural Math L2, fixed two-shard text preparation and declared reference exclusion. Reference-token counts are not final D5 packing or training authority. New larger source transactions are measured under a 2GiB observed-WAL envelope; this is not a hard WAL cap or a speed comparison.",
            "training_authority": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        durable_json(args.report, result)
    except BaseException as error:
        event("failed", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
