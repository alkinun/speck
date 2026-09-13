"""Prepare one approved source through acquisition, full reference exclusion, and counting."""

import json
import resource
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from speck.data.acquisition_units import _bound_identity, prepare_units
from speck.data.admitted_math import count_reference_tokens
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


def prepare_source_stock(
    plan_path, report_path, *, loader, category, result_format, boundary, resume=False
):
    plan_path, report_path = Path(plan_path).resolve(), Path(report_path).resolve()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("admitted source preparation requires a clean implementation")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if report_path.exists():
        raise FileExistsError(report_path)
    plan = loader(plan_path)
    if not plan["units"] or any(unit["category"] != category for unit in plan["units"]):
        raise ValueError("single-source stock category disagrees with its units")
    output = Path(plan["output_directory"])
    execution = {
        "repository_revision": revision,
        "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
    }
    if output.exists():
        if not resume or json.loads((output / "execution.json").read_text()) != execution:
            raise ValueError("preparation resume requires the same frozen execution")
    else:
        if resume:
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
            "resume": resume,
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
        parent_identity = _bound_identity(plan["reference_parent"], plan_path.parent)
        parent = json.loads(Path(parent_identity["path"]).read_text())
        policy_identity = _bound_identity(plan["sqlite_policy"], plan_path.parent)
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
        stock = analysis["retained"][category]
        if any(value["records"] for key, value in analysis["retained"].items() if key != category):
            raise RuntimeError("single-source preparation unexpectedly retained another category")
        count_path = output / "reference-count.json"
        if count_path.exists():
            capacity = json.loads(count_path.read_text())
        else:
            capacity = count_reference_tokens(
                output / "excluded" / stock["path"], plan["reference_tokenizer"]
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
            "format": result_format,
            "format_version": 1,
            "status": "reference_capacity_target_met"
            if capacity_pass and storage_pass
            else "prepared_text_requires_capacity_or_storage_followup",
            **execution,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            ("source_use_extension" if "source_use_extension" in plan else "source_use"): plan[
                "source_use"
            ]["identity"],
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
            "boundary": boundary,
            "training_authority": False,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        durable_json(report_path, result)
        return result
    except BaseException as error:
        event("failed", {"type": type(error).__name__, "message": str(error)})
        raise
