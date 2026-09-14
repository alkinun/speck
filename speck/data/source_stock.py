"""Prepare one approved source through acquisition, full reference exclusion, and counting."""

import json
import resource
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from speck.data.acquisition_units import _bound_identity, _digest, _unit_config, prepare_units
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


def domain_concentration(path):
    hosts = Counter()
    known_documents = unknown_documents = unknown_bytes = 0
    with Path(path).open() as handle:
        for raw in handle:
            row = json.loads(raw)
            size = len(row["text"].encode())
            if row.get("host"):
                hosts[row["host"]] += size
                known_documents += 1
            else:
                unknown_documents += 1
                unknown_bytes += size
    known_bytes = sum(hosts.values())
    return {
        "known_hosts": len(hosts),
        "known_host_documents": known_documents,
        "unknown_host_documents": unknown_documents,
        "known_host_utf8_bytes": known_bytes,
        "unknown_host_utf8_bytes": unknown_bytes,
        "known_host_byte_hhi": sum((size / known_bytes) ** 2 for size in hosts.values())
        if known_bytes
        else None,
        "largest_hosts": [
            {"host": host, "utf8_bytes": size, "share_of_known_host_bytes": size / known_bytes}
            for host, size in sorted(hosts.items(), key=lambda row: (-row[1], row[0]))[:20]
        ],
        "boundary": "Post-exclusion host concentration diagnostic only; no domain reweighting or corpus cap is imposed.",
    }


def reuse_completed_units(plan, acquired, plan_directory):
    """Copy only verified completed unit payloads into a new stock execution."""
    units = {unit["id"]: unit for unit in plan["units"]}
    receipts = []
    for unit_id, identity in plan.get("reuse_acquisition_units", {}).items():
        started = time.perf_counter()
        if unit_id not in units:
            raise ValueError("reused acquisition unit is outside the new plan")
        identity = _bound_identity(identity, plan_directory)
        manifest_path = Path(identity["path"])
        directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete_not_training_data" or manifest.get(
            "config_sha256"
        ) != _digest(_unit_config(plan, units[unit_id])):
            raise ValueError("reused unit differs from the new acquisition configuration")
        files = [(directory / "config.json", None), (manifest_path, identity["sha256"])]
        for entry in (manifest["output"], manifest["security_report"]):
            source = (directory / entry["path"]).resolve()
            if (
                not source.is_relative_to(directory.resolve())
                or file_sha256(source) != entry["sha256"]
            ):
                raise ValueError("reused acquisition payload identity mismatch")
            files.append((source, entry["sha256"]))
        if json.loads((directory / "config.json").read_text()) != _unit_config(
            plan, units[unit_id]
        ):
            raise ValueError("reused acquisition owner differs from its manifest")
        target = Path(acquired) / unit_id
        if not target.exists():
            staging = target.with_name(target.name + ".reuse-building")
            staging.mkdir(parents=True, exist_ok=False)
            for source, expected_hash in files:
                dest = staging / source.relative_to(directory)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                if expected_hash is not None and file_sha256(dest) != expected_hash:
                    raise ValueError("reused unit copy identity mismatch")
            staging.replace(target)
        receipts.append(
            {
                "unit_id": unit_id,
                "source_manifest": identity,
                "elapsed_seconds": time.perf_counter() - started,
                "timing_scope": "This invocation's source/copy verification; previous acquisition is excluded.",
            }
        )
    return receipts


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
            reused_units = reuse_completed_units(plan, output / "acquired", plan_path.parent)
            acquired = prepare_units(plan, output / "acquired")
            if reused_units:
                acquired["reused_unit_inputs"] = reused_units
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
        if plan.get("report_domain_concentration") is True:
            result["domain_concentration"] = domain_concentration(
                output / "excluded" / stock["path"]
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        durable_json(report_path, result)
        return result
    except BaseException as error:
        event("failed", {"type": type(error).__name__, "message": str(error)})
        raise
