"""Verify a stopped ordered acquisition and bind read-only completed-unit reuse."""

import fcntl
import json
import subprocess
import time
from pathlib import Path

from speck.data.acquisition_units import _digest
from speck.data.stack_edu_ordered_units import archive_batch, reopen_batch
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def bound_json(binding):
    path = Path(binding["path"])
    if file_sha256(path) != binding["sha256"]:
        raise ValueError("recovery input identity changed")
    return json.loads(path.read_text())


def inventory_completed(plan_path, result_path):
    """Hash all completed payloads and existing archives; never acquire or repair files."""
    plan_path, result_path = Path(plan_path).resolve(), Path(result_path).resolve()
    started = time.perf_counter()
    if result_path.exists():
        raise FileExistsError("preserve the previous recovery inventory")
    spec = json.loads(plan_path.read_text())
    if spec["format"] != "speck_stack_edu_ordered_acquisition" or spec["format_version"] != 1:
        raise ValueError("recovery inventory requires the original ordered plan")
    work, archive = Path(spec["working_directory"]), Path(spec["archive_directory"])
    with (work / "owner.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = identity(work / "execution.json")
        execution = bound_json(owner)
        if execution["plan"] != identity(plan_path):
            raise ValueError("recovery owner differs from predecessor plan")
        archive_owner = identity(archive / "execution.json")
        if bound_json(archive_owner) != execution:
            raise ValueError("recovery archive owner differs")
        progress_id = identity(work / "progress.json")
        progress = bound_json(progress_id)
        invocation_path = sorted(work.glob("invocation-*.json"))[-1]
        invocation_id = identity(invocation_path)
        if bound_json(invocation_id).get("status") != "failed_preserved":
            raise ValueError("recovery requires a preserved failed invocation")
        directories = sorted((work / "acquired").iterdir())
        rows = []
        for directory in directories:
            if directory.is_symlink() or not (directory / "manifest.json").is_file():
                raise ValueError("incomplete or linked unit needs separate recovery")
            config_id = identity(directory / "config.json")
            config = bound_json(config_id)
            manifest = reopen_batch(directory, config)
            if manifest["unit_id"] != directory.name:
                raise ValueError("recovery unit directory differs")
            archive_directory = archive / "units" / directory.name
            # archive_batch only verifies an existing receipt. Never let this audit create one.
            if not (archive_directory / "manifest.json").is_file():
                raise ValueError("completed unit lacks its preserved archive")
            receipt = archive_batch(directory, manifest, archive_directory)
            if receipt.get("all_archival_payload_hashes_reopened") is not True:
                raise ValueError("archive lacks original full-payload verification")
            rows.append(
                {
                    "unit_id": directory.name,
                    "directory": str(directory),
                    "config": config_id,
                    "manifest": identity(directory / "manifest.json"),
                    "archive_directory": str(archive_directory),
                    "archive_manifest": identity(archive_directory / "manifest.json"),
                    "tokens_before_full_exclusion": manifest["tokens_before_full_exclusion"],
                    "retained_records": manifest["retained_records"],
                }
            )
            if len(rows) % 50 == 0:
                print(json.dumps({"verified_completed_units": len(rows)}), flush=True)
        if (
            len(rows) != progress["complete_units"]
            or sum(r["tokens_before_full_exclusion"] for r in rows)
            != progress["tokens_before_full_exclusion"]
            or sum(r["retained_records"] for r in rows) != progress["retained_records"]
        ):
            raise ValueError("recovery inventory differs from last durable progress")
        result = {
            "format": "speck_ordered_stock_recovery_inventory",
            "format_version": 1,
            "status": "completed_units_and_archives_reopened",
            "plan": identity(plan_path),
            "execution": owner,
            "archive_execution": archive_owner,
            "failed_invocation": invocation_id,
            "progress": progress_id,
            "units": rows,
            "repository_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository_root(), text=True
            ).strip(),
            "implementation": [
                identity(repository_root() / name)
                for name in (
                    "speck/data/ordered_stock_recovery.py",
                    "speck/data/stack_edu_ordered_units.py",
                    "speck/data/acquisition_units.py",
                    "speck/provenance/io.py",
                )
            ],
            "wall_seconds_this_verification": time.perf_counter() - started,
            "training_authority": False,
            "full_exclusion_performed": False,
            "boundary": "Completed unit payloads and existing archive tar/inventory hashes reopened; original archive creation verified each member. No new acquisition, full exclusion, stock qualification or repair. Original files are retained in place.",
        }
        durable_json(result_path, result)
        return result


def load_recovery_successor(spec, predecessor, predecessor_identity):
    """Allow only fresh working/archive locations and exact verified predecessor reuse."""
    old = bound_json(predecessor_identity)
    allowed = {
        "format_version",
        "scope",
        "working_directory",
        "archive_directory",
        "fallback_caches",
        "supersedes",
        "reuse_inventory",
    }
    if {k: v for k, v in spec.items() if k not in allowed} != {
        k: v for k, v in old.items() if k not in allowed
    }:
        raise ValueError("recovery successor changes acquisition scope or storage guards")
    work, archive = Path(old["working_directory"]), Path(old["archive_directory"])
    expected_fallbacks = [str(work / "cache"), *old["fallback_caches"]]
    if spec["fallback_caches"] != expected_fallbacks:
        raise ValueError("recovery must preserve fallback order and prior cache")
    roots = [Path(spec[k]).resolve() for k in ("working_directory", "archive_directory")]
    protected = [work.resolve(), archive.resolve(), *map(Path, expected_fallbacks)]
    for a, b in [(roots[0], roots[1]), *[(a, b) for a in roots for b in protected]]:
        if a == b or a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError("recovery paths overlap preserved inputs")
    inventory = bound_json(spec["reuse_inventory"])
    if (
        inventory.get("format") != "speck_ordered_stock_recovery_inventory"
        or inventory.get("format_version") != 1
        or inventory.get("status") != "completed_units_and_archives_reopened"
        or inventory["plan"] != predecessor_identity
        or inventory.get("training_authority") is not False
        or inventory.get("full_exclusion_performed") is not False
    ):
        raise ValueError("unqualified recovery inventory")
    for key in ("execution", "archive_execution", "progress", "failed_invocation"):
        bound_json(inventory[key])
    units = {}
    for row in inventory["units"]:
        name = row["unit_id"]
        if name in units or not name or Path(name).name != name:
            raise ValueError("duplicate or invalid recovery unit")
        directory, archived = work / "acquired" / name, archive / "units" / name
        if row["directory"] != str(directory) or row["archive_directory"] != str(archived):
            raise ValueError("recovery unit path differs")
        for key, expected in (
            ("config", directory / "config.json"),
            ("manifest", directory / "manifest.json"),
            ("archive_manifest", archived / "manifest.json"),
        ):
            if row[key]["path"] != str(expected):
                raise ValueError("recovery binding path differs")
        units[name] = row
    if not units or len(units) != bound_json(inventory["progress"])["complete_units"]:
        raise ValueError("recovery inventory count differs")
    return {
        **predecessor,
        **{key: spec[key] for key in allowed},
        "reuse_units": units,
        "reuse_owner_directory": str(work),
    }


def reopen_reused_unit(row, expected_config):
    config = bound_json(row["config"])
    if config != expected_config:
        raise ValueError("recovery unit configuration differs from ordered prefix")
    manifest = bound_json(row["manifest"])
    if manifest["config_sha256"] != _digest(config):
        raise ValueError("recovery manifest configuration differs")
    bound_json(row["archive_manifest"])
    return reopen_batch(Path(row["directory"]), config)
