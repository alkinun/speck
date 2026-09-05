"""Qualify a dedicated checkpoint volume for the frozen Paper 1 baseline matrix."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.helmet_data_acquire import qualify_volume
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint
from speck.dataset import load_manifest, resolve_data_dir


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--volume-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def repository_revision():
    return subprocess.run(
        ["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def storage_contract_identity(matrix):
    planned = matrix["planned_primary_baselines"]
    storage = matrix["storage_contract"]
    payload = {
        "paper_id": matrix["paper_id"],
        "family_id": planned["family_id"],
        "output_root": planned["output_root"],
        "arms": planned["arms"],
        "shared_training": planned["shared_training"],
        "proxy_confirmation_pairs": planned["proxy_confirmation_pairs"],
        "storage_contract": {
            key: storage[key]
            for key in (
                "checkpoint_retention",
                "estimated_max_bytes_per_run",
                "proxy_confirmation_model_runs",
                "estimated_proxy_checkpoint_bytes",
                "future_finalist_model_runs",
                "estimated_finalist_checkpoint_bytes",
                "minimum_free_bytes_before_proxy_launch",
                "minimum_free_bytes_before_finalist_launch",
                "deletion_policy",
            )
        },
    }
    return bytes_sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _inputs(args, require_empty):
    root = Path(__file__).parents[1]
    matrix_path = args.matrix.expanduser().resolve()
    materialization_path = args.materialization.expanduser().resolve()
    volume = args.volume_dir.expanduser().absolute()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    storage = matrix["storage_contract"]
    if (
        matrix.get("format") != "speck_paper_baseline_matrix"
        or materialization.get("format") != "speck_paper_baseline_materialization"
        or materialization.get("status") != "materialized_unexecuted"
        or materialization.get("contract_sha256") != file_sha256(matrix_path)
    ):
        raise ValueError("Paper 1 baseline storage inputs do not match")
    mount = qualify_volume(volume, storage["minimum_free_bytes_before_proxy_launch"])
    experiment_root = root / matrix["planned_primary_baselines"]["output_root"] / "runs"
    experiments = sorted(path for path in experiment_root.glob("*/*") if path.is_dir())
    if len(experiments) != storage["proxy_confirmation_model_runs"]:
        raise ValueError("Paper 1 baseline experiment count changed")
    runs = []
    data_fingerprints = set()
    for experiment in experiments:
        configs = load_experiment(experiment, "data", "tokenizer", "model", "train")
        run = configs["train"]["run"]
        output = volume / run
        data_dir = resolve_data_dir(
            configs["data"].get("output_dir"), configs["data"].get("output_name")
        ).expanduser().resolve()
        fingerprint = manifest_fingerprint(load_manifest(data_dir))
        data_fingerprints.add(fingerprint)
        if configs["train"].get("output_dir") is not None:
            raise ValueError(f"baseline experiment unexpectedly embeds output path: {experiment}")
        if require_empty and output.exists():
            raise FileExistsError(f"Paper 1 checkpoint target already exists: {output}")
        runs.append(
            {
                "experiment": experiment.relative_to(root).as_posix(),
                "run": run,
                "checkpoint_directory": str(output),
                "data_directory": str(data_dir),
                "data_manifest_fingerprint": fingerprint,
                "launch": (
                    f"uv run --extra gpu --extra linear python -m scripts.base_train "
                    f"{experiment.relative_to(root).as_posix()} --output-dir {output}"
                ),
                "collect": (
                    "uv run --extra cpu python -m scripts.paper_baseline_analyze collect "
                    f"research/paper-1/baseline_analysis.json "
                    f"{experiment.relative_to(root).as_posix()} --checkpoint-dir {output} "
                    f"--output results/Speck-Paper1/runs/{run}.json"
                ),
            }
        )
    expected_fingerprint = matrix["planned_primary_baselines"]["shared_training"][
        "data_manifest"
    ]
    if data_fingerprints != {expected_fingerprint}:
        raise ValueError("Paper 1 baseline data fingerprint changed")
    allowed = {entry["run"] for entry in runs} | {"paper1-storage-qualification.json"}
    existing = sorted(path.name for path in volume.iterdir())
    unexpected = sorted(set(existing) - allowed)
    if unexpected:
        raise ValueError(f"Paper 1 volume contains unexpected paths: {unexpected}")
    return {
        "matrix_path": matrix_path,
        "matrix": matrix,
        "materialization_path": materialization_path,
        "materialization": materialization,
        "volume": volume,
        "mount": mount,
        "runs": runs,
        "existing": existing,
    }


def prepare(args):
    values = _inputs(args, require_empty=True)
    storage = values["matrix"]["storage_contract"]
    report = {
        "format": "speck_paper_baseline_storage_qualification",
        "format_version": 1,
        "status": "qualified_dedicated_proxy_and_finalist_checkpoint_volume",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrix": {
            "path": str(values["matrix_path"]),
            "storage_spec_sha256": storage_contract_identity(values["matrix"]),
        },
        "materialization": {
            "path": str(values["materialization_path"]),
            "sha256": file_sha256(values["materialization_path"]),
        },
        "volume": values["mount"],
        "capacity": {
            "estimated_proxy_checkpoint_bytes": storage["estimated_proxy_checkpoint_bytes"],
            "estimated_finalist_checkpoint_bytes": storage[
                "estimated_finalist_checkpoint_bytes"
            ],
            "minimum_free_bytes_before_proxy_launch": storage[
                "minimum_free_bytes_before_proxy_launch"
            ],
            "minimum_free_bytes_before_finalist_launch": storage[
                "minimum_free_bytes_before_finalist_launch"
            ],
            "proxy_floor_passed": values["mount"]["free_bytes"]
            >= storage["minimum_free_bytes_before_proxy_launch"],
            "finalist_floor_passed": values["mount"]["free_bytes"]
            >= storage["minimum_free_bytes_before_finalist_launch"],
        },
        "provenance": {
            "method": "new dedicated directory on a separately mounted physical ext4 filesystem",
            "initial_directory_empty": True,
            "existing_checkpoint_or_optimizer_artifacts_moved": 0,
            "existing_checkpoint_or_optimizer_artifacts_deleted": 0,
            "cleanup_counted_as_capacity": False,
        },
        "operational_binding": {
            "base_train_output_override": "--output-dir",
            "base_train_sha256": file_sha256(Path(__file__).parents[1] / "scripts/base_train.py"),
            "scientific_config_changed": False,
            "runs": values["runs"],
        },
        "limitations": [
            "mount is not configured in /etc/fstab and must be requalified after remount or reboot",
            "storage qualification does not satisfy SPE-58 or authorize baseline training by itself",
        ],
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }
    atomic_json(args.output, report)
    atomic_json(values["volume"] / "paper1-storage-qualification.json", report)
    return report


def check(args):
    values = _inputs(args, require_empty=False)
    report = json.loads(args.output.expanduser().resolve().read_text(encoding="utf-8"))
    current_runs = values["runs"]
    if (
        report.get("format") != "speck_paper_baseline_storage_qualification"
        or report.get("status")
        != "qualified_dedicated_proxy_and_finalist_checkpoint_volume"
        or report.get("matrix", {}).get("storage_spec_sha256")
        != storage_contract_identity(values["matrix"])
        or report.get("materialization", {}).get("sha256")
        != file_sha256(values["materialization_path"])
        or report.get("volume", {}).get("uuid") != values["mount"]["uuid"]
        or report.get("volume", {}).get("device_id") != values["mount"]["device_id"]
        or report.get("operational_binding", {}).get("runs") != current_runs
        or report.get("operational_binding", {}).get("base_train_sha256")
        != file_sha256(Path(__file__).parents[1] / "scripts/base_train.py")
        or not report.get("capacity", {}).get("proxy_floor_passed")
        or not report.get("capacity", {}).get("finalist_floor_passed")
    ):
        raise ValueError("Paper 1 storage qualification no longer matches its inputs")
    runner_source = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/paper_baseline_storage_qualify.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if bytes_sha256(runner_source) != report["runner_sha256"]:
        raise ValueError("Paper 1 storage qualifier runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    report = check(args) if args.check else prepare(args)
    print(f"Paper 1 storage: {report['status']} ({report['volume']['uuid'][:12]})")


if __name__ == "__main__":
    main()
