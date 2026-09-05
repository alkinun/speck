"""Qualify exact data-only HELMET inputs against their pinned upstream source."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from datasets.load import dataset_module_factory


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def protocol_identity(protocol):
    payload = {key: protocol[key] for key in protocol if key not in {"status", "result"}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_rows(dataset):
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(
            json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
            + b"\n"
        )
    return digest.hexdigest()


def features_sha256(dataset):
    encoded = json.dumps(
        dataset.features.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def runtime_identity():
    return {
        "datasets": importlib.metadata.version("datasets"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "dataset_module_factory_sha256": hashlib.sha256(
            inspect.getsource(dataset_module_factory).encode()
        ).hexdigest(),
    }


def validate_runtime(expected):
    observed = runtime_identity()
    if observed != expected:
        raise ValueError("HELMET data-only source runtime changed")
    return observed


def repository_revision():
    return subprocess.run(
        ["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_volume(path, storage):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise ValueError("HELMET data-only source directory must be real and existing")
    mount = json.loads(
        subprocess.run(
            ["findmnt", "-J", "-T", str(path), "-o", "FSTYPE,OPTIONS,UUID"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["filesystems"][0]
    if (
        mount["uuid"] != storage["device_uuid"]
        or mount["fstype"] != storage["filesystem"]
        or not set(storage["required_mount_options"]) <= set(mount["options"].split(","))
        or path.stat().st_dev == Path("/").stat().st_dev
        or path.stat().st_uid != os.getuid()
        or oct(path.stat().st_mode & 0o777) != storage["required_directory_mode"]
    ):
        raise ValueError("HELMET data-only source volume identity changed")


def download(spec, destination):
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "speck-evidence/1"})
    with urllib.request.urlopen(request) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
    if temporary.stat().st_size != spec["bytes"] or file_sha256(temporary) != spec[
        "sha256"
    ]:
        raise ValueError("HELMET data-only source payload changed")
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
    }


def split_identity(dataset):
    return {
        "rows": dataset.num_rows,
        "canonical_rows_sha256": canonical_rows(dataset),
        "features_sha256": features_sha256(dataset),
    }


def verify_upstream_rows(dataset, upstream, parts):
    labels = dataset.features["intent"].names
    observed = [[row["text"], labels[row["intent"]]] for row in dataset]
    expected = [row for part in parts for row in upstream[part]]
    if observed != expected:
        raise ValueError("CLINC Parquet does not match its pinned upstream row order")
    return {
        "parts": parts,
        "rows": len(expected),
        "all_text_and_intent_values_match_in_order": True,
    }


def prepare(protocol, protocol_path):
    runtime = validate_runtime(protocol["runtime"])
    source_dir = Path(protocol["storage"]["source_directory"])
    validate_volume(source_dir, protocol["storage"])
    if any(source_dir.iterdir()):
        raise ValueError("HELMET data-only source directory must be empty")
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise ValueError("HELMET data-only source qualification must be unauthenticated")
    source = protocol["source"]
    license_artifact = download(source["license_file"], source_dir / "LICENSE")
    upstream_artifact = download(source["upstream_data"], source_dir / "data_oos_plus.json")
    upstream = json.loads((source_dir / "data_oos_plus.json").read_text(encoding="utf-8"))
    split_artifacts = {}
    data_files = {}
    for split, spec in source["splits"].items():
        destination = source_dir / f"{split}.parquet"
        artifact = download(spec, destination)
        split_artifacts[split] = artifact
        data_files[split] = str(destination)
    loaded = load_dataset("parquet", data_files=data_files)
    for split, dataset in loaded.items():
        expected = {
            key: source["splits"][split][key]
            for key in ("rows", "canonical_rows_sha256", "features_sha256")
        }
        identity = split_identity(dataset)
        if identity != expected:
            raise ValueError(f"CLINC {split} identity changed")
        split_artifacts[split].update(identity)
        split_artifacts[split]["upstream_parity"] = verify_upstream_rows(
            dataset, upstream, source["splits"][split]["upstream_parts"]
        )
    return {
        "format": "speck_helmet_data_only_source_qualification",
        "format_version": 1,
        "status": "clinc_plus_required_splits_qualified_offline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "spec_identity_sha256": protocol_identity(protocol),
        },
        "runtime": runtime,
        "source": {
            "id": source["id"],
            "hub_revision": source["hub_revision"],
            "upstream_revision": source["upstream_revision"],
            "license": source["license"],
            "license_artifact": license_artifact,
            "upstream_artifact": upstream_artifact,
            "splits": split_artifacts,
        },
        "network": {
            "authentication": "none",
            "allowed_hosts": protocol["network"]["allowed_hosts"],
            "credentials_exposed": False,
        },
        "decision": {
            "source_snapshot_qualified": True,
            "helmet_icl_qualified": False,
            "helmet_execution_authorized": False,
            "contamination_scanned": False,
        },
        "non_claims": protocol["reporting"]["non_claims"],
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }


def check(protocol, report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_runtime(protocol["runtime"])
    if (
        report.get("format") != "speck_helmet_data_only_source_qualification"
        or report.get("status") != "clinc_plus_required_splits_qualified_offline"
        or report.get("protocol", {}).get("spec_identity_sha256")
        != protocol_identity(protocol)
        or report.get("decision", {}).get("source_snapshot_qualified") is not True
        or report.get("decision", {}).get("helmet_execution_authorized") is not False
    ):
        raise ValueError("HELMET CLINC source qualification report is invalid")
    source = protocol["source"]
    artifacts = [
        (source["license_file"], report["source"]["license_artifact"]),
        (source["upstream_data"], report["source"]["upstream_artifact"]),
    ]
    artifacts.extend(
        (source["splits"][split], artifact)
        for split, artifact in report["source"]["splits"].items()
    )
    for expected, artifact in artifacts:
        path = Path(artifact["path"])
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or file_sha256(path) != expected["sha256"]
        ):
            raise ValueError("HELMET CLINC retained artifact changed")
    upstream = json.loads(
        Path(report["source"]["upstream_artifact"]["path"]).read_text(encoding="utf-8")
    )
    data_files = {
        split: artifact["path"] for split, artifact in report["source"]["splits"].items()
    }
    loaded = load_dataset("parquet", data_files=data_files)
    for split, dataset in loaded.items():
        expected = {
            key: source["splits"][split][key]
            for key in ("rows", "canonical_rows_sha256", "features_sha256")
        }
        if split_identity(dataset) != expected:
            raise ValueError("HELMET CLINC offline identity changed")
        verify_upstream_rows(dataset, upstream, source["splits"][split]["upstream_parts"])
    runner = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/helmet_data_only_source_qualify.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(runner).hexdigest() != report["runner_sha256"]:
        raise ValueError("HELMET CLINC source qualification runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "speck_helmet_data_only_source_protocol":
        raise ValueError("HELMET data-only source protocol has the wrong format")
    if args.check:
        if protocol.get("status") not in {"frozen_unexecuted", "executed_qualified"}:
            raise ValueError("HELMET data-only source protocol has an invalid checked status")
        report = check(protocol, args.output.expanduser().resolve())
    else:
        if protocol.get("status") != "frozen_unexecuted":
            raise ValueError("HELMET data-only source protocol must be frozen and unexecuted")
        report = prepare(protocol, protocol_path)
        atomic_json(args.output, report)
    print(f"HELMET data-only source: {report['status']}")


if __name__ == "__main__":
    main()
