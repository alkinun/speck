"""Inspect the pinned HELMET tar archive without extracting its dataset payload."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.helmet_data_metadata_audit import CONFIGS, DATA_REVISION
from speck.external import validate_external_suite

LICENSE_PATTERN = re.compile(
    r"(^|/)(license|licenses|copying|copyright|notice|readme|citation|terms)(\.|$)",
    re.IGNORECASE,
)
MAX_METADATA_BYTES = 5 * 1024 * 1024


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--code-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
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


def validate_member(member):
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or member.issym()
        or member.islnk()
        or member.isdev()
        or not (member.isdir() or member.isfile())
    ):
        raise ValueError(f"unsafe HELMET archive member: {member.name}")
    return path


def _declared_archive_paths(code_checkout):
    paths = set()
    remote_datasets = set()
    for relative in CONFIGS:
        config = yaml.safe_load((code_checkout / relative).read_text(encoding="utf-8"))
        datasets = str(config["datasets"]).split(",")
        test_files = str(config.get("test_files", "")).split(",")
        demo_files = str(config.get("demo_files", "")).split(",")
        if len(test_files) != len(datasets) or len(demo_files) != len(datasets):
            raise ValueError(f"HELMET config path arity changed: {relative}")
        for dataset, test_file, demo_file in zip(datasets, test_files, demo_files):
            found = False
            for value in (test_file, demo_file):
                if value.startswith("data/"):
                    paths.add(value)
                    found = True
            if not found:
                remote_datasets.add(dataset)
    return sorted(paths), sorted(remote_datasets)


def inspect_archive(archive, code_checkout):
    declared_paths, remote_datasets = _declared_archive_paths(code_checkout)
    member_identity = hashlib.sha256()
    top_level = defaultdict(lambda: {"members": 0, "regular_files": 0, "bytes": 0})
    metadata_files = []
    regular_paths = set()
    members = 0
    regular_files = 0
    directories = 0
    uncompressed_bytes = 0
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            path = validate_member(member)
            entry = {
                "path": path.as_posix(),
                "type": "directory" if member.isdir() else "file",
                "bytes": member.size,
                "mode": member.mode,
            }
            member_identity.update(
                json.dumps(entry, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            )
            members += 1
            root = path.parts[0]
            top_level[root]["members"] += 1
            if member.isdir():
                directories += 1
                continue
            regular_files += 1
            uncompressed_bytes += member.size
            regular_paths.add(path.as_posix())
            top_level[root]["regular_files"] += 1
            top_level[root]["bytes"] += member.size
            if LICENSE_PATTERN.search(path.as_posix()) and member.size <= MAX_METADATA_BYTES:
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read HELMET metadata member: {member.name}")
                content = extracted.read()
                metadata_files.append(
                    {
                        "path": path.as_posix(),
                        "bytes": len(content),
                        "sha256": bytes_sha256(content),
                    }
                )
    missing = [path for path in declared_paths if path not in regular_paths]
    return {
        "members": members,
        "directories": directories,
        "regular_files": regular_files,
        "uncompressed_bytes": uncompressed_bytes,
        "member_identity_sha256": member_identity.hexdigest(),
        "top_level": [
            {"path": path, **values} for path, values in sorted(top_level.items())
        ],
        "license_and_metadata_files": metadata_files,
        "declared_local_paths": declared_paths,
        "declared_local_paths_found": len(declared_paths) - len(missing),
        "declared_local_paths_missing": missing,
        "runtime_download_datasets": remote_datasets,
    }


def _inputs(args):
    contract_path = args.contract.expanduser().resolve()
    acquisition_path = args.acquisition.expanduser().resolve()
    code = args.code_checkout.expanduser().resolve()
    contract = validate_external_suite(contract_path)
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if (
        contract["suite_id"] != "helmet"
        or contract["data"]["revision"] != DATA_REVISION
        or acquisition.get("format") != "speck_helmet_archive_acquisition"
        or acquisition.get("status")
        != "pinned_archive_downloaded_hash_qualified_extraction_blocked"
        or acquisition.get("archive", {}).get("sha256")
        != contract["data"]["required_archive"]["sha256"]
        or acquisition.get("archive", {}).get("bytes")
        != contract["data"]["required_archive"]["bytes"]
    ):
        raise ValueError("HELMET archive inspection inputs do not match the contract")
    archive = Path(acquisition["archive"]["path"])
    if (
        not archive.is_file()
        or archive.stat().st_size != contract["data"]["required_archive"]["bytes"]
        or file_sha256(archive) != contract["data"]["required_archive"]["sha256"]
    ):
        raise ValueError("HELMET archive changed before inspection")
    return contract_path, acquisition_path, code, archive


def prepare(args):
    contract_path, acquisition_path, code, archive = _inputs(args)
    inspection = inspect_archive(archive, code)
    if inspection["declared_local_paths_missing"]:
        raise ValueError(
            "HELMET archive misses config-declared paths: "
            + repr(inspection["declared_local_paths_missing"])
        )
    report = {
        "format": "speck_helmet_archive_inspection",
        "format_version": 1,
        "status": "path_safe_inventory_qualified_component_licenses_and_extraction_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {"path": str(contract_path)},
        "acquisition": {
            "path": str(acquisition_path),
            "sha256": file_sha256(acquisition_path),
        },
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": file_sha256(archive),
        },
        "inspection": inspection,
        "decision": {
            "path_safe": True,
            "config_declared_local_paths_complete": True,
            "component_licenses_qualified": False,
            "extraction_authorized": False,
            "execution_authorized": False,
            "reason": "review in-archive license metadata and runtime-download dataset terms before extraction",
        },
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }
    atomic_json(args.output, report)
    return report


def check(args):
    _, acquisition_path, code, archive = _inputs(args)
    report = json.loads(args.output.expanduser().resolve().read_text(encoding="utf-8"))
    current = inspect_archive(archive, code)
    if (
        report.get("format") != "speck_helmet_archive_inspection"
        or report.get("status")
        != "path_safe_inventory_qualified_component_licenses_and_extraction_blocked"
        or report.get("acquisition", {}).get("sha256") != file_sha256(acquisition_path)
        or report.get("inspection") != current
        or report.get("decision", {}).get("extraction_authorized") is not False
        or report.get("decision", {}).get("execution_authorized") is not False
    ):
        raise ValueError("HELMET archive inspection no longer matches pinned inputs")
    runner_source = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/helmet_archive_inspect.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if bytes_sha256(runner_source) != report["runner_sha256"]:
        raise ValueError("HELMET archive inspection runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    report = check(args) if args.check else prepare(args)
    print(
        f"HELMET archive inspection: {report['status']} "
        f"({report['inspection']['member_identity_sha256'][:12]})"
    )


if __name__ == "__main__":
    main()
