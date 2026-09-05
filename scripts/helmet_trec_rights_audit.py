"""Retain TREC metadata and block payload use when no affirmative grant is identified."""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


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


def repository_revision():
    return subprocess.run(
        ["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_volume(path, storage):
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise ValueError("TREC metadata directory must be real and existing")
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
        raise ValueError("TREC metadata storage identity changed")


def download(spec, directory):
    destination = directory / spec["filename"]
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "speck-evidence/1"})
    with urllib.request.urlopen(request) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
    if temporary.stat().st_size != spec["bytes"] or file_sha256(temporary) != spec[
        "sha256"
    ]:
        raise ValueError(f"TREC metadata changed: {spec['id']}")
    content = temporary.read_text(encoding="utf-8", errors="replace")
    for required in spec["required_substrings"]:
        if required not in content:
            raise ValueError(f"TREC metadata assertion changed: {spec['id']}")
    folded = content.casefold()
    for absent in spec.get("absent_casefold_substrings", ()):
        if absent.casefold() in folded:
            raise ValueError(f"TREC metadata now contains terms requiring review: {spec['id']}")
    os.replace(temporary, destination)
    return {
        "id": spec["id"],
        "path": str(destination),
        "url": spec["url"],
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "required_assertions_present": True,
        "absence_assertions_passed": True,
    }


def loader_analysis(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    urls = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "_URLs" for target in node.targets):
            urls = ast.literal_eval(node.value)
    if urls is None:
        raise ValueError("TREC loader no longer declares source URLs")
    return {
        "source_urls": urls,
        "license_assignment_present": "_LICENSE" in assignments,
    }


def prepare(protocol, protocol_path):
    directory = Path(protocol["storage"]["directory"])
    validate_volume(directory, protocol["storage"])
    if any(directory.iterdir()):
        raise ValueError("TREC metadata directory must be empty")
    artifacts = [download(spec, directory) for spec in protocol["evidence"]]
    loader = next(artifact for artifact in artifacts if artifact["id"] == "hub_loader")
    analysis = loader_analysis(Path(loader["path"]))
    if analysis["license_assignment_present"]:
        raise ValueError("TREC loader now declares a license and requires a new decision version")
    return {
        "format": "speck_helmet_rights_decision",
        "format_version": 1,
        "status": "trec_metadata_audited_payload_and_use_authority_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "spec_identity_sha256": protocol_identity(protocol),
        },
        "source": protocol["source"],
        "evidence": artifacts,
        "loader_analysis": analysis,
        "decision": protocol["decision"],
        "payload_files_acquired": 0,
        "contact_attempted": False,
        "non_claims": protocol["reporting"]["non_claims"],
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }


def check(protocol, report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("format") != "speck_helmet_rights_decision"
        or report.get("status")
        != "trec_metadata_audited_payload_and_use_authority_blocked"
        or report.get("protocol", {}).get("spec_identity_sha256")
        != protocol_identity(protocol)
        or report.get("payload_files_acquired") != 0
        or report.get("contact_attempted") is not False
        or report.get("decision", {}).get("payload_acquisition_authorized") is not False
        or report.get("decision", {}).get("evaluation_use_authorized") is not False
    ):
        raise ValueError("TREC rights decision report is invalid")
    expected = {spec["id"]: spec for spec in protocol["evidence"]}
    if {artifact["id"] for artifact in report["evidence"]} != set(expected):
        raise ValueError("TREC rights evidence inventory changed")
    for artifact in report["evidence"]:
        path = Path(artifact["path"])
        spec = expected[artifact["id"]]
        if (
            not path.is_file()
            or path.stat().st_size != spec["bytes"]
            or file_sha256(path) != spec["sha256"]
        ):
            raise ValueError("TREC retained metadata changed")
    runner = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/helmet_trec_rights_audit.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(runner).hexdigest() != report["runner_sha256"]:
        raise ValueError("TREC rights audit runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "speck_helmet_rights_decision_protocol":
        raise ValueError("TREC rights protocol has the wrong format")
    if args.check:
        if protocol.get("status") not in {"frozen_unexecuted", "executed_blocked"}:
            raise ValueError("TREC rights protocol has an invalid checked status")
        report = check(protocol, args.output.expanduser().resolve())
    else:
        if protocol.get("status") != "frozen_unexecuted":
            raise ValueError("TREC rights protocol must be frozen and unexecuted")
        report = prepare(protocol, protocol_path)
        atomic_json(args.output, report)
    print(f"HELMET TREC rights: {report['status']}")


if __name__ == "__main__":
    main()
