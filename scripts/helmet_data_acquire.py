"""Download and hash-qualify the pinned HELMET archive on its dedicated volume."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from speck.external import validate_external_suite

EXPECTED_UUID = "b64b59d1-ea2c-4206-9171-b7cd739f3eff"
REQUIRED_MOUNT_OPTIONS = {"rw", "nodev", "nosuid", "noexec"}
DATA_REVISION = "dddb209d03e38f1f0faf76d6d05ef4ccf96240ee"
ARCHIVE_NAME = "data.tar.gz"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--metadata-qualification", type=Path, required=True)
    parser.add_argument("--volume-dir", type=Path, required=True)
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


def _mount(path):
    result = subprocess.run(
        ["findmnt", "-J", "-T", str(path), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS,UUID"],
        check=True,
        capture_output=True,
        text=True,
    )
    filesystems = json.loads(result.stdout).get("filesystems", [])
    if len(filesystems) != 1:
        raise ValueError(f"expected one filesystem for {path}")
    value = filesystems[0]
    return {
        "target": value["target"],
        "source": value["source"],
        "filesystem": value["fstype"],
        "options": value["options"].split(","),
        "uuid": value["uuid"],
    }


def qualify_volume(path, minimum_free_bytes):
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise ValueError("HELMET volume directory must be a real existing directory")
    mount = _mount(path)
    target = Path(mount["target"]).resolve()
    if (
        mount["uuid"] != EXPECTED_UUID
        or mount["filesystem"] != "ext4"
        or not REQUIRED_MOUNT_OPTIONS.issubset(mount["options"])
        or target not in (path, *path.parents)
        or path.stat().st_dev == Path("/").stat().st_dev
        or path.stat().st_uid != os.getuid()
        or path.stat().st_mode & 0o077
        or not os.access(path, os.W_OK)
    ):
        raise ValueError("HELMET volume identity, ownership, isolation, or mount options changed")
    usage = shutil.disk_usage(path)
    if usage.free < minimum_free_bytes:
        raise ValueError(
            f"HELMET volume has {usage.free} free bytes, requires {minimum_free_bytes}"
        )
    return {
        **mount,
        "directory": str(path),
        "directory_mode": oct(path.stat().st_mode & 0o777),
        "directory_uid": path.stat().st_uid,
        "device_id": path.stat().st_dev,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "minimum_free_bytes": minimum_free_bytes,
    }


def _inputs(args):
    contract_path = args.contract.expanduser().resolve()
    metadata_path = args.metadata_qualification.expanduser().resolve()
    volume = args.volume_dir.expanduser().absolute()
    contract = validate_external_suite(contract_path)
    if contract["suite_id"] != "helmet" or contract["data"]["revision"] != DATA_REVISION:
        raise ValueError("HELMET archive acquisition received a different contract")
    if (
        metadata_path != Path(__file__).parents[1] / contract["data"]["metadata_qualification"]
        or file_sha256(metadata_path) != contract["data"]["metadata_qualification_sha256"]
    ):
        raise ValueError("HELMET metadata qualification does not match the contract")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("format") != "speck_helmet_data_metadata_qualification"
        or metadata.get("status")
        != "metadata_and_storage_plan_qualified_mount_and_payload_blocked"
        or metadata.get("decision", {}).get("download_authorized") is not False
    ):
        raise ValueError("HELMET metadata qualification is invalid")
    archive = contract["data"]["required_archive"]
    if archive["path"] != ARCHIVE_NAME:
        raise ValueError("HELMET required archive changed")
    live_volume = qualify_volume(volume, contract["data"]["minimum_free_bytes_before_download"])
    return {
        "contract": contract,
        "contract_path": contract_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "volume": volume,
        "live_volume": live_volume,
        "archive": archive,
        "url": (
            "https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/"
            f"{DATA_REVISION}/{ARCHIVE_NAME}"
        ),
    }


def _download(values):
    download_dir = values["volume"] / "downloads"
    download_dir.mkdir(mode=0o700, exist_ok=True)
    if download_dir.is_symlink() or download_dir.stat().st_uid != os.getuid():
        raise ValueError("HELMET download directory is not user-owned or is a symlink")
    final = download_dir / ARCHIVE_NAME
    partial = download_dir / f"{ARCHIVE_NAME}.partial"
    hf_staged = values["volume"] / "hf-staging" / ARCHIVE_NAME
    expected_bytes = values["archive"]["bytes"]
    expected_sha256 = values["archive"]["sha256"]
    if final.is_file():
        if final.stat().st_size != expected_bytes or file_sha256(final) != expected_sha256:
            raise ValueError("retained HELMET archive does not match its pin")
        return final, False, 0.0, "retained_qualified_archive", 0
    if hf_staged.is_file():
        if hf_staged.stat().st_size != expected_bytes or file_sha256(hf_staged) != expected_sha256:
            raise ValueError("Hugging Face staged HELMET archive does not match its pin")
        discarded_partial_bytes = partial.stat().st_size if partial.is_file() else 0
        os.replace(hf_staged, final)
        partial.unlink(missing_ok=True)
        return (
            final,
            True,
            0.0,
            "authenticated_huggingface_hub_xet_staging",
            discarded_partial_bytes,
        )
    if partial.exists() and (not partial.is_file() or partial.stat().st_size > expected_bytes):
        raise ValueError("HELMET partial download is not a valid resumable file")
    started = time.monotonic()
    print(
        f"Downloading pinned HELMET archive ({expected_bytes} bytes) to {partial}",
        flush=True,
    )
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--retry",
            "5",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            values["url"],
        ],
        check=True,
    )
    elapsed = time.monotonic() - started
    if partial.stat().st_size != expected_bytes:
        raise ValueError(
            f"HELMET archive has {partial.stat().st_size} bytes, expected {expected_bytes}"
        )
    observed = file_sha256(partial)
    if observed != expected_sha256:
        raise ValueError(f"HELMET archive SHA-256 is {observed}, expected {expected_sha256}")
    os.replace(partial, final)
    return final, True, elapsed, "anonymous_curl_resumable", 0


def prepare(args):
    values = _inputs(args)
    before = values["live_volume"]
    archive, transferred, elapsed, transport, discarded_partial_bytes = _download(values)
    after = qualify_volume(
        values["volume"], values["contract"]["data"]["minimum_free_bytes_before_download"]
    )
    report = {
        "format": "speck_helmet_archive_acquisition",
        "format_version": 1,
        "status": "pinned_archive_downloaded_hash_qualified_extraction_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(values["contract_path"]),
            "archive_spec_sha256": bytes_sha256(
                json.dumps(
                    {
                        "repository": values["contract"]["data"]["repository"],
                        "revision": DATA_REVISION,
                        "archive": values["archive"],
                        "minimum_free_bytes": values["contract"]["data"][
                            "minimum_free_bytes_before_download"
                        ],
                        "device_uuid": EXPECTED_UUID,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ),
        },
        "metadata_qualification": {
            "path": str(values["metadata_path"]),
            "sha256": file_sha256(values["metadata_path"]),
        },
        "archive": {
            "path": str(archive),
            "url": values["url"],
            "bytes": archive.stat().st_size,
            "sha256": file_sha256(archive),
            "transferred_this_run": transferred,
            "transfer_seconds": elapsed,
            "transport": transport,
            "discarded_superseded_partial_bytes": discarded_partial_bytes,
        },
        "storage": {"before": before, "after": after},
        "tools": {
            "curl": subprocess.run(
                ["curl", "--version"], check=True, capture_output=True, text=True
            ).stdout.splitlines()[0],
            "huggingface_cli": subprocess.run(
                ["hf", "--version"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "credential_exposed": False,
        },
        "decision": {
            "archive_qualified": True,
            "extraction_authorized": False,
            "execution_authorized": False,
            "reason": "inspect archive paths and component-license material before extraction",
        },
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }
    atomic_json(args.output, report)
    atomic_json(values["volume"] / "helmet-archive-acquisition.json", report)
    return report


def check(args):
    values = _inputs(args)
    report = json.loads(args.output.expanduser().resolve().read_text(encoding="utf-8"))
    archive = Path(report.get("archive", {}).get("path", ""))
    if (
        report.get("format") != "speck_helmet_archive_acquisition"
        or report.get("status")
        != "pinned_archive_downloaded_hash_qualified_extraction_blocked"
        or report.get("metadata_qualification", {}).get("sha256")
        != file_sha256(values["metadata_path"])
        or archive != values["volume"] / "downloads" / ARCHIVE_NAME
        or not archive.is_file()
        or archive.stat().st_size != values["archive"]["bytes"]
        or file_sha256(archive) != values["archive"]["sha256"]
        or report.get("decision", {}).get("extraction_authorized") is not False
        or report.get("decision", {}).get("execution_authorized") is not False
    ):
        raise ValueError("HELMET archive acquisition no longer matches pinned inputs")
    runner_source = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/helmet_data_acquire.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if bytes_sha256(runner_source) != report["runner_sha256"]:
        raise ValueError("HELMET archive acquisition runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    report = check(args) if args.check else prepare(args)
    print(
        f"HELMET archive: {report['status']} "
        f"({report['archive']['sha256'][:12]})"
    )


if __name__ == "__main__":
    main()
