"""Acquire a bounded complete-file tranche, preserving attempts and verifying prior metadata."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq

from scripts.qualify_stack_v3_metadata_ranges import projection
from speck.data.parquet_ranges import RecordedRangeFile
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def assemble_verified(source, destination):
    """Write complete ordered bytes; a failed assembly remains an unpublished attempt."""
    size = source.contract["size"]
    step = source.contract["maximum_range_bytes"]
    with destination.open("xb") as handle:
        for start in range(0, size, step):
            handle.write(source.fetch(start, min(step, size - start)))
        handle.flush()
        os.fsync(handle.fileno())
    if destination.stat().st_size != size or file_sha256(destination) != source.contract["sha256"]:
        raise ValueError("complete file identity mismatch; assembly preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = repository_root(__file__)
    if (
        args.result.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("requires new result and clean frozen checkout")
    spec = json.loads(args.plan.read_text())
    if (
        spec.get("format") != "speck_stack_v3_complete_file_intake"
        or spec.get("format_version") != 1
        or spec.get("training_authority") is not False
        or len(spec["files"]) != 2
        or spec["maximum_transfer_bytes_per_file"] != 536870912
        or spec["maximum_requests_per_file"] != 64
        or spec["maximum_range_bytes"] != 16777216
        or spec["workers"] != 4
    ):
        raise ValueError("unsupported bounded two-file contract")
    inputs = {}
    for key in (
        "metadata_discovery",
        "metadata_manifest",
        "cached_content_probe",
        "source_qualification",
        "source_use",
    ):
        binding = spec[key]
        path = Path(binding["path"])
        if file_sha256(path) != binding["sha256"]:
            raise ValueError(f"changed input: {key}")
        inputs[key] = json.loads(path.read_text())
    if any(
        inputs["source_qualification"]["source"][key] != spec["source"][key]
        or inputs["metadata_manifest"][key] != spec["source"][key]
        for key in ("repo", "revision")
    ):
        raise ValueError("approved source identity changed")
    for selected in spec["files"]:
        prior = next(
            row
            for row in inputs["metadata_discovery"]["files"]
            if row["declared_complete_file"] == selected["file"]
        )
        expected = {
            "repositories": prior["repositories_seen"],
            "physical_files": prior["physical_files_seen"],
            "sha256": prior["metadata"]["sha256"],
        }
        if (
            selected["metadata"] != expected
            or selected["file"] not in inputs["metadata_manifest"]["files"]
        ):
            raise ValueError("file or metadata binding mismatch")
    working = Path(spec["working_directory"])
    if not os.path.ismount("/mnt/speck-data"):
        raise ValueError("data drive is not mounted")
    if shutil.disk_usage(working.parent).free < spec["minimum_free_bytes"]:
        raise ValueError("insufficient free space")
    execution = {
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "plan": {"path": str(args.plan.resolve()), "sha256": file_sha256(args.plan)},
    }
    if working.exists():
        if not args.resume or json.loads((working / "execution.json").read_text()) != execution:
            raise ValueError("preserve prior state; resume requires identical execution")
    else:
        if args.resume:
            raise ValueError("cannot resume missing intake")
        working.mkdir()
        durable_json(working / "execution.json", execution)
    started = time.perf_counter()
    results = []
    for index, selected in enumerate(spec["files"]):
        directory = working / f"file-{index:03d}"
        directory.mkdir(exist_ok=True)
        declaration = selected["file"]
        contract = {
            "url": "https://huggingface.co/datasets/"
            + spec["source"]["repo"]
            + "/resolve/"
            + spec["source"]["revision"]
            + "/"
            + quote(declaration["path"], safe="/"),
            "size": declaration["bytes"],
            "sha256": declaration["sha256"],
            "maximum_transfer_bytes": spec["maximum_transfer_bytes_per_file"],
            "maximum_requests": spec["maximum_requests_per_file"],
            "maximum_range_bytes": spec["maximum_range_bytes"],
        }
        raw = directory / "complete.parquet"
        with RecordedRangeFile(
            directory / "ranges", contract, resume=(directory / "ranges").exists()
        ) as source:
            if not raw.exists():
                step = contract["maximum_range_bytes"]
                ranges = [
                    (start, min(step, contract["size"] - start))
                    for start in range(0, contract["size"], step)
                ]
                print(f"acquiring complete file: {declaration['path']}", flush=True)
                source.prefetch(ranges, workers=spec["workers"])
                attempt = (
                    directory / f"assembly-{len(list(directory.glob('assembly-*'))):05d}.parquet"
                )
                assemble_verified(source, attempt)
                if projection(pq.ParquetFile(attempt)) != selected["metadata"]:
                    raise ValueError("full-file metadata differs from preserved partial projection")
                os.link(attempt, raw)
            if (
                raw.stat().st_size != declaration["bytes"]
                or file_sha256(raw) != declaration["sha256"]
            ):
                raise ValueError("published complete file identity changed")
            if projection(pq.ParquetFile(raw)) != selected["metadata"]:
                raise ValueError("completed metadata reopen failed")
            result = {
                "source_file": declaration,
                "raw_path": str(raw),
                "full_file_hash_verified": True,
                "metadata_projection": selected["metadata"],
                "metadata_matches_prior_partial_projection": True,
                "range_attempts_all_invocations": len(source.attempts),
                "reserved_transfer_bytes_all_attempts": source.reserved_bytes,
                "completed_range_payload_bytes": sum(len(data) for _, data in source.chunks),
            }
            receipt = directory / "result.json"
            if receipt.exists():
                if json.loads(receipt.read_text()) != result:
                    raise ValueError("completed receipt changed")
            else:
                durable_json(receipt, result)
            results.append(result)
            print(f"complete hash and metadata verified: file {index}", flush=True)
    durable_json(
        args.result,
        {
            "format": "speck_stack_v3_complete_file_intake_result",
            "format_version": 1,
            "status": "complete_raw_files_verified_not_content_qualified_stock",
            **execution,
            "files": results,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "training_authority": False,
            "boundary": spec["scope"],
        },
    )
    print("bounded complete-file intake finished", flush=True)


if __name__ == "__main__":
    main()
