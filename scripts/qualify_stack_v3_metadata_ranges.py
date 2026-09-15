"""Compare bounded remote metadata ranges with one complete, hash-verified Stack v3 shard."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq

from speck.data.parquet_ranges import RecordedRangeFile, metadata_ranges
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root

PHYSICAL_COLUMNS = {
    "repo_path",
    "repo_id",
    "commit_id",
    "github_metadata.is_fork",
    "num_files",
    "files.list.element.content_id",
    "files.list.element.size_bytes",
    "files.list.element.file_path",
    "files.list.element.language",
    "files.list.element.is_vendor",
    "files.list.element.license_type",
    "files.list.element.detected_licenses.list.element",
}
PROJECTION_COLUMNS = sorted(
    column.removesuffix(".list.element")
    if column.endswith("detected_licenses.list.element")
    else column
    for column in PHYSICAL_COLUMNS
)


def projection(parquet, output=None):
    digest = hashlib.sha256()
    repositories = files = 0
    for batch in parquet.iter_batches(
        columns=PROJECTION_COLUMNS, batch_size=256, use_threads=False
    ):
        for row in batch.to_pylist():
            if any("content" in item for item in row.get("files") or []):
                raise ValueError("metadata projection exposed source text")
            payload = (
                json.dumps(
                    {"source_row": repositories, **row}, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            ).encode()
            digest.update(payload)
            if output is not None:
                output.write(payload)
            repositories += 1
            files += len(row.get("files") or [])
    if repositories != parquet.metadata.num_rows:
        raise ValueError("incomplete metadata projection")
    return {"repositories": repositories, "physical_files": files, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = repository_root(__file__)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (
        args.result.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("requires a new result and clean frozen implementation")
    spec = json.loads(args.plan.read_text())
    if (
        spec.get("format") != "speck_stack_v3_metadata_range_qualification"
        or spec.get("format_version") != 1
        or spec.get("training_authority") is not False
        or spec.get("maximum_transfer_bytes") != 67108864
        or spec.get("maximum_requests") != 128
        or spec.get("maximum_range_bytes") != 16777216
        or spec.get("merge_gap_bytes") != 65536
        or spec.get("workers") != 4
    ):
        raise ValueError("unsupported one-file metadata qualification contract")
    prior_path = (args.plan.parent / spec["reference_census"]["path"]).resolve()
    if file_sha256(prior_path) != spec["reference_census"]["sha256"]:
        raise ValueError("reference census identity changed")
    prior = json.loads(prior_path.read_text())
    declaration = spec["file"]
    local = Path(spec["local_complete_file"])
    reference = next(row for row in prior["files"] if row["raw"]["path"] == str(local))
    qualification_path = (
        root / "archive/pregrant-history/research/flagship/stack_v3_qualification.json"
    )
    qualification = json.loads(qualification_path.read_text())
    if (
        any(spec["source"][key] != qualification["source"][key] for key in ("repo", "revision"))
        or declaration != qualification["source"]["files"][0]
        or reference["raw"]["sha256"] != declaration["sha256"]
        or local.stat().st_size != declaration["size"]
        or file_sha256(local) != declaration["sha256"]
    ):
        raise ValueError("complete reference or approved source identity changed")

    def identity(path):
        return {"path": str(path.resolve()), "sha256": file_sha256(path)}

    execution = {
        "repository_revision": revision,
        "plan": identity(args.plan),
        "reference_census": identity(prior_path),
        "source_qualification": identity(qualification_path),
    }
    working = Path(spec["working_directory"])
    if working.exists():
        if not args.resume or json.loads((working / "execution.json").read_text()) != execution:
            raise ValueError("preserve qualification state; resume requires original execution")
    else:
        if args.resume:
            raise ValueError("cannot resume missing qualification")
        working.mkdir(parents=True)
        durable_json(working / "execution.json", execution)
    attempt = working / f"projection-attempt-{len(list(working.glob('projection-attempt-*'))):05d}"
    attempt.mkdir()
    url = (
        "https://huggingface.co/datasets/"
        + spec["source"]["repo"]
        + "/resolve/"
        + spec["source"]["revision"]
        + "/"
        + quote(declaration["path"], safe="/")
    )
    contract = {
        "url": url,
        "size": declaration["size"],
        "sha256": declaration["sha256"],
        **{
            key: spec[key]
            for key in ("maximum_transfer_bytes", "maximum_requests", "maximum_range_bytes")
        },
    }
    started = time.perf_counter()
    with RecordedRangeFile(working / "ranges", contract, resume=args.resume) as source:
        parquet = pq.ParquetFile(source, pre_buffer=False)
        ranges = metadata_ranges(
            parquet.metadata,
            PHYSICAL_COLUMNS,
            gap=spec["merge_gap_bytes"],
            maximum=spec["maximum_range_bytes"],
        )
        durable_json(attempt / "planned-ranges.json", ranges)
        print(
            f"metadata prefetch: {len(ranges)} ranges / {sum(n for _, n in ranges)} bytes",
            flush=True,
        )
        source.prefetch(ranges, workers=spec["workers"])
        transferred_seconds = time.perf_counter() - started
        print("bounded metadata ranges acquired", flush=True)
        with (attempt / "metadata.jsonl").open("xb") as handle:
            remote = projection(parquet, handle)
            handle.flush()
            os.fsync(handle.fileno())
        local_projection = projection(pq.ParquetFile(local))
        if remote != local_projection:
            raise ValueError("remote and complete-local metadata disagree")
        if (remote["repositories"], remote["physical_files"]) != (
            reference["repositories_seen"],
            reference["physical_files_seen"],
        ):
            raise ValueError("projection count disagrees with prior census")
        with local.open("rb") as handle:
            for offset, payload in source.chunks:
                handle.seek(offset)
                if handle.read(len(payload)) != payload:
                    raise ValueError("range payload differs from full-SHA verified reference")
        attempts = len(source.attempts)
        transferred = sum(len(payload) for _, payload in source.chunks)
        reserved = source.reserved_bytes
    with RecordedRangeFile(working / "ranges", contract, resume=True) as reopened:
        if projection(pq.ParquetFile(reopened, pre_buffer=False)) != remote or reopened.get_count:
            raise ValueError("completed metadata reopen changed or used the network")
    result = {
        "format": "speck_stack_v3_metadata_range_qualification_result",
        "format_version": 1,
        "status": "one_file_range_and_projection_parity_pass_not_stock",
        **execution,
        "local_complete_file": identity(local),
        "working_directory": str(working),
        "projection": {**remote, "path": str(attempt / "metadata.jsonl")},
        "remote_payload_bytes": transferred,
        "reserved_transfer_bytes_all_attempts": reserved,
        "range_attempts": attempts,
        "complete_file_bytes": declaration["size"],
        "remote_payload_fraction_of_complete_file": transferred / declaration["size"],
        "range_acquisition_seconds_this_invocation": transferred_seconds,
        "qualification_seconds_this_invocation": time.perf_counter() - started,
        "every_range_matches_verified_complete_file": True,
        "decoded_projection_matches_complete_file": True,
        "completed_reopen_without_network_pass": True,
        "training_authority": False,
        "boundary": spec["scope"]
        + " Range payload bytes exclude HTTP/TLS overhead and redirect headers. The Parquet footer read and merged small gaps can include adjacent unselected bytes; no content column is decoded. Failed/interrupted attempts retain conservative byte reservations; these are not measured wire bytes. This invocation's timings include local verification after initial complete-file hashing. Partial metadata from new files will still require full-file verification before stock qualification.",
    }
    durable_json(args.result, result)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "remote_payload_bytes", "range_attempts")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
