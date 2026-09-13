"""Measure bounded Python memory for old and streaming dedup resume verification."""

import argparse
import gc
import hashlib
import json
import platform
import sqlite3
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

from speck.data.production_data import accepted_document_chain
from speck.provenance.io import file_sha256
from speck.provenance.repository import repository_root


def measure(connection, limit, mode):
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        cursor = connection.execute(
            "SELECT dedup_sha256, content_sha256 FROM docs ORDER BY doc_seq LIMIT ?", (limit,)
        )
        if mode == "fetchall":
            # Reproduce the previous implementation rather than timing the successor twice.
            rows = cursor.fetchall()
            chain = hashlib.sha256(b"").hexdigest()
            for dedup_sha256, content_sha256 in rows:
                chain = hashlib.sha256(
                    bytes.fromhex(chain)
                    + bytes.fromhex(dedup_sha256)
                    + bytes.fromhex(content_sha256)
                ).hexdigest()
            count = len(rows)
        else:
            count, chain = accepted_document_chain(cursor)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "mode": mode,
        "requested_rows": limit,
        "rows": count,
        "chain_sha256": chain,
        "python_peak_traced_bytes": peak,
        "elapsed_seconds_with_tracing": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "database", type=Path, help="Completed retained SQLite index; opened read-only"
    )
    parser.add_argument("output", type=Path, help="New engineering result JSON")
    parser.add_argument("--rows", type=int, nargs="+", default=[10_000, 100_000])
    args = parser.parse_args()
    if any(value < 1 or value > 100_000 for value in args.rows):
        parser.error("each probe must request between 1 and 100,000 rows")
    if len(set(args.rows)) != len(args.rows):
        parser.error("row counts must be distinct")
    if args.output.exists():
        raise FileExistsError(f"resume probe result already exists: {args.output}")
    database = args.database.resolve()
    before = database.stat()
    digest = file_sha256(database)
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        measurements = [
            measure(connection, limit, mode)
            for limit in sorted(args.rows)
            for mode in ("fetchall", "streaming")
        ]
    finally:
        connection.close()
    after = database.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("probe input changed during measurement")
    for old, new in zip(measurements[::2], measurements[1::2], strict=True):
        if (
            old["rows"] != old["requested_rows"]
            or new["rows"] != old["rows"]
            or new["chain_sha256"] != old["chain_sha256"]
        ):
            raise RuntimeError("resume-chain parity or requested sample coverage failed")
    root = repository_root(__file__)
    result = {
        "format": "speck_production_dedup_resume_memory_probe",
        "format_version": 1,
        "status": "bounded_read_only_chain_parity_pass",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "database": {"path": str(database), "sha256": digest, "bytes": before.st_size},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "implementation": [
            {"path": name, "sha256": file_sha256(root / name)}
            for name in (
                "speck/data/production_data.py",
                "scripts/production_dedup_resume_probe.py",
                "tests/data/test_production_data.py",
            )
        ],
        "measurements": measurements,
        "measurement_boundary": (
            "Python allocations traced during ordered hash-pair retrieval and chain verification; "
            "not total process RSS, SQLite native cache, global-dedup throughput, or production-scale qualification"
        ),
        "command": [
            "python",
            "-m",
            "scripts.production_dedup_resume_probe",
            str(database),
            str(args.output),
            "--rows",
            *map(str, args.rows),
        ],
        "operations_authority": False,
        "training_authority": False,
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
