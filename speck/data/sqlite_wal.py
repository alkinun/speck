"""Experiment-scoped durable WAL policies and process-crash qualification hooks."""

import math
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import speck.data.production_data as production_data
from speck.provenance.io import durable_json, file_sha256

CRASH_EXIT_CODE = 75


@contextmanager
def wal_policy(autocheckpoint_pages, report, *, crash_receipt=None):
    """Change only connection policy for a serial, identity-bound engineering run.

    FULL synchronization and foreign keys are required. WAL thresholds are triggers,
    not hard size limits; checkpoint-boundary file-size observations are reported.
    """

    if (
        isinstance(autocheckpoint_pages, bool)
        or not isinstance(autocheckpoint_pages, int)
        or autocheckpoint_pages not in (1000, 65536)
    ):
        raise ValueError("WAL comparison supports only the frozen 1000/65536-page policies")
    if not isinstance(report, dict) or report:
        raise ValueError("WAL policy requires a fresh report dictionary")
    if crash_receipt is not None and Path(crash_receipt).exists():
        raise FileExistsError(crash_receipt)
    report.update(
        {
            "autocheckpoint_pages": autocheckpoint_pages,
            "samples": [],
            "observed_peak_wal_bytes": 0,
            "settings": None,
        }
    )
    original_database = production_data._database
    original_checkpoint = production_data._checkpoint
    database_path = None

    def sample(stage, state=None):
        wal = database_path.with_name(database_path.name + "-wal")
        size = wal.stat().st_size if wal.exists() else 0
        report["observed_peak_wal_bytes"] = max(report["observed_peak_wal_bytes"], size)
        report["samples"].append(
            {
                "stage": stage,
                "wal_bytes": size,
                "database_bytes": database_path.stat().st_size,
                "processed_records": state["processed_records"] if state else None,
            }
        )
        return size

    def database(path):
        nonlocal database_path
        if database_path is not None:
            raise ValueError("WAL comparison expects one preprocessing connection per invocation")
        database_path = Path(path)
        connection = original_database(path)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(f"PRAGMA wal_autocheckpoint={autocheckpoint_pages}")
            settings = {
                name: connection.execute(f"PRAGMA {name}").fetchone()[0]
                for name in (
                    "journal_mode",
                    "synchronous",
                    "foreign_keys",
                    "wal_autocheckpoint",
                    "page_size",
                    "cache_size",
                )
            }
            if (
                settings["journal_mode"] != "wal"
                or settings["synchronous"] != 2
                or settings["foreign_keys"] != 1
                or settings["wal_autocheckpoint"] != autocheckpoint_pages
                or settings["page_size"] != 4096
            ):
                raise ValueError("SQLite did not accept the frozen durable WAL settings")
            report["settings"] = settings
            report["nominal_trigger_bytes"] = autocheckpoint_pages * settings["page_size"]
            sample("opened")
            return connection
        except BaseException:
            connection.close()
            raise

    def checkpoint(connection, handles, removal, state_path, state, **kwargs):
        sample("before_commit", state)
        result = original_checkpoint(connection, handles, removal, state_path, state, **kwargs)
        wal_bytes = sample("after_durable_checkpoint", state)
        if crash_receipt is not None and 12 < state["source_index"] <= 18 and wal_bytes > 32:
            # Ignore WAL only to demonstrate that the committed checkpoint really
            # needs it. This private main file is stable while its sole writer pauses.
            main = sqlite3.connect(
                database_path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True
            )
            try:
                main_count = main.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            finally:
                main.close()
            if main_count < state["next_doc_seq"]:
                # Leave both an uncommitted SQL change and an uncommitted output tail.
                # os._exit deliberately bypasses connection/file cleanup.
                connection.execute("UPDATE docs SET content_sha256=? WHERE doc_seq=0", ("0" * 64,))
                if not connection.in_transaction:
                    raise RuntimeError("crash probe SQL change was unexpectedly committed")
                tail = b"UNCOMMITTED_WAL_RECOVERY_PROBE\n"
                handles[12].write(tail)
                handles[12].flush()
                durable_json(
                    crash_receipt,
                    {
                        "format": "speck_committed_wal_crash_receipt",
                        "format_version": 1,
                        "settings": report["settings"],
                        "config_fingerprint": state["contract"],
                        "processed_records": state["processed_records"],
                        "committed_documents": state["next_doc_seq"],
                        "main_file_documents_without_wal": main_count,
                        "wal_bytes_at_crash": wal_bytes,
                        "source_index": state["source_index"],
                        "checkpoint": {"path": str(state_path), "sha256": file_sha256(state_path)},
                        "uncommitted_sql_change": "private reference-row hash update without commit",
                        "uncommitted_tail": {
                            "path": str(handles[12].name),
                            "bytes": len(tail),
                            "committed_bytes": state["output_sizes"]["12"],
                        },
                        "exit_code": CRASH_EXIT_CODE,
                    },
                )
                os._exit(CRASH_EXIT_CODE)
        return result

    production_data._database = database
    production_data._checkpoint = checkpoint
    try:
        yield report
    finally:
        production_data._database = original_database
        production_data._checkpoint = original_checkpoint


def assess_wal_comparison(plan, runs, recovery_passed):
    """Apply the fixed two-pair speed, space, and correctness gates."""

    expected = plan["order"]
    if [run["id"] for run in runs] != [item["id"] for item in expected]:
        raise ValueError("WAL comparison requires the complete frozen run order")
    by_id = {run["id"]: run for run in runs}
    for run, declaration in zip(runs, expected, strict=True):
        elapsed = run["timing"].get("total_seconds")
        if (
            run["timing"].get("status") != "complete"
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
            or run["policy"]["autocheckpoint_pages"] != declaration["wal_autocheckpoint_pages"]
        ):
            raise ValueError(
                "WAL comparison requires complete finite timings for the declared policies"
            )
    pairs = []
    for baseline, candidate in plan["pairs"]:
        left, right = by_id[baseline], by_id[candidate]
        ratio = right["timing"]["total_seconds"] / left["timing"]["total_seconds"]
        pairs.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "candidate_over_baseline": ratio,
                "relative_reduction": 1 - ratio,
                "speed_gate_pass": ratio <= 1 - plan["minimum_relative_reduction"],
            }
        )
    space = all(
        run["policy"]["observed_peak_wal_bytes"] <= plan["maximum_observed_wal_bytes"]
        for run in runs
    )
    parity = all(run["parity_pass"] for run in runs)
    passing = (
        all(pair["speed_gate_pass"] for pair in pairs) and space and parity and recovery_passed
    )
    return {
        "pairs": pairs,
        "space_gate_pass": space,
        "parity_gate_pass": parity,
        "hard_crash_recovery_pass": recovery_passed,
        "recommendation": "65536_pages_for_this_qualified_envelope"
        if passing
        else "retain_default_pending_evidence",
        "scope": "Two paired engineering replays with reversed order, not a population confidence interval or production-scale speedup.",
    }
