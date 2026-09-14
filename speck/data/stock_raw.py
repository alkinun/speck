"""Checkpoint complete pinned Parquet input acquisition separately from text processing."""

import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from speck.data.acquisition_units import _raw_file
from speck.provenance.io import durable_json, file_sha256


def verify_raw_parquet(path, unit):
    path = Path(path)
    if path.stat().st_size != unit["raw"]["bytes"] or file_sha256(path) != unit["raw"]["sha256"]:
        raise ValueError("raw stock input identity mismatch")
    parquet = pq.ParquetFile(path)
    columns = {unit["reader"]["content_column"], *unit["reader"]["metadata_columns"].values()}
    if parquet.metadata.num_rows != unit["expected_file_rows"] or not columns.issubset(
        parquet.schema_arrow.names
    ):
        raise ValueError("raw stock input schema/physical row count mismatch")
    return {"physical_rows": parquet.metadata.num_rows, "schema_pass": True}


def acquire_stock_raw(plan, execution, *, resume=False):
    output = Path(plan["raw_acquisition_directory"])
    if output.exists():
        if not resume or json.loads((output / "execution.json").read_text()) != execution:
            raise ValueError("raw stock resume requires the same frozen execution")
    else:
        if resume:
            raise ValueError("cannot resume absent raw stock acquisition")
        output.mkdir(parents=True)
        durable_json(output / "execution.json", execution)
    reports = []
    for unit in plan["units"]:
        receipt = output / (unit["id"] + ".json")
        if receipt.exists():
            report = json.loads(receipt.read_text())
            if report["unit"] != unit:
                raise ValueError("completed raw stock unit configuration changed")
            verify_raw_parquet(report["raw"]["path"], unit)
        else:
            started = time.perf_counter()
            try:
                raw = _raw_file(plan, unit)
                verification = verify_raw_parquet(raw["path"], unit)
            except BaseException as error:
                durable_json(
                    output / f"{unit['id']}-failed-{time.time_ns()}.json",
                    {
                        "unit": unit,
                        "error_type": type(error).__name__,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
                raise
            report = {
                "unit": unit,
                "raw": raw,
                "verification": verification,
                "elapsed_seconds": time.perf_counter() - started,
            }
            durable_json(receipt, report)
        reports.append(report)
        print(
            f"verified raw stock: {unit['id']} / {report['verification']['physical_rows']} rows",
            flush=True,
        )
    return {
        "format": "speck_stock_raw_acquisition_result",
        "format_version": 1,
        "status": "complete_raw_verified_not_text_stock",
        **execution,
        "files": reports,
        "physical_rows": sum(r["verification"]["physical_rows"] for r in reports),
        "bytes": sum(r["raw"]["bytes"] for r in reports),
        "training_authority": False,
        "boundary": "Complete pinned Parquet input hashes, schema and physical rows only. No per-document eligibility, reference exclusion, selected-tokenizer capacity or training authority. Invocation timings include concurrent local preparation.",
    }
