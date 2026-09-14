"""Index metadata-eligible original rows and choose reproducible stratified capacity probes."""

import json
import os
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from speck.data.acquisition_units import _digest
from speck.data.stack_edu_metadata import REQUIRED_COLUMNS, verify_metadata_file
from speck.data.stack_edu_stock import metadata_rejection
from speck.provenance.io import durable_json, file_sha256


def build_eligible_index(unit, policy, output, *, maximum_index_bytes):
    output = Path(output)
    config = {"unit": unit, "policy": policy, "maximum_index_bytes": maximum_index_bytes}
    verify_metadata_file(unit["metadata_path"], unit)
    if output.exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("eligible index configuration changed")
        if not (output / "manifest.json").exists():
            raise ValueError("unpublished eligible index preserved; use explicit recovery")
        manifest = json.loads((output / "manifest.json").read_text())
        if (
            manifest["config_sha256"] != _digest(config)
            or file_sha256(output / "rows.jsonl") != manifest["output"]["sha256"]
        ):
            raise ValueError("eligible index identity mismatch")
        return manifest
    output.mkdir(parents=True)
    durable_json(output / "config.json", config)
    rows_path = output / "rows.jsonl"
    ordinal = offset = count = size = 0
    rejections = Counter()
    with rows_path.open("xb") as handle:
        for batch in pq.ParquetFile(unit["metadata_path"]).iter_batches(
            columns=sorted(REQUIRED_COLUMNS), batch_size=65536, use_threads=False
        ):
            low, high = (
                max(0, unit["start_row"] - offset),
                min(batch.num_rows, unit["stop_row"] - offset),
            )
            if high > low:
                selected = batch.slice(low, high - low)
                selected = selected.append_column(
                    "__physical_row", pa.array(range(offset + low, offset + high), type=pa.int64())
                )
                count += selected.num_rows
                mask = pc.and_(
                    pc.greater_equal(
                        selected.column("int_score"), policy["filters"]["minimum_integer_score"]
                    ),
                    pc.equal(selected.column("license_type"), "permissive"),
                )
                eligible = selected.filter(pc.fill_null(mask, False))
                rejections["coarse_score_or_license_type"] += selected.num_rows - eligible.num_rows
                for row in eligible.to_pylist():
                    physical = row.pop("__physical_row")
                    reason = metadata_rejection(row, unit["language"], policy)
                    if reason:
                        rejections[reason] += 1
                        continue
                    value = {"eligible_ordinal": ordinal, "source_row": physical, "metadata": row}
                    encoded = (
                        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                    if size + len(encoded) > maximum_index_bytes:
                        raise ValueError("eligible index exceeds its declared size envelope")
                    handle.write(encoded)
                    size += len(encoded)
                    ordinal += 1
            offset += batch.num_rows
            if offset >= unit["stop_row"]:
                break
        handle.flush()
        os.fsync(handle.fileno())
    if count != unit["stop_row"] - unit["start_row"] or ordinal + sum(rejections.values()) != count:
        raise ValueError("eligible index does not conserve physical rows")
    manifest = {
        "format": "speck_code_eligible_index",
        "format_version": 1,
        "config_sha256": _digest(config),
        "physical_rows": count,
        "eligible_rows": ordinal,
        "metadata_rejections": dict(rejections),
        "output": {
            "path": str(rows_path.resolve()),
            "sha256": file_sha256(rows_path),
            "bytes": size,
        },
        "boundary": "Original-order metadata eligibility only; coarse rejection counts combine score/license-type failures and do not reproduce legacy rejection precedence. No code-content, security, token capacity or training qualification.",
    }
    durable_json(output / "manifest.json", manifest)
    return manifest


def sample_index(manifest, *, per_stratum, strata, seed):
    if type(per_stratum) is not int or type(strata) is not int or per_stratum < 1 or strata < 1:
        raise ValueError("invalid eligible-index sample allocation")
    path = Path(manifest["output"]["path"])
    if file_sha256(path) != manifest["output"]["sha256"]:
        raise ValueError("eligible sample index changed")
    n = manifest["eligible_rows"]
    selected = {}
    for group in range(strata):
        low, high = n * group // strata, n * (group + 1) // strata
        population = high - low
        rng = random.Random(f"{seed}:{manifest['config_sha256']}:{group}")
        for ordinal in rng.sample(range(low, high), min(per_stratum, population)):
            selected[ordinal] = {
                "stratum": group,
                "population": population,
                "sample_size": min(per_stratum, population),
            }
    result = []
    with path.open() as handle:
        for ordinal, line in enumerate(handle):
            if ordinal in selected:
                row = json.loads(line)
                if row["eligible_ordinal"] != ordinal:
                    raise ValueError("eligible index ordinal mismatch")
                result.append({**row, **selected[ordinal]})
    if len(result) != len(selected):
        raise ValueError("eligible index sample coverage mismatch")
    return result
