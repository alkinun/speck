"""Acquire complete pinned Stack-Edu metadata files for the matched code-language view."""

import json
import os
import shutil
import time
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from speck.data.acquisition_units import _bound_identity, _raw_file
from speck.data.configuration import _validate_source
from speck.data.production_rehearsal import _raw_local_path
from speck.experiments.code_languages import load_code_languages
from speck.provenance.io import durable_json, file_sha256

REQUIRED_COLUMNS = {
    "blob_id",
    "language",
    "repo_name",
    "path",
    "src_encoding",
    "length_bytes",
    "score",
    "int_score",
    "detected_licenses",
    "license_type",
}


def load_metadata_plan(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        value.get("format") != "speck_stack_edu_metadata_acquisition"
        or value.get("format_version") != 1
        or value.get("training_authority") is not False
    ):
        raise ValueError("unsupported Stack-Edu metadata plan")
    inputs = {
        key: _bound_identity(value[key], path.parent)
        for key in ("metadata_manifest", "code_languages", "source_qualification", "source_use")
    }
    manifest, qualification, rights = (
        json.loads(Path(inputs[key]["path"]).read_text())
        for key in ("metadata_manifest", "source_qualification", "source_use")
    )
    code = load_code_languages(inputs["code_languages"]["path"])
    if (
        rights.get("status") != "all_sources_human_approved"
        or rights.get("automated_approval_made") is not False
        or "stack_edu" not in rights["approved_source_ids"]
        or manifest.get("format") != "speck_stack_edu_metadata_manifest"
        or manifest.get("format_version") != 1
        or any(manifest[key] != qualification["source"][key] for key in ("repo", "revision"))
    ):
        raise ValueError("Stack-Edu metadata requires its qualified approved source identity")
    units = []
    languages = set()
    for row in manifest["files"]:
        directory = row["language_directory"]
        language = "C++" if directory == "Cpp" else directory
        if (
            language in languages
            or language not in code["language_weights_percent"]
            or Path(row["filename"]).parts[0] != directory
            or len(Path(row["filename"]).parts) != 2
            or not Path(row["filename"]).name.startswith("train-00000-of-")
            or not row["filename"].endswith(".parquet")
            or type(row["rows"]) is not int
            or not 0 < row["rows"] <= 10000000
            or type(row["bytes"]) is not int
            or not 0 < row["bytes"] <= 1000000000
            or not REQUIRED_COLUMNS.issubset(row["columns"])
        ):
            raise ValueError("metadata shard differs from the matched complete-file scope")
        languages.add(language)
        reader = _validate_source(
            {
                "id": "stack_edu",
                "repo": manifest["repo"],
                "revision": manifest["revision"],
                "tree_path": directory,
                "file_format": "parquet",
                "content_column": "blob_id",
                "metadata_columns": {},
                "filters": {},
            }
        )
        units.append(
            {
                "id": f"stack_edu_metadata__{directory}",
                "language": language,
                "reader": reader,
                "raw": {
                    "source_id": "stack_edu",
                    **{k: row[k] for k in ("filename", "sha256", "bytes")},
                },
                "expected_file_rows": row["rows"],
            }
        )
    if languages != set(code["language_weights_percent"]):
        raise ValueError("metadata does not cover every matched code language")
    if set(value.get("reuse_local_files", {})) - languages:
        raise ValueError("local metadata reuse declares an unselected language")
    return {
        **value,
        "inputs": inputs,
        "units": units,
        "plan": {"path": str(path), "sha256": file_sha256(path)},
        "raw_directory": str((path.parent / value["raw_directory"]).resolve()),
        "output_directory": str((path.parent / value["output_directory"]).resolve()),
    }


def verify_metadata_file(path, unit):
    path = Path(path)
    if path.stat().st_size != unit["raw"]["bytes"] or file_sha256(path) != unit["raw"]["sha256"]:
        raise ValueError("metadata file identity mismatch")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != unit["expected_file_rows"] or not REQUIRED_COLUMNS.issubset(
        parquet.schema_arrow.names
    ):
        raise ValueError("metadata row count/schema mismatch")
    seen = 0
    for batch in parquet.iter_batches(columns=["language"], batch_size=65536, use_threads=False):
        if batch.column(0).null_count or pc.unique(batch.column(0)).to_pylist() != [
            unit["language"]
        ]:
            raise ValueError("metadata contains an unexpected or missing language")
        seen += batch.num_rows
    if seen != unit["expected_file_rows"]:
        raise ValueError("metadata language scan does not cover the complete file")
    return {
        "physical_rows": seen,
        "language": unit["language"],
        "schema_and_all_row_languages_pass": True,
    }


def acquire_metadata(plan, *, revision, resume=False):
    output = Path(plan["output_directory"])
    execution = {"plan": plan["plan"], "repository_revision": revision}
    if output.exists():
        if not resume or json.loads((output / "execution.json").read_text()) != execution:
            raise ValueError("metadata resume requires the same frozen execution")
    else:
        if resume:
            raise ValueError("cannot resume absent metadata acquisition")
        output.mkdir(parents=True)
        durable_json(output / "execution.json", execution)
    reports = []
    for unit in plan["units"]:
        started = time.perf_counter()
        receipt_path = output / (unit["id"] + ".json")
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            if previous["unit"] != unit:
                raise ValueError("published metadata unit changed")
            verify_metadata_file(previous["raw"]["path"], unit)
            reports.append(previous)
            print(f"verified completed metadata: {unit['language']}", flush=True)
            continue
        local = plan.get("reuse_local_files", {}).get(unit["language"])
        target = _raw_local_path(
            Path(plan["raw_directory"]),
            unit["reader"],
            unit["reader"]["revision"],
            unit["raw"]["filename"],
        )
        if local is not None and not target.exists():
            source = _bound_identity(local, Path(plan["plan"]["path"]).parent)
            verify_metadata_file(source["path"], unit)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_suffix(".copy-building")
            # Preserve an interrupted copy; explicit recovery decides its disposition.
            with staging.open("xb") as dest, Path(source["path"]).open("rb") as src:
                shutil.copyfileobj(src, dest, length=8 * 1024 * 1024)
                dest.flush()
                os.fsync(dest.fileno())
            verify_metadata_file(staging, unit)
            staging.rename(target)
            descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        raw = _raw_file(plan, unit)
        checked = verify_metadata_file(raw["path"], unit)
        report = {
            "unit": unit,
            "raw": raw,
            "verification": checked,
            "local_copy_input": local,
            "elapsed_seconds": time.perf_counter() - started,
        }
        durable_json(receipt_path, report)
        reports.append(report)
        print(
            f"verified complete metadata: {unit['language']} / {checked['physical_rows']} rows",
            flush=True,
        )
    return {
        "format": "speck_stack_edu_metadata_acquisition_result",
        "format_version": 1,
        "status": "complete_metadata_verified_not_code_stock",
        **execution,
        "inputs": plan["inputs"],
        "files": reports,
        "physical_rows": sum(row["verification"]["physical_rows"] for row in reports),
        "bytes": sum(row["raw"]["bytes"] for row in reports),
        "training_authority": False,
        "boundary": "Complete immutable metadata files and all-row language checks only. No code blob acquisition, per-document license/security qualification, eligible token capacity or training authority. Source text stock still requires a resumable SWH blob builder and per-language headroom after exclusion. Invocation costs overlap other local preparation.",
    }
