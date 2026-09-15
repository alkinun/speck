"""Bind the current Stack-Edu E1S allocation to preserved eligible metadata indices."""

import json
from pathlib import Path

from speck.data.acquisition_units import _bound_identity, _digest
from speck.data.stack_edu_metadata import load_metadata_plan, verify_metadata_file
from speck.data.stack_edu_stock import load_stack_edu_preparation, metadata_rejection
from speck.experiments.code_languages import load_code_languages
from speck.provenance.io import file_sha256


def load_ordered_stock(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if (
        spec.get("format") != "speck_stack_edu_ordered_acquisition"
        or spec.get("format_version") != 1
        or spec.get("source_id") != "stack_edu"
        or spec.get("training_authority") is not False
        or spec.get("eligible_rows_per_unit") != 512
        or (spec.get("workers"), spec.get("window")) != (32, 64)
        or spec.get("candidate_nominal_multiplier") != 2
        or spec.get("stop_on_pre_exclusion_language_shortfall") is not True
        or spec.get("maximum_cache_bytes") != 12884901888
        or spec.get("maximum_working_bytes") != 25769803776
        or spec.get("minimum_free_bytes") != 68719476736
        or spec.get("language_order")
        != [
            "Java",
            "TypeScript",
            "Python",
            "C++",
            "JavaScript",
            "C",
            "Go",
            "Rust",
            "Markdown",
            "SQL",
            "Shell",
        ]
    ):
        raise ValueError("unsupported E1S ordered acquisition contract")
    inputs = {
        key: _bound_identity(spec[key], path.parent)
        for key in ("old_stock_plan", "metadata_result", "code_languages", "fetch_qualification")
    }
    parent = load_stack_edu_preparation(inputs["old_stock_plan"]["path"])
    languages = load_code_languages(inputs["code_languages"]["path"])
    targets = {
        row["language"]: row
        for row in languages["source_language_capacity_envelope"]
        if row["source_id"] == "stack_edu"
    }
    if (
        set(targets) != set(spec["language_order"])
        or sum(row["nominal_tokens"] for row in targets.values()) != 300000000
    ):
        raise ValueError("ordered stock requires the selected 300M Stack-Edu treatment")
    qualification = json.loads(Path(inputs["fetch_qualification"]["path"]).read_text())
    if (
        qualification.get("training_authority") is not False
        or qualification.get("status")
        != "bounded_fetch_replay_and_content_checks_pass_full_exclusion_pending"
        or qualification.get("complete_reopen_pass") is not True
        or qualification.get("journal_parity_pass") is not True
        or qualification.get("stock_plan") != inputs["old_stock_plan"]
    ):
        raise ValueError("fetch qualification must remain preparation-only")
    metadata = json.loads(Path(inputs["metadata_result"]["path"]).read_text())
    intake_id = _bound_identity(metadata["plan"], Path(inputs["metadata_result"]["path"]).parent)
    intake = load_metadata_plan(intake_id["path"])
    if (
        metadata.get("status") != "complete_metadata_verified_not_code_stock"
        or metadata.get("training_authority") is not False
        or metadata["inputs"] != intake["inputs"]
        or [row["unit"] for row in metadata["files"]] != intake["units"]
        or len(metadata["files"]) != 28
        or len(spec["files"]) != 28
        or intake["inputs"]["source_qualification"]["sha256"]
        != parent["source_qualification"]["sha256"]
        or intake["inputs"]["source_use"]["sha256"] != parent["source_use"]["identity"]["sha256"]
    ):
        raise ValueError("ordered stock metadata/source-use lineage differs")
    files = []
    for entry, binding in zip(metadata["files"], spec["files"], strict=True):
        unit = entry["unit"]
        manifest_id = _bound_identity(binding["manifest"], path.parent)
        config_id = _bound_identity(binding["config"], path.parent)
        manifest = json.loads(Path(manifest_id["path"]).read_text())
        config = json.loads(Path(config_id["path"]).read_text())
        old = config["unit"]
        if (
            binding["metadata_sha256"] != unit["raw"]["sha256"]
            or manifest["format"] != "speck_code_eligible_index"
            or manifest["config_sha256"] != _digest(config)
            or config["policy"] != parent["base"]["stack_edu_policy"]
            or any(
                old[key] != unit[key] for key in ("raw", "reader", "language", "expected_file_rows")
            )
            or old["metadata_path"] != entry["raw"]["path"]
            or old["start_row"] != 0
            or old["stop_row"] != unit["expected_file_rows"]
            or manifest["physical_rows"] != unit["expected_file_rows"]
            or manifest["eligible_rows"] + sum(manifest["metadata_rejections"].values())
            != manifest["physical_rows"]
        ):
            raise ValueError("eligible index differs from complete metadata or unchanged policy")
        files.append(
            {
                "unit": {**unit, "metadata_path": entry["raw"]["path"]},
                "index": manifest,
                "index_identity": manifest_id,
            }
        )
    return {
        **spec,
        "inputs": inputs,
        "base": parent["base"],
        "checkpoint_rows": parent["checkpoint_rows"],
        "reference_tokenizer": parent["reference_tokenizer"],
        "source_use": parent["source_use"],
        "targets": targets,
        "files": files,
    }


def verify_index_file(entry):
    verify_metadata_file(entry["unit"]["metadata_path"], entry["unit"])
    payload = entry["index"]["output"]
    path = Path(payload["path"])
    if path.stat().st_size != payload["bytes"] or file_sha256(path) != payload["sha256"]:
        raise ValueError("preserved eligible index payload changed")


def index_batches(entry, policy, size):
    """Recheck indices while streaming fixed batches in original physical-row order."""
    previous = -1
    count = 0
    batch = []
    with Path(entry["index"]["output"]["path"]).open() as handle:
        for ordinal, line in enumerate(handle):
            row = json.loads(line)
            if (
                row["eligible_ordinal"] != ordinal
                or not previous < row["source_row"] < entry["unit"]["expected_file_rows"]
                or metadata_rejection(row["metadata"], entry["unit"]["language"], policy)
                is not None
            ):
                raise ValueError("eligible index order or source policy differs")
            previous = row["source_row"]
            count += 1
            batch.append({**row, "blob_id": row["metadata"]["blob_id"]})
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch
    if count != entry["index"]["eligible_rows"]:
        raise ValueError("eligible index count changed")
