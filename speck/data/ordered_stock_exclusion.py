"""Import completed ordered Stack-Edu units into the existing full-exclusion stock pipeline."""

import json
import time
from collections import defaultdict
from pathlib import Path

from speck.data.acquisition_units import _bound_identity, _digest, _unit_config
from speck.data.ordered_stock_recovery import bound_json
from speck.data.stack_edu_ordered_stock import load_ordered_stock
from speck.data.stack_edu_ordered_units import reopen_batch
from speck.data.stack_edu_stock import load_stack_edu_preparation
from speck.provenance.io import file_sha256


def validate_ordered_units(result, acquisition):
    """Require a complete finite result and recheck original unit configurations and payloads."""
    if (
        result.get("format") != "speck_stack_edu_ordered_acquisition_result"
        or result.get("format_version") not in (1, 2)
        or result.get("status") != "content_acquisition_complete"
        or result.get("training_authority") is not False
        or result.get("full_exclusion_performed") is not False
        or result["inputs"] != acquisition["inputs"]
        or result["source_use"] != acquisition["source_use"]["identity"]
        or result["reference_tokenizer"] != acquisition["reference_tokenizer"]
    ):
        raise ValueError("full exclusion requires complete scope-identical ordered acquisition")
    work = Path(acquisition["working_directory"])
    allowed = [work / "acquired"]
    if acquisition.get("reuse_owner_directory"):
        allowed.insert(0, Path(acquisition["reuse_owner_directory"]) / "acquired")
    if result["acquired_directories"] != [str(p) for p in allowed]:
        raise ValueError("ordered result names a different acquisition location")
    metadata = {entry["index_identity"]["sha256"]: entry for entry in acquisition["files"]}
    units, identities, seen = [], {}, set()
    file_positions = {
        entry["index_identity"]["sha256"]: i for i, entry in enumerate(acquisition["files"])
    }
    previous_position = None
    previous_stop = None
    totals = defaultdict(lambda: {"documents": 0, "tokens": 0})
    for item in result["units"]:
        unit = item["unit"]
        name, language = unit["id"], unit["language"]
        if (
            not name
            or Path(name).name != name
            or name in seen
            or language not in acquisition["language_order"]
            or unit["category"] != "code"
        ):
            raise ValueError("duplicate, unknown or invalid ordered unit")
        seen.add(name)
        entry = metadata.get(unit["index_identity"]["sha256"])
        if (
            entry is None
            or unit["index_identity"] != entry["index_identity"]
            or any(
                unit[key] != entry["unit"][key]
                for key in ("raw", "reader", "language", "metadata_path", "expected_file_rows")
            )
        ):
            raise ValueError("ordered unit differs from its bound metadata")
        position = (
            acquisition["language_order"].index(language),
            file_positions[unit["index_identity"]["sha256"]],
            unit["start_row"],
        )
        if (
            type(unit["start_row"]) is not int
            or type(unit["stop_row"]) is not int
            or not 0 <= unit["start_row"] < unit["stop_row"] <= unit["expected_file_rows"]
            or (
                previous_position is not None
                and (
                    position <= previous_position
                    or (position[:2] == previous_position[:2] and position[2] < previous_stop)
                )
            )
        ):
            raise ValueError("ordered unit order or physical-row ranges changed")
        previous_position, previous_stop = position, unit["stop_row"]
        identity = item["manifest_identity"]
        path = Path(identity["path"])
        if (
            path.name != "manifest.json"
            or path.parent.name != name
            or path.parent.parent not in allowed
        ):
            raise ValueError("ordered unit manifest escapes its original acquisition")
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("ordered acquisition manifest cannot be a redirected link")
        original = bound_json(identity)
        config = _unit_config(acquisition, unit)
        if (
            original != item["manifest"]
            or json.loads((path.parent / "config.json").read_text()) != config
            or reopen_batch(path.parent, config) != original
        ):
            raise ValueError("ordered acquisition payload or configuration changed")
        units.append(unit)
        identities[name] = identity
        totals[language]["documents"] += original["retained_records"]
        totals[language]["tokens"] += original["tokens_before_full_exclusion"]
    progress = result["progress"]
    if (
        not units
        or len(units) != progress["complete_units"]
        or progress["state"] != "content_acquisition_complete"
        or progress["unprocessed_languages"]
        or progress["shortfall_languages"]
        or set(totals) != set(acquisition["language_order"])
        or sum(r["documents"] for r in totals.values()) != progress["retained_records"]
        or sum(r["tokens"] for r in totals.values()) != progress["tokens_before_full_exclusion"]
        or set(progress["by_language_before_full_exclusion"]) != set(totals)
    ):
        raise ValueError("ordered acquisition coverage or totals differ")
    for language, count in totals.items():
        row = progress["by_language_before_full_exclusion"][language]
        if (
            any(row[k] != v for k, v in count.items())
            or count["tokens"] < acquisition["targets"][language]["preparation_target_tokens"]
        ):
            raise ValueError("ordered per-language totals or pre-exclusion headroom differ")
    return units, identities


def load_ordered_exclusion(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if (
        set(spec)
        != {
            "format",
            "format_version",
            "source_id",
            "ordered_result",
            "output_directory",
            "training_authority",
        }
        or spec["format"] != "speck_ordered_stack_edu_exclusion_preparation"
        or spec["format_version"] != 1
        or spec["source_id"] != "stack_edu"
        or spec["training_authority"] is not False
    ):
        raise ValueError("unsupported ordered Stack-Edu exclusion preparation")
    result_id = _bound_identity(spec["ordered_result"], path.parent)
    result = bound_json(result_id)
    acquisition_id = _bound_identity(result["plan"], Path(result_id["path"]).parent)
    acquisition = load_ordered_stock(acquisition_id["path"])
    parent_id = acquisition["inputs"]["old_stock_plan"]
    parent = load_stack_edu_preparation(parent_id["path"])
    output = (path.parent / spec["output_directory"]).resolve()
    protected = [
        path,
        Path(result_id["path"]),
        Path(acquisition_id["path"]),
        Path(acquisition["working_directory"]),
        Path(acquisition["archive_directory"]),
    ]
    if acquisition.get("reuse_owner_directory"):
        protected.append(Path(acquisition["reuse_owner_directory"]))
        protected.append(Path(bound_json(acquisition["supersedes"])["archive_directory"]))
    protected.extend(Path(p) for p in acquisition["fallback_caches"])
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("exclusion output overlaps preserved acquisition inputs")
    units, identities = validate_ordered_units(result, acquisition)
    targets = {
        language: row["preparation_target_tokens"]
        for language, row in acquisition["targets"].items()
    }
    return {
        **parent,
        **spec,
        "output_directory": str(output),
        "base": acquisition["base"],
        "checkpoint_rows": acquisition["checkpoint_rows"],
        "reference_tokenizer": acquisition["reference_tokenizer"],
        "source_use": acquisition["source_use"],
        "reference_parent": _bound_identity(
            parent["reference_parent"], Path(parent_id["path"]).parent
        ),
        "sqlite_policy": _bound_identity(parent["sqlite_policy"], Path(parent_id["path"]).parent),
        "units": units,
        "reuse_acquisition_units": identities,
        "source_language_targets": targets,
        "target_reference_tokens": sum(targets.values()),
        "ordered_result": result_id,
        "ordered_acquisition_plan": acquisition_id,
    }


def imported_units(plan, output):
    """Verify the copies made by source_stock.reuse_completed_units; never call a fetcher."""
    started = time.perf_counter()
    rows = []
    for unit in plan["units"]:
        original = bound_json(plan["reuse_acquisition_units"][unit["id"]])
        directory = Path(output) / unit["id"]
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest != original
            or manifest["config_sha256"] != _digest(_unit_config(plan, unit))
            or json.loads((directory / "config.json").read_text()) != _unit_config(plan, unit)
        ):
            raise ValueError("copied ordered unit differs from its original owner")
        for key in ("output", "security_report"):
            entry = manifest[key]
            payload = (directory / entry["path"]).resolve()
            if (
                not payload.is_relative_to(directory.resolve())
                or file_sha256(payload) != entry["sha256"]
            ):
                raise ValueError("copied ordered acquisition payload changed")
        rows.append(
            {
                "unit": unit,
                "manifest": manifest,
                "original_manifest": plan["reuse_acquisition_units"][unit["id"]],
                "copied_payload_keys": ["output", "security_report"],
            }
        )
    return {
        "units": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "ordered_acquisition_result": plan["ordered_result"],
        "boundary": "Only the output/security/config/manifest copies used by existing exclusion are present here. Complete raw/unscanned/fetch/archive history stays at the bound original paths. Copy/verification time is not original acquisition cost; no network acquisition performed.",
    }
