"""Bind a complete pinned FineWiki shard to the approved reference reader and source use."""

import json
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.configuration import _validate_source


def load_reference_preparation(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        value.get("format") != "speck_reference_stock_preparation"
        or value.get("format_version") != 1
        or value.get("source_id") != "finewiki_en"
        or value.get("training_authority") is not False
        or value.get("checkpoint_rows") != 256
        or value.get("target_reference_tokens") != 400_000_000
        or value.get("maximum_observed_wal_bytes") != 2_147_483_648
    ):
        raise ValueError("unsupported reference stock plan")
    identity = _bound_identity(value["base_plan"], path.parent)
    base_path = Path(identity["path"])
    base = json.loads(base_path.read_text())
    for key in ("source_registry", "rights_record", "deny_ledger", "contamination_plan"):
        base[key] = _bound_identity(base[key], base_path.parent)
    base["security"]["gitleaks_binary"] = _bound_identity(
        base["security"]["gitleaks_binary"], base_path.parent
    )
    rights = json.loads(Path(base["rights_record"]["path"]).read_text())
    if (
        rights.get("status") != "all_sources_human_approved"
        or rights.get("automated_approval_made") is not False
        or value["source_id"] not in rights["approved_source_ids"]
    ):
        raise ValueError("reference source needs its existing human approval")
    source = next(item for item in base["sources"] if item["id"] == value["source_id"])
    reader = _validate_source(source["reader"])
    if (
        source["category"] != "reference"
        or reader["repo"] != "HuggingFaceFW/finewiki"
        or reader["revision"] != "8bd13e72e6a002407649b3e898535f42ceb1aeb9"
        or reader["tree_path"] != "data/enwiki"
        or reader["filters"] != {"language": "en"}
    ):
        raise ValueError("reference reader differs from the qualified English FineWiki view")
    expected = {
        "filename": "data/enwiki/000_00000.parquet",
        "bytes": 2_510_037_970,
        "sha256": "8770000c2b1fd62743601ea24cdebd784f943cf3fa7bbe33161521a6f5ca7c74",
        "rows": 421456,
    }
    if value["raw_files"] != [expected]:
        raise ValueError("reference preparation requires its complete pinned first shard")
    unit = {
        "id": "finewiki_en__file_0",
        "category": "reference",
        "reader": reader,
        "raw": {key: item for key, item in expected.items() if key != "rows"}
        | {"source_id": value["source_id"]},
        "start_row": 0,
        "stop_row": expected["rows"],
        "expected_file_rows": expected["rows"],
    }
    return {
        **value,
        "base": base,
        "units": [unit],
        "source_use": {**rights, "identity": base["rights_record"]},
        "raw_directory": str((path.parent / value["raw_directory"]).resolve()),
        "output_directory": str((path.parent / value["output_directory"]).resolve()),
    }
