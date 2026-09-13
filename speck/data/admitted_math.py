"""Prepare admitted natural math text through the existing unit and exclusion paths."""

import json
import re
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.configuration import _validate_source
from speck.data.rights import load_source_use_extension
from speck.provenance.io import file_sha256
from speck.tokenization.tokenizer import Tokenizer


def load_math_preparation(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        value.get("format") != "speck_admitted_math_preparation"
        or value.get("format_version") != 1
        or value.get("source_id") != "ultradata_math_l2_preview"
        or value.get("training_authority") is not False
        or value.get("rows_per_file") != 100000
        or value.get("checkpoint_rows") != 256
    ):
        raise ValueError("unsupported admitted-math preparation plan")
    use_identity = _bound_identity(value["source_use_extension"], path.parent)
    use = load_source_use_extension(use_identity["path"])
    source = use["source"]
    if (
        source["id"] != value["source_id"]
        or source["category"] != "math"
        or source["repo"] != "openbmb/UltraData-Math"
        or source["config"] != "UltraData-Math-L2-preview"
        or source["content_field"] != "content"
    ):
        raise ValueError("source-use extension does not admit this natural math view")
    base_identity = _bound_identity(value["base_plan"], path.parent)
    base_path = Path(base_identity["path"])
    base = json.loads(base_path.read_text())
    for key in ("source_registry", "rights_record", "deny_ledger", "contamination_plan"):
        base[key] = _bound_identity(base[key], base_path.parent)
    if base["rights_record"]["sha256"] != use["parent_acceptance"]["sha256"]:
        raise ValueError("math preparation does not inherit the approved parent scope")
    base["security"]["gitleaks_binary"] = _bound_identity(
        base["security"]["gitleaks_binary"], base_path.parent
    )
    base["source_use_extension"] = use_identity
    base["math_english"] = {
        "language_policy": "math_prose",
        "minimum_prose_alphabetic_characters": 80,
        "minimum_detected_English_probability": 0.8,
    }
    reader = _validate_source(
        {
            "id": source["id"],
            "repo": source["repo"],
            "revision": source["revision"],
            "tree_path": "data/UltraData-Math-L2-preview",
            "file_format": "parquet",
            "content_column": "content",
            "metadata_columns": {"quality_label": "quality_label"},
            "filters": {},
        }
    )
    if len(value["raw_files"]) != 2 or len({item["filename"] for item in value["raw_files"]}) != 2:
        raise ValueError("math preparation requires the two frozen raw units")
    units = []
    for index, item in enumerate(value["raw_files"]):
        if (
            set(item) != {"filename", "sha256", "bytes"}
            or not item["filename"].startswith("data/UltraData-Math-L2-preview/")
            or ".." in Path(item["filename"]).parts
            or not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            or type(item["bytes"]) is not int
            or not 0 < item["bytes"] <= 1_000_000_000
        ):
            raise ValueError("math preparation raw view differs from the admitted subset")
        units.append(
            {
                "id": f"ultradata_math_l2_preview__file_{index}",
                "category": "math",
                "reader": reader,
                "raw": {**item, "source_id": source["id"]},
                "start_row": 0,
                "stop_row": value["rows_per_file"],
            }
        )
    if (
        value.get("target_reference_tokens") != 200_000_000
        or value.get("maximum_observed_wal_bytes") != 2_147_483_648
    ):
        raise ValueError("math preparation capacity/resource envelope differs from the plan")
    return {
        **value,
        "base": base,
        "units": units,
        "source_use": use,
        "raw_directory": str((path.parent / value["raw_directory"]).resolve()),
        "output_directory": str((path.parent / value["output_directory"]).resolve()),
    }


def count_reference_tokens(source_path, tokenizer_identity):
    if file_sha256(tokenizer_identity["path"]) != tokenizer_identity["sha256"]:
        raise ValueError("reference tokenizer identity mismatch")
    tokenizer = Tokenizer(tokenizer_identity["path"])
    documents = tokens = characters = 0
    batch = []
    with Path(source_path).open() as handle:
        for raw in handle:
            batch.append(json.loads(raw)["text"])
            characters += len(batch[-1])
            if len(batch) >= 128 or characters >= 1_000_000:
                encoded = tokenizer.encode_batch(batch, bos=True, eos=True)
                tokens += sum(len(row) for row in encoded)
                documents += len(batch)
                batch, characters = [], 0
        if batch:
            tokens += sum(len(row) for row in tokenizer.encode_batch(batch, bos=True, eos=True))
            documents += len(batch)
    return {
        "documents": documents,
        "tokens": tokens,
        "tokenizer": tokenizer_identity,
        "scope": "Reference-token capacity only; no training shards or final D5 selection",
    }
