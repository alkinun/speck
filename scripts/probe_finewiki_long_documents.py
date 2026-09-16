"""Inspect long candidates in the already pinned first FineWiki shard, with no network path."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from speck.data.long_document_probe import probe_rows
from speck.data.production_rehearsal import _raw_local_path
from speck.data.reference_stock import load_reference_preparation
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root
from speck.tokenization.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the previous long-document probe")
    spec = json.loads(args.inputs.read_text())
    if (
        spec.get("format") != "speck_finewiki_long_document_probe"
        or spec.get("format_version") != 1
    ):
        raise ValueError("unsupported long-document probe")
    binding = spec["stock_plan"]
    if file_sha256(binding["path"]) != binding["sha256"]:
        raise ValueError("pinned source plan changed")
    plan = load_reference_preparation(binding["path"])
    if spec["probe"]["minimum_chars"] != plan["base"]["filtering"]["max_chars"] + 1:
        raise ValueError(
            "probe must target documents beyond the preserved short-stock character cap"
        )
    tokenizer_id = spec["tokenizer"]
    decision_id = spec["tokenizer_decision"]
    if file_sha256(decision_id["path"]) != decision_id["sha256"]:
        raise ValueError("tokenizer decision changed")
    decision = json.loads(Path(decision_id["path"]).read_text())
    if (
        decision["status"] != "tokenizer_selected_and_frozen"
        or decision["tokenizer_fingerprint"] != tokenizer_id["sha256"]
    ):
        raise ValueError("probe does not use the selected tokenizer")
    if file_sha256(tokenizer_id["path"]) != tokenizer_id["sha256"]:
        raise ValueError("probe tokenizer changed")
    unit = plan["units"][0]
    raw = _raw_local_path(
        Path(plan["raw_directory"]),
        unit["reader"],
        unit["reader"]["revision"],
        unit["raw"]["filename"],
    )
    if (
        not raw.is_file()
        or raw.stat().st_size != unit["raw"]["bytes"]
        or file_sha256(raw) != unit["raw"]["sha256"]
    ):
        raise ValueError(
            "probe requires complete verified local raw file; no download is permitted"
        )
    print("Verified local pinned raw file; scanning length candidates", flush=True)
    parquet = pq.ParquetFile(raw)
    if parquet.metadata.num_rows != unit["expected_file_rows"]:
        raise ValueError("physical source row count differs")

    def rows():
        for batch in parquet.iter_batches(batch_size=8, columns=["text", "id", "in_language"]):
            yield from batch.to_pylist()

    result = probe_rows(rows(), Tokenizer(tokenizer_id["path"]), **spec["probe"])
    if result["physical_rows"] != unit["expected_file_rows"]:
        raise ValueError("incomplete raw candidate scan")
    durable_json(
        args.output,
        {
            "format": "speck_finewiki_long_document_probe_result",
            "format_version": 1,
            "inputs": {"path": str(args.inputs.resolve()), "sha256": file_sha256(args.inputs)},
            "source_plan": binding,
            "source_use": plan["source_use"]["identity"],
            "raw": {"path": str(raw), **unit["raw"]},
            "tokenizer": tokenizer_id,
            "implementation": [
                {
                    "path": str(repository_root() / name),
                    "sha256": file_sha256(repository_root() / name),
                }
                for name in (
                    "speck/data/long_document_probe.py",
                    "scripts/probe_finewiki_long_documents.py",
                )
            ],
            "probe": spec["probe"],
            "result": result,
            "training_authority": False,
            "production_filters_changed": False,
            "network_acquisition_performed": False,
        },
    )
    print(f"Completed bounded probe: {len(result['sample'])} long candidates tokenized", flush=True)


if __name__ == "__main__":
    main()
