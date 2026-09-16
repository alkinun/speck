"""Review padding sensitivity of an already measured, hash-bound FineWiki candidate sample."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from speck.data.long_document_structure import inspect_structure
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root
from speck.tokenization.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the previous candidate review")
    probe = json.loads(args.probe.read_text())
    if probe.get("format") != "speck_finewiki_long_document_probe_result":
        raise ValueError("expected the bound FineWiki raw candidate probe")
    for key in ("inputs", "source_plan", "raw", "tokenizer"):
        if file_sha256(probe[key]["path"]) != probe[key]["sha256"]:
            raise ValueError(f"candidate probe input changed: {key}")
    selected = {s["physical_row"]: s for s in probe["result"]["sample"]}
    if not 1 <= len(selected) <= 32 or len(selected) != len(probe["result"]["sample"]):
        raise ValueError("invalid bounded candidate sample")
    tokenizer = Tokenizer(probe["tokenizer"]["path"])
    reviewed = []
    ordinal = 0
    for batch in pq.ParquetFile(probe["raw"]["path"]).iter_batches(
        batch_size=8, columns=["text", "id", "title"]
    ):
        for row in batch.to_pylist():
            if ordinal in selected:
                candidate = selected[ordinal]
                text = row["text"]
                if (
                    hashlib.sha256(text.encode()).hexdigest() != candidate["text_sha256"]
                    or row["id"] != candidate["upstream_id"]
                    or len(tokenizer.encode(text, bos=True, eos=True))
                    != candidate["tokens_including_bos_eos"]
                ):
                    raise ValueError("candidate text, identity or token count changed")
                reviewed.append(
                    {
                        **candidate,
                        "title": row["title"],
                        "structure": inspect_structure(text, tokenizer),
                    }
                )
            ordinal += 1
    if ordinal != probe["result"]["physical_rows"] or len(reviewed) != len(selected):
        raise ValueError("incomplete candidate structure review")
    durable_json(
        args.output,
        {
            "format": "speck_finewiki_long_candidate_structure_review",
            "format_version": 1,
            "probe": {"path": str(args.probe.resolve()), "sha256": file_sha256(args.probe)},
            "raw": probe["raw"],
            "tokenizer": probe["tokenizer"],
            "sample": reviewed,
            "implementation": [
                {"path": str(repository_root() / p), "sha256": file_sha256(repository_root() / p)}
                for p in (
                    "speck/data/long_document_structure.py",
                    "scripts/review_finewiki_long_candidates.py",
                )
            ],
            "qualified_stock_tokens": None,
            "production_filters_changed": False,
            "training_authority": False,
            "boundary": "Diagnostic sensitivity to collapsing horizontal whitespace while preserving newlines and non-whitespace. This is not a selected normalization, parser-independent extraction, post-filter yield, quality score or family-separated training supply. The original biased sample and raw text remain unchanged. No source text is published.",
        },
    )
    print(f"Completed structural review of {len(reviewed)} pinned candidates", flush=True)


if __name__ == "__main__":
    main()
