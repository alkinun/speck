"""Publish a bounded-memory length census of selected existing document-token indices."""

import argparse
import json
from pathlib import Path

from speck.data.document_lengths import census
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the previous length census")
    spec = json.loads(args.inputs.read_text())
    if (
        spec.get("format") != "speck_document_length_census_inputs"
        or spec.get("format_version") != 1
    ):
        raise ValueError("unsupported document census inputs")
    rows = []
    for binding in spec["token_results"]:
        path = Path(binding["path"])
        if file_sha256(path) != binding["sha256"]:
            raise ValueError("token result changed")
        cache = json.loads(path.read_text())
        row = census(cache["manifest"])
        if row["tokens"] != cache["tokens"] or row["documents"] != cache["documents"]:
            raise ValueError("index count differs from cache receipt")
        if row["source_id"] in {r["source_id"] for r in rows}:
            raise ValueError("overlapping banks from the same source cannot be combined")
        if row["tokenizer"]["sha256"] != spec["tokenizer_sha256"]:
            raise ValueError("document census tokenizers differ")
        rows.append({"token_result": binding, **row})
        print(
            json.dumps(
                {
                    "source_id": row["source_id"],
                    "maximum_document_tokens": row["maximum_document_tokens"],
                }
            ),
            flush=True,
        )
    durable_json(
        args.output,
        {
            "format": "speck_document_length_census",
            "format_version": 1,
            "inputs": {"path": str(args.inputs.resolve()), "sha256": file_sha256(args.inputs)},
            "implementation": [
                {
                    "path": str(repository_root() / name),
                    "sha256": file_sha256(repository_root() / name),
                }
                for name in ("speck/data/document_lengths.py", "scripts/document_length_census.py")
            ],
            "sources": rows,
            "globally_unique_long_tokens": None,
            "training_authority": False,
        },
    )


if __name__ == "__main__":
    main()
