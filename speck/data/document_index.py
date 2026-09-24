"""Index the tokens of every document one preprocess pass retained for a source.

The joint family graph and the ladder builder need each retained document's content hash and
token count, in the order of the preprocessed text, bound to that exact text. This writes that
index and a manifest naming its input. It stores no token IDs and admits nothing.
"""

import json
from multiprocessing import Pool
from pathlib import Path

from speck.provenance.io import atomic_json, file_sha256
from speck.tokenization.tokenizer import Tokenizer

FORMAT = "speck_document_token_index"
_worker = {}


def _count(job):
    tokenizer_path, fields, lines = job
    tokenizer = _worker.get(tokenizer_path) or _worker.setdefault(
        tokenizer_path, Tokenizer(tokenizer_path)
    )
    rows = [json.loads(line) for line in lines]
    encoded = tokenizer.encode_batch([row["text"] for row in rows], bos=True, eos=True)
    return [
        {"released_content_sha256": row["released_content_sha256"], "token_count": len(tokens)}
        | {field: row[field] for field in fields}
        for row, tokens in zip(rows, encoded, strict=True)
    ]


def _jobs(path, tokenizer_path, fields, size=512):
    with path.open() as handle:
        batch = []
        for line in handle:
            batch.append(line)
            if len(batch) == size:
                yield tokenizer_path, fields, batch
                batch = []
        if batch:
            yield tokenizer_path, fields, batch


def index_documents(preprocessed, source_id, output, tokenizer, *, fields=(), workers=16):
    """Write documents.jsonl and manifest.json for one preprocessed source into a new directory."""
    preprocessed, output, tokenizer = Path(preprocessed), Path(output), Path(tokenizer)
    manifest_path = preprocessed / "manifest.json"
    entry = json.loads(manifest_path.read_text())["outputs"][source_id]
    text = preprocessed / entry["path"]
    if file_sha256(text) != entry["sha256"]:
        raise ValueError("retained text differs from its preprocess manifest")
    output.mkdir(parents=True, exist_ok=False)
    documents = tokens = 0
    with Pool(workers) as pool, (output / "documents.jsonl").open("w") as index:
        for rows in pool.imap(_count, _jobs(text, str(tokenizer), tuple(fields))):
            for row in rows:
                index.write(json.dumps({"ordinal": documents} | row, sort_keys=True) + "\n")
                documents += 1
                tokens += row["token_count"]
    manifest = {
        "format": FORMAT,
        "format_version": 1,
        "training_admitted": False,
        "plan": {
            "source_id": source_id,
            "input": {"path": str(text), "sha256": entry["sha256"]},
            "parent_manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        },
        "tokenizer": {"path": str(tokenizer), "sha256": file_sha256(tokenizer)},
        "documents": {"path": "documents.jsonl", "sha256": file_sha256(output / "documents.jsonl")},
        "document_count": documents,
        "tokens": tokens,
        "boundary": "Tokens with BOS/EOS per retained document; no token IDs, split or admission.",
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest
