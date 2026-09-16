"""Measure intact-document length capacity from bound token indices without repacking text."""

import hashlib
import heapq
import json
from pathlib import Path

from speck.provenance.io import file_sha256

LENGTHS = (4096, 32768, 65536, 131072)


def census(manifest_identity):
    path = Path(manifest_identity["path"]).resolve()
    if file_sha256(path) != manifest_identity["sha256"]:
        raise ValueError("token-stock manifest changed")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("format") != "speck_document_token_stock"
        or manifest.get("status") != "complete_document_token_cache_not_training_view"
        or manifest.get("training_authority") is not False
    ):
        raise ValueError("length census requires a complete document-token index")
    index = (path.parent / manifest["documents"]["path"]).resolve()
    if not index.is_relative_to(path.parent) or index == path:
        raise ValueError("document index escapes stock directory")
    counts = {
        length: {
            "documents_at_least_context_plus_one": 0,
            "tokens_in_those_documents": 0,
            "nonoverlapping_input_windows_with_lookahead": 0,
        }
        for length in LENGTHS
    }
    digest = hashlib.sha256()
    position = documents = utf8 = 0
    largest = []
    with index.open("rb") as handle:
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            if (
                type(row["ordinal"]) is not int
                or row["ordinal"] != documents
                or type(row["token_start"]) is not int
                or row["token_start"] != position
                or type(row["token_count"]) is not int
                or row["token_count"] <= 0
                or type(row["utf8_bytes"]) is not int
                or row["utf8_bytes"] < 0
                or not isinstance(row["content_id"], str)
                or not row["content_id"]
            ):
                raise ValueError("document index order, spans or counts differ")
            size = row["token_count"]
            for length, count in counts.items():
                if size >= length + 1:
                    count["documents_at_least_context_plus_one"] += 1
                    count["tokens_in_those_documents"] += size
                    count["nonoverlapping_input_windows_with_lookahead"] += (size - 1) // length
            item = (size, documents, row["content_id"], row["released_content_sha256"])
            heapq.heappush(largest, item)
            if len(largest) > 10:
                heapq.heappop(largest)
            position += size
            documents += 1
            utf8 += row["utf8_bytes"]
    if digest.hexdigest() != manifest["documents"]["sha256"]:
        raise ValueError("document index payload hash differs")
    if documents != manifest["document_count"] or position != manifest["token_count"]:
        raise ValueError("document index totals differ from token stock")
    return {
        "source_id": manifest["plan"]["source_id"],
        "category": manifest["plan"]["category"],
        "manifest": manifest_identity,
        "document_index": {"path": str(index), "sha256": digest.hexdigest()},
        "tokenizer": manifest["plan"]["tokenizer"],
        "documents": documents,
        "tokens": position,
        "utf8_bytes": utf8,
        "maximum_document_tokens": max((r[0] for r in largest), default=0),
        "context_capacity": {str(k): v for k, v in counts.items()},
        "longest_document_indices": [
            {"tokens": n, "ordinal": ordinal, "content_id": name, "released_content_sha256": sha}
            for n, ordinal, name, sha in sorted(largest, reverse=True)
        ],
        "qualified_coherent_family_tokens": None,
        "boundary": "Index-only, whole-document length capacity, including existing BOS/EOS. Every candidate window needs context+1 tokens within one source document; input spans are disjoint and one lookahead token can be shared. No concatenation, family/edition deduplication, window materialization, recurrent reset or useful-long-dependency claim. Token-shard contents are not reopened by this index census. Longest indices are candidates for provenance review, not selected training examples.",
    }
