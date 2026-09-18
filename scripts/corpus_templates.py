"""Census numeric-template families in an indexed stock without changing training data."""

import argparse
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from speck.provenance.io import atomic_json, file_sha256

NUMBER = re.compile(r"\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?")


def template_digest(text):
    # This intentionally erases mathematically meaningful values. Never use it as
    # an automatic duplicate/rejection decision without a separate reviewed policy.
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(NUMBER.sub("<number>", normalized).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def role_hint(row):
    url = urlsplit(row.get("url") or "")
    text = row["text"].lower()
    if (url.hostname or "").removeprefix("www.") == "nrich.maths.org":
        if (
            url.path == "/public/leg.php"
            and "search by topic" in text
            and re.search(r"there are \d+ results", text)
        ):
            return "nrich_topic_directory"
    if "conversion factor" in text and ("conversion table" in text or "conversion formula" in text):
        return "conversion_exposition"
    return "unclassified"


def audit(manifest_path, expected_sha256, output):
    started = time.perf_counter()
    manifest_path, output = Path(manifest_path), Path(output)
    if file_sha256(manifest_path) != expected_sha256:
        raise ValueError("manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete_document_token_cache_not_training_view":
        raise ValueError("requires a completed indexed stock")
    output.mkdir(parents=True, exist_ok=False)
    source = Path(manifest["plan"]["input"]["path"])
    index = manifest_path.parent / manifest["documents"]["path"]
    source_hash, index_hash = hashlib.sha256(), hashlib.sha256()
    connection = sqlite3.connect(output / "families.sqlite")
    try:
        connection.execute(
            "CREATE TABLE documents (ordinal INTEGER PRIMARY KEY, host TEXT, family TEXT, content_sha256 TEXT, tokens INTEGER, text_offset INTEGER, json_bytes INTEGER, role TEXT)"
        )
        counts, token_offset, text_offset = Counter(), 0, 0
        with source.open("rb") as handle, index.open("rb") as spans:
            for ordinal, (raw, span_raw) in enumerate(zip(handle, spans, strict=True)):
                source_hash.update(raw)
                index_hash.update(span_raw)
                row, span = json.loads(raw), json.loads(span_raw)
                digest = hashlib.sha256(row["text"].encode()).hexdigest()
                if (
                    span["ordinal"] != ordinal
                    or span["token_start"] != token_offset
                    or span["token_count"] < 2
                    or digest != row["released_content_sha256"]
                    or digest != span["released_content_sha256"]
                    or row["content_id"] != span["content_id"]
                    or len(row["text"].encode()) != span["utf8_bytes"]
                ):
                    raise ValueError("source and document index disagree")
                host = (
                    urlsplit(row.get("url") or "").hostname or row.get("host") or "unknown"
                ).removeprefix("www.")
                connection.execute(
                    "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)",
                    (
                        ordinal,
                        host,
                        template_digest(row["text"]),
                        digest,
                        span["token_count"],
                        text_offset,
                        len(raw),
                        role_hint(row),
                    ),
                )
                counts["documents"] += 1
                counts["tokens"] += span["token_count"]
                token_offset += span["token_count"]
                text_offset += len(raw)
        if (
            source_hash.hexdigest() != manifest["plan"]["input"]["sha256"]
            or index_hash.hexdigest() != manifest["documents"]["sha256"]
            or counts["documents"] != manifest["document_count"]
            or counts["tokens"] != manifest["token_count"]
        ):
            raise ValueError("input identity or totals mismatch")
        connection.execute("CREATE INDEX by_family ON documents(host, family)")
        connection.commit()
        groups = connection.execute(
            "SELECT host, family, COUNT(*), SUM(tokens) FROM documents GROUP BY host, family HAVING COUNT(*) > 1 ORDER BY SUM(tokens) DESC, host, family"
        ).fetchall()
        roles = connection.execute(
            "SELECT role, COUNT(*), SUM(tokens) FROM documents GROUP BY role ORDER BY role"
        ).fetchall()
        # A stable sample of two examples from each of the largest 30 families.
        examples = []
        with source.open("rb") as handle:
            for host, family, size, tokens in groups[:30]:
                rows = connection.execute(
                    "SELECT ordinal,text_offset,json_bytes FROM documents WHERE host=? AND family=? ORDER BY content_sha256,ordinal LIMIT 2",
                    (host, family),
                ).fetchall()
                for ordinal, offset, length in rows:
                    handle.seek(offset)
                    row = json.loads(handle.read(length))
                    examples.append(
                        {
                            "ordinal": ordinal,
                            "host": host,
                            "family": family,
                            "family_documents": size,
                            "family_tokens": tokens,
                            "record": row,
                        }
                    )
        with (output / "examples.jsonl").open("x") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
        result = {
            "status": "census_complete_not_a_selection_policy",
            "training_authority": False,
            "manifest": {"path": str(manifest_path), "sha256": expected_sha256},
            "source": manifest["plan"]["input"],
            "counts": dict(counts),
            "method": "Group within hostname by exact SHA256 of NFKC/casefold/whitespace-normalized full text with decimal number runs replaced by <number>. No fuzzy matching or word-number replacement.",
            "limits": "Numeric variants can be distinct useful problems. Families and role hints are not quality labels, decontamination, or automatic rejection. No training data changed.",
            "repeated_families": len(groups),
            "documents_in_repeated_families": sum(row[2] for row in groups),
            "tokens_in_repeated_families": sum(row[3] for row in groups),
            "roles": [
                {"role": role, "documents": size, "tokens": tokens} for role, size, tokens in roles
            ],
            "largest_families": [
                {"host": host, "family": family, "documents": size, "tokens": tokens}
                for host, family, size, tokens in groups[:30]
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        connection.close()
    result["artifacts"] = {
        name: {"path": str(output / name), "sha256": file_sha256(output / name)}
        for name in ("families.sqlite", "examples.jsonl")
    }
    atomic_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest, args.sha256, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "counts",
                    "repeated_families",
                    "tokens_in_repeated_families",
                    "roles",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
