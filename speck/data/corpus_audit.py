"""Reproducible content-review samples; diagnostics never authorize training or rejection."""

import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from speck.provenance.io import atomic_json, file_sha256


def length_band(size):
    for upper in (2048, 8192, 32768):
        if size <= upper:
            return f"bytes_le_{upper}"
    return "bytes_gt_32768"


def diagnostic_flags(text):
    """Review hints, including intentional markup/code; not correctness or quality scores."""
    flags = []
    lower = text.lower()
    phrases = ("accept cookies", "all rights reserved", "subscribe to", "sign up", "javascript")
    if any(phrase in lower for phrase in phrases):
        flags.append("possible_boilerplate")
    if "\ufffd" in text:
        flags.append("replacement_character")
    if re.search(r"<(?:div|span|script|style)\b", text, re.I):
        flags.append("html_markup")
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 30]
    if len(lines) >= 5 and 1 - len(set(lines)) / len(lines) >= 0.2:
        flags.append("repeated_long_lines")
    return flags


def _offer(heaps, census, source, ordinal, size, tokens, language, seed, per_stratum):
    key = f"{language or 'all'}/{length_band(size)}"
    counts = census[key]
    counts["documents"] += 1
    counts["utf8_bytes"] += size
    if tokens is not None:
        counts["tokens"] += tokens
        if tokens > 4096:
            counts["documents_over_4096_tokens"] += 1
            counts["tokens_in_documents_over_4096"] += tokens
    rank = int.from_bytes(hashlib.sha256(f"{seed}:{source}:{ordinal}".encode()).digest(), "big")
    heap = heaps[key]
    candidate = (-rank, ordinal)
    if len(heap) < per_stratum:
        heapq.heappush(heap, candidate)
    elif candidate > heap[0]:
        heapq.heapreplace(heap, candidate)


def _identity(path, expected):
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"identity mismatch: {path}")
    return {"path": str(Path(path).resolve()), "sha256": actual}


def audit_source(spec, seed, per_stratum):
    """Census immutable indices, then hash the full text while recovering selected documents."""
    heaps, census = defaultdict(list), defaultdict(Counter)
    identities = {}
    sampled_spans = {}
    indexed = spec["kind"] == "token_stock"
    expected_count = None
    if indexed:
        manifest_path = Path(spec["path"])
        identities["manifest"] = _identity(manifest_path, spec["sha256"])
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete_document_token_cache_not_training_view":
            raise ValueError("audit requires a completed stock manifest")
        source_path = Path(manifest["plan"]["input"]["path"])
        source_sha = manifest["plan"]["input"]["sha256"]
        index_path = manifest_path.parent / manifest["documents"]["path"]
        digest = hashlib.sha256()
        offset = count = 0
        with index_path.open("rb") as handle:
            for ordinal, raw in enumerate(handle):
                digest.update(raw)
                row = json.loads(raw)
                if row["ordinal"] != ordinal or row["token_start"] != offset:
                    raise ValueError("noncontiguous document index")
                if row["token_count"] < 2 or row["utf8_bytes"] < 0:
                    raise ValueError("invalid document size")
                _offer(
                    heaps,
                    census,
                    spec["id"],
                    ordinal,
                    row["utf8_bytes"],
                    row["token_count"],
                    None,
                    seed,
                    per_stratum,
                )
                offset += row["token_count"]
                count += 1
        if digest.hexdigest() != manifest["documents"]["sha256"]:
            raise ValueError("document index identity mismatch")
        if (count, offset) != (manifest["document_count"], manifest["token_count"]):
            raise ValueError("document index totals mismatch")
        identities["index"] = {"path": str(index_path), "sha256": digest.hexdigest()}
        expected_count = count
        ordinals = {ordinal for heap in heaps.values() for _, ordinal in heap}
        with index_path.open() as handle:
            for ordinal, raw in enumerate(handle):
                if ordinal in ordinals:
                    sampled_spans[ordinal] = json.loads(raw)
    elif spec["kind"] == "jsonl":
        source_path, source_sha = Path(spec["path"]), spec["sha256"]
        # Unindexed sources use byte lengths. Tokens are deliberately not estimated.
        digest = hashlib.sha256()
        with source_path.open("rb") as handle:
            for ordinal, raw in enumerate(handle):
                digest.update(raw)
                row = json.loads(raw)
                size = len(row["text"].encode())
                _offer(
                    heaps,
                    census,
                    spec["id"],
                    ordinal,
                    size,
                    None,
                    row.get("language"),
                    seed,
                    per_stratum,
                )
        if digest.hexdigest() != source_sha:
            raise ValueError("source identity mismatch")
        ordinals = {ordinal for heap in heaps.values() for _, ordinal in heap}
    else:
        raise ValueError("unsupported audit source kind")
    selected = {ordinal: key for key, heap in heaps.items() for _, ordinal in heap}
    samples, digest, count = [], hashlib.sha256(), 0
    with source_path.open("rb") as handle:
        for ordinal, raw in enumerate(handle):
            digest.update(raw)
            count += 1
            if ordinal not in ordinals:
                continue
            row = json.loads(raw)
            text_hash = hashlib.sha256(row["text"].encode()).hexdigest()
            if text_hash != row["released_content_sha256"]:
                raise ValueError("sample content hash mismatch")
            span = sampled_spans.get(ordinal)
            if span and (
                text_hash != span["released_content_sha256"]
                or row["content_id"] != span["content_id"]
                or len(row["text"].encode()) != span["utf8_bytes"]
            ):
                raise ValueError("sample and index disagree")
            samples.append(
                {
                    "sample_id": f"{spec['id']}:{ordinal}",
                    "source": spec["id"],
                    "ordinal": ordinal,
                    "stratum": selected[ordinal],
                    "token_count": span["token_count"] if span else None,
                    "flags": diagnostic_flags(row["text"]),
                    "record": row,
                }
            )
    if digest.hexdigest() != source_sha or (expected_count is not None and count != expected_count):
        raise ValueError("source identity or document count mismatch")
    if len(samples) != len(selected):
        raise ValueError("selected documents missing")
    identities["text"] = {"path": str(source_path), "sha256": digest.hexdigest()}
    strata = {}
    for key, counts in sorted(census.items()):
        group = [sample for sample in samples if sample["stratum"] == key]
        strata[key] = {
            **dict(counts),
            "sample_documents": len(group),
            "sample_flag_counts": dict(Counter(flag for s in group for flag in s["flags"])),
        }
    return {
        "id": spec["id"],
        "scope": spec["scope"],
        "identities": identities,
        "documents": count,
        "token_counts_available": indexed,
        "strata": strata,
    }, samples


def build_audit(plan, output):
    if type(plan["per_stratum"]) is not int or not 1 <= plan["per_stratum"] <= 1000:
        raise ValueError("per_stratum must be an integer in [1, 1000]")
    names = [source["id"] for source in plan["sources"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate source IDs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    results, total = [], 0
    with (output / "samples.jsonl").open("x") as handle:
        for spec in plan["sources"]:
            result, samples = audit_source(spec, plan["seed"], plan["per_stratum"])
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            results.append(result)
            total += len(samples)
            print(
                f"Audited {spec['id']}: {result['documents']:,} documents; {len(samples)} samples",
                flush=True,
            )
    result = {
        "format": "speck_corpus_review_sample",
        "format_version": 1,
        "status": "sampled_not_quality_approved",
        "training_authority": False,
        "method": "Lowest SHA256(seed:source:ordinal) ranks per language/UTF-8-byte-length stratum. Equal stratum quotas are not corpus proportions. Census token totals include BOS/EOS. Hashes verify text and indices, not token shards.",
        "limitations": "Flags are review hints, not rejection decisions or correctness scores. No labels, population defect estimates, joint exclusions, or model-quality comparison are implied. Source scope is explicit, including candidate-only code.",
        "sources": results,
        "sample_documents": total,
        "samples": {
            "path": str(output / "samples.jsonl"),
            "sha256": file_sha256(output / "samples.jsonl"),
        },
    }
    atomic_json(output / "report.json", result)
    return result
