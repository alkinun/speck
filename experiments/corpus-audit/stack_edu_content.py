"""Measure Stack-Edu content yield at int_score 3 against 4+ with one unchanged screen.

PYTHONPATH=. python experiments/corpus-audit/stack_edu_content.py LISTING OUTPUT_DIR RECEIPT

LISTING is the census listing with local metadata paths. For every language it takes the 512
lowest SHA-256(seed:blob_id) licence-eligible rows at int_score 3 and at 4 or 5, fetches their
Software Heritage blobs, and applies the retained acquisition's per-document screen restored from
Git history (`fc54dafb^`): content identity and length, decoding, high-confidence secrets, English
prose, the 200-100,000 character envelope, raw email/IPv4, duplicate lines, benchmark matches and
Gitleaks. Both tiers pass the same screen, so the yield ratio between them is the measurement; it
scales the census projection for int_score 3. No code is executed and nothing is admitted.
"""

import argparse
import gzip
import hashlib
import ipaddress
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_stack_edu_metadata import POLICY, file_mask  # noqa: E402

from speck.data.sources.stack_v3_refine import _english_prose_result  # noqa: E402
from speck.evaluation.protocol import BenchmarkExclusion  # noqa: E402
from speck.provenance.io import atomic_json, file_sha256  # noqa: E402
from speck.tokenization.tokenizer import Tokenizer  # noqa: E402

SEED = "speck-stack-edu-yield-v1"
PER_TIER = 512
BLOBS = "https://softwareheritage.s3.amazonaws.com/content/"
TOKENIZER = Path("/mnt/speck-data/speck/tokenizer-final-mistral-v1/tokenizer.model")
GITLEAKS = Path("/mnt/speck-data/speck/tools/gitleaks/8.30.1/gitleaks")
BENCHMARKS = Path("/mnt/speck-data/speck/h100-pilot-relocation-20260918/preview/prepared.json")
# Historical screen settings (data_calibration_2b_v1 production plan, stack_v3 secret patterns).
MIN_CHARS, MAX_CHARS, MAX_DUPLICATE_LINES = 200, 100000, 0.5
ALLOWED_EMAILS, ALLOWED_IPV4 = (
    {"email@example.com", "firstname.lastname@example.org"},
    {"127.0.0.1"},
)
EMAIL = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9.-]+\.[a-z]{2,}(?![\w.-])")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
SECRETS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{30,255}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
)
COLUMNS = [
    "blob_id",
    "language",
    "repo_name",
    "path",
    "src_encoding",
    "length_bytes",
    "int_score",
    "detected_licenses",
    "license_type",
]


def tier(score):
    return "3" if score == 3 else "4+"


def select(listing):
    """Lowest-ranked licence-eligible rows per (language, tier), across a language's files."""
    policy = json.loads(POLICY.read_text())
    best = defaultdict(list)
    for item in listing["files"]:
        table = pq.read_table(item["local"], columns=COLUMNS)
        mask = file_mask(table, policy) & (table["int_score"].to_numpy() >= 3)
        rows = table.filter(mask).to_pylist()
        for row in rows:
            rank = hashlib.sha256(f"{SEED}:{row['blob_id']}".encode()).hexdigest()
            best[(row["language"], tier(row["int_score"]))].append(
                (rank, row | {"file": item["path"]})
            )
        for key in best:
            best[key] = sorted(best[key], key=lambda pair: pair[0])[:PER_TIER]
    return {key: [row for _, row in pairs] for key, pairs in best.items()}


def fetch(blob_id):
    for attempt in range(5):
        try:
            response = requests.get(BLOBS + blob_id, timeout=60)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return gzip.decompress(response.content)
        except (OSError, requests.RequestException):
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def screen(row, raw, prose_policy, exclusion):
    """Return a rejection reason or the decoded text, in the historical order."""
    if raw is None:
        return "blob_missing_404", None
    if hashlib.sha1(raw).hexdigest() != row["blob_id"]:
        return "content_hash_mismatch", None
    if len(raw) != row["length_bytes"]:
        return "content_length_metadata_mismatch", None
    try:
        text = raw.decode(row["src_encoding"], errors="strict")
    except (LookupError, UnicodeDecodeError):
        return "content_decode", None
    if text.encode("utf-8") != raw:
        return "content_non_identity_utf8", None
    if any(pattern.search(text) for pattern in SECRETS):
        return "code_high_confidence_secret", None
    if _english_prose_result(text, row["language"], prose_policy)[0] == "non_English":
        return "code_non_English_prose", None
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        return "code_character_envelope", None
    emails = {value.lower() for value in EMAIL.findall(text)} - ALLOWED_EMAILS
    ips = set()
    for value in IPV4.findall(text):
        try:
            ips.add(str(ipaddress.ip_address(value)))
        except ValueError:
            continue
    if emails or ips - ALLOWED_IPV4:
        return "raw_email_or_ipv4", None
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and 1 - len(set(lines)) / len(lines) > MAX_DUPLICATE_LINES:
        return "duplicate_lines", None
    if exclusion.matches(text):
        return "benchmark_contamination", None
    return None, text


def gitleaks(records, output):
    """Line numbers of records with Gitleaks findings, scanning them as one JSONL file."""
    source, report = output / "screened.jsonl", output / "gitleaks.json"
    source.write_text("".join(json.dumps({"text": r["text"]}) + "\n" for r in records))
    subprocess.run(
        [str(GITLEAKS), "detect", "--no-git", "--source", str(source), "--report-format", "json"]
        + ["--report-path", str(report), "--redact=100", "--exit-code", "0", "--no-banner"]
        + ["--log-level", "error"],
        check=True,
    )
    return {finding["StartLine"] for finding in json.loads(report.read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("listing", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    listing = json.loads(args.listing.read_text())
    selected = select(listing)
    targets = [row for key in sorted(selected) for row in selected[key]]
    atomic_json(args.output / "selection.json", targets)

    started = time.perf_counter()
    with ThreadPoolExecutor(32) as pool:
        blobs = list(pool.map(lambda row: fetch(row["blob_id"]), targets))
    fetch_seconds = time.perf_counter() - started

    prose_policy = json.loads(POLICY.read_text())["filters"]["English_prose"]
    exclusion = BenchmarkExclusion(json.loads(BENCHMARKS.read_text()))
    tokenizer = Tokenizer(str(TOKENIZER))
    reasons, kept = defaultdict(Counter), []
    for row, raw in zip(targets, blobs, strict=True):
        reason, text = screen(row, raw, prose_policy, exclusion)
        key = (row["language"], tier(row["int_score"]))
        if reason:
            reasons[key][reason] += 1
        else:
            kept.append(row | {"text": text})
    flagged = gitleaks(kept, args.output)
    strata = defaultdict(Counter)
    for row in targets:
        strata[(row["language"], tier(row["int_score"]))].update(
            rows=1, declared_bytes=row["length_bytes"]
        )
    for line, row in enumerate(kept, 1):
        key = (row["language"], tier(row["int_score"]))
        if line in flagged:
            reasons[key]["gitleaks"] += 1
            continue
        strata[key].update(kept_rows=1, tokens=len(tokenizer.encode(row["text"], True, True)))

    by_language = {}
    for language in sorted({language for language, _ in strata}):
        entry = {}
        for name in ("3", "4+"):
            counts = strata[(language, name)]
            entry[name] = dict(counts) | {
                "rejections": dict(reasons[(language, name)]),
                "tokens_per_declared_byte": counts["tokens"] / counts["declared_bytes"]
                if counts["declared_bytes"]
                else None,
            }
        low, high = entry["3"]["tokens_per_declared_byte"], entry["4+"]["tokens_per_declared_byte"]
        entry["yield_ratio_3_to_4plus"] = low / high if low is not None and high else None
        by_language[language] = entry
    pooled = {
        name: sum(strata[(language, name)]["tokens"] for language in by_language)
        / sum(strata[(language, name)]["declared_bytes"] for language in by_language)
        for name in ("3", "4+")
    }
    atomic_json(
        args.receipt,
        {
            "format": "speck_stack_edu_yield_probe",
            "format_version": 1,
            "status": "bounded_content_probe_not_training_admission",
            "training_admitted": False,
            "eligible_tokens_established": 0,
            "gpu_hours": 0,
            "corpus_code_executed": False,
            "seed": SEED,
            "rows_per_language_and_tier": PER_TIER,
            "listing": {"path": str(args.listing), "sha256": file_sha256(args.listing)},
            "selection": {
                "path": str(args.output / "selection.json"),
                "sha256": file_sha256(args.output / "selection.json"),
            },
            "benchmarks": {"path": str(BENCHMARKS), "sha256": file_sha256(BENCHMARKS)},
            "gitleaks": {"path": str(GITLEAKS), "sha256": file_sha256(GITLEAKS)},
            "tokenizer": {"path": str(TOKENIZER), "sha256": file_sha256(TOKENIZER)},
            "fetch_seconds": fetch_seconds,
            "pooled_tokens_per_declared_byte": pooled,
            "pooled_yield_ratio_3_to_4plus": pooled["3"] / pooled["4+"],
            "by_language": by_language,
            "boundary": (
                "Hash-ranked rows per stratum, not a census. The screen is restored from the "
                "retained acquisition, with the current frozen five-benchmark scanner in place of "
                "its historical contamination plan; comparing tiers under one screen is what makes "
                "the ratio meaningful. Origin, notice, family and near-duplicate gates are not "
                "applied, so eligible tokens stay zero."
            ),
        },
    )
    print(json.dumps({"pooled": pooled, "fetch_seconds": fetch_seconds}, indent=2))


if __name__ == "__main__":
    main()
