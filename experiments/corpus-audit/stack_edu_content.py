"""Fetch and screen Stack-Edu content: a yield probe and resumable bulk acquisition.

PYTHONPATH=. python experiments/corpus-audit/stack_edu_content.py probe LISTING OUTPUT_DIR RECEIPT
PYTHONPATH=. python experiments/corpus-audit/stack_edu_content.py acquire LISTING CENSUS LANGUAGE TIER OUTPUT_DIR

LISTING is the census listing with local metadata paths. Both commands fetch Software Heritage
blobs for licence-eligible rows and apply the retained acquisition's per-document screen restored
from Git history (`fc54dafb^`): content identity and length, decoding, high-confidence secrets,
English prose, the 200-100,000 character envelope, raw email/IPv4, duplicate lines, benchmark
matches and Gitleaks. `probe` takes the 512 lowest SHA-256(seed:blob_id) rows per language at
int_score 3 and at 4 or 5; both tiers pass the same screen, so their yield ratio is the
measurement. `acquire` takes every row of one language and tier in physical listing order, in
4,096-row units that each publish a record file and manifest, so a rerun resumes at the first
missing unit. Tier 4+ skips the rows the retained acquisition already consumed. No code is
executed and nothing is admitted.
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

import numpy as np
import pyarrow as pa
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
UNIT_ROWS = 4096
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


def gitleaks(texts, output):
    """Indexes of texts with Gitleaks findings, scanning them as one JSONL file."""
    source, report = output / "screened.jsonl", output / "gitleaks.json"
    source.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
    subprocess.run(
        [str(GITLEAKS), "detect", "--no-git", "--source", str(source), "--report-format", "json"]
        + ["--report-path", str(report), "--redact=100", "--exit-code", "0", "--no-banner"]
        + ["--log-level", "error"],
        check=True,
    )
    findings = {finding["StartLine"] - 1 for finding in json.loads(report.read_text())}
    source.unlink()
    return findings


class Screen:
    """The per-document screen with its settings, benchmarks and tokenizer loaded once."""

    def __init__(self, workers):
        self.workers = workers
        self.prose = json.loads(POLICY.read_text())["filters"]["English_prose"]
        self.exclusion = BenchmarkExclusion(json.loads(BENCHMARKS.read_text()))
        self.tokenizer = Tokenizer(str(TOKENIZER))

    def run(self, rows, workdir):
        """Return (reason, text, tokens) per row in order; rejected rows carry no text."""
        with ThreadPoolExecutor(self.workers) as pool:
            blobs = list(pool.map(lambda row: fetch(row["blob_id"]), rows))
        results = [
            screen(row, raw, self.prose, self.exclusion)
            for row, raw in zip(rows, blobs, strict=True)
        ]
        kept = [index for index, (reason, _) in enumerate(results) if reason is None]
        flagged = gitleaks([results[index][1] for index in kept], workdir)
        outcomes = [(reason, None, 0) for reason, _ in results]
        for position, index in enumerate(kept):
            text = results[index][1]
            outcomes[index] = (
                ("gitleaks", None, 0)
                if position in flagged
                else (None, text, len(self.tokenizer.encode(text, True, True)))
            )
        return outcomes


def identity(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def probe(listing_path, output, receipt):
    output.mkdir(parents=True, exist_ok=False)
    targets = [
        row
        for key, rows in sorted(select(json.loads(listing_path.read_text())).items())
        for row in rows
    ]
    atomic_json(output / "selection.json", targets)
    started = time.perf_counter()
    outcomes = Screen(32).run(targets, output)
    seconds = time.perf_counter() - started
    strata, reasons = defaultdict(Counter), defaultdict(Counter)
    for row, (reason, _, tokens) in zip(targets, outcomes, strict=True):
        key = (row["language"], tier(row["int_score"]))
        strata[key].update(rows=1, declared_bytes=row["length_bytes"])
        if reason:
            reasons[key][reason] += 1
        else:
            strata[key].update(kept_rows=1, tokens=tokens)

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
        receipt,
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
            "listing": identity(listing_path),
            "selection": identity(output / "selection.json"),
            "benchmarks": identity(BENCHMARKS),
            "gitleaks": identity(GITLEAKS),
            "tokenizer": identity(TOKENIZER),
            "fetch_seconds": seconds,
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
    print(json.dumps({"pooled": pooled, "fetch_seconds": seconds}, indent=2))


def tranche(listing, census, language, name):
    """Every licence-eligible row of one language and tier, in physical listing order.

    Tier 4+ is exactly the historical predicate's eligible set, whose first rows the retained
    acquisition consumed in this same order; those are skipped so nothing is fetched twice.
    """
    policy = json.loads(POLICY.read_text())
    parts = []
    for item in listing["files"]:
        if (
            pq.ParquetFile(item["local"])
            .read_row_group(0, columns=["language"])["language"][0]
            .as_py()
            != language
        ):
            continue
        table = pq.read_table(item["local"], columns=COLUMNS)
        scores = table["int_score"].to_numpy()
        mask = file_mask(table, policy) & ((scores == 3) if name == "3" else (scores >= 4))
        rows = np.nonzero(mask)[0]
        parts.append(
            table.take(rows)
            .append_column("file", pa.array([item["path"]] * len(rows)))
            .append_column("source_row", pa.array(rows))
        )
    if not parts:
        raise ValueError(f"no metadata files for {language}")
    rows = pa.concat_tables(parts)
    if name == "4+":
        history = census["by_language"][language].get("historical_acquisition")
        rows = rows.slice(history["consumed_eligible_rows"] if history else 0)
    return rows


def acquire(listing_path, census_path, language, name, output, workers=128):
    if name not in ("3", "4+"):
        raise ValueError("tier must be 3 or 4+")
    rows = tranche(
        json.loads(listing_path.read_text()), json.loads(census_path.read_text()), language, name
    )
    directory = output / f"{language}-{name}".replace("+", "plus").replace("#", "sharp")
    directory.mkdir(parents=True, exist_ok=True)
    screen_ = None
    units = []
    for start in range(0, rows.num_rows, UNIT_ROWS):
        manifest = directory / f"unit-{start // UNIT_ROWS:05d}.json"
        if not manifest.exists():
            screen_ = screen_ or Screen(workers)
            unit = rows.slice(start, UNIT_ROWS).to_pylist()
            started = time.perf_counter()
            outcomes = screen_.run(unit, directory)
            records = directory / manifest.name.replace(".json", ".jsonl.gz")
            reasons, tokens = Counter(), 0
            with gzip.open(records, "wt") as handle:
                for row, (reason, text, count) in zip(unit, outcomes, strict=True):
                    if reason:
                        reasons[reason] += 1
                        continue
                    tokens += count
                    handle.write(json.dumps(row | {"text": text, "tokens": count}) + "\n")
            atomic_json(
                manifest,
                {
                    "rows": len(unit),
                    "first_row": start,
                    "declared_bytes": sum(row["length_bytes"] for row in unit),
                    "kept_rows": len(unit) - sum(reasons.values()),
                    "tokens": tokens,
                    "rejections": dict(reasons),
                    "records": identity(records),
                    "seconds": time.perf_counter() - started,
                },
            )
            print(f"{directory.name} unit {start // UNIT_ROWS} tokens {tokens:,}", flush=True)
        units.append(json.loads(manifest.read_text()))
    totals = Counter()
    for unit in units:
        totals.update(
            {key: unit[key] for key in ("rows", "declared_bytes", "kept_rows", "tokens", "seconds")}
        )
    atomic_json(
        directory / "tranche.json",
        {
            "format": "speck_stack_edu_tranche",
            "format_version": 1,
            "training_admitted": False,
            "eligible_tokens_established": 0,
            "language": language,
            "tier": name,
            "listing": identity(listing_path),
            "census": identity(census_path),
            "units": len(units),
            "totals": dict(totals),
            "rejections": dict(sum((Counter(unit["rejections"]) for unit in units), Counter())),
            "tokens_per_declared_byte": totals["tokens"] / totals["declared_bytes"]
            if totals["declared_bytes"]
            else None,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe_parser = commands.add_parser("probe")
    for name in ("listing", "output", "receipt"):
        probe_parser.add_argument(name, type=Path)
    acquire_parser = commands.add_parser("acquire")
    acquire_parser.add_argument("listing", type=Path)
    acquire_parser.add_argument("census", type=Path)
    acquire_parser.add_argument("language")
    acquire_parser.add_argument("tier", choices=("3", "4+"))
    acquire_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "probe":
        probe(args.listing, args.output, args.receipt)
    else:
        acquire(args.listing, args.census, args.language, args.tier, args.output)


if __name__ == "__main__":
    main()
