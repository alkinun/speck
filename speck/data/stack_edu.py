"""Fetch and screen Stack-Edu content: a yield probe and resumable bulk acquisition.

python -m scripts.stack_edu probe LISTING OUTPUT_DIR RECEIPT
python -m scripts.stack_edu acquire LISTING CENSUS LANGUAGE TIER OUTPUT_DIR
python -m scripts.stack_edu verify OUTPUT_DIR [--repair]
python -m scripts.stack_edu summarize OUTPUT_DIR RECEIPT
python -m scripts.stack_edu convert RETAINED OUTPUT_DIR BASE_PLAN OUT

LISTING is the census listing with local metadata paths. Both commands fetch Software Heritage
blobs for licence-eligible rows and apply the retained acquisition's per-document screen restored
from Git history (`fc54dafb^`): content identity and length, decoding, high-confidence secrets,
English prose, the 200-100,000 character envelope, raw email/IPv4, duplicate lines, benchmark
matches and Gitleaks. `probe` takes the 512 lowest SHA-256(seed:blob_id) rows per language at
int_score 3 and at 4 or 5; both tiers pass the same screen, so their yield ratio is the
measurement. `acquire` takes every row of one language and tier in physical listing order, in
4,096-row units that each publish a record file and manifest, so a rerun resumes at the first
missing unit. Tier 4+ skips the rows the retained acquisition already consumed. No code is
executed and nothing is admitted. `verify` re-derives every completed unit from its stored records
(each text's SHA-1 and length against its blob identity, and the unit's kept-row and token totals);
`--repair` deletes failed units so the next `acquire` refetches them. `summarize` records every
completed tranche in one receipt.
`convert` writes the retained stock (RETAINED is its acquisition receipt) and every completed
tranche as one input for `scripts.production_data_preprocess`, verifying each archive and unit
file first, with a plan that keeps the base plan's firewall references and policy. Each record
carries its Stack-Edu `int_score` and `tier`. Acquired rows keep the listing's int_score. The
retained stock was selected by the historical predicate, whose score floor is 4, so its tier is
4+; its int_score is the one its record's metadata carries, or null when the record has none.
"""

import argparse
import ast
import gzip
import hashlib
import io
import ipaddress
import json
import re
import subprocess
import tarfile
import tempfile
import time
import tokenize
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from speck.data.acquisition import _py3langid_identifier
from speck.data.stack_edu_census import POLICY, file_mask
from speck.evaluation.protocol import BenchmarkExclusion
from speck.provenance.io import atomic_json, file_sha256, fsync_path
from speck.tokenization.tokenizer import Tokenizer

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


_C_STYLE_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_SQL_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


def _python_prose(text):
    values = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                values.append(token.string.lstrip("#"))
    except (IndentationError, SyntaxError, tokenize.TokenError):
        pass
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return "\n".join(values)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            value = ast.get_docstring(node, clean=False)
            if value:
                values.append(value)
    return "\n".join(values)


def _extract_prose(text, language):
    if language == "Markdown":
        return text
    if language == "Python":
        return _python_prose(text)
    if language == "Shell":
        return "\n".join(
            line.lstrip()[1:]
            for line in text.splitlines()
            if line.lstrip().startswith("#") and not line.lstrip().startswith("#!")
        )
    if language == "SQL":
        return "\n".join(_SQL_COMMENTS.findall(text))
    return "\n".join(_C_STYLE_COMMENTS.findall(text))


def english_prose_result(text, language, settings):
    """Classify a file's comments and docstrings (all of Markdown) as English or not."""
    prose = _extract_prose(text, language)
    alphabetic = sum(character.isalpha() for character in prose)
    if alphabetic < settings["minimum_alphabetic_characters"]:
        return "insufficient_prose", None
    detected, probability = _py3langid_identifier().classify(prose)
    probability = float(probability)
    if detected == "en" and probability >= settings["minimum_probability"]:
        return "English", probability
    return "non_English", probability


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
    if english_prose_result(text, row["language"], prose_policy)[0] == "non_English":
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


def gitleaks(texts):
    """Indexes of texts with Gitleaks findings, scanning them as one JSONL file in TMPDIR."""
    with tempfile.TemporaryDirectory() as scratch:
        source, report = Path(scratch, "screened.jsonl"), Path(scratch, "gitleaks.json")
        source.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
        subprocess.run(
            [str(GITLEAKS), "detect", "--no-git", "--source", str(source)]
            + ["--report-format", "json", "--report-path", str(report), "--redact=100"]
            + ["--exit-code", "0", "--no-banner", "--log-level", "error"],
            check=True,
        )
        return {finding["StartLine"] - 1 for finding in json.loads(report.read_text())}


class Screen:
    """The per-document screen with its settings, benchmarks and tokenizer loaded once."""

    def __init__(self, workers):
        self.workers = workers
        self.prose = json.loads(POLICY.read_text())["filters"]["English_prose"]
        self.exclusion = BenchmarkExclusion(json.loads(BENCHMARKS.read_text()))
        self.tokenizer = Tokenizer(str(TOKENIZER))

    def run(self, rows):
        """Return (reason, text, tokens) per row in order; rejected rows carry no text."""
        with ThreadPoolExecutor(self.workers) as pool:
            blobs = list(pool.map(lambda row: fetch(row["blob_id"]), rows))
        results = [
            screen(row, raw, self.prose, self.exclusion)
            for row, raw in zip(rows, blobs, strict=True)
        ]
        kept = [index for index, (reason, _) in enumerate(results) if reason is None]
        flagged = gitleaks([results[index][1] for index in kept])
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
    outcomes = Screen(32).run(targets)
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
            outcomes = screen_.run(unit)
            records = directory / manifest.name.replace(".json", ".jsonl.gz")
            reasons, tokens = Counter(), 0
            with gzip.open(records, "wt") as handle:
                for row, (reason, text, count) in zip(unit, outcomes, strict=True):
                    if reason:
                        reasons[reason] += 1
                        continue
                    tokens += count
                    handle.write(json.dumps(row | {"text": text, "tokens": count}) + "\n")
            # The manifest marks the unit complete, so its records must be on disk first.
            fsync_path(records)
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
                fsync=True,
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


def summarize(output, receipt):
    """Record completed tranches; a tranche without its receipt is still running and omitted."""
    tranches, totals = [], defaultdict(Counter)
    for path in sorted(output.glob("*/tranche.json")):
        record = json.loads(path.read_text())
        tranches.append(
            {
                "language": record["language"],
                "tier": record["tier"],
                "units": record["units"],
                "receipt": identity(path),
            }
            | {key: record["totals"].get(key, 0) for key in ("rows", "kept_rows", "tokens")}
        )
        totals[record["tier"]].update(
            {key: record["totals"].get(key, 0) for key in ("rows", "kept_rows", "tokens")}
        )
    atomic_json(
        receipt,
        {
            "format": "speck_stack_edu_acquisition",
            "format_version": 1,
            "status": "screened_candidate_stock_not_training_admission",
            "training_admitted": False,
            "eligible_tokens_established": 0,
            "gpu_hours": 0,
            "corpus_code_executed": False,
            "output_directory": str(output),
            "tranches": tranches,
            "totals_by_tier": {tier: dict(counts) for tier, counts in sorted(totals.items())},
            "tokens_before_full_exclusion": sum(counts["tokens"] for counts in totals.values()),
            "boundary": (
                "Completed tranches only, screened like the retained stock and disjoint from it by "
                "row; tokens include BOS/EOS. Origin and notice recovery, family partition, "
                "near-duplicate and full exclusion remain open, so eligible tokens stay zero."
            ),
        },
    )


def _verify_unit(manifest_path):
    """Return the problems of one completed unit, re-derived from its stored records."""
    manifest = json.loads(Path(manifest_path).read_text())
    records = Path(manifest["records"]["path"])
    if not records.is_file() or file_sha256(records) != manifest["records"]["sha256"]:
        return [f"{manifest_path}: records file changed or missing"]
    problems, kept, tokens = [], 0, 0
    with gzip.open(records, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            raw = row["text"].encode("utf-8")
            if hashlib.sha1(raw).hexdigest() != row["blob_id"] or len(raw) != row["length_bytes"]:
                problems.append(f"{manifest_path}: record {row['blob_id']} differs from its blob")
            kept += 1
            tokens += row["tokens"]
    if (kept, tokens) != (manifest["kept_rows"], manifest["tokens"]):
        problems.append(f"{manifest_path}: kept rows or tokens differ from the manifest")
    return problems


def verify(output, repair=False, workers=8):
    """Re-verify every completed unit by content; with repair, delete failed units to refetch."""
    manifests = sorted(Path(output).glob("*/unit-*.json"))
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(_verify_unit, manifests))
    failed = [path for path, problems in zip(manifests, results, strict=True) if problems]
    if repair:
        for path in failed:
            for stale in (path.with_suffix(".jsonl.gz"), path.parent / "tranche.json", path):
                stale.unlink(missing_ok=True)
    return {
        "units": len(manifests),
        "failed_units": len(failed),
        "problems": [problem for problems in results for problem in problems],
        "repaired": repair,
    }


def _code_record(
    text, content_id, repository, path, language, source_file, source_row, origin, int_score, name
):
    return {
        "text": text,
        "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "content_id": content_id,
        "url": None,
        "host": None,
        "repository": repository,
        "path": path,
        "language": language,
        "source_file": source_file,
        "source_row": int(source_row),
        "origin": origin,
        "int_score": int_score,
        "tier": name,
    }


def _retained_score(row):
    """The retained record's int_score, or None when its metadata does not carry one.

    The historical predicate admitted int_score 4 and above only, so a lower score contradicts
    the stock's provenance and is rejected rather than recorded.
    """
    metadata = row.get("metadata")
    score = metadata.get("int_score") if isinstance(metadata, dict) else None
    if score is not None and (isinstance(score, bool) or not isinstance(score, int) or score < 4):
        raise ValueError(f"retained record {row['content_id']} has int_score {score!r}")
    return score


def _retained_records(receipt):
    """Records of the retained acquisition, unit by unit, from verified archives."""
    for unit in json.loads(receipt.read_text())["units"]:
        tar, manifest = unit["archive"]["tar"], unit["manifest"]
        if file_sha256(tar["path"]) != tar["sha256"]:
            raise ValueError(f"retained archive changed: {tar['path']}")
        with tarfile.open(tar["path"]) as archive:
            data = archive.extractfile("unit/" + manifest["output"]["path"]).read()
        if hashlib.sha256(data).hexdigest() != manifest["output"]["sha256"]:
            raise ValueError(f"retained records changed: {tar['path']}")
        for line in data.splitlines():
            row = json.loads(line)
            yield _code_record(
                row["text"],
                row["content_id"],
                row["repo_path"],
                row["file_path"],
                row["language"],
                row["source_file"],
                row["source_row"],
                "retained",
                _retained_score(row),
                "4+",
            )


def _acquired_records(output):
    """Records of every completed tranche, unit by unit, from verified unit files."""
    for tranche_path in sorted(output.glob("*/tranche.json")):
        directory = tranche_path.parent
        for index in range(json.loads(tranche_path.read_text())["units"]):
            manifest = json.loads((directory / f"unit-{index:05d}.json").read_text())
            records = Path(manifest["records"]["path"])
            if file_sha256(records) != manifest["records"]["sha256"]:
                raise ValueError(f"acquired records changed: {records}")
            with gzip.open(records, "rt") as handle:
                for line in handle:
                    row = json.loads(line)
                    yield _code_record(
                        row["text"],
                        row["blob_id"],
                        row["repo_name"],
                        row["path"],
                        row["language"],
                        row["file"],
                        row["source_row"],
                        "acquired",
                        row["int_score"],
                        tier(row["int_score"]),
                    )


def convert(retained, output, base_plan, destination):
    destination.mkdir(parents=True, exist_ok=False)
    records = destination / "input.jsonl"
    counts = Counter()
    with records.open("w") as handle:
        for source in (_retained_records(retained), _acquired_records(output)):
            for row in source:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["origin"]] += 1
    plan = json.loads(base_plan.read_text())
    plan["sources"] = [s for s in plan["sources"] if s["id"].startswith("firewall_reference__")]
    plan["sources"].append(
        {
            "id": "acquired_train__code",
            "precedence": len(plan["sources"]) + 1,
            "path": str(records),
            "sha256": file_sha256(records),
            "text_field": "text",
            "content_sha256_field": "released_content_sha256",
            "url_field": None,
            "domain_field": None,
            "blob_field": "content_id",
        }
    )
    plan["output_directory"] = str(destination / "excluded")
    plan["cleanup_files"] = []
    atomic_json(destination / "preprocess-plan.json", plan)
    atomic_json(
        destination / "conversion.json",
        {
            "format": "speck_stack_edu_conversion",
            "format_version": 1,
            "training_admitted": False,
            "retained": identity(retained),
            "tranche_receipts": [identity(path) for path in sorted(output.glob("*/tranche.json"))],
            "base_plan": identity(base_plan),
            "input": {"path": str(records), "sha256": plan["sources"][-1]["sha256"]},
            "documents": dict(counts),
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
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("output", type=Path)
    verify_parser.add_argument("--repair", action="store_true")
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("output", type=Path)
    summary_parser.add_argument("receipt", type=Path)
    convert_parser = commands.add_parser("convert")
    for name in ("retained", "output", "base_plan", "destination"):
        convert_parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "probe":
        probe(args.listing, args.output, args.receipt)
    elif args.command == "acquire":
        acquire(args.listing, args.census, args.language, args.tier, args.output)
    elif args.command == "verify":
        print(json.dumps(verify(args.output, args.repair), indent=2))
    elif args.command == "summarize":
        summarize(args.output, args.receipt)
    else:
        convert(args.retained, args.output, args.base_plan, args.destination)


if __name__ == "__main__":
    main()
