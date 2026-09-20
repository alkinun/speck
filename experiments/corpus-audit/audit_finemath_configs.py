"""Audit bounded FineMath and InfiWebMath samples without acquiring the corpora.

The command reads fixed rows through the Hugging Face datasets server, validates the two
published schemas separately, and compares only normalized text and URL keys within the bounded
sample. It stores no corpus text and does not establish correctness, population overlap, source
rights, or training eligibility.

Run from the repository root::

    python experiments/corpus-audit/audit_finemath_configs.py --output PATH
"""

import argparse
import collections
import hashlib
import json
import re
import statistics
import unicodedata
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DATASET = "HuggingFaceTB/finemath"
SPLIT = "train"
OFFSETS_BY_CONFIG = {
    "finemath-4plus": (0, 1_000_000, 5_000_000, 6_000_000),
    "infiwebmath-3plus": (0, 1_000_000, 5_000_000, 7_000_000),
}
LENGTH = 16
CONFIGS = {
    "finemath-4plus": (
        "url",
        "fetch_time",
        "content_mime_type",
        "warc_filename",
        "warc_record_offset",
        "warc_record_length",
        "text",
        "token_count",
        "char_count",
        "metadata",
        "score",
        "int_score",
        "crawl",
        "snapshot_type",
        "language",
        "language_score",
    ),
    "infiwebmath-3plus": (
        "url",
        "metadata",
        "score",
        "int_score",
        "token_count",
        "char_count",
        "text",
    ),
}


def fetch(config, offset):
    query = {
        "dataset": DATASET,
        "config": config,
        "split": SPLIT,
        "offset": offset,
        "length": LENGTH,
    }
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(query)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    data = json.loads(payload)
    features = tuple(feature["name"] for feature in data.get("features", []))
    if features != CONFIGS[config]:
        raise ValueError(f"unexpected {config} schema at offset {offset}: {features}")
    rows = data.get("rows", [])
    if len(rows) != LENGTH:
        raise ValueError(f"{config}: expected {LENGTH} rows at {offset}, received {len(rows)}")
    if [item["row_idx"] for item in rows] != list(range(offset, offset + LENGTH)):
        raise ValueError(f"{config}: unexpected row indexes at {offset}")
    if any(item.get("truncated_cells") for item in rows):
        raise ValueError(f"{config}: truncated cells at {offset}")
    if any(
        not isinstance(item["row"].get("text"), str) or not item["row"]["text"].strip()
        for item in rows
    ):
        raise ValueError(f"{config}: missing or empty text at {offset}")
    return url, hashlib.sha256(payload).hexdigest(), rows


def normalize(value):
    value = unicodedata.normalize("NFKC", value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip().casefold()


def summarize(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"present": 0}
    return {
        "present": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def audit():
    samples = {}
    responses = []
    for config in CONFIGS:
        rows_for_config = []
        for offset in OFFSETS_BY_CONFIG[config]:
            url, response_sha256, rows = fetch(config, offset)
            responses.append(
                {
                    "config": config,
                    "offset": offset,
                    "length": LENGTH,
                    "url": url,
                    "response_sha256": response_sha256,
                }
            )
            rows_for_config.extend(item["row"] for item in rows)
        samples[config] = rows_for_config

    config_summaries = {}
    text_keys = {}
    url_keys = {}
    for config, rows in samples.items():
        hosts = collections.Counter()
        languages = collections.Counter()
        metadata_present = 0
        scores = []
        int_scores = []
        token_counts = []
        char_counts = []
        texts = set()
        urls = set()
        for row in rows:
            url = row.get("url")
            if isinstance(url, str) and url:
                parsed = urllib.parse.urlparse(url)
                hosts[parsed.hostname or "missing_host"] += 1
                urls.add(url)
            text = row.get("text")
            if isinstance(text, str):
                texts.add(hashlib.sha256(normalize(text).encode()).hexdigest())
            metadata = row.get("metadata")
            metadata_present += metadata is not None
            if isinstance(row.get("language"), str):
                languages[row["language"]] += 1
            if isinstance(row.get("score"), (int, float)):
                scores.append(row["score"])
            if isinstance(row.get("int_score"), (int, float)):
                int_scores.append(row["int_score"])
            if isinstance(row.get("token_count"), (int, float)):
                token_counts.append(row["token_count"])
            if isinstance(row.get("char_count"), (int, float)):
                char_counts.append(row["char_count"])
        text_keys[config] = texts
        url_keys[config] = urls
        config_summaries[config] = {
            "rows": len(rows),
            "url_hosts": dict(sorted(hosts.items())),
            "metadata_present": metadata_present,
            "language_counts": dict(sorted(languages.items())),
            "score": summarize(scores),
            "int_score": summarize(int_scores),
            "token_count": summarize(token_counts),
            "char_count": summarize(char_counts),
        }

    first, second = CONFIGS
    return {
        "format": "speck_finemath_config_bounded_sample",
        "format_version": 1,
        "checked_utc": datetime.now(UTC).isoformat(),
        "dataset": DATASET,
        "split": SPLIT,
        "configs": {
            config: {
                "features": list(features),
                "schema_validated": True,
            }
            for config, features in CONFIGS.items()
        },
        "sample_design": {
            "offsets": {config: list(offsets) for config, offsets in OFFSETS_BY_CONFIG.items()},
            "rows_per_offset": LENGTH,
            "rows_per_config": {
                config: len(offsets) * LENGTH for config, offsets in OFFSETS_BY_CONFIG.items()
            },
            "selection": "Fixed valid offsets within each documented train split; diagnostic sample, not a population estimate.",
        },
        "config_summaries": config_summaries,
        "bounded_overlap": {
            "normalized_text_keys": {
                "finemath-4plus": len(text_keys[first]),
                "infiwebmath-3plus": len(text_keys[second]),
            },
            "exact_normalized_text_overlap": len(text_keys[first] & text_keys[second]),
            "exact_url_overlap": len(url_keys[first] & url_keys[second]),
            "boundary": "Text uses NFKC, collapsed whitespace and case folding; URLs use literal equality. Matches cover these 128 rows only; semantic, n-gram, benchmark and population overlap are not measured.",
        },
        "responses": responses,
        "training_admitted": False,
        "corpus_code_executed": False,
        "revision_boundary": "The live dataset viewer is not revision-pinned. Response hashes identify the sampled API payloads, not a verified join to the separately pinned dataset card or retained stock.",
        "boundary": "Schema, URL hosts, metadata presence, score/length summaries and bounded exact-key overlap only. No raw text is retained; no correctness, source-use, finite-supply, contamination clearance or training admission follows.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
