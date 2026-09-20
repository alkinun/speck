"""Audit a bounded OpenMathInstruct-2 sample without acquiring the corpus wholesale.

The command reads fixed rows through the Hugging Face datasets server, reports schema/source and
length diagnostics, and compares a final boxed answer with the card-provided expected answer when
both are present. That comparison is self-consistency only; it is not independent correctness or
contamination certification.

Run from the repository root::

    python experiments/corpus-audit/audit_openmath_sample.py --output PATH
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

DATASET = "nvidia/OpenMathInstruct-2"
CONFIG = "default"
SPLIT = "train"
OFFSETS = (0, 1_000_000, 5_000_000, 10_000_000, 13_000_000)
LENGTH = 16
EXPECTED_FEATURES = ("problem", "generated_solution", "expected_answer", "problem_source")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def fetch(offset):
    query = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": LENGTH,
    }
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(query)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    data = json.loads(payload)
    features = tuple(feature["name"] for feature in data.get("features", []))
    if features != EXPECTED_FEATURES:
        raise ValueError(f"unexpected feature schema at offset {offset}: {features}")
    rows = data.get("rows", [])
    if len(rows) != LENGTH:
        raise ValueError(f"expected {LENGTH} rows at offset {offset}, received {len(rows)}")
    return url, hashlib.sha256(payload).hexdigest(), rows


def normalize(value):
    value = unicodedata.normalize("NFKC", value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip().casefold()


def audit(gsm8k_path=None):
    source_counts = collections.Counter()
    problem_lengths = []
    solution_lengths = []
    expected_empty = 0
    boxed_count = 0
    boxed_expected_matches = 0
    responses = []
    sample_rows = []

    for offset in OFFSETS:
        url, response_sha256, rows = fetch(offset)
        responses.append(
            {
                "offset": offset,
                "length": LENGTH,
                "url": url,
                "response_sha256": response_sha256,
            }
        )
        sample_rows.extend(item["row"] for item in rows)
        for item in rows:
            row = item["row"]
            source_counts[row["problem_source"]] += 1
            problem_lengths.append(len(row["problem"]))
            solution_lengths.append(len(row["generated_solution"]))
            expected = row["expected_answer"].strip()
            expected_empty += not bool(expected)
            matches = BOXED.findall(row["generated_solution"])
            boxed_count += bool(matches)
            boxed_expected_matches += bool(matches and expected and matches[-1].strip() == expected)

    def length_summary(values):
        return {
            "min_chars": min(values),
            "median_chars": statistics.median(values),
            "max_chars": max(values),
        }

    contamination = {
        "status": "not_run",
        "boundary": "No benchmark comparison was requested.",
    }
    if gsm8k_path:
        import pyarrow.parquet as parquet

        benchmark = Path(gsm8k_path)
        benchmark_questions = parquet.read_table(benchmark, columns=["question"])[
            "question"
        ].to_pylist()
        benchmark_keys = {normalize(question) for question in benchmark_questions}
        matches = sum(normalize(row["problem"]) in benchmark_keys for row in sample_rows)
        contamination = {
            "benchmark": "gsm8k",
            "benchmark_path": str(benchmark),
            "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
            "benchmark_rows": len(benchmark_questions),
            "exact_normalized_problem_matches": matches,
            "sample_rows_with_gsm8k_source_label": sum(
                row["problem_source"] in {"gsm8k", "augmented_gsm8k"} for row in sample_rows
            ),
            "status": "exact_match_diagnostic_only",
            "boundary": "Exact normalized problem matching against GSM8K only. MATH, derived variants, semantic overlap and the full candidate are not covered; zero matches does not establish contamination clearance.",
        }

    return {
        "format": "speck_openmathinstruct2_bounded_sample",
        "format_version": 1,
        "checked_utc": datetime.now(UTC).isoformat(),
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "sample_design": {
            "offsets": list(OFFSETS),
            "rows_per_offset": LENGTH,
            "rows": len(OFFSETS) * LENGTH,
            "selection": "Fixed offsets across the documented train range; diagnostic sample, not a population estimate.",
        },
        "schema": {"features": list(EXPECTED_FEATURES), "schema_validated": True},
        "source_counts": dict(sorted(source_counts.items())),
        "lengths": {
            "problem": length_summary(problem_lengths),
            "generated_solution": length_summary(solution_lengths),
        },
        "answer_consistency": {
            "rows_with_empty_expected_answer": expected_empty,
            "rows_with_boxed_solution": boxed_count,
            "boxed_solution_equals_expected": boxed_expected_matches,
            "independent_correctness_established": False,
        },
        "contamination": contamination,
        "responses": responses,
        "training_admitted": False,
        "corpus_code_executed": False,
        "boundary": "Schema, source labels, character lengths and boxed-answer self-consistency only. No independent oracle, contamination clearance, source-use decision, finite-supply estimate or training admission.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gsm8k-path", type=Path)
    args = parser.parse_args()
    result = audit(args.gsm8k_path)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
