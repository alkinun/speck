"""Exhaustive, full-text near-copy diagnostics for a finite reviewed code cohort.

No corpus execution, automatic deletion, partition changes or training admission.
Run with a hash-bound input plan and a fresh external output path.
"""

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

from speck.provenance.io import file_sha256

TOKEN_PATTERN = r"\w+|[^\w\s]"
POLICY = {
    "token_pattern": TOKEN_PATTERN,
    "normalization": "none; case and numeric literals preserved",
    "shingle_tokens": 5,
    "minimum_shared_shingles": 50,
    "jaccard_threshold": 0.8,
    "containment_threshold": 0.8,
    "candidate_generation": "all unordered pairs; no truncation or LSH",
    "interpretation": "review leads only; no calibrated production rejection rule",
}


def shingles(text):
    tokens = re.findall(TOKEN_PATTERN, text)
    return {tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4)}


def analyze(records, assessments):
    """Preserve denominators and existing holds; report only IDs and measurements."""
    keys = [(r["cohort"], r["id"]) for r in records]
    by_key = {(r["cohort"], r["id"]): r for r in assessments}
    if len(set(keys)) != len(keys) or len(by_key) != len(assessments) or set(keys) != set(by_key):
        raise ValueError("cohort identities must be unique and match the assessment")
    rows = sorted(records, key=lambda r: (r["cohort"], r["id"]))
    sets = []
    for row in rows:
        assessment = by_key[row["cohort"], row["id"]]
        digest = hashlib.sha256(row["text"].encode()).hexdigest()
        if digest != row["consumed_sha256"] or digest != assessment["consumed_sha256"]:
            raise ValueError("consumed text identity mismatch")
        if row["tokens"] != assessment["tokens"] or row["sampling"] != assessment["sampling"]:
            raise ValueError("original token counts or sampling weights changed")
        if assessment["training_admitted"] is not False:
            raise ValueError("diagnostic requires non-admitted records")
        sets.append(shingles(row["text"]))
    matches = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        a, b = sets[i], sets[j]
        shared = len(a & b)
        union = len(a) + len(b) - shared
        jaccard = shared / union if union else 0.0
        containment = shared / min(len(a), len(b)) if a and b else 0.0
        exact = rows[i]["consumed_sha256"] == rows[j]["consumed_sha256"]
        if not exact and not (
            shared >= POLICY["minimum_shared_shingles"]
            and (
                jaccard >= POLICY["jaccard_threshold"]
                or containment >= POLICY["containment_threshold"]
            )
        ):
            continue
        endpoints = []
        for k in (i, j):
            row = rows[k]
            assessment = by_key[row["cohort"], row["id"]]
            endpoints.append(
                {
                    "cohort": row["cohort"],
                    "id": row["id"],
                    "consumed_sha256": row["consumed_sha256"],
                    "prior_held": assessment["candidate_partition"] == "quarantine",
                    "family_component_sha256": assessment["family_component_sha256"],
                }
            )
        matches.append(
            {
                "endpoints": endpoints,
                "exact": exact,
                "shared_shingles": shared,
                "jaccard": jaccard,
                "containment": containment,
            }
        )
    return {
        "format": "speck_cohort_similarity_diagnostic",
        "format_version": 1,
        "policy": POLICY,
        "records": len(rows),
        "pairs_compared": len(rows) * (len(rows) - 1) // 2,
        "prior_held_records": sum(r["candidate_partition"] == "quarantine" for r in assessments),
        "matches": matches,
        "training_admitted": False,
        "records_removed": 0,
        "limitations": [
            "Fixed review cohorts only; no population yield or complete family graph.",
            "Lexical similarity can reflect boilerplate; no semantic-copy clearance.",
            "All existing holds and partitions remain unchanged.",
        ],
    }


def run(plan_path, output):
    plan = json.loads(Path(plan_path).read_text())
    values = {}
    for name in ("records", "assessment"):
        binding = plan[name]
        path = Path(binding["path"])
        if file_sha256(path) != binding["sha256"]:
            raise ValueError(f"input checksum mismatch: {name}")
        values[name] = json.loads(path.read_text())
    result = analyze(values["records"], values["assessment"]["records"])
    result["inputs"] = plan
    result["implementation_sha256"] = file_sha256(__file__)
    with Path(output).open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    print(json.dumps({k: result[k] for k in ("records", "pairs_compared", "prior_held_records")}))
    print(f"review pairs: {len(result['matches'])}")
