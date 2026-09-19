"""Replay the bounded exclusion/family packet offline, without executing corpus code.

Run: PYTHONPATH=. python experiments/main-data/check_qualification.py INPUTS.json
Only summary counts and source identities are emitted; benchmark task text stays local.
"""

import argparse
import json
from pathlib import Path

from speck.data.code_families import partition_code_families
from speck.evaluation.protocol import BenchmarkExclusion
from speck.provenance.io import file_sha256


def verified(artifact):
    path = Path(artifact["path"])
    if file_sha256(path) != artifact["sha256"]:
        raise ValueError(f"artifact checksum mismatch: {path}")
    return path


def screen(records, benchmarks, policy):
    """Union separate benchmark lanes so translations do not suppress matching anchors."""
    counts = {r["id"]: {} for r in records}
    for benchmark in benchmarks:
        exclusion = BenchmarkExclusion({"benchmarks": [benchmark], "policy": policy})
        for row in records:
            count = len(exclusion.matches(row["text"]))
            if count:
                counts[row["id"]][benchmark["id"]] = count
    return counts


def audit(inputs):
    for artifact in inputs["evidence"]:
        verified(artifact)
    records = json.loads(verified(inputs["records"]).read_text())
    policy = json.loads(verified(inputs["rules"]).read_text())
    split = policy["family_split"]
    if [
        split[k] for k in ("train_buckets", "development_buckets", "final_buckets", "total_buckets")
    ] != [9000, 500, 500, 10000]:
        raise ValueError("unsupported family split; version the implementation before changing it")
    # Validate identities and lineage before any expensive indexing.
    partition_code_families(records, seed=policy["family_split"]["seed"])
    counts = screen(records, inputs["benchmarks"], policy["content_exclusion"])
    rows = [{**r, "benchmark_overlap": bool(counts[r["id"]])} for r in records]
    partitions = partition_code_families(
        rows,
        aliases=inputs["aliases"],
        held_repositories=inputs["held_repositories"],
        seed=policy["family_split"]["seed"],
    )
    return {
        "format_version": 1,
        "status": "bounded_screen_complete_main_admission_blocked",
        "training_admitted": False,
        "records": [{**r, "benchmark_match_counts": counts[r["id"]]} for r in partitions],
        "summary": {
            "files": len(records),
            "benchmark_lanes": len(inputs["benchmarks"]),
            "benchmark_rows": sum(b["expected_tasks"] for b in inputs["benchmarks"]),
            "held_repository_names": len(inputs["held_repositories"]),
            "content_flagged_files": sum(bool(v) for v in counts.values()),
            "quarantined_files": sum(r["candidate_partition"] == "quarantine" for r in partitions),
        },
        "remaining_gates": policy["remaining_gates"],
        "boundary": "Counts include translations of the same tasks, not independent tasks. "
        "No benchmark tasks scored. Candidate partitions do not authorize training; "
        "full-corpus family graph, coverage and eligibility are still required.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.inputs.read_text())), indent=2, sort_keys=True))
