"""Reconstruct and classify every reference behind a frozen contamination result."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.contamination_audit import check_report, file_sha256, find_subsequence
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def repository_revision():
    return subprocess.run(
        ["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def spec_identity(protocol):
    payload = {key: protocol[key] for key in protocol if key not in {"status", "result"}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def probe_id(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<u2").tobytes()).hexdigest()


def case_probe_references(row, tokens, n, fractions, reference, tokenizer):
    """Yield the exact probe/reference pairs constructed by contamination audit v1."""
    if n <= len(tokens) <= 4096:
        yield tokens, {**reference, "slot": "full"}
    if len(tokens) >= n:
        maximum_start = len(tokens) - n
        for fraction in fractions:
            start = min(maximum_start, round(maximum_start * fraction))
            yield tokens[start : start + n], {
                **reference,
                "slot": f"context_{fraction}",
            }
    for answer_number, answer in enumerate(row["expected_answer"]):
        answer_tokens = tokenizer.encode(answer, bos=False, eos=False)
        for occurrence, location in enumerate(find_subsequence(tokens, answer_tokens)):
            start = max(0, location - (n - len(answer_tokens)) // 2)
            start = min(start, max(0, len(tokens) - n))
            probe = tokens[start : start + n]
            if len(probe) == n:
                yield probe, {
                    **reference,
                    "slot": f"answer_{answer_number}_{occurrence}",
                }

def reconstruct_references(protocol, audit, repository_root, tokenizer):
    patterns = {pattern["id"]: pattern for pattern in audit["matches"]["patterns"]}
    references = defaultdict(list)
    cases = Counter()
    contamination_protocol = json.loads(
        (repository_root / protocol["audit"]["protocol_path"]).read_text(encoding="utf-8")
    )
    ruler = next(
        entry for entry in contamination_protocol["evaluations"] if entry["suite"] == "ruler"
    )
    n = contamination_protocol["probes"]["token_ngram_length"]
    fractions = contamination_protocol["probes"]["context_fractions"]
    for length_report in ruler["length_reports"]:
        report_path = repository_root / length_report["path"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        case_root = Path(report["local_output_dir"]) / "cases"
        for task in report["tasks"]:
            with (case_root / task / "test.jsonl").open(encoding="utf-8") as handle:
                for row_number, line in enumerate(handle):
                    row = json.loads(line)
                    tokens = tokenizer.encode(row["question"], bos=False, eos=False)
                    reference = {
                        "suite": "ruler",
                        "length": length_report["length"],
                        "task": task,
                        "row": row_number,
                    }
                    cases[(length_report["length"], task)] += 1
                    for probe, probe_reference in case_probe_references(
                        row, tokens, n, fractions, reference, tokenizer
                    ):
                        identifier = probe_id(probe)
                        if identifier in patterns:
                            references[identifier].append(probe_reference)
    unresolved = sorted(set(patterns) - set(references))
    if unresolved:
        raise ValueError(f"matched contamination patterns could not be reconstructed: {unresolved}")
    for identifier, pattern in patterns.items():
        if len(references[identifier]) != pattern["reference_count"]:
            raise ValueError(
                f"contamination reference count changed for {identifier}: "
                f"{len(references[identifier])} != {pattern['reference_count']}"
            )
    return patterns, references, cases


def classify(protocol, audit, patterns, references, cases):
    critical_kinds = set(protocol["decision"]["critical_probe_kinds"])
    expected_tasks = set(protocol["scope"]["expected_tasks"])
    observed_tasks = {task for _, task in cases}
    if observed_tasks != expected_tasks:
        raise ValueError("RULER task inventory changed while classifying contamination")
    task_patterns = defaultdict(lambda: defaultdict(set))
    task_references = defaultdict(Counter)
    matched_cases = defaultdict(set)
    matched_cells = defaultdict(set)
    pattern_summaries = []
    for identifier, pattern in sorted(patterns.items()):
        kind = pattern["kind"]
        pattern_tasks = Counter(reference["task"] for reference in references[identifier])
        pattern_lengths = Counter(reference["length"] for reference in references[identifier])
        pattern_slots = Counter(
            reference["slot"].split("_", 1)[0] for reference in references[identifier]
        )
        for reference in references[identifier]:
            task = reference["task"]
            task_patterns[task][kind].add(identifier)
            task_references[task][kind] += 1
            matched_cases[kind].add((reference["length"], task, reference["row"]))
            matched_cells[kind].add((reference["length"], task))
        pattern_summaries.append(
            {
                "id": identifier,
                "audit_kind": kind,
                "reference_count": len(references[identifier]),
                "references_by_task": dict(sorted(pattern_tasks.items())),
                "references_by_length": {
                    str(length): count for length, count in sorted(pattern_lengths.items())
                },
                "references_by_slot_family": dict(sorted(pattern_slots.items())),
            }
        )
    quarantine_tasks = sorted(
        task
        for task in expected_tasks
        if any(task_patterns[task].get(kind) for kind in critical_kinds)
    )
    no_critical_match_tasks = sorted(expected_tasks - set(quarantine_tasks))
    context_tasks = sorted(task for task in expected_tasks if task_patterns[task].get("context"))
    task_summary = []
    for task in sorted(expected_tasks):
        task_summary.append(
            {
                "task": task,
                "cases": sum(count for (length, name), count in cases.items() if name == task),
                "critical_disposition": (
                    "quarantined" if task in quarantine_tasks else "no_detected_critical_match"
                ),
                "unique_matched_patterns_by_kind": {
                    kind: len(values) for kind, values in sorted(task_patterns[task].items())
                },
                "matched_probe_references_by_kind": dict(sorted(task_references[task].items())),
            }
        )
    source_counts = Counter()
    for pattern in audit["matches"]["patterns"]:
        for location in pattern["locations"]:
            source_counts[(pattern["kind"], location["source"])] += 1
    return {
        "all_matched_references_reconstructed": True,
        "matched_pattern_count": len(patterns),
        "matched_patterns_by_kind": audit["matches"]["unique_patterns_by_kind"],
        "matched_training_occurrences_by_kind": audit["matches"]["occurrences_by_kind"],
        "matched_stored_locations_by_kind_and_source": [
            {"kind": kind, "source": source, "locations": count}
            for (kind, source), count in sorted(source_counts.items())
        ],
        "critical_quarantine_tasks": quarantine_tasks,
        "no_detected_critical_match_tasks": no_critical_match_tasks,
        "context_overlap_tasks": context_tasks,
        "critical_affected_cases": len(
            set().union(*(matched_cases.get(kind, set()) for kind in critical_kinds))
        ),
        "critical_affected_cells": [
            {"length": length, "task": task}
            for length, task in sorted(
                set().union(*(matched_cells.get(kind, set()) for kind in critical_kinds))
            )
        ],
        "task_summary": task_summary,
        "pattern_reference_summary": pattern_summaries,
    }


def check_disposition(protocol, output, repository_root):
    result = json.loads(output.expanduser().resolve().read_text(encoding="utf-8"))
    if (
        result.get("format") != "speck_contamination_disposition"
        or result.get("format_version") != 1
        or result.get("status") != protocol["decision"]["failed_status"]
        or result.get("protocol", {}).get("spec_identity_sha256") != spec_identity(protocol)
        or result.get("audit", {}).get("sha256") != protocol["audit"]["sha256"]
        or result.get("decision", {}).get("ruler_v1") != "failed"
    ):
        raise ValueError("contamination disposition no longer matches its frozen protocol")
    runner_source = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "show",
            f"{result['runner_revision']}:scripts/contamination_disposition.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(runner_source).hexdigest() != result["runner_sha256"]:
        raise ValueError("contamination disposition runner changed")
    print(f"Contamination disposition: {result['status']}")
    return result


def main(argv=None):
    args = arguments(argv)
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "speck_contamination_disposition_protocol":
        raise ValueError("contamination disposition protocol has the wrong format")
    repository_root = Path(__file__).parents[1]
    if args.check:
        if protocol.get("status") not in {"frozen_unexecuted", "executed_failed"}:
            raise ValueError("contamination disposition protocol has an invalid checked status")
        return check_disposition(protocol, args.output, repository_root)
    if protocol.get("status") != "frozen_unexecuted":
        raise ValueError("contamination disposition protocol must be frozen and unexecuted")
    audit_path = repository_root / protocol["audit"]["path"]
    if file_sha256(audit_path) != protocol["audit"]["sha256"]:
        raise ValueError("contamination audit changed before disposition")
    contamination_protocol_path = repository_root / protocol["audit"]["protocol_path"]
    contamination_protocol = json.loads(contamination_protocol_path.read_text(encoding="utf-8"))
    audit = check_report(contamination_protocol_path, contamination_protocol, audit_path)
    if audit["status"] != "failed_critical_overlap_detected":
        raise ValueError("contamination disposition requires a failed critical-overlap audit")
    patterns, references, cases = reconstruct_references(
        protocol, audit, repository_root, get_tokenizer()
    )
    derived = classify(protocol, audit, patterns, references, cases)
    quarantine = derived["critical_quarantine_tasks"]
    if not quarantine:
        raise ValueError("failed contamination audit did not resolve to any quarantine task")
    result = {
        "format": "speck_contamination_disposition",
        "format_version": 1,
        "status": protocol["decision"]["failed_status"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "spec_identity_sha256": spec_identity(protocol),
        },
        "audit": {
            "path": protocol["audit"]["path"],
            "sha256": protocol["audit"]["sha256"],
            "status": audit["status"],
        },
        "derived": derived,
        "decision": {
            "ruler_v1": "failed",
            "quarantine_tasks": quarantine,
            "threshold_changed": False,
            "candidate_execution_authorized": False,
            "required_next": protocol["decision"]["required_next"],
            "allowed_reporting": protocol["decision"]["allowed_reporting"],
            "forbidden_claim": protocol["decision"]["forbidden_claim"],
        },
        "non_claims": protocol["reporting"]["non_claims"],
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }
    atomic_json(args.output, result)
    print(f"Contamination disposition: {result['status']}")


if __name__ == "__main__":
    main()
