"""Scan frozen evaluation probes against exact packed Paper 1 training windows."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.ruler_case_prepare import directory_identity
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import load_manifest
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def protocol_identity(protocol):
    payload = {key: protocol[key] for key in protocol if key not in {"status", "result"}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prefix_signature(tokens):
    if len(tokens) < 4:
        raise ValueError("contamination probe prefix requires four tokens")
    return sum(int(tokens[index]) << (16 * index) for index in range(4))


def find_subsequence(values, query):
    if not query or len(query) > len(values):
        return []
    first = query[0]
    return [
        index
        for index in range(len(values) - len(query) + 1)
        if values[index] == first and values[index : index + len(query)] == query
    ]


def _add_probe(probes, tokens, kind, reference):
    key = tuple(int(token) for token in tokens)
    if key not in probes:
        probes[key] = {"tokens": key, "kind": kind, "references": []}
    elif probes[key]["kind"] != kind:
        # Preserve the more critical interpretation when the exact same tokens serve two roles.
        priority = {"context": 0, "answer_anchored": 1, "full_prompt": 2}
        if priority[kind] > priority[probes[key]["kind"]]:
            probes[key]["kind"] = kind
    probes[key]["references"].append(reference)


def _encode(tokenizer, text):
    return tokenizer.encode(text, bos=False, eos=False)


def build_ruler_probes(protocol, tokenizer, repository_root, hf_tokenizer=None):
    n = protocol["probes"]["token_ngram_length"]
    fractions = protocol["probes"]["context_fractions"]
    ruler = next(entry for entry in protocol["evaluations"] if entry["suite"] == "ruler")
    probes = {}
    cases = Counter()
    skipped_answers = 0
    parity_samples = 0
    special_literals = tuple(hf_tokenizer.all_special_tokens) if hf_tokenizer is not None else ()
    for length_report in ruler["length_reports"]:
        report_path = repository_root / length_report["path"]
        if file_sha256(report_path) != length_report["sha256"]:
            raise ValueError(f"RULER contamination report changed: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        output = Path(report["local_output_dir"]) / "cases"
        for task in report["tasks"]:
            case_path = output / task / "test.jsonl"
            if not case_path.is_file():
                raise FileNotFoundError(f"RULER contamination cases missing: {case_path}")
            with case_path.open(encoding="utf-8") as handle:
                for row_number, line in enumerate(handle):
                    row = json.loads(line)
                    if any(literal and literal in row["question"] for literal in special_literals):
                        raise ValueError("RULER prompt contains a tokenizer special-token literal")
                    tokens = _encode(tokenizer, row["question"])
                    if row_number == 0 and hf_tokenizer is not None:
                        expected = hf_tokenizer.encode(
                            row["question"], add_special_tokens=False
                        )
                        if tokens != expected:
                            raise ValueError(
                                f"native/Transformers tokenizer mismatch: {task} "
                                f"at {length_report['length']}"
                            )
                        parity_samples += 1
                    reference = {
                        "suite": "ruler",
                        "length": length_report["length"],
                        "task": task,
                        "row": row_number,
                    }
                    cases[(length_report["length"], task)] += 1
                    if n <= len(tokens) <= protocol["training"]["sequence_length"]:
                        _add_probe(probes, tokens, "full_prompt", {**reference, "slot": "full"})
                    if len(tokens) >= n:
                        maximum_start = len(tokens) - n
                        for fraction in fractions:
                            start = min(maximum_start, round(maximum_start * fraction))
                            _add_probe(
                                probes,
                                tokens[start : start + n],
                                "context",
                                {**reference, "slot": f"context_{fraction}"},
                            )
                    for answer_number, answer in enumerate(row["expected_answer"]):
                        answer_tokens = _encode(tokenizer, answer)
                        locations = find_subsequence(tokens, answer_tokens)
                        if not locations:
                            skipped_answers += 1
                            continue
                        for occurrence, location in enumerate(locations):
                            start = max(0, location - (n - len(answer_tokens)) // 2)
                            start = min(start, max(0, len(tokens) - n))
                            if len(tokens[start : start + n]) == n:
                                _add_probe(
                                    probes,
                                    tokens[start : start + n],
                                    "answer_anchored",
                                    {
                                        **reference,
                                        "slot": f"answer_{answer_number}_{occurrence}",
                                    },
                                )
    return probes, cases, skipped_answers, parity_samples


def _batch_signatures(rows):
    values = rows.astype(np.uint64, copy=False)
    return (
        values[:, :-3]
        | (values[:, 1:-2] << np.uint64(16))
        | (values[:, 2:-1] << np.uint64(32))
        | (values[:, 3:] << np.uint64(48))
    )


def scan_windows(protocol, probes, tokenizer):
    by_signature = defaultdict(list)
    for key, probe in probes.items():
        by_signature[prefix_signature(key)].append(probe)
    signatures = np.array(sorted(by_signature), dtype=np.uint64)
    hits = Counter()
    locations = defaultdict(list)
    per_window = []
    training = protocol["training"]
    batches = training["tokens_per_window"] // (
        training["sequence_length"] * training["device_batch_size"]
    )
    if batches * training["sequence_length"] * training["device_batch_size"] != training[
        "tokens_per_window"
    ]:
        raise ValueError("contamination training window does not align with loader microbatches")
    for window in training["windows"]:
        loader = packed_loader(
            tokenizer,
            batch_size=training["device_batch_size"],
            sequence_length=training["sequence_length"],
            split="train",
            device="cpu",
            data_dir=training["data_directory"],
            initial_token_offset=window["data_token_offset"],
        )
        source_counts = Counter()
        matched_by_kind = Counter()
        started = time.monotonic()
        for batch_index in range(batches):
            inputs, _, state = next(loader)
            rows = inputs.numpy().astype(np.uint16, copy=False)
            batch_signatures = _batch_signatures(rows)
            indices = np.searchsorted(signatures, batch_signatures)
            clipped = np.minimum(indices, len(signatures) - 1)
            candidates = np.argwhere(
                (indices < len(signatures)) & (signatures[clipped] == batch_signatures)
            )
            for row_index, position in candidates:
                signature = int(batch_signatures[row_index, position])
                row = rows[row_index]
                for probe in by_signature[signature]:
                    probe_tokens = probe["tokens"]
                    if position + len(probe_tokens) > len(row):
                        continue
                    if np.array_equal(
                        row[position : position + len(probe_tokens)],
                        np.asarray(probe_tokens, dtype=np.uint16),
                    ):
                        identifier = hashlib.sha256(
                            np.asarray(probe_tokens, dtype="<u2").tobytes()
                        ).hexdigest()
                        hits[(identifier, probe["kind"])] += 1
                        matched_by_kind[probe["kind"]] += 1
                        if len(locations[identifier]) < 3:
                            locations[identifier].append(
                                {
                                    "pair": window["pair"],
                                    "source": state["selected_source"],
                                    "batch": batch_index,
                                    "row": int(row_index),
                                    "token_position": int(position),
                                }
                            )
            source_counts[state["selected_source"]] += rows.size
            if (batch_index + 1) % 500 == 0:
                print(
                    f"contamination pair {window['pair']}: {batch_index + 1}/{batches} batches",
                    flush=True,
                )
        per_window.append(
            {
                **window,
                "tokens_scanned": sum(source_counts.values()),
                "source_tokens": dict(sorted(source_counts.items())),
                "matched_occurrences_by_kind": dict(sorted(matched_by_kind.items())),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    matched_patterns = []
    for key, probe in probes.items():
        identifier = hashlib.sha256(np.asarray(key, dtype="<u2").tobytes()).hexdigest()
        count = hits[(identifier, probe["kind"])]
        if count:
            matched_patterns.append(
                {
                    "id": identifier,
                    "kind": probe["kind"],
                    "tokens": len(key),
                    "reference_count": len(probe["references"]),
                    "occurrences": count,
                    "locations": locations[identifier],
                    "references": probe["references"][:10],
                }
            )
    return per_window, matched_patterns


def check_report(protocol_path, protocol, output):
    report = json.loads(output.expanduser().resolve().read_text(encoding="utf-8"))
    if (
        report.get("format") != "speck_contamination_audit"
        or report.get("status")
        not in {
            "qualified_no_full_or_answer_overlap_context_risk_reported",
            "failed_critical_overlap_detected",
        }
        or report.get("protocol", {}).get("spec_identity_sha256")
        != protocol_identity(protocol)
        or report.get("training", {}).get("total_tokens_scanned")
        != protocol["training"]["total_scanned_tokens"]
        or report.get("evaluation", {}).get("total_cases") != 7800
        or set(report.get("decisions", {})) != {"full_prompt", "answer_anchored", "context"}
    ):
        raise ValueError("contamination result no longer matches the frozen protocol")
    runner_source = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/contamination_audit.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(runner_source).hexdigest() != report["runner_sha256"]:
        raise ValueError("contamination audit runner changed")
    for evaluation in protocol["evaluations"]:
        for length_report in evaluation.get("length_reports", ()):
            path = Path(__file__).parents[1] / length_report["path"]
            if not path.is_file() or file_sha256(path) != length_report["sha256"]:
                raise ValueError(f"contamination evaluation input changed: {path}")
    print(f"Contamination audit: {report['status']} ({protocol_path.name})")
    return report


def main(argv=None):
    args = arguments(argv)
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "speck_contamination_protocol":
        raise ValueError("contamination protocol has the wrong format")
    if args.check:
        if protocol.get("status") not in {
            "frozen_unexecuted",
            "executed_failed_critical_overlap_detected",
            "executed_qualified_no_full_or_answer_overlap_context_risk_reported",
        }:
            raise ValueError("contamination protocol has an invalid checked status")
        return check_report(protocol_path, protocol, args.output)
    if protocol.get("status") != "frozen_unexecuted":
        raise ValueError("contamination protocol must be frozen and unexecuted")
    repository_root = Path(__file__).parents[1]
    data_manifest = load_manifest(protocol["training"]["data_directory"])
    if manifest_fingerprint(data_manifest) != protocol["training"]["manifest_fingerprint"]:
        raise ValueError("contamination training manifest changed")
    export_identity = directory_identity(protocol["tokenizer"]["path"])
    if export_identity["sha256"] != protocol["tokenizer"]["identity_sha256"]:
        raise ValueError("contamination tokenizer export changed")
    hf_tokenizer = AutoTokenizer.from_pretrained(
        protocol["tokenizer"]["path"], trust_remote_code=True, local_files_only=True
    )
    packed_tokenizer = get_tokenizer()
    probes, cases, skipped_answers, parity_samples = build_ruler_probes(
        protocol, packed_tokenizer, repository_root, hf_tokenizer
    )
    counts = Counter(probe["kind"] for probe in probes.values())
    references = Counter()
    for probe in probes.values():
        references[probe["kind"]] += len(probe["references"])
    windows, matches = scan_windows(protocol, probes, packed_tokenizer)
    matched_unique = Counter(match["kind"] for match in matches)
    matched_occurrences = Counter()
    for match in matches:
        matched_occurrences[match["kind"]] += match["occurrences"]
    full_pass = matched_unique["full_prompt"] <= protocol["decision"][
        "full_prompt_matches_allowed"
    ]
    answer_pass = matched_unique["answer_anchored"] <= protocol["decision"][
        "answer_anchored_matches_allowed"
    ]
    report = {
        "format": "speck_contamination_audit",
        "format_version": 1,
        "status": (
            "qualified_no_full_or_answer_overlap_context_risk_reported"
            if full_pass and answer_pass
            else "failed_critical_overlap_detected"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "spec_identity_sha256": protocol_identity(protocol),
        },
        "training": {
            "manifest_fingerprint": manifest_fingerprint(data_manifest),
            "windows": windows,
            "total_tokens_scanned": sum(window["tokens_scanned"] for window in windows),
        },
        "evaluation": {
            "ruler_cases": [
                {"length": length, "task": task, "cases": count}
                for (length, task), count in sorted(cases.items())
            ],
            "total_cases": sum(cases.values()),
            "skipped_answer_occurrences": skipped_answers,
            "native_transformers_parity_samples": parity_samples,
            "unique_probes_by_kind": dict(sorted(counts.items())),
            "probe_references_by_kind": dict(sorted(references.items())),
        },
        "matches": {
            "unique_patterns_by_kind": dict(sorted(matched_unique.items())),
            "occurrences_by_kind": dict(sorted(matched_occurrences.items())),
            "patterns": matches,
        },
        "decisions": {
            "full_prompt": full_pass,
            "answer_anchored": answer_pass,
            "context": "descriptive_only",
        },
        "unscanned_suites": [
            {"suite": entry["suite"], "status": entry["status"]}
            for entry in protocol["evaluations"]
            if entry["suite"] != "ruler"
        ],
        "non_claims": protocol["reporting"]["non_claims"],
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }
    atomic_json(args.output, report)
    print(f"Contamination audit: {report['status']}")


if __name__ == "__main__":
    main()
