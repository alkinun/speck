"""Replay a bounded natural-code lineage audit offline, without executing source code.

Run with the repository environment and PYTHONPATH=.:
  python experiments/corpus-audit/audit_natural_code.py INPUTS.json
This produces evidence for review, never a training-admission manifest.
"""

import argparse
import ast
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote, urlencode


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verified(artifact):
    data = Path(artifact["path"]).read_bytes()
    if digest(data) != artifact["sha256"]:
        raise ValueError(f"SHA256 mismatch: {artifact['path']}")
    if "bytes" in artifact and len(data) != artifact["bytes"]:
        raise ValueError(f"size mismatch: {artifact['path']}")
    return data


def require(condition, message):
    if not condition:
        raise ValueError(message)


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def check_lineage(candidate, record, acquired):
    """Verify the source/commit/tree binding; missing evidence stays unresolved."""
    require(acquired["candidate"] == candidate, "acquisition candidate mismatch")
    requests = acquired["requests"]
    for request in requests:
        verified(request)

    def response(artifact, expected_url):
        require(artifact in requests, "response absent from acquisition log")
        require(artifact["url"] == expected_url, "repository/revision/path URL mismatch")
        require(artifact["status"] == 200, "unsuccessful upstream response")
        return verified(artifact)

    raw = record["text"].encode()
    require(digest(raw) == candidate["expected_sha256"], "retained source SHA mismatch")
    require(hashlib.sha1(raw).hexdigest() == candidate["content_id"], "content ID mismatch")
    repo, path = candidate["repo"], candidate["path"]
    require(
        repo == record["repo_path"] and path == record["file_path"].lstrip("/"),
        "archive origin mismatch",
    )
    base = "https://api.github.com/repos/" + repo
    meta = acquired.get("repository_metadata")
    family = None
    if meta and meta["status"] == 200:
        metadata = json.loads(response(meta, base))
        family = metadata.get("source", metadata)["full_name"].lower()
    revision = acquired.get("matched_revision")
    notices, source_match, tree_complete = [], False, False
    if revision is not None:
        require(re.fullmatch(r"[0-9a-f]{40}", revision), "invalid immutable revision")
        commits = json.loads(
            response(
                acquired["commits"], base + "/commits?" + urlencode({"path": path, "per_page": 5})
            )
        )
        require(revision in [item["sha"] for item in commits], "revision absent from path history")
        prefix = "https://raw.githubusercontent.com/" + repo + "/" + revision + "/"
        source = response(acquired["source"], prefix + quote(path, safe="/"))
        require(source == raw, "upstream bytes do not match retained source")
        source_match = True
        if acquired.get("tree") and acquired["tree"]["status"] == 200:
            tree = json.loads(
                response(acquired["tree"], base + "/git/trees/" + revision + "?recursive=1")
            )
            entries = {entry["path"]: entry for entry in tree["tree"]}
            require(len(entries) == len(tree["tree"]), "duplicate tree path")
            entry = entries.get(path, {})
            require(
                entry.get("type") == "blob"
                and entry.get("mode") in ("100644", "100755")
                and entry.get("sha") == blob_sha(source),
                "source/tree blob mismatch",
            )
            ancestors = {".", *[str(p) for p in Path(path).parents]}
            expected = {
                name
                for name, entry in entries.items()
                if entry["type"] == "blob"
                and str(Path(name).parent) in ancestors
                and re.match(
                    r"^(licen[sc]e|copying|notice|copyright)([._-].*)?$", Path(name).name, re.I
                )
            }
            for notice in acquired["notices"]:
                name = notice["repository_path"]
                require(name in expected, "notice is outside source ancestor scope")
                payload = response(notice, prefix + quote(name, safe="/"))
                require(
                    entries[name].get("mode") in ("100644", "100755")
                    and blob_sha(payload) == entries[name]["sha"] == notice["git_blob_sha1"],
                    "notice/tree blob mismatch",
                )
                notices.append(
                    {
                        "repository_path": name,
                        "sha256": digest(payload),
                        "bytes": len(payload),
                        "url": notice["url"],
                    }
                )
            found = [n["repository_path"] for n in notices]
            require(len(found) == len(set(found)), "duplicate license notice")
            tree_complete = not tree.get("truncated", True) and set(found) == expected
    reasons = []
    if not source_match:
        reasons.append("unresolved_matching_revision")
    if not family:
        reasons.append("unresolved_repository_family")
    if not tree_complete:
        reasons.append("incomplete_tree_or_notice_evidence")
    if not any(
        re.match(r"^(licen[sc]e|copying)([._-].*)?$", Path(n["repository_path"]).name, re.I)
        and n["bytes"] > 0
        for n in notices
    ):
        reasons.append("no_license_file_in_inspected_ancestor_scope")
    if re.search(
        r"(^|/)(vendor|vendored|third_party|3rdparty|site-packages|node_modules)(/|$)", path, re.I
    ):
        reasons.append("vendor_origin_requires_review")
    return {
        "matching_revision": revision,
        "upstream_bytes_match": source_match,
        "repository_family": family,
        "family_basis": "observed GitHub source/full_name",
        "license_notices": notices,
        "tree_and_notice_inventory_complete": tree_complete,
        "lineage_evidence_ready": not reasons,
        "lineage_hold_reasons": reasons,
    }


def audit(inputs):
    # Imports are the repository's trusted machinery; corpus text is parsed, never imported.
    from speck.evaluation.protocol import BenchmarkExclusion
    from speck.tokenization.tokenizer import Tokenizer

    for artifact in inputs.values():
        verified(artifact)
    selection = json.loads(verified(inputs["selection"]))
    acquisition = json.loads(verified(inputs["acquisition"]))
    require(
        acquisition["selection_sha256"] == inputs["selection"]["sha256"],
        "acquisition selection identity mismatch",
    )
    candidates = selection["candidates"]
    require(
        len(candidates) == 16 and len({c["ordinal"] for c in candidates}) == 16,
        "cohort must contain 16 unique ordinals",
    )
    manifest = json.loads(verified(selection["natural_code_manifest"]))
    inventory = json.loads(verified(manifest["inventory"]))
    verified(manifest["tar"])
    member = "unit/attempt-00000/records.jsonl"
    with tarfile.open(manifest["tar"]["path"]) as archive:
        payload = archive.extractfile(member).read()
    expected = next(item for item in inventory if item["path"] == member)
    require(digest(payload) == expected["sha256"], "archive records identity mismatch")
    records = [json.loads(line) for line in payload.splitlines()]
    by_ordinal = {row["eligible_ordinal"]: row for row in records}
    require(len(by_ordinal) == len(records), "duplicate archive ordinal")
    acquired = {r["candidate"]["ordinal"]: r for r in acquisition["results"]}
    require(
        len(acquired) == len(acquisition["results"]) == 16
        and set(acquired) == {c["ordinal"] for c in candidates},
        "acquisition coverage mismatch",
    )
    protocol = json.loads(verified(inputs["benchmark_protocol"]))
    exclusion = BenchmarkExclusion(protocol)
    verified(inputs["tokenizer_model"])
    tokenizer = Tokenizer(inputs["tokenizer_model"]["path"])
    results = []
    for candidate in candidates:
        record = by_ordinal[candidate["ordinal"]]
        result = {**candidate, **check_lineage(candidate, record, acquired[candidate["ordinal"]])}
        try:
            ast.parse(record["text"])
            result["python_syntax"] = "pass"
        except SyntaxError as error:
            result["python_syntax"] = f"SyntaxError at line {error.lineno}"
        matches = exclusion.matches(record["text"])
        result.update(
            {
                "benchmark_match_count": len(matches),
                "tokens_with_bos_eos": len(tokenizer.encode(record["text"], bos=True, eos=True)),
                "dataset_detected_licenses": record["metadata"]["detected_licenses"],
                "decision": "hold_outside_new_training_intervention",
                "source_family_partition": "quarantine",
                "remaining_gates": [
                    "file_license_and_attribution_review",
                    "broader_code_benchmark_and_repository_exclusions",
                    "source_family_split_and_near_deduplication",
                    "independent_quality_and_dependency_checks",
                ],
            }
        )
        if matches:
            result["remaining_gates"].append("existing_benchmark_overlap")
        if result["python_syntax"] != "pass":
            result["remaining_gates"].append("python_runtime_compatibility")
        result["ready_for_independent_quality_review"] = (
            result["lineage_evidence_ready"] and result["python_syntax"] == "pass" and not matches
        )
        results.append(result)
    counts = Counter(r["expected_sha256"] for r in results)
    return {
        "format_version": 1,
        "inputs": inputs,
        "status": "bounded_lineage_qualification_complete_not_training_admission",
        "training_authority": False,
        "corpus_code_executed": False,
        "syntax_runtime": sys.version,
        "selection_boundary": selection["selection_boundary"],
        "revision_boundary": "Matching immutable revision, not necessarily the original crawl commit.",
        "license_boundary": "Notice evidence presence and identity; no automated legal acceptance. "
        "Missing conventional files does not prove absence of a license.",
        "exclusion_boundary": "Existing frozen five-benchmark exclusion only, both partitions; "
        "no scoring or exposure of final tasks. MBPP+, MultiPL-E, "
        "LiveCodeBench and repository repair exclusions remain unqualified.",
        "summary": {
            "candidates": len(results),
            "exact_upstream_matches": sum(r["upstream_bytes_match"] for r in results),
            "lineage_evidence_ready": sum(r["lineage_evidence_ready"] for r in results),
            "benchmark_flagged_files": sum(bool(r["benchmark_match_count"]) for r in results),
            "syntax_pass_files": sum(r["python_syntax"] == "pass" for r in results),
            "ready_for_independent_quality_review": sum(
                r["ready_for_independent_quality_review"] for r in results
            ),
            "quality_review_candidate_tokens_with_bos_eos": sum(
                r["tokens_with_bos_eos"]
                for r in results
                if r["ready_for_independent_quality_review"]
            ),
            "distinct_observed_families": len({r["repository_family"] for r in results} - {None}),
            "exact_duplicate_groups": sum(v > 1 for v in counts.values()),
            "candidate_tokens_with_bos_eos": sum(r["tokens_with_bos_eos"] for r in results),
            "admitted_files": 0,
            "admitted_tokens": 0,
        },
        "records": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.inputs.read_text())), indent=2, sort_keys=True))
