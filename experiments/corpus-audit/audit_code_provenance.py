"""Replay the bounded provenance inspection offline; never execute or admit corpus code.

This checks retained evidence, not legal eligibility or a corpus-wide L2/L3 join.
Run: python experiments/corpus-audit/audit_code_provenance.py INPUTS.json > RECEIPT.json
"""

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verified(artifact):
    data = Path(artifact["path"]).read_bytes()
    if sha256(data) != artifact["sha256"]:
        raise ValueError(f"SHA256 mismatch: {artifact['path']}")
    if "bytes" in artifact and len(data) != artifact["bytes"]:
        raise ValueError(f"size mismatch: {artifact['path']}")
    return data


def audit(inputs):
    acquisition = json.loads(verified(inputs["acquisition_receipt"]))
    revision = inputs["revision"]
    if acquisition["revision"] != revision:
        raise ValueError("acquisition revision mismatch")
    windows = {}
    probes = {}
    for artifact in acquisition["requests"]:
        if artifact["revision"] != revision:
            raise ValueError("response revision mismatch")
        response = json.loads(verified(artifact))
        if response["partial"] or any(r["truncated_cells"] for r in response["rows"]):
            raise ValueError("incomplete response")
        tier = "L2" if "UltraData-Code-L2" in artifact["url"] else "L3"
        fields = sorted(f["name"] for f in response["features"])
        if any(sorted(r["row"]) != fields for r in response["rows"]):
            raise ValueError("schema/row mismatch")
        target = probes if "schema" in artifact["path"] else windows
        if tier in target:
            raise ValueError("duplicate response role")
        target[tier] = response
    if set(windows) != {"L2", "L3"} or set(probes) != {"L2", "L3"}:
        raise ValueError("missing response role")
    for tier in windows:
        if windows[tier]["features"] != probes[tier]["features"]:
            raise ValueError("schema probe disagrees with window")
        if len(windows[tier]["rows"]) != 16 or len(probes[tier]["rows"]) != 1:
            raise ValueError("unexpected bounded sample size")
    for artifact in inputs["primary_sources"]:
        verified(artifact)

    tiers = {}
    for tier, response in windows.items():
        rows = response["rows"]
        text_key = "content" if tier == "L2" else "raw_content"
        counts = Counter(sha256(r["row"][text_key].encode()) for r in rows)
        tiers[tier] = {
            "sample_rows": len(rows),
            "schema_fields": sorted(f["name"] for f in response["features"]),
            "nonempty_field_counts": {
                field: sum(bool(r["row"][field]) for r in rows) for field in sorted(rows[0]["row"])
            },
            "raw_text_field": text_key,
            "raw_text_distinct_sha256": len(counts),
            "raw_text_duplicate_group_sizes": sorted(c for c in counts.values() if c > 1),
            "raw_text_license_marker_rows": [
                r["row_idx"]
                for r in rows
                if re.search(r"copyright|license|SPDX", r["row"][text_key], re.I)
            ],
        }
    l2 = windows["L2"]["rows"] + probes["L2"]["rows"]
    l3 = windows["L3"]["rows"] + probes["L3"]["rows"]
    uuid_matches = sorted({r["row"]["uuid"] for r in l2} & {r["row"]["uuid"] for r in l3})
    text_matches = sorted(
        {sha256(r["row"]["content"].encode()) for r in l2}
        & {sha256(r["row"]["raw_content"].encode()) for r in l3}
    )

    # Read archive members as bytes; do not extract paths or import any dataset code.
    manifest = json.loads(verified(inputs["natural_code_manifest"]))
    inventory = json.loads(verified(manifest["inventory"]))
    verified(manifest["tar"])
    member = "unit/attempt-00000/records.jsonl"
    with tarfile.open(manifest["tar"]["path"]) as archive:
        records_data = archive.extractfile(member).read()
    expected = next(item for item in inventory if item["path"] == member)
    if sha256(records_data) != expected["sha256"]:
        raise ValueError("archive records mismatch")
    records = [json.loads(line) for line in records_data.splitlines()]
    candidates = [r for r in records if r["eligible_ordinal"] == inputs["route_ordinal"]]
    if len(candidates) != 1:
        raise ValueError("ambiguous natural-code candidate")
    candidate = candidates[0]
    upstream = verified(inputs["route_source"])
    verified(inputs["route_license"])
    if (
        upstream != candidate["text"].encode()
        or sha256(upstream) != candidate["released_content_sha256"]
        or hashlib.sha1(upstream).hexdigest() != candidate["content_id"]
    ):
        raise ValueError("upstream source does not match retained natural code")

    return {
        "format_version": 1,
        "inputs": inputs,
        "status": "bounded_lineage_audit_complete_candidates_not_admitted",
        "training_authority": False,
        "corpus_code_executed": False,
        "tiers": tiers,
        "join": {
            "scope": "17 retained rows per tier including schema probes; not a global search",
            "uuid_intersection": uuid_matches,
            "exact_l2_content_l3_raw_content_sha256_intersection": text_matches,
            "documented_shared_uuid_semantics": False,
            "verified_origin_join": False,
            "interpretation": "No supported join established; no overlap does not disprove a global join.",
        },
        "rejections": [
            {
                "row_idx": r["row_idx"],
                "uuid": r["row"]["uuid"],
                "raw_content_sha256": sha256(r["row"]["raw_content"].encode()),
                "reason_codes": [
                    "unverified_source_origin",
                    "missing_source_revision",
                    "missing_source_license_evidence",
                    "unverified_l2_join",
                ],
                "decision": "hold_outside_training",
            }
            for r in windows["L3"]["rows"]
        ],
        "next_source_route": {
            "inspected_unit_rows": len(records),
            "rows_with_commit_id": sum(bool(r.get("commit_id")) for r in records),
            "candidate": {k: v for k, v in candidate.items() if k != "text"},
            "verified_matching_upstream_revision": inputs["route_revision"],
            "upstream_bytes_equal_retained_text": True,
            "content_id_algorithm_verified": "SHA1 of plain file bytes, not Git blob SHA1",
            "source_license_notice_available": True,
            "admitted": False,
            "remaining_gates": inputs["route_remaining_gates"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.inputs.read_text())), indent=2, sort_keys=True))
