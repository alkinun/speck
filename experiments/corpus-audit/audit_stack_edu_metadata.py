"""Census every pinned Stack-Edu metadata file under the historical file predicate.

PYTHONPATH=. python experiments/corpus-audit/audit_stack_edu_metadata.py scan LISTING OUTPUT_DIR
PYTHONPATH=. python experiments/corpus-audit/audit_stack_edu_metadata.py summarize OUTPUT_DIR RECEIPT

Metadata only: rows and declared bytes by language and int_score, never file content. The
predicate is the one that produced the retained stock, with its score floor lifted so each
int_score is counted separately; licence, encoding, size and path conditions are unchanged.
Projected tokens multiply declared bytes by the content yield the retained acquisition
measured, so they are planning figures for int_score 4 and 5 and assumptions for 3.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speck.provenance.io import atomic_json, file_sha256  # noqa: E402

DATA = Path("/mnt/speck-data/speck/data-qualification-20260919")
POLICY = DATA / "code-expansion/metadata-policy.json"
ACQUISITION = DATA / "code-supply/acquisition.json"
HISTORICAL_FLOOR = 4
COLUMNS = ["language", "path", "src_encoding", "length_bytes", "int_score"]


def _per_row(lists, predicate, rows):
    """Count list elements matching `predicate` per parent row."""
    parents = pc.list_parent_indices(lists).to_numpy()
    hits = predicate(pc.list_flatten(lists)).to_numpy(zero_copy_only=False)
    return np.bincount(parents[hits], minlength=rows)


def file_mask(table, policy):
    """Vectorized historical metadata predicate, without its score floor."""
    filters, rows = policy["filters"], table.num_rows
    licences = table["detected_licenses"].combine_chunks()
    allowed = pa.array(filters["accepted_detected_licenses"])
    licence = (
        pc.equal(table["license_type"], "permissive")
        .fill_null(False)
        .to_numpy(zero_copy_only=False)
        & (pc.list_value_length(licences).fill_null(0).to_numpy() > 0)
        & (_per_row(licences, lambda v: pc.invert(pc.is_in(v, allowed)), rows) == 0)
    )
    encoding = pc.is_in(table["src_encoding"], pa.array(filters["accepted_encodings"]))
    length = table["length_bytes"].to_numpy(zero_copy_only=False)
    parts = pc.split_pattern(pc.utf8_lower(pc.replace_substring(table["path"], "\\", "/")), "/")
    vendor = _per_row(
        parts, lambda v: pc.is_in(v, pa.array(policy["excluded_path_components"])), rows
    )
    identity = (
        pc.and_(
            pc.match_substring_regex(table["blob_id"], "^[0-9a-f]{40}$"),
            pc.and_(
                pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(table["repo_name"])), 0),
                pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(table["path"])), 0),
            ),
        )
        .fill_null(False)
        .to_numpy(zero_copy_only=False)
    )
    return (
        identity
        & licence
        & encoding.to_numpy(zero_copy_only=False)
        & (length >= filters["min_file_bytes"])
        & (length <= filters["max_file_bytes"])
        & (vendor == 0)
    )


def scan_file(item):
    policy = json.loads(POLICY.read_text())
    table = pq.read_table(
        item["local"],
        columns=COLUMNS + ["blob_id", "repo_name", "detected_licenses", "license_type"],
    )
    languages = pc.unique(table["language"]).to_pylist()
    if len(languages) != 1:
        raise ValueError(f"expected one language in {item['path']}: {languages}")
    score = table["int_score"].to_numpy(zero_copy_only=False)
    length = table["length_bytes"].to_numpy(zero_copy_only=False)
    unlicensed = (
        pc.equal(table["license_type"], "no_license")
        .fill_null(False)
        .to_numpy(zero_copy_only=False)
    )
    passing = file_mask(table, policy)
    by_score = {}
    for value in sorted(set(score.tolist())):
        tier = score == value
        by_score[str(value)] = {
            "rows": int(tier.sum()),
            "declared_bytes": int(length[tier].sum()),
            "no_license_rows": int((tier & unlicensed).sum()),
            "no_license_declared_bytes": int(length[tier & unlicensed].sum()),
            "pass_rows": int((tier & passing).sum()),
            "pass_declared_bytes": int(length[tier & passing].sum()),
        }
    # Physical-order declared bytes of historically eligible rows, to rebuild yield denominators.
    ordered = length[passing & (score >= HISTORICAL_FLOOR)]
    return {
        "path": item["path"],
        "language": languages[0],
        "rows": table.num_rows,
        "by_score": by_score,
    }, ordered


def scan(listing, output):
    listing = json.loads(Path(listing).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(4) as pool:
        results = list(pool.map(scan_file, listing["files"]))
    files = []
    for index, (summary, ordered) in enumerate(results):
        np.save(output / f"ordered-{index:02d}.npy", ordered)
        files.append(summary | {"ordered_bytes": f"ordered-{index:02d}.npy"})
    atomic_json(output / "scan.json", {"revision": listing["revision"], "files": files})


def historical_yield():
    """Tokens and consumed eligible rows per language from the retained acquisition."""
    consumed = defaultdict(Counter)
    for unit in json.loads(ACQUISITION.read_text())["units"]:
        manifest = unit["manifest"]
        consumed[manifest["language"]].update(
            rows=manifest["consumed_eligible_rows"], tokens=manifest["tokens_before_full_exclusion"]
        )
    return consumed


def summarize(output, receipt):
    output = Path(output)
    scan_record = json.loads((output / "scan.json").read_text())
    consumed = historical_yield()
    languages = defaultdict(lambda: {"files": [], "by_score": defaultdict(Counter), "ordered": []})
    for item in scan_record["files"]:
        entry = languages[item["language"]]
        entry["files"].append(item["path"])
        for value, counts in item["by_score"].items():
            entry["by_score"][value].update(counts)
        entry["ordered"].append(np.load(output / item["ordered_bytes"]))

    by_language, pooled = {}, Counter()
    for language, entry in sorted(languages.items()):
        ordered = np.concatenate(entry["ordered"])
        record = {
            "files": sorted(entry["files"]),
            "by_score": {k: dict(v) for k, v in sorted(entry["by_score"].items())},
        }
        if language in consumed:
            rows = consumed[language]["rows"]
            # Acquisition consumed eligible rows in physical listing order, so the first
            # `rows` historically eligible rows are exactly the declared bytes it fetched.
            declared = int(ordered[:rows].sum())
            record["historical_acquisition"] = {
                "consumed_eligible_rows": rows,
                "consumed_declared_bytes": declared,
                "tokens_before_full_exclusion": consumed[language]["tokens"],
                "tokens_per_declared_byte": consumed[language]["tokens"] / declared,
                "all_release_rows_consumed": rows == len(ordered),
            }
            pooled.update(tokens=consumed[language]["tokens"], declared=declared)
        by_language[language] = record

    pooled_yield = pooled["tokens"] / pooled["declared"]
    totals = defaultdict(Counter)
    for language, record in by_language.items():
        rate = record.get("historical_acquisition", {}).get(
            "tokens_per_declared_byte", pooled_yield
        )
        record["yield_basis"] = (
            "own_acquisition" if "historical_acquisition" in record else "pooled_acquisition"
        )
        projected = {}
        for value, counts in record["by_score"].items():
            projected[value] = int(counts["pass_declared_bytes"] * rate)
            totals[value].update(counts)
            totals[value]["projected_tokens"] += projected[value]
        record["projected_tokens_by_score"] = projected

    receipt_value = {
        "format": "speck_stack_edu_metadata_census",
        "format_version": 1,
        "status": "complete_metadata_census_not_training_admission",
        "training_admitted": False,
        "eligible_tokens_established": 0,
        "gpu_hours": 0,
        "repository": "HuggingFaceTB/stack-edu",
        "revision": scan_record["revision"],
        "metadata_files": len(scan_record["files"]),
        "physical_rows": sum(item["rows"] for item in scan_record["files"]),
        "predicate": {
            "policy": {"path": str(POLICY), "sha256": file_sha256(POLICY)},
            "historical_score_floor": HISTORICAL_FLOOR,
            "changed": "score floor lifted; each int_score reported separately",
            "unchanged": "permissive licence type with every detected licence on the allowlist, encoding, 100 B to 1 MB size, vendor path components",
        },
        "historical_acquisition": {"path": str(ACQUISITION), "sha256": file_sha256(ACQUISITION)},
        "scan": {"path": str(output / "scan.json"), "sha256": file_sha256(output / "scan.json")},
        "pooled_tokens_per_declared_byte": pooled_yield,
        "totals_by_score": {k: dict(v) for k, v in sorted(totals.items())},
        "by_language": by_language,
        "boundary": (
            "Declared metadata bytes and projections only; no content fetched. Projected tokens apply "
            "the retained acquisition's content yield (prose, length, benchmark, security and "
            "Gitleaks rejections, then Mistral tokens) measured on int_score 4 and 5 rows; its "
            "transfer to int_score 3 is an assumption until a content probe measures it. Origin "
            "and notice recovery, family partition and exclusion remain open, so eligible tokens "
            "are zero."
        ),
    }
    atomic_json(receipt, receipt_value)
    print(json.dumps(receipt_value["totals_by_score"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser("scan")
    scan_parser.add_argument("listing")
    scan_parser.add_argument("output")
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("output")
    summary_parser.add_argument("receipt")
    args = parser.parse_args()
    if args.command == "scan":
        scan(args.listing, args.output)
    else:
        summarize(args.output, args.receipt)


if __name__ == "__main__":
    main()
