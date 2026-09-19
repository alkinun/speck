"""Census retained Stack-Edu archives without executing code or admitting training data.

PYTHONPATH=. python experiments/corpus-audit/audit_code_supply.py scan PLAN.json FRESH_OUTPUT
PYTHONPATH=. python experiments/corpus-audit/audit_code_supply.py summarize PLAN.json OUTPUT
Raw metadata stays outside Git. Summarize verifies the completed index before replaying metrics.
"""

import argparse
import hashlib
import io
import json
import re
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sentencepiece as spm

from speck.data.code_families import repository_key
from speck.provenance.io import file_sha256

TOKENIZER = None
VENDOR = re.compile(
    r"(^|/)(vendor|vendored|third[-_]party|3rdparty|site-packages|node_modules)(/|$)", re.I
)
TEST = re.compile(
    r"(^|/)(__tests__|tests?|specs?)(/|$)|(^|/)(test_[^/]*|[^/]*(_test|[.]test|[.]spec))[.][^/]+$",
    re.I,
)
DOC = re.compile(r"(^|/)(docs?|tutorials?|examples?)(/|$)|[.](md|rst)$", re.I)
GENERATED = re.compile(
    r"(?im)^.{0,50}(auto[- ]?generated|automatically generated|generated (by|using)|do not edit).{0,80}$"
)


def role_hint(path):
    return next(
        (
            name
            for name, pattern in (("vendor", VENDOR), ("test", TEST), ("docs_example", DOC))
            if pattern.search(path)
        ),
        "other",
    )


def identity(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def verified(item):
    path = Path(item["path"])
    if file_sha256(path) != item["sha256"]:
        raise ValueError(f"input identity mismatch: {path}")
    return path


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def initialize(tokenizer):
    global TOKENIZER
    TOKENIZER = spm.SentencePieceProcessor(model_file=str(tokenizer))


def census_unit(item):
    archive, expected = item["archive"], item["manifest"]
    raw = verified(archive["tar"]).read_bytes()
    if len(raw) != archive["tar"]["bytes"]:
        raise ValueError("archive size mismatch")
    inventory = {row["path"]: row for row in json.loads(verified(archive["inventory"]).read_text())}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:

        def member(name):
            data = tar.extractfile(name).read()
            binding = inventory[name]
            if (
                len(data) != binding["bytes"]
                or hashlib.sha256(data).hexdigest() != binding["sha256"]
            ):
                raise ValueError(f"member identity mismatch: {name}")
            return data

        manifest_bytes = member("unit/manifest.json")
        if hashlib.sha256(manifest_bytes).hexdigest() != archive["unit_manifest_sha256"]:
            raise ValueError("unit manifest identity mismatch")
        manifest = json.loads(manifest_bytes)
        if manifest != expected or manifest["status"] != "complete_not_training_data":
            raise ValueError("unit differs from archived acquisition receipt")
        config_bytes = member("unit/config.json")
        config = json.loads(config_bytes)
        canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(canonical_config).hexdigest() != manifest["config_sha256"]:
            raise ValueError("unit configuration mismatch")
        records = member("unit/" + manifest["output"]["path"])
        if hashlib.sha256(records).hexdigest() != manifest["output"]["sha256"]:
            raise ValueError("unit output mismatch")
    rows = []
    for line in records.splitlines():
        row = json.loads(line)
        raw_text = row["text"].encode()
        sha = hashlib.sha256(raw_text).hexdigest()
        if sha != row["released_content_sha256"]:
            raise ValueError("released text identity mismatch")
        language = row["language"]
        if language != manifest["language"]:
            raise ValueError("language mismatch")
        source = config["unit"]["reader"]
        if (row["source_repo"], row["source_revision"]) != (source["repo"], source["revision"]):
            raise ValueError("source revision mismatch")
        tokens = len(TOKENIZER.encode(row["text"])) + 2
        repository = repository_key(row["repo_path"]) if row.get("repo_path") else None
        path = row.get("file_path") or ""
        rows.append(
            {
                "id": f"{manifest['unit_id']}:{row['eligible_ordinal']}",
                "language": language,
                "repository": repository,
                "path": path,
                "sha256": sha,
                "tokens": tokens,
                "utf8_bytes": len(raw_text),
                "role_hint": role_hint(path),
                "generated_header_hint": bool(GENERATED.search(row["text"][:4000])),
                "commit_id": row.get("commit_id"),
                "license_labels": row.get("metadata", {}).get("detected_licenses", []),
                "blob_sha1_matches_text": hashlib.sha1(raw_text).hexdigest() == row["content_id"],
                "source_revision": row["source_revision"],
                "source_file": row["source_file"],
                "source_row": row["source_row"],
            }
        )
    if (len(rows), sum(r["tokens"] for r in rows), sum(r["utf8_bytes"] for r in rows)) != (
        manifest["retained_records"],
        manifest["tokens_before_full_exclusion"],
        manifest["retained_utf8_bytes"],
    ):
        raise ValueError(f"recount differs from acquisition: {manifest['unit_id']}")
    return rows, config["contamination_plan"]["sha256"]


def scan(plan, output):
    acquisition = json.loads(verified(plan["acquisition"]).read_text())
    tokenizer = verified(acquisition["reference_tokenizer"])
    units = sorted(acquisition["units"], key=lambda item: item["manifest"]["unit_id"])
    if len({item["manifest"]["unit_id"] for item in units}) != len(units):
        raise ValueError("duplicate acquisition unit")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    policies = Counter()
    with ProcessPoolExecutor(max_workers=4, initializer=initialize, initargs=(tokenizer,)) as pool:
        with (output / "documents.jsonl").open("w") as handle:
            for index, (rows, policy) in enumerate(pool.map(census_unit, units, chunksize=4), 1):
                policies[policy] += 1
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                if index % 100 == 0 or index == len(units):
                    print(f"Verified and retokenized {index}/{len(units)} archives", flush=True)
    save(
        output / "scan.json",
        {
            "acquisition": plan["acquisition"],
            "script": identity(__file__),
            "documents": identity(output / "documents.jsonl"),
            "tokenizer": identity(tokenizer),
            "units": len(units),
            "archive_bytes": sum(item["archive"]["tar"]["bytes"] for item in units),
            "historical_contamination_plan_units": dict(policies),
            "seconds": time.monotonic() - start,
            "training_admitted": False,
        },
    )


def summarize(plan, output):
    acquisition = json.loads(verified(plan["acquisition"]).read_text())
    scan_result = json.loads((output / "scan.json").read_text())
    if scan_result["acquisition"] != plan["acquisition"]:
        raise ValueError("scan acquisition mismatch")
    verified(scan_result["script"])
    qualification = json.loads(verified(plan["qualification_inputs"]).read_text())
    rows = [
        json.loads(line) for line in verified(scan_result["documents"]).read_text().splitlines()
    ]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate record identity")
    counts, roles, repositories = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    exact, links = {}, {}
    duplicate_copies = duplicate_tokens = 0

    def find(repo):
        links.setdefault(repo, repo)
        while links[repo] != repo:
            links[repo] = links[links[repo]]
            repo = links[repo]
        return repo

    def union(a, b):
        if a and b:
            a, b = find(a), find(b)
            links[max(a, b)] = min(a, b)

    for a, b in qualification["aliases"]:
        union(repository_key(a), repository_key(b))
    for row in rows:
        language, repo, tokens = row["language"], row["repository"], row["tokens"]
        count = counts[language]
        count.update(documents=1, tokens=tokens, utf8_bytes=row["utf8_bytes"])
        for key, flag in {
            "repository_and_path": bool(repo and row["path"]),
            "commit_populated": bool(row["commit_id"]),
            "license_label_present": bool(row["license_labels"]),
            "blob_sha1_matches_text": row["blob_sha1_matches_text"],
            "generated_header_hint": row["generated_header_hint"],
            "over_4096": tokens > 4096,
            "over_32768": tokens > 32768,
            "over_131072": tokens > 131072,
        }.items():
            if flag:
                count[key + "_documents"] += 1
                count[key + "_tokens"] += tokens
        role = row["role_hint"]
        roles[language][role + "_documents"] += 1
        roles[language][role + "_tokens"] += tokens
        if repo:
            find(repo)
            repositories[repo][role] += 1
            repositories[repo]["tokens"] += tokens
        sha = row["sha256"]
        if sha in exact:
            duplicate_copies += 1
            duplicate_tokens += tokens
            union(repo, exact[sha])
        else:
            exact[sha] = repo
    expected = acquisition["progress"]["by_language_before_full_exclusion"]
    if set(counts) != set(expected) or any(
        (counts[lang]["documents"], counts[lang]["tokens"]) != (value["documents"], value["tokens"])
        for lang, value in expected.items()
    ):
        raise ValueError("language census differs from acquisition")
    held = {repository_key(repo) for repo in qualification["held_repositories"]}
    held_roots = {find(repo) for repo in held}
    held_counts = Counter()
    for row in rows:
        repo = row["repository"]
        for kind, hit in (
            ("direct_name", repo in held),
            ("name_alias_exact_component", bool(repo and find(repo) in held_roots)),
        ):
            if hit:
                held_counts[kind + "_documents"] += 1
                held_counts[kind + "_tokens"] += row["tokens"]
    co_presence = Counter()
    for repo, value in repositories.items():
        for name, required in (
            ("test_and_other", ("test", "other")),
            ("docs_and_other", ("docs_example", "other")),
            ("test_docs_and_other", ("test", "docs_example", "other")),
        ):
            if all(value[role] for role in required):
                co_presence[name + "_repositories"] += 1
                co_presence[name + "_tokens"] += value["tokens"]
    total_tokens = sum(c["tokens"] for c in counts.values())
    result = {
        "training_admitted": False,
        "eligible_unique_tokens_established": 0,
        "documents": len(rows),
        "tokens_with_bos_eos": total_tokens,
        "by_language": {
            lang: {**counts[lang], "role_hints": dict(roles[lang])} for lang in sorted(counts)
        },
        "exact_text": {
            "distinct_texts": len(exact),
            "extra_copies": duplicate_copies,
            "extra_copy_tokens": duplicate_tokens,
            "distinct_text_tokens": total_tokens - duplicate_tokens,
        },
        "named_repositories": len(repositories),
        "named_repository_co_presence": dict(co_presence),
        "top_named_repositories_by_tokens": sorted(
            repositories.items(), key=lambda pair: (-pair[1]["tokens"], pair[0])
        )[:20],
        "known_benchmark_repository_holds": dict(held_counts),
        "source_revisions": sorted({row["source_revision"] for row in rows}),
        "scan": identity(output / "scan.json"),
        "qualification_inputs": plan["qualification_inputs"],
        "limitations": [
            "Complete retained finite acquisition, not the entire Stack-Edu release or a random sample.",
            "Path roles and generated-header wording are hints; other does not mean practical implementation.",
            "Repository-name co-presence does not establish same-commit linkage, runnable bundles or complete repositories.",
            "Exact equality and the existing aliases/hold names are only a partial family graph; no production partitions assigned.",
            "No new benchmark content scan, near-duplicate screen, license determination, execution or training admission.",
        ],
    }
    save(output / "summary.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "documents",
                    "tokens_with_bos_eos",
                    "exact_text",
                    "known_benchmark_repository_holds",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scan", "summarize"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.mode == "scan":
        scan(plan, args.output)
    summarize(plan, args.output)
