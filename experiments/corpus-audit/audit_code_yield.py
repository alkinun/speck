"""Offline stratified preflight; never execute corpus code or infer training eligibility.

PYTHONPATH=. python experiments/corpus-audit/audit_code_yield.py PLAN.json FRESH_OUTPUT
"""

import argparse
import hashlib
import importlib.util
import json
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import sentencepiece as spm

from speck.data.code_families import repository_key
from speck.provenance.io import file_sha256


def identity(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def verified(item):
    path = Path(item["path"])
    if file_sha256(path) != item["sha256"]:
        raise ValueError(f"input identity mismatch: {path}")
    return path


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def select(rows, seed, per_stratum):
    """Equal-probability file sample within strata; keep failures in the original frame."""
    if not isinstance(seed, str) or not seed or type(per_stratum) is not int or per_stratum < 1:
        raise ValueError("nonempty seed and positive integer sample size required")
    counts, selected, seen = defaultdict(Counter), defaultdict(list), set()
    for row in rows:
        key = row["id"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("duplicate or invalid record identity")
        seen.add(key)
        if type(row["tokens"]) is not int or row["tokens"] < 2:
            raise ValueError("invalid token count")
        if row["role_hint"] not in {"vendor", "test", "docs_example", "other"}:
            raise ValueError("unknown census role")
        band = "le_4096" if row["tokens"] <= 4096 else "gt_4096"
        stratum = f"{row['language']}/{row['role_hint']}/{band}"
        counts[stratum].update(documents=1, tokens=row["tokens"])
        rank = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
        selected[stratum].append((rank, key, row))
        selected[stratum].sort(key=lambda item: item[:2])
        del selected[stratum][per_stratum:]
    strata, samples = {}, []
    for stratum in sorted(counts):
        n, population = len(selected[stratum]), counts[stratum]["documents"]
        strata[stratum] = {
            **counts[stratum],
            "sample_documents": n,
            "sample_tokens": sum(r["tokens"] for _, _, r in selected[stratum]),
            "inclusion_probability": n / population,
            "expansion_weight": population / n,
        }
        for rank, _, row in selected[stratum]:
            samples.append({**row, "stratum": stratum, "rank": rank})
    return strata, samples


def recover(samples, acquisition, tokenizer):
    """Read bound members directly; no tar extraction, imports or execution of source text."""
    groups = defaultdict(list)
    for row in samples:
        groups[row["id"].rsplit(":", 1)[0]].append(row)
    units = {u["manifest"]["unit_id"]: u for u in acquisition["units"]}
    records = []
    for unit_id, expected_rows in sorted(groups.items()):
        unit = units[unit_id]
        archive = unit["archive"]
        path = verified(archive["tar"])
        if path.stat().st_size != archive["tar"]["bytes"]:
            raise ValueError("archive size mismatch")
        inventory = {r["path"]: r for r in json.loads(verified(archive["inventory"]).read_text())}
        with tarfile.open(path) as tar:

            def member(name):
                data = tar.extractfile(name).read()
                if (
                    len(data) != inventory[name]["bytes"]
                    or hashlib.sha256(data).hexdigest() != inventory[name]["sha256"]
                ):
                    raise ValueError("archive member mismatch")
                return data

            manifest_bytes = member("unit/manifest.json")
            manifest = json.loads(manifest_bytes)
            if (
                manifest != unit["manifest"]
                or hashlib.sha256(manifest_bytes).hexdigest() != archive["unit_manifest_sha256"]
            ):
                raise ValueError("unit manifest mismatch")
            data = member("unit/" + manifest["output"]["path"])
            if hashlib.sha256(data).hexdigest() != manifest["output"]["sha256"]:
                raise ValueError("unit output mismatch")
        wanted = {r["id"]: r for r in expected_rows}
        for line in data.splitlines():
            row = json.loads(line)
            key = f"{unit_id}:{row['eligible_ordinal']}"
            if key not in wanted:
                continue
            expected = wanted.pop(key)
            text = row["text"]
            raw = text.encode()
            checks = (
                hashlib.sha256(raw).hexdigest()
                == expected["sha256"]
                == row["released_content_sha256"],
                hashlib.sha1(raw).hexdigest() == row["content_id"],
                len(raw) == expected["utf8_bytes"],
                len(tokenizer.encode(text)) + 2 == expected["tokens"],
                repository_key(row["repo_path"]) == expected["repository"],
                row["file_path"] == expected["path"],
                row["language"] == expected["language"],
                row["source_revision"] == expected["source_revision"],
                row["source_file"] == expected["source_file"],
                row["source_row"] == expected["source_row"],
            )
            if not all(checks):
                raise ValueError(f"selected source binding mismatch: {key}")
            records.append({**expected, "text": text})
        if wanted:
            raise ValueError("selected records missing from archive")
    return sorted(records, key=lambda r: (r["stratum"], r["rank"], r["id"]))


def held_closure(names, aliases):
    held = {repository_key(name) for name in names}
    pairs = [(repository_key(a), repository_key(b)) for a, b in aliases]
    while True:
        updated = held | {name for pair in pairs if held.intersection(pair) for name in pair}
        if updated == held:
            return held
        held = updated


def run(plan_path, output):
    plan = json.loads(plan_path.read_text())
    scan = json.loads(verified(plan["scan"]).read_text())
    acquisition = json.loads(verified(scan["acquisition"]).read_text())
    inputs = json.loads(verified(plan["qualification_inputs"]).read_text())
    policy = json.loads(verified(inputs["rules"]).read_text())
    prior = json.loads(verified(plan["prior_hold_assessment"]).read_text())
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with verified(scan["documents"]).open() as handle:
        strata, samples = select(
            (json.loads(line) for line in handle), plan["seed"], plan["per_stratum"]
        )
    expected = acquisition["progress"]["by_language_before_full_exclusion"]
    for language, value in expected.items():
        observed = [v for k, v in strata.items() if k.split("/")[0] == language]
        if (sum(v["documents"] for v in observed), sum(v["tokens"] for v in observed)) != (
            value["documents"],
            value["tokens"],
        ):
            raise ValueError("sample frame differs from census")
    if {k.split("/")[0] for k in strata} != set(expected):
        raise ValueError("sample frame languages differ from census")
    if len(samples) > plan["resource_bounds"]["maximum_selected_files"]:
        raise ValueError("sample exceeds protocol bound")
    archive_count = len({r["id"].rsplit(":", 1)[0] for r in samples})
    if archive_count > plan["resource_bounds"]["maximum_selected_archives"]:
        raise ValueError("archive count exceeds protocol bound")
    save(
        output / "selection.json",
        {"plan": identity(plan_path), "strata": strata, "samples": samples},
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(verified(scan["tokenizer"])))
    records = recover(samples, acquisition, tokenizer)
    save(output / "records.json", records)
    screen_path = Path(__file__).resolve().parents[1] / "main-data/check_qualification.py"
    spec = importlib.util.spec_from_file_location("qualification", screen_path)
    qualification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualification)
    print(
        f"Recovered {len(records)} files; screening {len(inputs['benchmarks'])} lanes", flush=True
    )
    counts = qualification.screen(records, inputs["benchmarks"], policy["content_exclusion"])
    prior_names = set(inputs["held_repositories"]) | set(
        prior["known_content_and_family_holds"]["by_named_repository"]
    )
    aliases = list(inputs["aliases"])
    for artifact in plan["additional_prior_screens"]:
        prior_screen = json.loads(verified(artifact).read_text())
        prior_names.update(
            r["repository"] for r in prior_screen["records"] if r["benchmark_match_counts"]
        )
        aliases.extend(prior_screen.get("aliases", []))
    previous = held_closure(prior_names, aliases)
    held = held_closure(previous | {r["repository"] for r in records if counts[r["id"]]}, aliases)
    results = [
        {
            "id": r["id"],
            "stratum": r["stratum"],
            "tokens": r["tokens"],
            "repository": r["repository"],
            "benchmark_match_counts": counts[r["id"]],
            "previous_named_family_hold": r["repository"] in previous,
            "named_family_hold_after_screen": r["repository"] in held,
            "natural_code_eligibility": "hold" if r["repository"] in held else "unresolved",
            "verified_exercise_eligibility": "not_assessed",
        }
        for r in records
    ]
    save(output / "screen.json", results)
    summary = {
        "status": "stratified_preflight_complete_eligibility_unresolved",
        "training_admitted": False,
        "eligible_yield_estimate": None,
        "plan": identity(plan_path),
        "script": identity(__file__),
        "screen_implementation": identity(screen_path),
        "selection": identity(output / "selection.json"),
        "records": identity(output / "records.json"),
        "screen": identity(output / "screen.json"),
        "population_documents": sum(v["documents"] for v in strata.values()),
        "population_tokens": sum(v["tokens"] for v in strata.values()),
        "strata": strata,
        "selected_files": len(records),
        "selected_archives": archive_count,
        "selected_tokens": sum(r["tokens"] for r in records),
        "content_flagged_files": sum(bool(r["benchmark_match_counts"]) for r in results),
        "named_family_held_files": sum(r["named_family_hold_after_screen"] for r in results),
        "benchmark_lanes": len(inputs["benchmarks"]),
        "benchmark_rows": sum(b["expected_tasks"] for b in inputs["benchmarks"]),
        "seconds": time.monotonic() - start,
        "gpu_hours": 0,
        "boundary": "Stratified sample of retained stock only. Content flags are conservative "
        "matches, not semantic leakage findings. No full-family graph, source-use/quality review, "
        "independent behavioral verification, natural-code admission or corpus-yield estimate.",
    }
    save(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "strata"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
