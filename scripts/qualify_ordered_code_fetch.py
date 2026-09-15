"""Qualify ordered NVMe blob fetching and estimate bounded pre-exclusion language yield."""

import argparse
import json
import math
import shutil
import statistics
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.code_eligible_index import build_eligible_index, sample_index
from speck.data.ordered_blob_fetch import fetch_targets
from speck.data.production_rehearsal import (
    _contamination_indexes,
    _document_rejection,
    _gitleaks_filter,
)
from speck.data.stack_edu_metadata import load_metadata_plan
from speck.data.stack_edu_stock import decode_code, load_stack_edu_preparation
from speck.data.swh_cache import read_cached_blob
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root
from speck.tokenization.tokenizer import Tokenizer


def estimate_language(targets, token_counts, language):
    observations = defaultdict(list)
    populations, sizes = {}, {}
    for ordinal, target in enumerate(targets):
        if target["language"] != language:
            continue
        group = (target["metadata_sha256"], target["stratum"])
        population, size = target["population"], target["sample_size"]
        if group in populations and (populations[group], sizes[group]) != (population, size):
            raise ValueError("code sample stratum population/size changed")
        populations[group], sizes[group] = population, size
        observations[group].append(token_counts.get(ordinal, 0))
    estimate = variance = 0
    for group, values in observations.items():
        n, population = len(values), populations[group]
        if n != sizes[group] or not 1 <= n <= population:
            raise ValueError("code sample does not cover its declared stratum")
        estimate += population * statistics.mean(values)
        if n > 1:
            variance += population**2 * (1 - n / population) * statistics.variance(values) / n
    return estimate, math.sqrt(variance), observations


def additional_metadata_units(spec, metadata, intake, plan):
    """Bind a probe to exactly the new files of its declared metadata generation."""
    version = spec["format_version"]
    start, count = {2: (11, 26), 3: (26, 28)}[version]
    if (
        intake["format_version"] != version
        or len(intake["units"]) != count
        or metadata.get("status") != "complete_metadata_verified_not_code_stock"
        or metadata.get("training_authority") is not False
        or metadata.get("inputs") != intake["inputs"]
        or [entry["unit"] for entry in metadata["files"]] != intake["units"]
        or spec.get("selected_unit_ids") != [unit["id"] for unit in intake["units"][start:]]
        or intake["inputs"]["code_languages"]["sha256"] != plan["code_languages"]["sha256"]
        or intake["inputs"]["source_qualification"]["sha256"]
        != plan["source_qualification"]["sha256"]
    ):
        raise ValueError(
            "expanded code probe requires the exact additional metadata files and policy"
        )
    return [
        {
            **entry["unit"],
            "metadata_path": entry["raw"]["path"],
            "start_row": 0,
            "stop_row": entry["unit"]["expected_file_rows"],
        }
        for entry in metadata["files"][start:]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = repository_root(__file__)
    if (
        args.report.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("qualification requires a new report and clean frozen implementation")
    spec = json.loads(args.plan.read_text())
    expanded = spec.get("format_version") in (2, 3)
    if (
        spec.get("format") != "speck_ordered_code_fetch_qualification"
        or type(spec.get("format_version")) is not int
        or spec.get("format_version") not in (1, 2, 3)
        or spec.get("training_authority") is not False
        or spec.get("strata") != 4
        or spec.get("samples_per_stratum") != 32
        or spec.get("seed") != 42
        or spec.get("workers") != 32
        or spec.get("window") != 64
        or spec.get("maximum_cache_bytes") != 12884901888
        or spec.get("maximum_index_bytes_per_language") != 536870912
        or spec.get("minimum_free_bytes") != 68719476736
    ):
        raise ValueError("unsupported ordered code qualification contract")
    parent_id = _bound_identity(spec["stock_plan"], args.plan.parent)
    plan = load_stack_edu_preparation(parent_id["path"])
    metadata_id = None
    if expanded:
        metadata_id = _bound_identity(spec["metadata_result"], args.plan.parent)
        metadata = json.loads(Path(metadata_id["path"]).read_text())
        metadata_plan_id = _bound_identity(metadata["plan"], Path(metadata_id["path"]).parent)
        intake = load_metadata_plan(metadata_plan_id["path"])
        plan["units"] = additional_metadata_units(spec, metadata, intake, plan)
    policy = plan["base"]["stack_edu_policy"]
    working = Path(spec["working_directory"]).resolve()
    archive = Path(spec["archive_directory"]).resolve()
    if working.exists() or archive.exists():
        raise FileExistsError("preserve earlier qualification outputs; do not restart in place")
    working.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(working.parent).free < spec["minimum_free_bytes"] + 21474836480:
        raise ValueError("qualification requires 20GiB working headroom above its free-space floor")
    working.mkdir()
    archive.mkdir(parents=True)
    execution = {
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "plan": {"path": str(args.plan.resolve()), "sha256": file_sha256(args.plan)},
        "stock_plan": parent_id,
    }
    if metadata_id is not None:
        execution["metadata_result"] = metadata_id
    durable_json(working / "execution.json", execution)
    indices = []
    targets = []
    started = time.perf_counter()
    for unit in plan["units"]:
        index = build_eligible_index(
            unit,
            policy,
            working / "indices" / unit["id"],
            maximum_index_bytes=spec["maximum_index_bytes_per_language"],
        )
        indices.append({"unit_id": unit["id"], "language": unit["language"], "index": index})
        samples = sample_index(
            index, per_stratum=spec["samples_per_stratum"], strata=spec["strata"], seed=spec["seed"]
        )
        targets.extend(
            {
                **row,
                "blob_id": row["metadata"]["blob_id"],
                "unit_id": unit["id"],
                "language": unit["language"],
                "metadata_sha256": unit["raw"]["sha256"],
            }
            for row in samples
        )
        print(
            f"indexed {unit['language']}: {index['eligible_rows']} eligible / {len(samples)} sampled",
            flush=True,
        )
    indexing_seconds = time.perf_counter() - started
    durable_json(working / "targets.json", targets)
    fetch_args = dict(
        cache=working / "cache",
        fallback=plan["blob_cache"],
        settings=policy["fetch"],
        workers=spec["workers"],
        window=spec["window"],
        maximum_working_bytes=spec["maximum_cache_bytes"],
        minimum_free_bytes=spec["minimum_free_bytes"],
    )
    started = time.perf_counter()
    if expanded:
        # Replay was qualified by v1. This execution measures newly pinned source supply.
        resumed = fetch_targets(targets, working / "resumed", **fetch_args)
        interrupted_and_resumed_seconds = time.perf_counter() - started
        warm_replay_seconds = None
    else:
        try:
            fetch_targets(targets, working / "resumed", **fetch_args, interrupt_after=64)
        except RuntimeError as error:
            if str(error) != "injected ordered fetch interruption":
                raise
            durable_json(working / "interruption.json", {"expected": True, "error": str(error)})
        else:
            raise RuntimeError("expected ordered fetch interruption did not occur")
        resumed = fetch_targets(targets, working / "resumed", **fetch_args, resume=True)
        interrupted_and_resumed_seconds = time.perf_counter() - started
        started = time.perf_counter()
        clean = fetch_targets(targets, working / "clean", **fetch_args)
        warm_replay_seconds = time.perf_counter() - started
        if resumed["journal"]["sha256"] != clean["journal"]["sha256"]:
            raise RuntimeError("ordered fetch clean/resume journal parity failed")
    if fetch_targets(targets, working / "resumed", **fetch_args, resume=True) != resumed:
        raise RuntimeError("completed ordered fetch changed on reopen")
    print("complete fetch reopen passed", flush=True)
    contamination = _contamination_indexes(plan["base"])
    tokenizer = Tokenizer(plan["reference_tokenizer"]["path"])
    rejections = defaultdict(Counter)
    records = working / "accepted-before-exclusion.jsonl"
    journals = [
        json.loads(line) for line in Path(resumed["journal"]["path"]).read_text().splitlines()
    ]
    with records.open("x") as handle:
        for ordinal, (target, journal) in enumerate(zip(targets, journals, strict=True)):
            raw, _ = read_cached_blob(
                Path(journal["manifest"]["path"]).parent, target["blob_id"], policy["fetch"]
            )
            language = target["language"]
            if raw is None:
                rejections[language]["blob_missing_404"] += 1
                continue
            reason, text, prose = decode_code(target["metadata"], raw, policy)
            if (
                reason is None
                and not plan["base"]["filtering"]["min_chars"]
                <= len(text)
                <= plan["base"]["filtering"]["max_chars"]
            ):
                reason = "code_character_envelope"
            if reason is None:
                reason = _document_rejection(
                    {"content": text, "metadata": {}}, plan["base"]["security"], contamination
                )
            if reason:
                rejections[language][reason] += 1
                continue
            record = {
                "text": text,
                "sample_ordinal": ordinal,
                "language": language,
                "stratum": target["stratum"],
                "source_row": target["source_row"],
                "eligible_ordinal": target["eligible_ordinal"],
                "metadata_sha256": target["metadata_sha256"],
                "metadata": target["metadata"],
                "blob_manifest": journal["manifest"],
                **prose,
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    before = [json.loads(line)["sample_ordinal"] for line in records.read_text().splitlines()]
    removed, security = _gitleaks_filter(
        records, plan["base"]["security"]["gitleaks_binary"]["path"], working / "security"
    )
    token_counts = {}
    retained = Counter()
    verified_bytes = Counter()
    with records.open() as handle:
        for line in handle:
            row = json.loads(line)
            token_counts[row["sample_ordinal"]] = len(
                tokenizer.encode(row["text"], bos=True, eos=True)
            )
            retained[row["language"]] += 1
            verified_bytes[row["language"]] += len(row["text"].encode())
    for ordinal in set(before) - set(token_counts):
        rejections[targets[ordinal]["language"]]["gitleaks"] += 1
    yields = []
    for language in plan["source_language_targets"]:
        estimate, standard_error, observations = estimate_language(targets, token_counts, language)
        if not observations:
            continue
        yields.append(
            {
                "language": language,
                "sampled_eligible_rows": sum(map(len, observations.values())),
                "retained_after_content_and_security": retained[language],
                "verified_retained_utf8_bytes": verified_bytes[language],
                "observed_mistral_tokens": sum(sum(v) for v in observations.values()),
                "rejections": dict(rejections[language]),
                (
                    "estimated_tokens_in_additional_files_before_full_exclusion"
                    if expanded
                    else "estimated_tokens_in_current_file_before_full_exclusion"
                ): estimate,
                "sampling_standard_error_tokens": standard_error,
                "e1s_language_headroom_target_tokens": plan["source_language_targets"][language]
                // 4,
            }
        )
    with records.open("rb") as handle:
        import os

        os.fsync(handle.fileno())
    preliminary = {
        "format": "speck_ordered_code_fetch_qualification_result",
        "format_version": spec["format_version"],
        "status": "bounded_fetch_replay_and_content_checks_pass_full_exclusion_pending",
        **execution,
        "working_directory": str(working),
        "indices": indices,
        "targets": {
            "path": str(working / "targets.json"),
            "sha256": file_sha256(working / "targets.json"),
        },
        "indexing_seconds": indexing_seconds,
        (
            "fetch_seconds" if expanded else "fetch_including_interruption_resume_seconds"
        ): interrupted_and_resumed_seconds,
        "warm_replay_seconds": warm_replay_seconds,
        "journal_parity_pass": None if expanded else True,
        "complete_reopen_pass": True,
        "fetch": resumed,
        "security": security,
        "security_removed": removed,
        "accepted_records": {"path": str(records), "sha256": file_sha256(records)},
        "language_yield": yields,
        "training_authority": False,
        "boundary": "128 metadata-eligible rows per language, sampled with fixed seed in four eligible-order strata from the currently pinned first file. Actual code/content/security/Gitleaks and Mistral checks; no full reference or candidate deduplication. Stratified estimates and sampling SE describe these files before full exclusion; they are not qualified capacity, cross-language independence or guaranteed targets. Fetch timing includes cache reuse, interruption, replay and persistence. Warm replay is not an independent network speed benchmark. Original cache and checkpoint remain preserved; no full-stock resume or recipe change.",
    }
    if expanded:
        preliminary["status"] = "additional_file_supply_probe_complete_full_exclusion_pending"
        preliminary["boundary"] = (
            "128 metadata-eligible rows per additional complete file in four equal eligible-order strata. "
            "Fixed seed 42 and unchanged source/content/security/Gitleaks/Mistral policy. Estimates sum "
            "independent file-position strata with their own population sizes; no original file is sampled "
            "again or included in the additional-file estimate. No full reference/candidate deduplication "
            "or guaranteed capacity. V1 qualified interruption/replay; this run only fetches new supply "
            "samples and verifies completed reopen. New and preserved caches/attempts remain intact. "
            "No full-stock resume, language/recipe change, source approval or training authority."
        )
    durable_json(working / "qualification.json", preliminary)
    files = [p for p in sorted(working.rglob("*")) if p.is_file()]
    inventory = [
        {"path": str(p.relative_to(working)), "bytes": p.stat().st_size, "sha256": file_sha256(p)}
        for p in files
    ]
    durable_json(working / "archive-inventory.json", inventory)
    files.append(working / "archive-inventory.json")
    bundle = archive / "qualification.tar"
    with tarfile.open(bundle, "x") as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(working)), recursive=False)
    with bundle.open("rb") as handle:
        os.fsync(handle.fileno())
    with tarfile.open(bundle, "r") as tar:
        expected = {row["path"]: row for row in inventory}
        expected["archive-inventory.json"] = {
            "sha256": file_sha256(working / "archive-inventory.json")
        }
        for member in tar.getmembers():
            import hashlib

            h = hashlib.sha256()
            with tar.extractfile(member) as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    h.update(chunk)
            if member.name not in expected or h.hexdigest() != expected.pop(member.name)["sha256"]:
                raise ValueError("archival copy payload mismatch")
        if expected:
            raise ValueError("archival copy lacks files")
    preliminary["archive"] = {
        "path": str(bundle),
        "sha256": file_sha256(bundle),
        "bytes": bundle.stat().st_size,
        "inventory": {
            "path": str(working / "archive-inventory.json"),
            "sha256": file_sha256(working / "archive-inventory.json"),
        },
        "all_archived_payloads_reopened_and_verified": True,
    }
    durable_json(args.report, preliminary)
    print(
        json.dumps(
            {
                "status": preliminary["status"],
                "retained_documents": sum(retained.values()),
                "tokens": sum(token_counts.values()),
                "fetch_seconds": interrupted_and_resumed_seconds,
                "warm_replay_seconds": warm_replay_seconds,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
