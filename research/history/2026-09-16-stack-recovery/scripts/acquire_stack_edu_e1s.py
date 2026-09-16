"""Acquire finite, archived Stack-Edu E1S units in source order, without full exclusion."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from speck.data.acquisition_units import _digest
from speck.data.production_rehearsal import _contamination_indexes
from speck.data.stack_edu_ordered_stock import index_batches, load_ordered_stock, verify_index_file
from speck.data.stack_edu_ordered_units import acquire_batch, archive_batch
from speck.data.stock_blob_store import StockBlobStore, directory_bytes
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root
from speck.tokenization.tokenizer import Tokenizer


def run_units(plan, output, archive, store, contamination, tokenizer, *, pause_after_units=None):
    """Replay completed units by exact identity, then advance the deterministic prefix."""
    started = time.perf_counter()
    reports, languages = [], {}
    # Scan the owned working tree once. Charge each new attempt's actual size thereafter.
    non_cache_bytes = directory_bytes(output) - store.used
    newly_completed = 0
    progress = {"complete_units": 0, "by_language_before_full_exclusion": languages}
    for language in plan["language_order"]:
        target = plan["targets"][language]
        candidate_target = target["nominal_tokens"] * plan["candidate_nominal_multiplier"]
        totals = {"documents": 0, "tokens": 0, "pre_gitleaks_candidate_tokens": 0}
        languages[language] = totals
        for entry in plan["files"]:
            if entry["unit"]["language"] != language:
                continue
            batches = index_batches(
                entry, plan["base"]["stack_edu_policy"], plan["eligible_rows_per_unit"]
            )
            for batch in batches:
                remaining = candidate_target - totals["pre_gitleaks_candidate_tokens"]
                if remaining <= 0:
                    break
                unit = {
                    **entry["unit"],
                    "id": entry["unit"]["id"] + f"__eligible_{batch[0]['eligible_ordinal']:09d}",
                    "category": "code",
                    "index_identity": entry["index_identity"],
                    "targets_sha256": _digest(batch),
                    "reference_tokenizer": plan["reference_tokenizer"],
                    "candidate_tokens_remaining": remaining,
                    "start_row": batch[0]["source_row"],
                    "stop_row": batch[-1]["source_row"] + 1,
                }
                directory = output / "acquired" / unit["id"]
                completed = (directory / "manifest.json").exists()
                before = directory_bytes(directory)
                # JSON escaping and two retained text copies fit within this conservative
                # per-record reservation. Cache retries have an independent live reservation.
                content_bound = len(batch) * (
                    16 * plan["base"]["stack_edu_policy"]["fetch"]["maximum_blob_bytes"] + 65536
                )
                working_bound = non_cache_bytes + plan["maximum_cache_bytes"] + content_bound
                if not completed and (
                    working_bound > plan["maximum_working_bytes"]
                    or shutil.disk_usage(output).free < plan["minimum_free_bytes"] + content_bound
                ):
                    raise ValueError("ordered acquisition working/free-space bound reached")
                manifest = acquire_batch(
                    plan, unit, batch, output / "acquired", store, contamination, tokenizer
                )
                # A verified sequential archive is durable before the next network batch.
                archived = archive_batch(directory, manifest, archive / "units" / unit["id"])
                non_cache_bytes += directory_bytes(directory) - before
                totals["documents"] += manifest["retained_records"]
                totals["tokens"] += manifest["tokens_before_full_exclusion"]
                totals["pre_gitleaks_candidate_tokens"] += manifest["pre_gitleaks_candidate_tokens"]
                reports.append(
                    {
                        "unit": unit,
                        "manifest": manifest,
                        "manifest_identity": {
                            "path": str(directory / "manifest.json"),
                            "sha256": file_sha256(directory / "manifest.json"),
                        },
                        "archive": archived,
                    }
                )
                newly_completed += not completed
                progress = {
                    "state": "acquiring",
                    "complete_units": len(reports),
                    "current_language": language,
                    "retained_records": sum(row["documents"] for row in languages.values()),
                    "tokens_before_full_exclusion": sum(
                        row["tokens"] for row in languages.values()
                    ),
                    "by_language_before_full_exclusion": languages,
                    "cache_bytes": store.used,
                    "elapsed_seconds_this_invocation": time.perf_counter() - started,
                    "full_exclusion_performed": False,
                }
                durable_json(output / "progress.json", progress)
                print(
                    json.dumps(
                        {
                            k: progress[k]
                            for k in (
                                "complete_units",
                                "current_language",
                                "tokens_before_full_exclusion",
                            )
                        }
                    ),
                    flush=True,
                )
                if pause_after_units is not None and newly_completed >= pause_after_units:
                    progress["state"] = "paused_after_requested_completed_units"
                    durable_json(output / "progress.json", progress)
                    return None
            if totals["pre_gitleaks_candidate_tokens"] >= candidate_target:
                break
        totals.update(
            {
                "nominal_tokens": target["nominal_tokens"],
                "preparation_target_tokens": target["preparation_target_tokens"],
                "candidate_target_tokens": candidate_target,
                "candidate_target_pass": totals["pre_gitleaks_candidate_tokens"]
                >= candidate_target,
                "pre_exclusion_headroom_pass": totals["tokens"]
                >= target["preparation_target_tokens"],
                "complete_declared_prefix_or_metadata_exhaustion": True,
            }
        )
        if not totals["pre_exclusion_headroom_pass"]:
            break
    short = [name for name, row in languages.items() if not row["pre_exclusion_headroom_pass"]]
    progress.update(
        {
            "state": "pre_exclusion_language_shortfall"
            if short
            else "content_acquisition_complete",
            "by_language_before_full_exclusion": languages,
            "shortfall_languages": short,
            "unprocessed_languages": [
                name for name in plan["language_order"] if name not in languages
            ],
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
        }
    )
    durable_json(output / "progress.json", progress)
    return {"units": reports, "progress": progress}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after-units", type=int)
    args = parser.parse_args()
    if args.pause_after_units is not None and args.pause_after_units <= 0:
        raise ValueError("pause limit must be positive")
    root = repository_root(__file__)
    if (
        args.result.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("requires new result and clean frozen implementation")
    if not os.path.ismount("/mnt/speck-data"):
        raise ValueError("data drive is not mounted")
    plan = load_ordered_stock(args.plan)
    output, archive = Path(plan["working_directory"]), Path(plan["archive_directory"])
    execution = {
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "plan": {"path": str(args.plan.resolve()), "sha256": file_sha256(args.plan)},
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = output / "execution.json"
        if owner.exists():
            if not args.resume or json.loads(owner.read_text()) != execution:
                raise ValueError("resume requires identical frozen execution")
        else:
            if args.resume or set(p.name for p in output.iterdir()) != {"owner.lock"}:
                raise ValueError("cannot claim unowned output or resume absent acquisition")
            durable_json(owner, execution)
        archive.mkdir(parents=True, exist_ok=True)
        archive_owner = archive / "execution.json"
        if archive_owner.exists():
            if json.loads(archive_owner.read_text()) != execution:
                raise ValueError("archive execution differs")
        elif list(archive.iterdir()):
            raise ValueError("archive has no bound owner")
        else:
            durable_json(archive_owner, execution)
        attempt = output / f"invocation-{len(list(output.glob('invocation-*'))):05d}.json"
        invocation = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            **execution,
            "resume": args.resume,
            "pause_after_units": args.pause_after_units,
        }
        durable_json(attempt, {**invocation, "status": "started"})
        try:
            for index, entry in enumerate(plan["files"]):
                verify_index_file(entry)
                print(
                    f"verified metadata and eligible index {index + 1}/{len(plan['files'])}",
                    flush=True,
                )
            store = StockBlobStore(
                output / "cache",
                plan["fallback_caches"],
                plan["base"]["stack_edu_policy"]["fetch"],
                maximum_bytes=plan["maximum_cache_bytes"],
                minimum_free_bytes=plan["minimum_free_bytes"],
            )
            result = run_units(
                plan,
                output,
                archive,
                store,
                _contamination_indexes(plan["base"]),
                Tokenizer(plan["reference_tokenizer"]["path"]),
                pause_after_units=args.pause_after_units,
            )
            if result is not None:
                durable_json(
                    args.result,
                    {
                        "format": "speck_stack_edu_ordered_acquisition_result",
                        "format_version": 1,
                        "status": result["progress"]["state"],
                        **execution,
                        **result,
                        "source_use": plan["source_use"]["identity"],
                        "reference_tokenizer": plan["reference_tokenizer"],
                        "inputs": plan["inputs"],
                        "acquired_directory": str(output / "acquired"),
                        "training_authority": False,
                        "full_exclusion_performed": False,
                        "boundary": plan["scope"],
                    },
                )
            durable_json(attempt, {**invocation, "status": "complete" if result else "paused"})
        except BaseException as error:
            durable_json(
                attempt,
                {**invocation, "status": "failed_preserved", "error_type": type(error).__name__},
            )
            raise


if __name__ == "__main__":
    main()
