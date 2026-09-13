"""Qualify complete reference exclusion, production-cadence resume, and source-bank handoff."""

import argparse
import json
import resource
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from speck.data.acquisition_units import _bound_identity, load_unit_plan, prepare_units
from speck.data.firewall_integration import (
    CATEGORIES,
    analyze_exclusion,
    exclusion_config,
    group_acquisition_units,
    make_reference_controls,
    reference_sources,
    run_exclusion,
)
from speck.data.source_bank import prepare_source_bank
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--resume", action="store_true", help="Continue the same frozen implementation and runtime"
    )
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("firewall integration requires a clean checkout")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if args.report.exists():
        raise FileExistsError(args.report)
    path = args.plan.resolve()
    plan = json.loads(path.read_text())
    if (
        plan.get("format") != "speck_firewall_integration_rehearsal"
        or plan.get("format_version") != 1
        or plan.get("training_authority") is not False
        or plan.get("dedup_checkpoint_records") != 10000
        or plan.get("bank_target_utf8_bytes_per_category") != 16384
        or plan.get("interruption")
        != "first candidate after all reference records; resume from the complete reference checkpoint"
        or plan.get("controls")
        != "exact replay and one-token near replay of the first eligible web_unseen reference with at least 100 lexical tokens, uncapped tail and a shared MinHash band"
        or plan.get("bank_shortfall")
        != "record insufficient source capacity; do not search new windows or reduce quota"
    ):
        raise ValueError("unsupported firewall integration contract")
    acquisition_identity = _bound_identity(plan["acquisition_plan"], path.parent)
    acquisition_plan = load_unit_plan(acquisition_identity["path"])
    if acquisition_plan["dedup_checkpoint_records"] != 10000:
        raise ValueError("acquisition plan and integration checkpoint cadence differ")
    references = reference_sources(plan["firewall_plan"], path.parent)
    tokenizer = _bound_identity(plan["reference_tokenizer"], path.parent)
    output = Path(plan["output_directory"])
    execution = {
        "repository_revision": revision,
        "plan": {"path": str(path), "sha256": file_sha256(path)},
    }
    if output.exists():
        if not args.resume or json.loads((output / "execution.json").read_text()) != execution:
            raise ValueError(
                "existing integration runtime requires --resume at the frozen revision"
            )
    else:
        if args.resume:
            raise ValueError("cannot resume an absent integration runtime")
        output.mkdir(parents=True)
        durable_json(output / "execution.json", execution)
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"events": []}

    def event(stage, result):
        progress["events"].append(
            {
                "stage": stage,
                "at": datetime.now(timezone.utc).isoformat(),
                "result": result,
                "free_bytes": shutil.disk_usage(output).free,
                "main_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            }
        )
        durable_json(progress_path, progress)
        print(f"{stage}: {json.dumps(result, sort_keys=True)}", flush=True)

    event("started", {"reference_records": references["records"], "resume": args.resume})
    try:
        acquisition_report = output / "acquisition-report.json"
        if acquisition_report.exists():
            acquired = json.loads(acquisition_report.read_text())
        else:
            acquired = prepare_units(acquisition_plan, output / "acquired")
            durable_json(acquisition_report, acquired)
        event("acquired", {"elapsed_seconds": acquired["elapsed_seconds"]})
        grouped = group_acquisition_units(acquisition_plan, output / "acquired", output / "groups")
        controls = make_reference_controls(references, output / "controls")
        config = exclusion_config(
            acquisition_plan, references, output / "groups", controls, output / "excluded"
        )
        config_path = output / "exclusion-config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("resumed exclusion configuration changed")
        durable_json(config_path, config)
        interruption_path = output / "reference-interruption.json"
        if not interruption_path.exists():
            started = time.perf_counter()
            try:
                run_exclusion(config, crash_after_records=references["records"] + 1)
            except RuntimeError as error:
                if str(error) != "injected production preprocess crash":
                    raise
                seconds = time.perf_counter() - started
                state_path = output / "excluded.building/state.json"
                state = json.loads(state_path.read_text())
                if (
                    state["processed_records"] != references["records"]
                    or state["source_index"] != 12
                ):
                    raise RuntimeError("full reference checkpoint boundary mismatch")
                interruption = {
                    "elapsed_seconds": seconds,
                    "processed_records": state["processed_records"],
                    "accepted_references": state["next_doc_seq"],
                    "index_chain": state["index_chain"],
                    "checkpoint": {
                        "path": str(state_path),
                        "sha256_at_interruption": file_sha256(state_path),
                    },
                }
                # Retain the exact interrupted metadata even though resume will advance state.json.
                durable_json(output / "reference-checkpoint-state.json", state)
                interruption["retained_checkpoint"] = {
                    "path": str(output / "reference-checkpoint-state.json"),
                    "sha256": file_sha256(output / "reference-checkpoint-state.json"),
                }
                durable_json(interruption_path, interruption)
                event("reference_checkpoint_interrupted", interruption)
            else:
                raise RuntimeError("integration did not reach the declared reference interruption")
        else:
            interruption = json.loads(interruption_path.read_text())
        exclusion_report_path = output / "exclusion-report.json"
        if exclusion_report_path.exists():
            excluded = json.loads(exclusion_report_path.read_text())
        else:
            already_complete = (output / "excluded/manifest.json").exists()
            excluded = run_exclusion(config)
            excluded["completed_before_invocation"] = already_complete
            durable_json(exclusion_report_path, excluded)
        event("exclusion_complete", {"elapsed_seconds": excluded["elapsed_seconds"]})
        analysis = analyze_exclusion(output / "excluded", references)
        durable_json(output / "exclusion-analysis.json", analysis)
        target = plan["bank_target_utf8_bytes_per_category"]
        shortfalls = [
            category
            for category, value in analysis["retained"].items()
            if value["utf8_bytes"] < target
        ]
        bank = {"status": "insufficient_source_capacity", "shortfall_categories": shortfalls}
        if not shortfalls:
            bank_plan = {
                "format": "speck_bounded_source_bank_plan",
                "format_version": 2,
                "purpose": "engineering_rehearsal_not_training_data",
                "parent_manifest": analysis["parent_manifest"],
                "firewall_plan": references["firewall_plan"],
                "reference_tokenizer": tokenizer,
                "sources": [
                    {
                        "category": category,
                        "parent_source_id": f"acquired_train__{category}",
                        "target_utf8_bytes": target,
                    }
                    for category in CATEGORIES
                ],
                "checkpoint_records": 16,
                "shard_tokens": 1048576,
                "output_directory": str(output / "bank"),
            }
            bank_path = output / "bank-plan.json"
            durable_json(bank_path, bank_plan)
            bank_report_path = output / "bank-report.json"
            if bank_report_path.exists():
                bank = json.loads(bank_report_path.read_text())
            else:
                bank = prepare_source_bank(bank_path)
                durable_json(bank_report_path, bank)
            bank["status"] = "six_category_excluded_reference_bank_complete"
            bank["plan"] = {"path": str(bank_path), "sha256": file_sha256(bank_path)}
        event("bank_handoff", {"status": bank["status"], "shortfalls": shortfalls})
        report = {
            "format": "speck_complete_firewall_integration_qualification",
            "format_version": 1,
            "status": "complete_reference_exclusion_and_bank_handoff_pass"
            if not shortfalls
            else "reference_exclusion_pass_bank_capacity_failed",
            **execution,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "resumed_driver": args.resume,
            "acquisition": acquired,
            "grouped": grouped,
            "reference_controls": controls,
            "interruption": interruption,
            "exclusion": excluded,
            "analysis": analysis,
            "bank": bank,
            "progress": {"path": str(progress_path), "sha256": file_sha256(progress_path)},
            "timing_boundary": "Recorded invocations; full reference build is in interruption.elapsed_seconds. Costs before external driver termination may be incomplete with --resume. No second full uninterrupted reference build is claimed.",
            "evidence_boundary": "Full twelve-view reference exclusion under the declared exact/MinHash-candidate/verified-near policy, larger bounded raw windows, 10000-record checkpoints and reference-tokenizer bank mechanics. Not exhaustive all-pairs near matching, E1/E3 supply, final tokenizer selection, production-scale throughput, or training authority.",
            "operations_authority": False,
            "training_authority": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(args.report),
                    "retained": {k: v["records"] for k, v in analysis["retained"].items()},
                },
                indent=2,
            )
        )
    except BaseException as error:
        event("failed", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
