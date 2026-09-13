"""Publish recovered tokenizer endpoint reports from verified completed checkpoint artifacts."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root
from speck.tokenization.pilot_orchestration import load_execution_record, macro_bpb
from speck.tokenization.pilot_recovery import recover_terminal_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "execution", type=Path, help="Original execution record in its frozen checkout"
    )
    parser.add_argument("interruption", type=Path)
    parser.add_argument("report", type=Path, help="New checked recovery record")
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("report recovery requires a clean analysis implementation")
    execution = load_execution_record(args.execution)
    output = Path(execution["output_directory"])
    result_path, summary_path = output / "run-result.json", output / "run-summary.json"
    if args.report.exists() or result_path.exists() or summary_path.exists():
        raise FileExistsError("recovery outputs must be new; inspect any partial publication")
    interruption = json.loads(args.interruption.read_text())
    result, summary = recover_terminal_report(execution, interruption)
    durable_json(result_path, result)
    summary["result"] = {"path": result_path.name, "sha256": file_sha256(result_path)}
    durable_json(summary_path, summary)
    record = {
        "format": "speck_tokenizer_pilot_report_recovery",
        "format_version": 1,
        "status": "terminal_results_recovered_missing_memory_disclosed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "execution": execution["execution_record"],
        "interruption": {"path": str(args.interruption), "sha256": file_sha256(args.interruption)},
        "result": {"path": str(result_path), "sha256": file_sha256(result_path)},
        "summary": {"path": str(summary_path), "sha256": file_sha256(summary_path)},
        "run_fingerprint": result["recovery"]["run_fingerprint"],
        "metrics": {
            "fixed_document_macro_bpb": macro_bpb(result["fixed_document"]["categories"]),
            "fixed_flop_macro_bpb": macro_bpb(result["fixed_flop"]["categories"]),
            "active_seconds": summary["active_seconds"],
            "peak_memory_bytes": None,
        },
        "missing_measurements": result["missing_measurements"],
        "training_replayed": False,
        "evaluation_replayed": False,
        "D5_opening": False,
        "final_selection": False,
    }
    durable_json(args.report, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
