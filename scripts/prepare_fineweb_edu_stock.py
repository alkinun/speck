"""Prepare the pinned FineWeb-Edu incumbent stock and report retained domain concentration."""

import argparse
import subprocess
from pathlib import Path

from speck.data.fineweb_edu_stock import load_fineweb_edu_preparation
from speck.data.source_stock import prepare_source_stock
from speck.data.stock_raw import acquire_stock_raw
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    if args.raw_only:
        root = repository_root(__file__)
        if args.report.exists():
            raise FileExistsError(args.report)
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
            raise ValueError("raw acquisition requires a clean implementation")
        execution = {
            "repository_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "plan": {"path": str(args.plan.resolve()), "sha256": file_sha256(args.plan)},
        }
        result = acquire_stock_raw(
            load_fineweb_edu_preparation(args.plan), execution, resume=args.resume
        )
        durable_json(args.report, result)
        return
    prepare_source_stock(
        args.plan,
        args.report,
        loader=load_fineweb_edu_preparation,
        category="web",
        result_format="speck_fineweb_edu_stock_preparation_result",
        resume=args.resume,
        boundary="Complete sample/10BT files and capacity target selected by the hash-bound FineWeb-Edu plan at the approved revision. Frozen score>=3, valid released language confidence, independent English classification, size/prose/host/PII/security checks and full reference/candidate exclusion. Natural postfilter host distribution without the tokenizer-sampler 2MB host cap. Capacity is measured with frozen Mistral, not the upstream 10BT label. Overlapping successor stocks are not additive; no joint training-view or launch authority. Invocation timings include contention.",
    )


if __name__ == "__main__":
    main()
