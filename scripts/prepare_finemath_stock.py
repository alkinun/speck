"""Prepare the pinned FineMath incumbent stock and report retained domain concentration."""

import argparse
from pathlib import Path

from speck.data.finemath_stock import load_finemath_preparation
from speck.data.source_stock import prepare_source_stock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prepare_source_stock(
        args.plan,
        args.report,
        loader=load_finemath_preparation,
        category="math",
        result_format="speck_finemath_stock_preparation_result",
        resume=args.resume,
        boundary="Eight complete pinned FineMath-4+ shards. Qualified metadata/per-document criteria plus common math-prose English and acquisition filters; tokenizer-sample host diversity cap deliberately not used for corpus selection. Post-exclusion domain concentration reported. Target 800M plus 20% headroom; joint background eligibility and experiment manifests remain pending. Timings and WAL peaks cover observed invocations only.",
    )


if __name__ == "__main__":
    main()
