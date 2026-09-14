"""Prepare the frozen peS2o science stock under the selected base tokenizer."""

import argparse
from pathlib import Path

from speck.data.science_stock import load_science_preparation
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
        loader=load_science_preparation,
        category="science",
        result_format="speck_science_stock_preparation_result",
        resume=args.resume,
        boundary="Complete qualified peS2o v3 shard with approved document licenses, English/OCR/text criteria and inherited benchmark/security checks. Full reference exclusion. Selected-tokenizer count only: joint background eligibility and packing remain pending. Timings and WAL peaks cover observed invocations, not lost attempts or production throughput.",
    )


if __name__ == "__main__":
    main()
