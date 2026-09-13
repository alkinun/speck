"""Prepare the frozen English FineWiki stock and measure its reference-token capacity."""

import argparse
from pathlib import Path

from speck.data.reference_stock import load_reference_preparation
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
        loader=load_reference_preparation,
        category="reference",
        result_format="speck_reference_stock_preparation_result",
        resume=args.resume,
        boundary="Single complete English FineWiki shard under existing source use, inherited reader/security filters and full reference exclusion. Reference-token capacity only; final-tokenizer packing, joint background eligibility and launch manifests remain pending. Separate banks are not summed. Observed WAL envelope is 2 GiB, not a hard cap or production forecast.",
    )


if __name__ == "__main__":
    main()
