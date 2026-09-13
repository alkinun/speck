"""Prepare admitted Math L2 text, exclude all firewall references, and measure usable capacity."""

import argparse
from pathlib import Path

from speck.data.admitted_math import load_math_preparation
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
        loader=load_math_preparation,
        category="math",
        result_format="speck_admitted_math_preparation_result",
        resume=args.resume,
        boundary="Admitted natural Math L2, fixed two-shard text preparation and declared reference exclusion. Reference-token counts are not final D5 packing or training authority. New larger source transactions are measured under a 2GiB observed-WAL envelope; this is not a hard WAL cap or a speed comparison.",
    )


if __name__ == "__main__":
    main()
