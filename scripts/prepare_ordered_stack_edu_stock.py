"""Exclude complete finite ordered Stack-Edu acquisition without downloading its content again."""

import argparse
from pathlib import Path

from speck.data.ordered_stock_exclusion import imported_units, load_ordered_exclusion
from speck.data.source_stock import prepare_source_stock
from speck.data.stack_edu_stock import count_code_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prepare_source_stock(
        args.plan,
        args.report,
        loader=load_ordered_exclusion,
        category="code",
        result_format="speck_stack_edu_stock_preparation_result",
        resume=args.resume,
        acquire=imported_units,
        counter=count_code_tokens,
        boundary="Full reference/candidate exclusion of the unchanged finite ordered acquisition, with original per-language headroom measured after exclusion. No new network acquisition or filter relaxation. Original raw, attempts, unscanned text and archives remain at their hash-bound owners; copied inputs are not extra unique supply. Historical E1S quotas identify this preparation and do not reinstate retired experiments or substitute Stack-Edu for selected restricted Stack v3. No joint-view or model-launch authority.",
    )


if __name__ == "__main__":
    main()
