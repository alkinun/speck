"""Prepare matched Stack-Edu code stocks from completed metadata and retained SWH blobs."""

import argparse
from pathlib import Path

from speck.data.source_stock import prepare_source_stock
from speck.data.stack_edu_stock import count_code_tokens, load_stack_edu_preparation
from speck.data.stack_edu_units import prepare_stack_edu_units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prepare_source_stock(
        args.plan,
        args.report,
        loader=load_stack_edu_preparation,
        category="code",
        result_format="speck_stack_edu_stock_preparation_result",
        resume=args.resume,
        acquire=prepare_stack_edu_units,
        counter=count_code_tokens,
        boundary="Matched eleven-language Stack-Edu stock with retained SWH fetch attempts, qualified permissive file licenses, strict content identity, English-prose and security checks, and full reference exclusion. Each language independently requires 20% headroom. Whole-document candidate prefixes target twice nominal before Gitleaks/exclusion. Natural repository proportions replace the tokenizer-sampler repository cap. Path guards do not establish complete vendor/fork ancestry. No joint training-view or model-launch authority; timings overlap other preparation.",
    )


if __name__ == "__main__":
    main()
