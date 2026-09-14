"""Prepare pinned complete Cosmopedia v2 shards with prompt lineage and full exclusion."""

import argparse
from pathlib import Path

from speck.data.cosmopedia_stock import load_cosmopedia_preparation
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
        loader=load_cosmopedia_preparation,
        category="synthetic",
        result_format="speck_cosmopedia_stock_preparation_result",
        resume=args.resume,
        boundary="Five complete pinned Cosmopedia v2 shards. Qualified generated-text filters, corrected full-prompt overlap and hash lineage, and full reference exclusion. Natural postfilter proportions without tokenizer-sampler template/domain byte quotas; concentration and repeated prompt hashes reported. Generator/seed revisions and original seed ancestry remain undisclosed. Target 800M plus 20% headroom; no model-quality, answer-correctness, final-view or training-authority claim. Invocation timings are not isolated hardware benchmarks.",
    )


if __name__ == "__main__":
    main()
