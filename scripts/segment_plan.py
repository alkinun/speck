"""build a document-aligned segment plan for calibrated search."""

import argparse
import json
from pathlib import Path

from speck.search.segments import build_segment_plan_from_dataset


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("data_dir")
    value.add_argument("output")
    value.add_argument("--data-seed", type=int, default=42)
    value.add_argument("--train-tokens", type=int, required=True)
    value.add_argument("--monitor-tokens", type=int, required=True)
    value.add_argument("--promotion-tokens", type=int, required=True)
    value.add_argument("--audit-tokens", type=int, required=True)
    value.add_argument("--final-tokens", type=int, required=True)
    return value


def create(args):
    plan = build_segment_plan_from_dataset(
        args.data_dir,
        args.data_seed,
        args.train_tokens,
        {
            "monitor": args.monitor_tokens,
            "promotion": args.promotion_tokens,
            "audit": args.audit_tokens,
            "final": args.final_tokens,
        },
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan.export(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return {
        "digest": plan.digest,
        "output": str(output.resolve()),
        "partitions": {
            partition.name: partition.tokens for partition in plan.partitions
        },
    }


def main():
    result = create(parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
