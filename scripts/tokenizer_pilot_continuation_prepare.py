"""Pack the tokenizer pilot's shared equal-FLOP continuation."""

import argparse

from speck.tokenizer_pilot_continuation import (
    load_continuation_plan,
    materialize_continuation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    result = materialize_continuation(load_continuation_plan(args.plan), restart=args.restart)
    continuation = result["continuation"]
    print(
        f"Materialized {continuation['actual_reference_tokens']:,} continuation reference tokens."
    )


if __name__ == "__main__":
    main()
