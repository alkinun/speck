"""Materialize the fixed whole-document tokenizer-pilot stream and packs."""

import argparse

from speck.tokenizer_pilot_data import load_pilot_stream_plan, materialize_pilot_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    result = materialize_pilot_stream(load_pilot_stream_plan(args.plan), restart=args.restart)
    stream = result["document_stream"]
    print(
        f"Materialized {stream['actual_reference_tokens']:,} reference tokens "
        f"across {sum(x['documents'] for x in stream['categories']):,} documents."
    )


if __name__ == "__main__":
    main()
