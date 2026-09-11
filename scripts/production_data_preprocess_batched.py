"""Run production preprocessing with the qualified exact-equivalent batched MinHash path."""

import argparse

import speck.production_data as production_data
from speck.production_data import load_preprocess_config, preprocess_sources


def _batched_signature(shingles, num_perm, seed):
    from datasketch import MinHash

    value = MinHash(num_perm=num_perm, seed=seed)
    value.update_batch(shingles)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    original = production_data._signature
    production_data._signature = _batched_signature
    try:
        result = preprocess_sources(load_preprocess_config(args.config), restart=args.restart)
    finally:
        production_data._signature = original
    counts = result["manifest"]["counts"]
    print(
        f"Retained {counts.get('records_retained', 0):,} records; "
        f"removed {counts.get('records_seen', 0) - counts.get('records_retained', 0):,}."
    )


if __name__ == "__main__":
    main()
