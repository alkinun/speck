"""Verify a bounded parser-independent held-out evaluation bundle."""

import argparse

from speck.heldout_evaluation import build_heldout_manifest, load_heldout_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    result = build_heldout_manifest(load_heldout_config(args.config))
    print(f"Verified {len(result['documents'])} paired held-out documents.")


if __name__ == "__main__":
    main()
