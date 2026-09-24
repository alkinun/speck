"""Run the disk-backed global production text preprocessor."""

import argparse

from speck.data.production_data import load_preprocess_config, preprocess_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--index-directory",
        help="build the SQLite index on this (fast, local) disk and move it into the output "
        "at publication; the published result is identical",
    )
    parser.add_argument(
        "--per-shingle-minhash",
        action="store_true",
        help="build MinHash signatures one shingle at a time; bitwise identical to the "
        "default batched path and about 5.69x slower, for differential debugging only",
    )
    args = parser.parse_args()
    result = preprocess_sources(
        load_preprocess_config(args.config),
        restart=args.restart,
        batched_minhash=not args.per_shingle_minhash,
        index_directory=args.index_directory,
    )
    counts = result["manifest"]["counts"]
    print(
        f"Retained {counts.get('records_retained', 0):,} records; "
        f"removed {counts.get('records_seen', 0) - counts.get('records_retained', 0):,}."
    )


if __name__ == "__main__":
    main()
