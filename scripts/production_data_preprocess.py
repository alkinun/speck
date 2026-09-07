"""Run the disk-backed global production text preprocessor."""

import argparse

from speck.production_data import load_preprocess_config, preprocess_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    result = preprocess_sources(load_preprocess_config(args.config), restart=args.restart)
    counts = result["manifest"]["counts"]
    print(
        f"Retained {counts.get('records_retained', 0):,} records; "
        f"removed {counts.get('records_seen', 0) - counts.get('records_retained', 0):,}."
    )


if __name__ == "__main__":
    main()
