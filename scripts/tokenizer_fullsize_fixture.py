"""Generate the deterministic six-category full-vocabulary tokenizer fixture."""

import argparse
import json

from speck.tokenizer_fixture import prepare_fullsize_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_fullsize_fixture(args.output_directory, restart=args.restart),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
