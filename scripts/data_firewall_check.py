"""Authorize declared files for one non-sealed firewall consumer."""

import argparse
import json

from speck.data_firewall import authorize_consumer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument(
        "role",
        choices=[
            "fixture_validation",
            "tokenizer_training",
            "tokenizer_static_evaluation",
            "data_selection",
            "model_training",
        ],
    )
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    print(json.dumps(authorize_consumer(args.manifest, args.role, args.paths), sort_keys=True))


if __name__ == "__main__":
    main()
