"""Verify all blocked real tokenizer input identities without materializing them."""

import argparse
import json

from speck.tokenizer_inputs import load_tokenizer_inputs_freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    value = load_tokenizer_inputs_freeze(args.manifest, verify_files=True)
    print(
        json.dumps(
            {
                "status": value["status"],
                "sources": sum(len(category["inputs"]) for category in value["categories"]),
                "training_authority": value["training_authority"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
