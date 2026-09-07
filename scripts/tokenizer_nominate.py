"""Apply the frozen static tokenizer nomination policy."""

import argparse
import json

from speck.tokenizer_nomination import nominate_from_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation")
    parser.add_argument("policy")
    parser.add_argument("output")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            nominate_from_files(args.evaluation, args.policy, args.output, fixture=args.fixture),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
