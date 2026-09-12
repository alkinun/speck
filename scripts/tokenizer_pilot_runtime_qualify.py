"""Qualify corrected tokenizer-pilot data and resume cursors without model output."""

import argparse
import json
from pathlib import Path

from speck.io import atomic_json
from speck.tokenizer_pilot_qualification import qualify_corrected_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materialization")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"tokenizer pilot runtime qualification exists: {args.output}")
    result = qualify_corrected_runtime(args.materialization)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
