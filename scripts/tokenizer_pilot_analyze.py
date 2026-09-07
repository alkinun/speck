"""Analyze seven frozen tokenizer pilot run records without opening D5 audit."""

import argparse
import json
from pathlib import Path

from speck.tokenizer_pilot import analyze_tokenizer_pilot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("nominations")
    parser.add_argument("output")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"tokenizer pilot analysis already exists: {output}")
    result = analyze_tokenizer_pilot(
        json.loads(Path(args.plan).read_text()),
        json.loads(Path(args.nominations).read_text()),
        [json.loads(Path(path).read_text()) for path in args.runs],
        fixture=args.fixture,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
