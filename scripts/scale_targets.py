"""Write or check deterministic flagship scale-target accounting."""

import argparse
import json
from pathlib import Path

from speck.model.accounting import load_and_generate
from speck.provenance.repository import repository_root

ROOT = repository_root()
TARGETS = ROOT / "archive/pregrant-history/research/flagship/targets"
DEFAULT_SPEC = TARGETS / "scale-targets-v2.json"
DEFAULT_OUTPUT = TARGETS / "accounting-v2.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--output", type=Path, help="new accounting output; required unless --check"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        if not args.check:
            parser.error("--output is required when generating a new accounting record")
        args.output = DEFAULT_OUTPUT
    result = load_and_generate(args.spec, ROOT)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text() != rendered:
            raise SystemExit(f"scale target accounting is stale: {args.output}")
        return
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
