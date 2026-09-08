"""Write or check deterministic flagship scale-target accounting."""

import argparse
import json
from pathlib import Path

from speck.scale_targets import load_and_generate

ROOT = Path(__file__).parents[1]
DEFAULT_SPEC = ROOT / "research" / "flagship" / "targets" / "scale-targets-v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "flagship" / "targets" / "accounting-v1.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = load_and_generate(args.spec, ROOT)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text() != rendered:
            raise SystemExit(f"scale target accounting is stale: {args.output}")
        return
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
