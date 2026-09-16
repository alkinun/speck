"""Publish or exactly reopen a source-specific base demand/exposure report."""

import argparse
import json
from pathlib import Path

from speck.data.base_supply import analyze, render_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze(args.manifest)
    outputs = {
        "report.json": json.dumps(result, sort_keys=True, indent=2) + "\n",
        "table.md": render_markdown(result),
    }
    if not args.check:
        args.output_directory.mkdir(parents=True, exist_ok=False)
    for name, value in outputs.items():
        path = args.output_directory / name
        if args.check:
            if path.read_text() != value:
                raise ValueError("base supply report changed; publish a successor")
        else:
            path.write_text(value)
    print("Verified base supply report" if args.check else "Published base supply report")


if __name__ == "__main__":
    main()
