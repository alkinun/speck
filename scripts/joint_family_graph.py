"""Link retained stocks across sources into one exact/near-duplicate family graph."""

import argparse
import json
from pathlib import Path

from speck.data.joint_graph import build


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", help="joint family graph plan JSON")
    parser.add_argument("output", help="new output directory")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = build(json.loads(Path(args.plan).read_text()), args.output)
    print(
        json.dumps({key: report[key] for key in ("edges", "linked_tokens", "families")}, indent=2)
    )


if __name__ == "__main__":
    main()
