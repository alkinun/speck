"""Issue one immutable flagship data launch preflight receipt."""

import argparse
import json
from pathlib import Path

from speck.data_launch import issue_launch_receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    args = parser.parse_args()
    path = Path(args.request).resolve()
    request = json.loads(path.read_text(encoding="utf-8"))
    receipt = issue_launch_receipt(request, config_dir=path.parent)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
