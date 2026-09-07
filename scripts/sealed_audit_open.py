"""Claim the sole opening of one frozen sealed-audit identity."""

import argparse
import json
from pathlib import Path

from speck.data_firewall import open_sealed_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("request")
    parser.add_argument("receipts_directory")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    print(
        json.dumps(
            open_sealed_audit(args.manifest, request, args.receipts_directory),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
