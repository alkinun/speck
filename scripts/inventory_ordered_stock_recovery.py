"""Verify completed units and archives of a stopped Stack-Edu acquisition, without downloads."""

import argparse
from pathlib import Path

from speck.data.ordered_stock_recovery import inventory_completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = inventory_completed(args.plan, args.result)
    print(f"Reopened {len(result['units'])} completed units and preserved archives", flush=True)


if __name__ == "__main__":
    main()
