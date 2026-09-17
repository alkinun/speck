"""Check finite real-data loading and replay; torchrun can supply multiple independent ranks."""

import argparse

from speck.data.loader_check import check_loader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--mode", choices=("scan", "replay"), default="scan")
    args = parser.parse_args(argv)
    result = check_loader(args.experiment, args.output_dir, batches=args.batches, mode=args.mode)
    print(f"Rank {result['contract']['rank']}: {result['status']}")


if __name__ == "__main__":
    main()
