"""Create a v2 preparation config with a checked explicit SQLite policy and binding receipt."""

import argparse
import json
import os
from pathlib import Path

from speck.data.preparation_policy import bind_preparation_policy
from speck.provenance.io import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    receipt = args.output.with_name(args.output.name + ".binding.json")
    if args.output.exists() or receipt.exists():
        raise FileExistsError("preparation config and binding receipt must be new")
    config, binding = bind_preparation_policy(args.config, args.policy, args.output_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    binding["config"] = {"path": str(args.output.resolve()), "sha256": file_sha256(args.output)}
    with receipt.open("x") as handle:
        handle.write(json.dumps(binding, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(args.output.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(binding, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
