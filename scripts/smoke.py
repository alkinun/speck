"""Run the offline CPU data, training, exact-resume, and evaluation smoke workflow."""

import argparse
import json
import tempfile
from pathlib import Path

from speck.training.smoke import run_smoke


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="retain outputs in a new directory")
    args = parser.parse_args(argv)
    if args.output_dir:
        result = run_smoke(args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="speck-smoke-") as directory:
            result = run_smoke(Path(directory) / "run")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
