"""Pack a ladder experiment's corpus from preprocessed sources and their family partition."""

import argparse
import json

from speck.data.ladder import prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--inputs", default="experiments/ladder/inputs.json")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.experiment, args.inputs), indent=2))


if __name__ == "__main__":
    main()
