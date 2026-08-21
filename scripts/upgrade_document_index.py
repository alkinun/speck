"""upgrade a persisted format-one packed dataset with a verified document index."""

import argparse
import json

from speck.search.upgrade_dataset import upgrade_document_index


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("data_dir")
    return value


def main():
    result = upgrade_document_index(parser().parse_args().data_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
