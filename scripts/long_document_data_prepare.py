"""Derive a filtered long-document packed dataset from an existing experiment."""

import argparse
from pathlib import Path

from speck.config import load_experiment
from speck.dataset import resolve_data_dir
from speck.long_data import derive_long_document_dataset


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    configs = load_experiment(args.experiment, "data", "long_data")
    settings = configs["long_data"]
    source_experiment = (args.experiment / settings["source_experiment"]).resolve()
    source_data = load_experiment(source_experiment, "data")["data"]
    manifest = derive_long_document_dataset(
        resolve_data_dir(source_data.get("output_dir"), source_data.get("output_name")),
        resolve_data_dir(configs["data"].get("output_dir"), configs["data"].get("output_name")),
        source_weights=settings["source_weights"],
        requested_train_tokens=settings["requested_train_tokens"],
        validation_tokens_per_source=settings["validation_tokens_per_source"],
        minimum_document_tokens=settings["minimum_document_tokens"],
        shard_tokens=settings["shard_tokens"],
        maximum_loader_microbatch_tokens=settings["maximum_loader_microbatch_tokens"],
        restart=args.restart,
    )
    print(
        f"Prepared {manifest['splits']['train']['tokens']:,} long-document train tokens "
        f"at {resolve_data_dir(configs['data'].get('output_dir'), configs['data'].get('output_name'))}"
    )


if __name__ == "__main__":
    main()
