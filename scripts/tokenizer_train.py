"""Train pinned custom tokenizers and prepare declared baseline tokenizers."""

import argparse

from speck.tokenizer_experiment import (
    load_experiment_config,
    prepare_baseline,
    train_candidate,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="tokenizer experiment JSON")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="candidate ID to train; repeat for multiple candidates (default: all)",
    )
    parser.add_argument(
        "--prepare-baselines",
        action="store_true",
        help="download and pin all declared baseline tokenizers",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard and replace incomplete candidate/baseline builds",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    candidate_ids = args.candidate or [candidate["id"] for candidate in config["candidates"]]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must not be repeated")
    for candidate_id in candidate_ids:
        manifest = train_candidate(config, candidate_id, restart=args.restart)
        print(
            f"Trained {candidate_id}: {manifest['model']['vocab_size']:,} pieces, "
            f"sha256={manifest['model']['sha256']}"
        )
    if args.prepare_baselines:
        for baseline in config["baselines"]:
            manifest = prepare_baseline(config, baseline["id"], restart=args.restart)
            print(
                f"Prepared {baseline['id']}: {manifest['model']['vocab_size']:,} pieces, "
                f"sha256={manifest['model']['sha256']}"
            )


if __name__ == "__main__":
    main()
