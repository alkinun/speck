"""Evaluate prepared tokenizer candidates on disjoint category-balanced text."""

import argparse

from speck.tokenizer_experiment import evaluate_tokenizers, load_experiment_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="tokenizer experiment JSON")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    report = evaluate_tokenizers(config)
    for result in report["tokenizers"]:
        print(
            f"{result['id']}: vocab={result['vocab_size']:,}, "
            f"macro tokens/KiB={result['macro']['tokens_per_kib']:.3f}, "
            f"embedding+head={result['embedding_and_head_parameters']:,} parameters"
        )
    print(f"Static report: {config['output_dir']}/evaluation.json")


if __name__ == "__main__":
    main()
