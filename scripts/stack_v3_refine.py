"""Apply pinned license, Gitleaks, English-prose, and tokenizer-yield gates to Stack v3."""

import argparse

from speck.stack_v3_refine import load_refinement_config, refine_stack_v3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Stack v3 refinement JSON")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard and rebuild an incomplete refinement output",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_refinement_config(args.config)
    report = refine_stack_v3(config, restart=args.restart)
    observed = report["downstream_partition"]["observed"]
    print(
        f"Refined {report['counts']['records_accepted']:,} files: "
        f"train={observed['train_bytes']:,} bytes, eval={observed['eval_bytes']:,} bytes"
    )
    print(f"Non-authoritative result: {config['output_directory']}/report.json")


if __name__ == "__main__":
    main()
