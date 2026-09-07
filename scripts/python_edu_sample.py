"""Build a bounded score-4+ Python-Edu source sample."""

import argparse

from speck.python_edu import load_python_edu_config, sample_python_edu


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Python-Edu sample plan JSON")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = sample_python_edu(load_python_edu_config(args.config), restart=args.restart)
    observed = report["downstream_partition"]["observed"]
    print(
        f"Sampled {report['counts']['records_sampled']:,} Python files: "
        f"{observed['train_bytes']:,} train bytes, {observed['eval_bytes']:,} eval bytes."
    )


if __name__ == "__main__":
    main()
