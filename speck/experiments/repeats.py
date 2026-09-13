"""Prepare repeat-seed experiments around an existing completed baseline."""

import argparse
import json
import os
import shutil
from pathlib import Path

from speck.config import load_experiment

_INHERITED_CONFIGS = ("data", "long_context", "model", "tokenizer")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_experiment", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="at least two new seeds; the source experiment supplies the baseline seed",
    )
    return parser.parse_args(argv)


def repeat_seeds(source_seed, seeds):
    """Validate two or more distinct seeds that differ from the baseline."""

    if not isinstance(source_seed, int) or isinstance(source_seed, bool):
        raise ValueError("source training seed must be an integer")
    if len(seeds) < 2:
        raise ValueError("noise-floor preparation requires at least two repeat seeds")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("repeat seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("repeat seeds must be unique")
    if source_seed in seeds:
        raise ValueError("repeat seeds must differ from the completed baseline seed")
    return tuple(seeds)


def _write_extension(path, parent, overrides=None):
    values = {"extends": os.path.relpath(parent, path.parent)}
    values.update(overrides or {})
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(source_experiment, output_dir, seeds):
    source = Path(source_experiment).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"noise-floor family already exists: {output}")
    configs = load_experiment(source, *_INHERITED_CONFIGS, "train")
    train = configs["train"]
    source_seed = train.get("seed", 42)
    seeds = repeat_seeds(source_seed, seeds)
    contract = {
        "format": "speck_seed_noise_floor",
        "format_version": 1,
        "source_experiment": str(source),
        "comparison": "training-seed repeats with identical model, packed data, and recipe",
        "baseline": {
            "run": train["run"],
            "seed": source_seed,
        },
        "repeat_seeds": list(seeds),
        "train_tokens": train["train_tokens"],
        "batch_tokens": train["batch_tokens"],
        "sequence_length": train["sequence_length"],
    }

    building = output.with_name(output.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        for seed in seeds:
            directory = building / f"seed-{seed}"
            directory.mkdir()
            for name in _INHERITED_CONFIGS:
                _write_extension(directory / f"{name}.json", source / f"{name}.json")
            _write_extension(
                directory / "train.json",
                source / "train.json",
                {
                    "device_batch_size": train["device_batch_size"],
                    "output_dir": None,
                    "run": f"{output.name}-seed-{seed}",
                    "seed": seed,
                    "wandb_group": output.name,
                },
            )
        (building / "noise_floor.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(building, output)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return contract


def main(argv=None):
    args = arguments(argv)
    contract = prepare(args.source_experiment, args.output_dir, args.seeds)
    total = 1 + len(contract["repeat_seeds"])
    print(f"Prepared {total}-seed noise floor under {args.output_dir}")


if __name__ == "__main__":
    main()
