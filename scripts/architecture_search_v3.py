"""initialize and inspect version three architecture search studies."""

import argparse
import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from speck.common import base_dir
from speck.config import load_experiment
from speck.search.initialize_v3 import initialize_study
from speck.search.spec_v3 import V3SearchSettings
from speck.search.study_v3 import V3Study
from speck.tokenizer import get_tokenizer


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init")
    initialize.add_argument("experiment")
    initialize.add_argument("--study", required=True)
    initialize.add_argument("--config", default=None)
    initialize.add_argument("--data-dir", default=None)

    status = commands.add_parser("status")
    status.add_argument("study")
    status.add_argument("--output", default=None)
    return value


def study_dir(name):
    return Path(base_dir()) / "search-v3" / name


@contextmanager
def study_lock(directory):
    path = Path(directory) / "coordinator.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("study already has a running coordinator") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def display(value, output=None):
    text = json.dumps(value, indent=2, sort_keys=True)
    print(text)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def initialize_command(args):
    experiment = Path(args.experiment)
    config_path = Path(args.config or experiment / "search-v3.json")
    settings = V3SearchSettings.from_dict(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    configs = load_experiment(experiment, "data", "model", "tokenizer")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    directory = study_dir(args.study)
    with study_lock(directory):
        result = initialize_study(
            directory / "study.sqlite3",
            directory / "artifacts",
            settings,
            experiment=experiment,
            model_settings=configs["model"],
            tokenizer_settings=configs["tokenizer"],
            data_settings=configs["data"],
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            config_path=config_path,
        )
    display(result)


def status_command(args):
    path = study_dir(args.study) / "study.sqlite3"
    if not path.is_file():
        raise FileNotFoundError(f"v3 search study not found: {args.study}")
    study = V3Study(path, readonly=True)
    try:
        actions = study.actions()
        runs = study.runs()
        result = {
            "actions": {
                status: sum(action["status"] == status for action in actions)
                for status in ("pending", "running", "completed", "failed")
            },
            "events": len(study.events()),
            "runs": {
                status: sum(run["status"] == status for run in runs)
                for status in ("pending", "running", "paused", "completed", "failed")
            },
            "study": study.study(),
        }
    finally:
        study.close()
    display(result, args.output)


def main():
    args = parser().parse_args()
    if args.command == "init":
        initialize_command(args)
    else:
        status_command(args)


if __name__ == "__main__":
    main()
