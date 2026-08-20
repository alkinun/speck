"""run and inspect architecture search studies."""

import argparse
import json
from pathlib import Path

import torch

from speck.common import base_dir
from speck.config import load_experiment
from speck.model import build_model
from speck.search.runner import (
    SearchSettings,
    prepare_study,
    run_search,
    run_worker,
    study_lock,
)
from speck.search.store import StudyStore
from speck.tokenizer import get_tokenizer


def arguments():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("experiment")
    run.add_argument("--study", required=True)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    for name in ("status", "frontier"):
        command = commands.add_parser(name)
        command.add_argument("study")
        command.add_argument("--output", default=None)

    lineage = commands.add_parser("lineage")
    lineage.add_argument("study")
    lineage.add_argument("candidate", type=int)
    lineage.add_argument("--output", default=None)

    worker = commands.add_parser("_evaluate")
    worker.add_argument("input")
    worker.add_argument("output")
    worker.add_argument("--device", required=True)
    return parser.parse_args()


def study_dir(name):
    return Path(base_dir()) / "search" / name


def display(value, output=None):
    text = json.dumps(value, indent=2, sort_keys=True)
    print(text)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def run_command(args):
    configs = load_experiment(
        args.experiment, "data", "tokenizer", "model", "train", "search"
    )
    settings = SearchSettings.from_dict(configs["search"])
    tokenizer = get_tokenizer(**configs["tokenizer"])
    with torch.device("meta"):
        baseline = build_model(
            configs["model"], tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id
        ).config
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("architecture search requires cuda for peak memory objectives")
    directory = study_dir(args.study)
    with study_lock(directory):
        store = StudyStore(directory / "study.sqlite3")
        try:
            prepare_study(
                store,
                args.experiment,
                configs,
                baseline,
                tokenizer,
                settings,
                device,
            )
            run_search(
                store,
                directory,
                baseline,
                configs["tokenizer"],
                settings,
                args.device,
            )
            display(store.summary())
        finally:
            store.close()


def query_command(args):
    database = study_dir(args.study) / "study.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(f"search study not found: {args.study}")
    store = StudyStore(database)
    try:
        if args.command == "status":
            value = store.summary()
        elif args.command == "frontier":
            value = [
                {
                    "id": candidate["id"],
                    "config": candidate["config"],
                    "objectives": candidate["result"]["objectives"],
                    "mutation": candidate["mutation"],
                    "parents": candidate["parents"],
                }
                for candidate in store.frontier()
            ]
        else:
            value = store.lineage(args.candidate)
        display(value, args.output)
    finally:
        store.close()


def main():
    args = arguments()
    if args.command == "_evaluate":
        success = run_worker(args.input, args.output, args.device)
        raise SystemExit(0 if success else 1)
    if args.command == "run":
        run_command(args)
    else:
        query_command(args)


if __name__ == "__main__":
    main()
