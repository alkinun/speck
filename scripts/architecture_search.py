"""run and inspect architecture search studies."""

import argparse
import json
import sqlite3
from pathlib import Path

import torch

from speck.common import base_dir
from speck.config import load_experiment
from speck.model import build_model
from speck.search.runner import (
    prepare_study,
    run_search,
    run_worker,
    study_lock,
)
from speck.search.scheduler import rung_frontier
from speck.search.spec import SearchSettings
from speck.search.store import StudyStore
from speck.search.study import SearchStudy
from speck.tokenizer import get_tokenizer


def arguments():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("experiment")
    run.add_argument("--study", required=True)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    status = commands.add_parser("status")
    status.add_argument("study")
    status.add_argument("--output", default=None)

    frontier = commands.add_parser("frontier")
    frontier.add_argument("study")
    frontier.add_argument("--rung", type=int, default=None)
    frontier.add_argument("--output", default=None)

    lineage = commands.add_parser("lineage")
    lineage.add_argument("study")
    lineage.add_argument("candidate", type=int)
    lineage.add_argument("--output", default=None)

    worker = commands.add_parser("_evaluate")
    worker.add_argument("input")
    worker.add_argument("output")
    worker.add_argument("--device", required=True)
    worker.add_argument("--start-gate", default=None)
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
        store = SearchStudy(directory / "study.sqlite3")
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
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro", uri=True
    )
    try:
        legacy = connection.execute(
            "select 1 from sqlite_master where type = 'table' and name = 'candidates'"
        ).fetchone()
    finally:
        connection.close()
    if legacy:
        if args.command == "frontier" and args.rung is not None:
            raise ValueError("legacy studies do not contain rungs")
        store = StudyStore(database, readonly=True)
        try:
            if args.command == "status":
                value = store.summary()
            elif args.command == "frontier":
                value = [
                    {
                        "id": item["id"],
                        "config": item["config"],
                        "objectives": item["result"]["objectives"],
                        "mutation": item["mutation"],
                        "parents": item["parents"],
                    }
                    for item in store.frontier()
                ]
            else:
                value = store.lineage(args.candidate)
        finally:
            store.close()
    else:
        store = SearchStudy(database, readonly=True)
        try:
            if args.command == "status":
                value = store.summary()
            elif args.command == "frontier":
                settings = SearchSettings.from_dict(store.study()["config"])
                value = rung_frontier(store, settings, args.rung)
            else:
                value = store.lineage(args.candidate)
        finally:
            store.close()
    display(value, args.output)


def main():
    args = arguments()
    if args.command == "_evaluate":
        success = run_worker(
            args.input, args.output, args.device, args.start_gate
        )
        raise SystemExit(0 if success else 1)
    if args.command == "run":
        run_command(args)
    else:
        query_command(args)


if __name__ == "__main__":
    main()
