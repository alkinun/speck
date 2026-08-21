"""initialize and inspect version three architecture search studies."""

import argparse
import fcntl
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path

import torch

from speck.common import base_dir
from speck.config import load_experiment
from speck.profile.schema import ProfileScenario
from speck.search.coordinator_v3 import coordinate_bootstrap
from speck.search.evaluation_worker import run_evaluation_worker
from speck.search.initialize_v3 import initialize_study
from speck.search.profile_worker import backend_plugin, run_profile_worker
from speck.search.protocol import SeedBundle, TrainingProtocol, derive_seed
from speck.search.quality_worker import run_quality_worker
from speck.search.segments import load_segment_plan
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

    schedule = commands.add_parser("schedule-quality")
    schedule.add_argument("study")
    schedule.add_argument("--architecture", default=None)
    schedule.add_argument("--seed-index", type=int, required=True)
    schedule.add_argument("--numerical-repeat", type=int, default=0)
    schedule.add_argument("--priority", type=float, default=1.0)
    schedule.add_argument("--estimated-cost", type=float, required=True)

    schedule_profile = commands.add_parser("schedule-profile")
    schedule_profile.add_argument("study")
    schedule_profile.add_argument("--profile", required=True)
    schedule_profile.add_argument("--objective-set", default=None)
    schedule_profile.add_argument("--architecture", default=None)
    schedule_profile.add_argument("--priority", type=float, default=1.0)
    schedule_profile.add_argument("--estimated-cost", type=float, required=True)

    schedule_evaluation = commands.add_parser("schedule-evaluation")
    schedule_evaluation.add_argument("study")
    schedule_evaluation.add_argument("--run", type=int, required=True)
    schedule_evaluation.add_argument("--priority", type=float, default=1.0)
    schedule_evaluation.add_argument("--estimated-cost", type=float, required=True)

    worker = commands.add_parser("worker")
    worker.add_argument("study")
    worker.add_argument(
        "--owner",
        default=f"{socket.gethostname()}:{os.getpid()}",
    )
    worker.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    worker.add_argument("--lease-seconds", type=int, default=300)
    worker.add_argument("--once", action="store_true")

    profile_worker = commands.add_parser("profile-worker")
    profile_worker.add_argument("study")
    profile_worker.add_argument(
        "--owner",
        default=f"{socket.gethostname()}:{os.getpid()}",
    )
    profile_worker.add_argument("--backend", default="torch_native")
    profile_worker.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    profile_worker.add_argument("--lease-seconds", type=int, default=300)

    evaluation_worker = commands.add_parser("evaluation-worker")
    evaluation_worker.add_argument("study")
    evaluation_worker.add_argument(
        "--owner",
        default=f"{socket.gethostname()}:{os.getpid()}",
    )
    evaluation_worker.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    evaluation_worker.add_argument("--lease-seconds", type=int, default=300)

    coordinate = commands.add_parser("coordinate")
    coordinate.add_argument("study")
    coordinate.add_argument("--quality-cost", type=float, required=True)
    coordinate.add_argument("--evaluation-cost", type=float, required=True)
    coordinate.add_argument("--profile-cost", type=float, required=True)
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


def schedule_quality_command(args):
    path = study_dir(args.study) / "study.sqlite3"
    study = V3Study(path)
    try:
        stored = study.study()
        architecture_digest = (
            args.architecture or stored["provenance"]["model_digest"]
        )
        protocol = TrainingProtocol.from_dict(
            stored["provenance"]["resolved_protocol"]
        )
        seeds = SeedBundle.create(
            stored["config"]["seed"],
            args.seed_index,
            args.numerical_repeat,
        )
        run_id = study.add_run(architecture_digest, protocol, seeds)
        action_id = study.add_quality_action(
            run_id,
            args.priority,
            args.estimated_cost,
        )
    finally:
        study.close()
    display({"action_id": action_id, "run_id": run_id})


def worker_command(args):
    directory = study_dir(args.study)
    results = []
    while True:
        result = run_quality_worker(
            directory / "study.sqlite3",
            directory / "artifacts",
            owner=args.owner,
            device=args.device,
            lease_seconds=args.lease_seconds,
        )
        if result is None:
            break
        results.append(result)
        if args.once:
            break
    display({"completed": results})


def schedule_profile_command(args):
    path = study_dir(args.study) / "study.sqlite3"
    study = V3Study(path)
    try:
        stored = study.study()
        settings = V3SearchSettings.from_dict(stored["config"])
        templates = {profile.name: profile for profile in settings.profiles}
        if args.profile not in templates:
            raise ValueError(f"unknown profile template: {args.profile}")
        template = templates[args.profile]
        plugin = backend_plugin(template.backend)
        scenario = ProfileScenario(
            name=template.name,
            backend=plugin.identity,
            device=template.device,
            dtype=template.dtype,
            cache_dtype=template.cache_dtype,
            batch_size=template.batch_size,
            prompt_tokens=template.prompt_tokens,
            generated_tokens=template.generated_tokens,
            warmup_requests=template.warmup_requests,
            measured_requests=template.measured_requests,
            process_repetitions=template.process_repetitions,
        )
        objective_name = args.objective_set or template.name
        objective_sets = {
            objectives.name: objectives for objectives in settings.objective_sets
        }
        if objective_name not in objective_sets:
            raise ValueError(f"unknown objective set: {objective_name}")
        architecture_digest = (
            args.architecture or stored["provenance"]["model_digest"]
        )
        action_ids = []
        for repetition in range(scenario.process_repetitions):
            prompt_seed = derive_seed(
                settings.seed,
                "profile",
                architecture_digest,
                scenario.digest,
                repetition,
            )
            action_ids.append(
                study.add_profile_action(
                    architecture_digest,
                    scenario,
                    objective_sets[objective_name].digest,
                    repetition,
                    prompt_seed,
                    args.priority,
                    args.estimated_cost,
                )
            )
    finally:
        study.close()
    display({"action_ids": action_ids, "scenario_digest": scenario.digest})


def profile_worker_command(args):
    directory = study_dir(args.study)
    result = run_profile_worker(
        directory / "study.sqlite3",
        directory / "artifacts",
        owner=args.owner,
        device=args.device,
        backend=args.backend,
        lease_seconds=args.lease_seconds,
    )
    display({"completed": result})


def schedule_evaluation_command(args):
    path = study_dir(args.study) / "study.sqlite3"
    study = V3Study(path)
    try:
        stored = study.study()
        settings = V3SearchSettings.from_dict(stored["config"])
        run = study.run(args.run)
        plan = load_segment_plan(stored["provenance"]["segment_plan"]["path"])
        partition = next(
            (
                partition
                for partition in plan.partitions
                if partition.name == run["protocol"].evaluation_partition
            ),
            None,
        )
        if partition is None:
            raise ValueError("evaluation partition is not in the segment plan")
        objective_sets = tuple(
            objectives.digest
            for objectives in settings.objective_sets
            if any(
                objective.name == "quality.target_nll"
                for objective in objectives.objectives
            )
        )
        action_id = study.add_evaluation_action(
            args.run,
            objective_sets,
            partition.tokens - 1,
            args.priority,
            args.estimated_cost,
        )
    finally:
        study.close()
    display({"action_id": action_id, "run_id": args.run})


def evaluation_worker_command(args):
    directory = study_dir(args.study)
    result = run_evaluation_worker(
        directory / "study.sqlite3",
        directory / "artifacts",
        owner=args.owner,
        device=args.device,
        lease_seconds=args.lease_seconds,
    )
    display({"completed": result})


def coordinate_command(args):
    directory = study_dir(args.study)
    with study_lock(directory):
        study = V3Study(directory / "study.sqlite3")
        try:
            settings = V3SearchSettings.from_dict(study.study()["config"])
            result = coordinate_bootstrap(
                study,
                settings,
                quality_cost=args.quality_cost,
                evaluation_cost=args.evaluation_cost,
                profile_cost=args.profile_cost,
                artifact_root=directory / "artifacts",
            )
        finally:
            study.close()
    display(result)


def main():
    args = parser().parse_args()
    if args.command == "init":
        initialize_command(args)
    elif args.command == "status":
        status_command(args)
    elif args.command == "schedule-quality":
        schedule_quality_command(args)
    elif args.command == "worker":
        worker_command(args)
    elif args.command == "schedule-profile":
        schedule_profile_command(args)
    elif args.command == "profile-worker":
        profile_worker_command(args)
    elif args.command == "schedule-evaluation":
        schedule_evaluation_command(args)
    elif args.command == "evaluation-worker":
        evaluation_worker_command(args)
    else:
        coordinate_command(args)


if __name__ == "__main__":
    main()
