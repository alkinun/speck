"""Qualify Paper 1 finalist configs, data windows, and storage without launching them."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.helmet_data_acquire import qualify_volume
from speck.config import load_experiment
from speck.dataloader import loader_state_for_offset, manifest_fingerprint, packed_loader
from speck.dataset import load_manifest, resolve_data_dir
from speck.paper_finalist import file_sha256, materialize_finalist
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--volume-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def token_sha256(value):
    array = value.detach().cpu().contiguous().numpy()
    return bytes_sha256(array.tobytes())


def repository_revision():
    return subprocess.run(
        ["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_object(path):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return path, value


def unique_windows(pairs, training_tokens):
    by_offset = {}
    for pair in pairs:
        start = pair["data_token_offset"]
        end = pair["end_token_offset"]
        if end - start != training_tokens:
            raise ValueError("finalist data window length changed")
        by_offset.setdefault(start, {"start": start, "end": end, "seeds": []})["seeds"].append(
            pair["seed"]
        )
    windows = [by_offset[offset] for offset in sorted(by_offset)]
    for window in windows:
        window["seeds"].sort()
        if window["seeds"] != [42, 43, 44]:
            raise ValueError("finalist data window is not crossed with every seed")
    for left, right in zip(windows, windows[1:]):
        if left["end"] > right["start"]:
            raise ValueError("finalist distinct data windows overlap")
    return windows


def window_points(window, training_tokens):
    if training_tokens % 4:
        raise ValueError("finalist training horizon does not have exact quartiles")
    quarter = training_tokens // 4
    return [window["start"] + index * quarter for index in range(5)]


def _external_outputs(repository_root, contract):
    identity = contract["identity"]
    return {
        "checkpoint_root": Path(identity["checkpoint_root"]),
        "result_root": repository_root / identity["result_root"],
        "analysis_result": repository_root / identity["analysis_result"],
        "time_to_quality_lock": repository_root / identity["time_to_quality_lock"],
    }


def _inputs(args):
    repository_root = Path(__file__).parents[1]
    contract_path, contract = _load_object(args.contract)
    materialization_path, materialization = _load_object(args.materialization)
    expected_path = repository_root / contract["identity"]["output_root"] / "finalist_materialization.json"
    expected_manifest = materialize_finalist(contract_path, check=True)
    if (
        materialization_path != expected_path.resolve()
        or materialization != expected_manifest
        or materialization.get("training_authorized") is not False
    ):
        raise ValueError("finalist materialization input changed")
    training = contract["materialized_training"]
    experiment_root = materialization_path.parent / "runs"
    experiments = sorted(path for path in experiment_root.glob("*/*") if path.is_dir())
    if len(experiments) != 12:
        raise ValueError("finalist experiment count changed")
    runs = []
    outputs = set()
    fingerprints = set()
    data_directories = set()
    tokenizer_configs = []
    for experiment in experiments:
        configs = load_experiment(experiment, "data", "model", "runtime", "tokenizer", "train")
        train = configs["train"]
        output = Path(train["output_dir"])
        if output.parent != Path(contract["identity"]["checkpoint_root"]):
            raise ValueError("finalist checkpoint output is outside its frozen root")
        if str(output) in outputs:
            raise ValueError("finalist checkpoint output is not unique")
        outputs.add(str(output))
        if output.exists():
            raise FileExistsError(f"finalist checkpoint target already exists: {output}")
        if (
            train["train_tokens"] != training["training_tokens"]
            or train["save_every"] != training["save_every"]
            or train["eval_every"] != training["evaluation_every_steps"]
            or train["eval_tokens"] != training["evaluation_tokens"]
            or train["final_eval_tokens"] != training["final_evaluation_tokens"]
            or train["checkpoint_tokens"] != []
            or configs["runtime"] != {"device_batch_size": 4}
        ):
            raise ValueError("finalist resolved training config changed")
        data_dir = resolve_data_dir(
            configs["data"].get("output_dir"), configs["data"].get("output_name")
        ).expanduser().resolve()
        fingerprint = manifest_fingerprint(load_manifest(data_dir))
        fingerprints.add(fingerprint)
        data_directories.add(str(data_dir))
        tokenizer_configs.append(configs["tokenizer"])
        runs.append(
            {
                "experiment": experiment.relative_to(repository_root).as_posix(),
                "run": train["run"],
                "seed": train["seed"],
                "data_token_offset": train["data_token_offset"],
                "checkpoint_directory": str(output),
                "launch": (
                    "uv run --extra gpu --extra linear python -m scripts.base_train "
                    f"{experiment.relative_to(repository_root).as_posix()}"
                ),
            }
        )
    expected_fingerprint = contract["inputs"]["finalist_analysis"]
    _, analysis = _load_object(repository_root / expected_fingerprint)
    if fingerprints != {analysis["shared_training"]["data_manifest"]} or len(data_directories) != 1:
        raise ValueError("finalist data identity changed")
    if len({json.dumps(value, sort_keys=True) for value in tokenizer_configs}) != 1:
        raise ValueError("finalist tokenizer configs differ")
    outputs_by_kind = _external_outputs(repository_root, contract)
    present = {name: str(path) for name, path in outputs_by_kind.items() if path.exists()}
    if present:
        raise FileExistsError(f"finalist external output already exists: {present}")
    mount = qualify_volume(args.volume_dir, contract["qualification_requirements"]["storage"]["minimum_free_bytes_before_launch"])
    if (
        mount["uuid"] != contract["qualification_requirements"]["storage"]["filesystem_uuid"]
        or mount["filesystem"]
        != contract["qualification_requirements"]["storage"]["filesystem_type"]
    ):
        raise ValueError("finalist storage identity changed")
    return {
        "repository_root": repository_root,
        "contract_path": contract_path,
        "contract": contract,
        "materialization_path": materialization_path,
        "materialization": materialization,
        "analysis": analysis,
        "runs": runs,
        "data_dir": Path(next(iter(data_directories))),
        "tokenizer_config": tokenizer_configs[0],
        "mount": mount,
        "external_outputs": {name: str(path) for name, path in outputs_by_kind.items()},
    }


def _replay_data(values):
    contract = values["contract"]
    training_tokens = contract["materialized_training"]["training_tokens"]
    windows = unique_windows(contract["pairs"], training_tokens)
    manifest = load_manifest(values["data_dir"])
    tokenizer = get_tokenizer(**values["tokenizer_config"])
    replays = []
    for window in windows:
        for point in window_points(window, training_tokens):
            state = loader_state_for_offset(manifest, "train", point, 4096, 4, 1)
            direct = packed_loader(
                tokenizer,
                4,
                4096,
                "train",
                device="cpu",
                data_dir=values["data_dir"],
                initial_token_offset=point,
            )
            resumed = packed_loader(
                tokenizer,
                4,
                4096,
                "train",
                device="cpu",
                data_dir=values["data_dir"],
                resume_state_dict=state,
            )
            direct_inputs, direct_targets, direct_state = next(direct)
            resume_inputs, resume_targets, resume_state = next(resumed)
            if (
                direct_state != state
                or resume_state != state
                or not torch.equal(direct_inputs, resume_inputs)
                or not torch.equal(direct_targets, resume_targets)
            ):
                raise ValueError(f"finalist loader replay mismatch at {point}")
            replays.append(
                {
                    "global_token_offset": point,
                    "phase": state["phase"],
                    "selected_source": state["selected_source"],
                    "state_sha256": bytes_sha256(
                        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
                    ),
                    "input_sha256": token_sha256(direct_inputs),
                    "target_sha256": token_sha256(direct_targets),
                    "direct_resume_equal": True,
                }
            )
    return {"windows": windows, "replay_points": replays}


def prepare(args):
    values = _inputs(args)
    data = _replay_data(values)
    report = {
        "format": "speck_paper_finalist_qualification",
        "format_version": 1,
        "status": "materialization_data_and_storage_qualified_runtime_analysis_and_release_gates_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_id": values["contract"]["paper_id"],
        "contract": {
            "path": str(values["contract_path"]),
            "sha256": file_sha256(values["contract_path"]),
        },
        "materialization": {
            "path": str(values["materialization_path"]),
            "sha256": file_sha256(values["materialization_path"]),
            "generated_configs": len(values["materialization"]["generated_files"]),
        },
        "analysis": {
            "path": values["contract"]["inputs"]["finalist_analysis"],
            "sha256": values["contract"]["inputs"]["finalist_analysis_sha256"],
            "status": values["analysis"]["status"],
        },
        "runs": values["runs"],
        "data": {
            "directory": str(values["data_dir"]),
            "manifest": values["analysis"]["shared_training"]["data_manifest"],
            **data,
            "distinct_windows_disjoint": True,
            "every_window_crossed_with_all_seeds": True,
            "all_replays_exact": True,
        },
        "storage": {
            **values["mount"],
            "checkpoint_root": values["external_outputs"]["checkpoint_root"],
            "checkpoint_root_absent": True,
            "estimated_finalist_checkpoint_bytes": values["contract"][
                "qualification_requirements"
            ]["storage"]["estimated_finalist_checkpoint_bytes"],
            "finalist_floor_passed": True,
            "artifacts_deleted": 0,
            "cleanup_counted_as_capacity": False,
        },
        "output_absence": {
            **{f"{name}_absent": True for name in values["external_outputs"]},
            "unique_checkpoint_targets": len({run["checkpoint_directory"] for run in values["runs"]}),
        },
        "implementation": {
            "materializer_sha256": file_sha256(values["repository_root"] / "speck/paper_finalist.py"),
            "base_train_sha256": file_sha256(values["repository_root"] / "scripts/base_train.py"),
            "runner_revision": repository_revision(),
            "runner_sha256": file_sha256(__file__),
        },
        "decision": {
            "materialization_qualified": True,
            "data_windows_qualified": True,
            "storage_qualified": True,
            "output_absence_qualified": True,
            "runtime_preflight_qualified": False,
            "collector_and_analysis_implementation_qualified": False,
            "release_dependencies_qualified": False,
            "training_authorized": False,
            "automatic_launch_authorized": False,
            "next_action": "qualify exact-config CUDA runtime and finalist collector/analysis behavior; preserve release-suite blockers and do not launch training",
        },
    }
    atomic_json(args.output, report)
    return report


def check(args):
    values = _inputs(args)
    report_path, report = _load_object(args.output)
    current_data = _replay_data(values)
    if (
        report.get("format") != "speck_paper_finalist_qualification"
        or report.get("status")
        != "materialization_data_and_storage_qualified_runtime_analysis_and_release_gates_blocked"
        or report.get("contract", {}).get("sha256") != file_sha256(values["contract_path"])
        or report.get("materialization", {}).get("sha256")
        != file_sha256(values["materialization_path"])
        or report.get("runs") != values["runs"]
        or report.get("data", {}).get("windows") != current_data["windows"]
        or report.get("data", {}).get("replay_points") != current_data["replay_points"]
        or report.get("storage", {}).get("uuid") != values["mount"]["uuid"]
        or report.get("storage", {}).get("device_id") != values["mount"]["device_id"]
        or report.get("storage", {}).get("finalist_floor_passed") is not True
        or report.get("output_absence", {}).get("unique_checkpoint_targets") != 12
        or report.get("implementation", {}).get("materializer_sha256")
        != file_sha256(values["repository_root"] / "speck/paper_finalist.py")
        or report.get("implementation", {}).get("base_train_sha256")
        != file_sha256(values["repository_root"] / "scripts/base_train.py")
        or report.get("decision", {}).get("training_authorized") is not False
    ):
        raise ValueError("finalist qualification no longer matches live inputs")
    runner_source = subprocess.run(
        [
            "git",
            "-C",
            str(values["repository_root"]),
            "show",
            f"{report['implementation']['runner_revision']}:scripts/paper_finalist_qualify.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if bytes_sha256(runner_source) != report["implementation"]["runner_sha256"]:
        raise ValueError("finalist qualification runner changed")
    return report_path, report


def main(argv=None):
    args = arguments(argv)
    result = check(args)[1] if args.check else prepare(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
