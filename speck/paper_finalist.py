"""Materialize the frozen Paper 1 six-pair finalist without launching it."""

import copy
import hashlib
import json
import os
from pathlib import Path

from speck.config import load_experiment

CONFIG_NAMES = ("data", "long_context", "model", "runtime", "tokenizer", "train")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def value_file_sha256(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _load_object(path):
    path = Path(path).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load finalist artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"finalist artifact must contain an object: {path}")
    return path, value


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _relative_config(source, destination):
    return Path(os.path.relpath(source, destination.parent)).as_posix()


def load_contract(path):
    path, contract = _load_object(path)
    if (
        contract.get("format") != "speck_paper_finalist_materialization_contract"
        or contract.get("format_version") != 1
        or contract.get("status") != "frozen_after_proxy_pass_before_finalist_materialization"
    ):
        raise ValueError("finalist materialization contract is not frozen format version 1")
    repository_root = path.parents[2]
    for name, relative in (
        ("baseline matrix", "baseline_matrix"),
        ("proxy disposition", "proxy_disposition"),
        ("proxy analysis", "proxy_analysis"),
        ("proxy materialization", "proxy_materialization"),
        ("finalist analysis", "finalist_analysis"),
    ):
        source = repository_root / contract["inputs"].get(relative, "")
        if not source.is_file() or file_sha256(source) != contract["inputs"].get(
            f"{relative}_sha256"
        ):
            raise ValueError(f"finalist {name} does not match its pin")
    _, matrix = _load_object(repository_root / contract["inputs"]["baseline_matrix"])
    _, disposition = _load_object(repository_root / contract["inputs"]["proxy_disposition"])
    _, proxy = _load_object(repository_root / contract["inputs"]["proxy_analysis"])
    _, analysis = _load_object(repository_root / contract["inputs"]["finalist_analysis"])
    future = matrix.get("future_finalist_design", {})
    disposition_branches = {entry.get("id"): entry for entry in disposition.get("branches", ())}
    pairs = contract.get("pairs", ())
    analysis_pairs = [
        {key: pair[key] for key in ("pair", "seed", "data_token_offset")} for pair in pairs
    ]
    expected_pair_crossing = [
        (seed, offset)
        for seed in future.get("seeds", ())
        for offset in future.get("data_token_offsets", ())
    ]
    if (
        future.get("status") != "not_materialized_until_proxy_confirmation"
        or future.get("training_tokens_per_run")
        != contract.get("materialized_training", {}).get("training_tokens")
        or future.get("paired_runs") != len(pairs)
        or future.get("total_model_runs") != 2 * len(pairs)
        or future.get("compute_matched_dense_tokens")
        != contract.get("materialized_training", {}).get("compute_matched_dense_tokens")
        or [(pair.get("seed"), pair.get("data_token_offset")) for pair in pairs]
        != expected_pair_crossing
        or any(
            pair.get("end_token_offset") - pair.get("data_token_offset")
            != future.get("training_tokens_per_run")
            for pair in pairs
        )
        or analysis.get("pairs") != analysis_pairs
        or analysis.get("execution", {}).get("training_authorized") is not False
        or proxy.get("proxy_quality_screen_pass") is not True
        or proxy.get("fixed_tokens", {}).get("non_inferiority_pass") is not True
        or not all(value.get("pass") for value in proxy.get("source_guardrails", {}).values())
        or disposition_branches.get("quality_screen_passed", {}).get(
            "finalist_materialization_authorized"
        )
        is not True
        or contract.get("decision", {}).get("materialization_authorized") is not True
        or contract.get("decision", {}).get("training_authorized") is not False
    ):
        raise ValueError("finalist materialization eligibility or design is invalid")
    expected_arms = {"dense_global_param_match", "five_cache_kda_gqa"}
    if set(contract.get("parent_arms", {})) != expected_arms:
        raise ValueError("finalist parent arms are invalid")
    for arm in contract["parent_arms"].values():
        directory = repository_root / arm.get("directory", "")
        expected_hashes = arm.get("config_sha256", {})
        if set(expected_hashes) != {f"{name}.json" for name in CONFIG_NAMES}:
            raise ValueError("finalist parent config pins are incomplete")
        for name, expected_hash in expected_hashes.items():
            source = directory / name
            if not source.is_file() or file_sha256(source) != expected_hash:
                raise ValueError(f"finalist parent config does not match its pin: {source}")
    return path, repository_root, contract


def _arm_configs(repository_root, contract, arm_id):
    arm = contract["parent_arms"][arm_id]
    parent = repository_root / arm["directory"]
    configs = load_experiment(parent, *CONFIG_NAMES)
    values = {f"{name}.json": copy.deepcopy(configs[name]) for name in CONFIG_NAMES}
    family = contract["identity"]["family_id"]
    training = contract["materialized_training"]
    values["train.json"].update(
        {
            "checkpoint_tokens": training["checkpoint_tokens"],
            "data_token_offset": 0,
            "eval_every": training["evaluation_every_steps"],
            "eval_tokens": training["evaluation_tokens"],
            "final_eval_tokens": training["final_evaluation_tokens"],
            "global_token_offset": 0,
            "output_dir": None,
            "run": f"{family}-{arm_id}-template",
            "save_every": training["save_every"],
            "seed": 42,
            "train_tokens": training["training_tokens"],
            "training_phase": training["training_phase"],
            "wandb_group": family,
        }
    )
    return values


def expected_materialization(contract_path, output_root=None):
    contract_path, repository_root, contract = load_contract(contract_path)
    identity = contract["identity"]
    output_root = (
        repository_root / identity["output_root"]
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    files = {}
    arm_directories = {}
    arm_summaries = {}
    for arm_id, arm in contract["parent_arms"].items():
        directory = output_root / "arms" / arm_id
        arm_directories[arm_id] = directory
        configs = _arm_configs(repository_root, contract, arm_id)
        if configs["model.json"].get("expected_parameters") != arm["parameters"]:
            raise ValueError(f"finalist arm parameter identity changed: {arm_id}")
        arm_summaries[arm_id] = {
            "parameters": arm["parameters"],
            "flops_per_token_at_4096": arm["flops_per_token_at_4096"],
        }
        for name, value in configs.items():
            files[directory / name] = value
    for pair in contract["pairs"]:
        pair_id = (
            f"pair-{pair['pair']}-seed-{pair['seed']}-order-{pair['data_token_offset']}"
        )
        for arm_id in contract["parent_arms"]:
            directory = output_root / "runs" / pair_id / arm_id
            source = arm_directories[arm_id]
            for name in ("data.json", "long_context.json", "model.json", "runtime.json", "tokenizer.json"):
                files[directory / name] = {
                    "extends": _relative_config(source / name, directory / name)
                }
            run_name = f"{identity['family_id']}-{pair_id}-{arm_id}"
            files[directory / "train.json"] = {
                "extends": _relative_config(source / "train.json", directory / "train.json"),
                "data_token_offset": pair["data_token_offset"],
                "output_dir": str(Path(identity["checkpoint_root"]) / run_name),
                "run": run_name,
                "seed": pair["seed"],
            }
    generated = {
        path.relative_to(output_root).as_posix(): value_file_sha256(value)
        for path, value in sorted(files.items(), key=lambda item: item[0].as_posix())
    }
    manifest_path = output_root / "finalist_materialization.json"
    manifest = {
        "format": "speck_paper_finalist_materialization",
        "format_version": 1,
        "status": "materialized_unexecuted_training_blocked",
        "contract": str(contract_path.relative_to(repository_root)),
        "contract_sha256": file_sha256(contract_path),
        "analysis": contract["inputs"]["finalist_analysis"],
        "analysis_sha256": contract["inputs"]["finalist_analysis_sha256"],
        "family_id": identity["family_id"],
        "arms": arm_summaries,
        "pairs": contract["pairs"],
        "generated_files": list(generated),
        "generated_config_sha256": generated,
        "training_authorized": False,
    }
    files[manifest_path] = manifest
    return output_root, files, manifest


def _external_output_paths(repository_root, contract):
    identity = contract["identity"]
    return [
        Path(identity["checkpoint_root"]),
        repository_root / identity["result_root"],
        repository_root / identity["analysis_result"],
        repository_root / identity["time_to_quality_lock"],
    ]


def materialize_finalist(contract_path, output_root=None, check=False):
    contract_path, repository_root, contract = load_contract(contract_path)
    expected_root, files, manifest = expected_materialization(contract_path, output_root)
    if check:
        missing = []
        changed = []
        for path, expected in files.items():
            if not path.is_file():
                missing.append(path.relative_to(expected_root).as_posix())
                continue
            try:
                actual = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                changed.append(path.relative_to(expected_root).as_posix())
                continue
            if actual != expected:
                changed.append(path.relative_to(expected_root).as_posix())
        observed = {
            path.relative_to(expected_root).as_posix()
            for path in expected_root.rglob("*.json")
            if path.is_file()
        }
        unexpected = sorted(
            observed - {path.relative_to(expected_root).as_posix() for path in files}
        )
        if missing or changed or unexpected:
            raise ValueError(
                "finalist materialization drift: "
                f"missing={missing}, changed={changed}, unexpected={unexpected}"
            )
        return manifest
    forbidden = [expected_root, *_external_output_paths(repository_root, contract)]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise FileExistsError(f"finalist output already exists: {present}")
    for path, value in files.items():
        _write_json(path, value)
    return manifest
