"""Materialize and audit the matched Speck Paper 1 baseline program."""

import copy
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from speck.architecture import ArchitectureConfig
from speck.checkpoint import checkpoint_identity, load_metadata, load_timing
from speck.config import load_experiment
from speck.dataloader import loader_state_for_offset, manifest_fingerprint
from speck.dataset import load_manifest, resolve_data_dir
from speck.model import SpeckForCausalLM

CONFIG_NAMES = ("data", "long_context", "model", "tokenizer", "train")
PREFLIGHT_RESULT = Path("results/Speck-Paper1/baseline-preflight.json")


def load_matrix(path):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != "speck_paper_baseline_matrix" or value.get("format_version") != 1:
        raise ValueError("baseline matrix must use the checked format version")
    return path, value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value):
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_geometry(model_config):
    model = SpeckForCausalLM(ArchitectureConfig.from_dict(model_config))
    try:
        return {
            "parameters": model.parameter_count(),
            "flops_per_token_at_4096": int(model.flops_per_token(4_096)),
        }
    finally:
        del model
        gc.collect()


def mixer_counts(model_config):
    counts = {}
    architecture = ArchitectureConfig.from_dict(model_config)
    for invocation in architecture.execution_plan:
        for stage in invocation.block.stages:
            for branch in stage.branches:
                kind = getattr(branch, "kind", None)
                if kind == "swiglu":
                    continue
                if kind == "attention":
                    if branch.scope == "global":
                        key = "attention_global"
                    else:
                        key = f"attention_sliding_{branch.window_size}"
                else:
                    key = kind
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_config(source, destination):
    return Path(os.path.relpath(source, destination.parent)).as_posix()


def _arm_configs(repository_root, matrix, arm):
    source = repository_root / arm["template_experiment"]
    configs = load_experiment(source, *CONFIG_NAMES)
    model = copy.deepcopy(configs["model"])
    if size := arm["transform"].get("all_swiglu_intermediate_size"):
        for group in model["blocks"]:
            for stage in group["block"]["stages"]:
                for branch in stage["branches"]:
                    if branch["kind"] == "swiglu":
                        branch["intermediate_size"] = size
    model["expected_parameters"] = arm["parameters"]
    shared = matrix["planned_primary_baselines"]["shared_training"]
    train = copy.deepcopy(configs["train"])
    train.update(
        {
            "batch_tokens": shared["batch_tokens"],
            "checkpoint_tokens": [],
            "data_token_offset": 0,
            "eval_every": shared["evaluation_every_steps"],
            "eval_tokens": shared["evaluation_tokens"],
            "final_eval_tokens": shared["final_evaluation_tokens"],
            "global_token_offset": 0,
            "grad_clip": shared["gradient_clip"],
            "loss_backend": shared["loss_backend"],
            "lr": shared["learning_rate"],
            "lr_schedule": shared["schedule"],
            "min_lr": shared["minimum_learning_rate_multiplier"],
            "optimizer": shared["optimizer"],
            "output_dir": None,
            "run": f"{matrix['planned_primary_baselines']['family_id']}-{arm['id']}-template",
            "save_every": shared["training_tokens"] // shared["batch_tokens"],
            "seed": 42,
            "sequence_length": shared["sequence_length"],
            "train_tokens": shared["training_tokens"],
            "training_phase": "base",
            "wandb_group": matrix["planned_primary_baselines"]["family_id"],
            "weight_decay": shared["weight_decay"],
            "warmup_steps": shared["warmup_steps"],
        }
    )
    train.pop("device_batch_size", None)
    return {
        "data.json": configs["data"],
        "long_context.json": configs["long_context"],
        "model.json": model,
        "runtime.json": {"device_batch_size": arm["device_batch_size"]},
        "tokenizer.json": configs["tokenizer"],
        "train.json": train,
    }


def expected_materialization(matrix_path, output_root=None, geometry_fn=model_geometry):
    matrix_path, matrix = load_matrix(matrix_path)
    repository_root = matrix_path.parents[2]
    planned = matrix["planned_primary_baselines"]
    output_root = (
        repository_root / planned["output_root"]
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    files = {}
    arm_directories = {}
    geometries = {}
    for arm in planned["arms"]:
        directory = output_root / "arms" / arm["id"]
        arm_directories[arm["id"]] = directory
        configs = _arm_configs(repository_root, matrix, arm)
        geometry = geometry_fn(configs["model.json"])
        if geometry != {
            "parameters": arm["parameters"],
            "flops_per_token_at_4096": arm["flops_per_token_at_4096"],
        }:
            raise ValueError(f"planned baseline {arm['id']} geometry does not match its contract")
        geometries[arm["id"]] = geometry
        for name, value in configs.items():
            files[directory / name] = value
    for pair in planned["proxy_confirmation_pairs"]:
        pair_id = f"pair-{pair['pair']}-seed-{pair['seed']}-order-{pair['data_token_offset']}"
        for arm in planned["arms"]:
            directory = output_root / "runs" / pair_id / arm["id"]
            source = arm_directories[arm["id"]]
            for name in ("data.json", "long_context.json", "model.json", "tokenizer.json"):
                files[directory / name] = {
                    "extends": _relative_config(source / name, directory / name)
                }
            files[directory / "runtime.json"] = {
                "extends": _relative_config(source / "runtime.json", directory / "runtime.json")
            }
            files[directory / "train.json"] = {
                "extends": _relative_config(source / "train.json", directory / "train.json"),
                "data_token_offset": pair["data_token_offset"],
                "run": f"{planned['family_id']}-{pair_id}-{arm['id']}",
                "seed": pair["seed"],
            }
    manifest_path = output_root / "baseline_materialization.json"
    manifest = {
        "format": "speck_paper_baseline_materialization",
        "format_version": 1,
        "contract": str(matrix_path.relative_to(repository_root)),
        "contract_sha256": file_sha256(matrix_path),
        "family_id": planned["family_id"],
        "arms": geometries,
        "pairs": planned["proxy_confirmation_pairs"],
        "generated_files": sorted(path.relative_to(output_root).as_posix() for path in files),
        "status": "materialized_unexecuted",
    }
    files[manifest_path] = manifest
    return output_root, files, manifest


def materialize_baselines(matrix_path, output_root=None, check=False, geometry_fn=model_geometry):
    output_root, files, manifest = expected_materialization(
        matrix_path,
        output_root,
        geometry_fn,
    )
    if check:
        missing = []
        changed = []
        for path, expected in files.items():
            if not path.is_file():
                missing.append(path.relative_to(output_root).as_posix())
                continue
            try:
                actual = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                changed.append(path.relative_to(output_root).as_posix())
                continue
            if actual != expected:
                changed.append(path.relative_to(output_root).as_posix())
        observed = {
            path.relative_to(output_root).as_posix()
            for path in output_root.rglob("*.json")
            if path.is_file()
        }
        unexpected = sorted(observed - {path.relative_to(output_root).as_posix() for path in files})
        if missing or changed or unexpected:
            raise ValueError(
                "baseline materialization drift: "
                f"missing={missing}, changed={changed}, unexpected={unexpected}"
            )
        return manifest
    if output_root.exists():
        raise FileExistsError(f"baseline output already exists: {output_root}")
    for path, value in files.items():
        _write_json(path, value)
    return manifest


def _checkpoint_bytes(directory):
    return sum(path.stat().st_size for path in directory.iterdir() if path.is_file())


def _largest_directories(directory, limit):
    completed = subprocess.run(
        ["du", "-x", "-B1", "--max-depth=1", str(directory)],
        check=True,
        capture_output=True,
        text=True,
    )
    root = Path(directory).resolve()
    entries = []
    for line in completed.stdout.splitlines():
        size, raw_path = line.split("\t", 1)
        path = Path(raw_path).resolve()
        if path == root:
            continue
        entries.append({"path": path.name, "bytes": int(size)})
    return sorted(entries, key=lambda entry: (-entry["bytes"], entry["path"]))[:limit]


def _audit_historical_arm(arm, shared, repository_root, cache_root):
    experiment = repository_root / arm["experiment"]
    checkpoint_dir = cache_root / "checkpoints" / arm["checkpoint_run"]
    step = shared["checkpoint_step"]
    metadata = load_metadata(checkpoint_dir, step)
    resolved = metadata["resolved"]
    expected = {
        "parameters": arm["parameters"],
        "training_tokens": shared["training_tokens"],
        "sequence_length": shared["sequence_length"],
        "batch_tokens": shared["batch_tokens"],
        "optimizer": shared["optimizer"],
        "learning_rate": shared["learning_rate"],
        "warmup_steps": shared["warmup_steps"],
        "seed": shared["seed"],
        "data_manifest": shared["data_manifest"],
        "tokenizer_revision": shared["tokenizer_revision"],
        "validation_loss": arm["validation_loss"],
        "mixer_counts": arm["mixer_counts"],
    }
    actual = {
        "parameters": resolved["parameters"],
        "training_tokens": metadata["global_tokens"],
        "sequence_length": resolved["sequence_length"],
        "batch_tokens": resolved["batch_tokens"],
        "optimizer": resolved["optimizer"],
        "learning_rate": resolved["lr"],
        "warmup_steps": resolved["warmup_steps"],
        "seed": resolved["seed"],
        "data_manifest": metadata["manifest"],
        "tokenizer_revision": resolved["tokenizer"]["revision"],
        "validation_loss": metadata["validation_loss"],
        "mixer_counts": mixer_counts(metadata["config"]),
    }
    if actual != expected:
        differences = sorted(key for key in expected if actual[key] != expected[key])
        raise ValueError(f"historical baseline {arm['id']} drifted: {differences}")
    configs = load_experiment(experiment, "data", "model", "tokenizer", "train")
    if (
        ArchitectureConfig.from_dict(configs["model"]).settings()
        != ArchitectureConfig.from_dict(metadata["config"]).settings()
    ):
        raise ValueError(f"historical baseline {arm['id']} model config changed")
    timing = load_timing(checkpoint_dir, step) or metadata.get("timing", {})
    return {
        "id": arm["id"],
        "role": arm["role"],
        "experiment": arm["experiment"],
        "checkpoint": checkpoint_identity(checkpoint_dir, step),
        "checkpoint_bytes": _checkpoint_bytes(checkpoint_dir),
        "config_sha256": {
            f"{name}.json": file_sha256(experiment / f"{name}.json")
            for name in ("data", "model", "tokenizer", "train")
        },
        "parameters": actual["parameters"],
        "flops_per_token_at_4096": arm["flops_per_token_at_4096"],
        "training_tokens": actual["training_tokens"],
        "validation_loss": actual["validation_loss"],
        "validation_tokens": metadata["validation_tokens"],
        "optimizer_seconds": timing.get("optimizer_seconds"),
        "active_seconds": timing.get("active_seconds"),
        "device_batch_size": resolved["device_batch_size"],
        "accumulation_steps": resolved["accumulation_steps"],
        "mixer_counts": actual["mixer_counts"],
        "status": "qualified_historical_discovery_only",
    }


def audit_baselines(matrix_path, cache_root, runner_revision):
    matrix_path, matrix = load_matrix(matrix_path)
    repository_root = matrix_path.parents[2]
    cache_root = Path(cache_root).expanduser().resolve()
    historical = matrix["historical_evidence"]
    source_artifacts = []
    for source in historical["source_artifacts"]:
        path = repository_root / source["path"]
        actual = file_sha256(path)
        if actual != source["sha256"]:
            raise ValueError(f"historical source artifact drifted: {source['path']}")
        source_artifacts.append(source)
    arms = [
        _audit_historical_arm(
            arm,
            historical["shared_expected"],
            repository_root,
            cache_root,
        )
        for arm in historical["arms"]
    ]
    materialization = materialize_baselines(matrix_path, check=True)
    planned = matrix["planned_primary_baselines"]
    template = repository_root / planned["arms"][0]["template_experiment"]
    data_config = load_experiment(template, "data")["data"]
    data_dir = resolve_data_dir(data_config.get("output_dir"), data_config.get("output_name"))
    data_manifest = load_manifest(data_dir)
    if manifest_fingerprint(data_manifest) != planned["shared_training"]["data_manifest"]:
        raise ValueError("planned baseline packed-data manifest does not match")
    training_tokens = planned["shared_training"]["training_tokens"]
    data_windows = []
    for pair in planned["proxy_confirmation_pairs"]:
        start = pair["data_token_offset"]
        end = start + training_tokens
        sequence_length = planned["shared_training"]["sequence_length"]
        device_batch_size = planned["arms"][0]["device_batch_size"]
        start_state = loader_state_for_offset(
            data_manifest, "train", start, sequence_length, device_batch_size, 1
        )
        end_state = loader_state_for_offset(
            data_manifest, "train", end, sequence_length, device_batch_size, 1
        )
        data_windows.append(
            {
                **pair,
                "end_token_offset": end,
                "start_phase": start_state["phase"],
                "end_phase": end_state["phase"],
                "start_state_sha256": value_sha256(start_state),
                "end_state_sha256": value_sha256(end_state),
            }
        )
    by_id = {arm["id"]: arm for arm in arms}
    dense = by_id["dense_global"]
    kda = by_id["kda_global_sigmoid_nope"]
    gdn = by_id["gdn_global_sigmoid_nope"]
    storage = matrix["storage_contract"]
    disk = shutil.disk_usage(cache_root)
    matrix_sha256 = file_sha256(matrix_path)
    analysis_contract = matrix.get("analysis_contract", {})
    analysis_path = repository_root / analysis_contract.get("path", "")
    analysis = (
        json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.is_file() else {}
    )
    analysis_qualified = (
        analysis_contract.get("status") == "frozen_before_results"
        and analysis.get("format") == "speck_paper_baseline_analysis_plan"
        and analysis.get("format_version") == 1
        and analysis.get("status") == "frozen_before_results"
        and analysis.get("baseline_matrix_sha256") == matrix_sha256
    )
    preflight_path = repository_root / PREFLIGHT_RESULT
    preflight = (
        json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
    )
    preflight_valid = (
        preflight.get("format") == "speck_paper_baseline_preflight"
        and preflight.get("format_version") == 1
        and preflight.get("matrix_sha256") == matrix_sha256
        and len(preflight.get("arms", ())) == len(planned["arms"])
    )
    blockers = ["SPE-58 evaluation-manifest dependency remains open"]
    if not preflight:
        blockers.append("paired compiled/runtime/export preflight has not run")
    elif not preflight_valid:
        blockers.append("paired compiled/runtime/export preflight artifact is invalid")
    elif preflight.get("status") != "qualified":
        blockers.append("paired compiled/runtime/export preflight failed")
    if not analysis_qualified:
        blockers.append("paired analysis and stopping script is not frozen")
    if disk.free < storage["minimum_free_bytes_before_proxy_launch"]:
        blockers.append("free storage is below the proxy launch minimum")
    return {
        "format": "speck_paper_baseline_audit",
        "format_version": 1,
        "paper_id": matrix["paper_id"],
        "status": "historical_evidence_qualified_proxy_launch_blocked",
        "runner_revision": runner_revision,
        "contract": str(matrix_path.relative_to(repository_root)),
        "contract_sha256": matrix_sha256,
        "analysis_contract": {
            "path": analysis_contract.get("path"),
            "sha256": file_sha256(analysis_path) if analysis_path.is_file() else None,
            "status": "qualified_frozen_before_results" if analysis_qualified else "unqualified",
        },
        "preflight": {
            "path": PREFLIGHT_RESULT.as_posix(),
            "sha256": file_sha256(preflight_path) if preflight_path.is_file() else None,
            "status": preflight.get("status", "not_run"),
            "runner_revision": preflight.get("runner_revision"),
            "arms": [
                {"arm_id": arm.get("arm_id"), "passed": arm.get("passed")}
                for arm in preflight.get("arms", ())
            ],
        },
        "source_artifacts": source_artifacts,
        "historical_arms": arms,
        "historical_comparisons": {
            "dense_global_vs_kda_hybrid": {
                "dense_minus_kda_validation_loss": dense["validation_loss"]
                - kda["validation_loss"],
                "dense_over_kda_parameters": dense["parameters"] / kda["parameters"],
                "dense_over_kda_flops_per_token_at_4096": dense["flops_per_token_at_4096"]
                / kda["flops_per_token_at_4096"],
                "interpretation": "whole-architecture historical context only; multiple axes differ",
            },
            "gdn_nope_vs_kda_nope": {
                "kda_minus_gdn_validation_loss": kda["validation_loss"] - gdn["validation_loss"],
                "kda_over_gdn_parameters": kda["parameters"] / gdn["parameters"],
                "kda_over_gdn_flops_per_token_at_4096": kda["flops_per_token_at_4096"]
                / gdn["flops_per_token_at_4096"],
                "interpretation": "close operator isolation at one seed and one packed-data order",
            },
        },
        "planned_materialization": materialization,
        "planned_data_orders": {
            "manifest": manifest_fingerprint(data_manifest),
            "data_directory": str(data_dir),
            "windows": data_windows,
            "status": "aligned_disjoint_windows_qualified",
        },
        "storage": {
            "cache_root": str(cache_root),
            "filesystem_total_bytes": disk.total,
            "filesystem_used_bytes": disk.used,
            "filesystem_free_bytes": disk.free,
            "historical_checkpoint_bytes": sum(arm["checkpoint_bytes"] for arm in arms),
            "minimum_free_bytes_before_proxy_launch": storage[
                "minimum_free_bytes_before_proxy_launch"
            ],
            "proxy_storage_deficit_bytes": max(
                0,
                storage["minimum_free_bytes_before_proxy_launch"] - disk.free,
            ),
            "estimated_proxy_checkpoint_bytes": storage["estimated_proxy_checkpoint_bytes"],
            "top_level_directories": _largest_directories(cache_root, 20),
            "largest_checkpoint_directories": _largest_directories(cache_root / "checkpoints", 25),
        },
        "blockers": blockers,
        "conclusion": "historical arms are reproducible discovery context, but no existing result is a promotion-authority Paper 1 baseline comparison; the new paired matrix is materialized and remains blocked from launch",
    }


def proxy_run_count(matrix):
    planned = matrix["planned_primary_baselines"]
    return len(planned["arms"]) * len(planned["proxy_confirmation_pairs"])


def aligned_steps(tokens, batch_tokens):
    return math.ceil(tokens / batch_tokens)
