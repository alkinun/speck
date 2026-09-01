"""Run and summarize the benchmarks on the pinned Open SLM Leaderboard."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from speck.checkpoint import directory_identity

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments" / "Speck1-140M" / "open_slm.json"
DEFAULT_OUTPUT_ROOT = (
    Path(os.environ.get("speck_base_dir", Path.home() / ".cache" / "speck"))
    / "evaluations"
    / "open-slm"
)


def _load_config(path):
    path = Path(path)
    values = json.loads(path.read_text(encoding="utf-8"))
    parent = values.pop("extends", None)
    config = _load_config(path.parent / parent) if parent is not None else {}
    config.update(values)
    required = {"leaderboard", "model", "lm_eval", "arithmark_2", "arithmark_3"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Open SLM config is missing: {', '.join(sorted(missing))}")
    return config


def _default_output_dir(config, local_model=None):
    if local_model is not None:
        identity = directory_identity(local_model)
        return DEFAULT_OUTPUT_ROOT / f"{Path(local_model).name}--{identity['sha256'][:16]}"
    return DEFAULT_OUTPUT_ROOT / config["model"]["repo"].rsplit("/", 1)[-1]


def _validate_local_export(local_model):
    local_model = Path(local_model).expanduser().resolve()
    required = ("config.json", "model.safetensors", "speck_parity.json")
    missing = [name for name in required if not (local_model / name).is_file()]
    if missing:
        raise ValueError(f"local export is missing parity artifacts: {', '.join(missing)}")
    config = json.loads((local_model / "config.json").read_text(encoding="utf-8"))
    parity = json.loads((local_model / "speck_parity.json").read_text(encoding="utf-8"))
    if parity.get("format") != "speck_export_parity" or parity.get("passed") is not True:
        raise ValueError("local export has no successful native/Transformers parity gate")
    if parity.get("parameters") != config.get("expected_parameters"):
        raise ValueError("local export parameter parity does not match its config")
    return local_model


def _bind_local_model(output_dir, local_model):
    identity = directory_identity(local_model)
    path = output_dir / "local-model-identity.json"
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != identity:
            raise ValueError("Open SLM output directory is bound to a different local export")
    else:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    return identity


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path, expected, label):
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _download_benchmark_files(config):
    paths = {}
    for name in ("arithmark_2", "arithmark_3"):
        benchmark = config[name]
        downloaded = {}
        for field, checksum_field in (
            ("runner", "runner_sha256"),
            ("data", "data_sha256"),
        ):
            path = Path(
                hf_hub_download(
                    benchmark["repo"],
                    benchmark[field],
                    repo_type="dataset",
                    revision=benchmark["revision"],
                )
            )
            _verify_checksum(path, benchmark[checksum_field], f"{name} {field}")
            downloaded[field] = path
        paths[name] = downloaded
    return paths


def _assert_revision(api, repo, repo_type, expected, *, require_main=False):
    resolved = api.repo_info(repo, repo_type=repo_type, revision=expected).sha
    if resolved != expected:
        raise RuntimeError(f"{repo}@{expected} resolved to {resolved}")
    if require_main:
        current = api.repo_info(repo, repo_type=repo_type).sha
        if current != expected:
            raise RuntimeError(
                f"{repo} main moved from pinned revision {expected} to {current}; "
                "refresh and review the evaluation config before running"
            )


def _assert_implicit_revisions(config, *, check_model=True):
    api = HfApi()
    if check_model:
        model = config["model"]
        _assert_revision(api, model["repo"], "model", model["revision"], require_main=True)
    for repo, revision in config["lm_eval"]["datasets"].items():
        _assert_revision(api, repo, "dataset", revision, require_main=True)


def _lm_eval_command(config, output_path, device, limit=None, local_model=None):
    model = config["model"]
    evaluation = config["lm_eval"]
    model_args = [
        f"pretrained={local_model or model['repo']}",
        "trust_remote_code=True",
        f"dtype={evaluation['dtype']}",
        "use_cache=False",
    ]
    if local_model is None:
        model_args.insert(1, f"revision={model['revision']}")
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        "hf",
        "--model_args",
        *model_args,
        "--tasks",
        *evaluation["tasks"],
        "--num_fewshot",
        str(evaluation["num_fewshot"]),
        "--batch_size",
        str(evaluation["batch_size"]),
        "--device",
        device,
        "--output_path",
        str(output_path),
        "--trust_remote_code",
        "--confirm_run_unsafe_code",
    ]
    if limit is not None:
        command.extend(("--limit", str(limit)))
    return command


def _new_result(output_path, before):
    matches = set(output_path.parent.glob(f"{output_path.stem}_*.json"))
    created = matches - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new result beside {output_path}, found {len(created)}")
    return created.pop()


def _record_lm_eval_result(result, local_identity=None):
    """Atomically bind later summaries to one checksummed full result."""

    result = Path(result)
    selection = result.parent / "selected-result.json"
    temporary = selection.with_suffix(".json.tmp")
    value = {
        "format_version": 1,
        "path": result.name,
        "sha256": _sha256(result),
    }
    if local_identity is not None:
        value["local_model_sha256"] = local_identity["sha256"]
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, selection)


def _run_lm_eval(config, output_dir, device, limit=None, local_model=None):
    _assert_implicit_revisions(config, check_model=local_model is None)
    name = "smoke-results.json" if limit is not None else "results.json"
    output_path = output_dir / "lm-eval" / name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_identity = directory_identity(local_model) if local_model is not None else None
    before = set(output_path.parent.glob(f"{output_path.stem}_*.json"))
    subprocess.run(_lm_eval_command(config, output_path, device, limit, local_model), check=True)
    _assert_implicit_revisions(config, check_model=local_model is None)
    result = _new_result(output_path, before)
    if limit is None:
        _record_lm_eval_result(result, local_identity)
    if local_identity is not None:
        identity_dir = output_path.parent / "local-identities"
        identity_dir.mkdir(exist_ok=True)
        identity = {
            "local_model": local_identity,
            "result": {"path": result.name, "sha256": _sha256(result)},
        }
        (identity_dir / f"{result.name}.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"lm-eval result: {result}")
    return result


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load benchmark runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _disable_arithmark_2_cache(official):
    original_load = official.load_hf_model

    def load_without_cache(*args, **kwargs):
        model, tokenizer = original_load(*args, **kwargs)
        model.config.use_cache = False
        return model, tokenizer

    official.load_hf_model = load_without_cache


def _run_arithmark_2(config, files, output_dir, local_model=None, local_identity=None):
    _assert_implicit_revisions(config, check_model=local_model is None)
    benchmark = config["arithmark_2"]
    results_dir = output_dir / "arithmark-2"
    results_dir.mkdir(parents=True, exist_ok=True)
    before = {
        path: _sha256(path)
        for path in results_dir.glob("*.json")
        if path.name != "selected-result.json"
    }
    official = _load_module(files["runner"], "official_arithmark_2")
    official.CACHE_DIR = str(results_dir)

    # The pinned runner right-pads but omits use_cache=False. Keep its scoring
    # code byte-for-byte intact and apply the required model runtime setting.
    _disable_arithmark_2_cache(official)
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            str(files["runner"]),
            "--model",
            str(local_model or config["model"]["repo"]),
            "--batch-size",
            str(benchmark["batch_size"]),
            "--data-path",
            str(files["data"]),
        ]
        official.main()
    finally:
        sys.argv = original_argv
    _assert_implicit_revisions(config, check_model=local_model is None)
    created = {
        path
        for path in results_dir.glob("*.json")
        if path.name != "selected-result.json" and before.get(path) != _sha256(path)
    }
    if len(created) != 1:
        raise RuntimeError(f"ArithMark 2.0 produced {len(created)} new result files")
    result = created.pop()
    _record_lm_eval_result(result, local_identity)
    print(f"ArithMark 2.0 result: {result}")
    return result


def _run_arithmark_3(config, files, output_dir, device, local_model=None, local_identity=None):
    _assert_implicit_revisions(config, check_model=local_model is None)
    benchmark = config["arithmark_3"]
    results_dir = output_dir / "arithmark-3"
    results_dir.mkdir(parents=True, exist_ok=True)
    before = {
        path: _sha256(path)
        for path in results_dir.glob("*.json")
        if path.name != "selected-result.json"
    }
    command = [
        sys.executable,
        str(files["runner"]),
        "--model",
        str(local_model or config["model"]["repo"]),
        "--batch-size",
        str(benchmark["batch_size"]),
        "--max-context",
        str(benchmark["max_context"]),
        "--data-path",
        str(files["data"]),
        "--device",
        device,
        "--dtype",
        benchmark["dtype"],
        "--primary-metric",
        benchmark["primary_metric"],
        "--results-dir",
        str(results_dir),
    ]
    subprocess.run(command, check=True)
    _assert_implicit_revisions(config, check_model=local_model is None)
    created = {
        path
        for path in results_dir.glob("*.json")
        if path.name != "selected-result.json" and before.get(path) != _sha256(path)
    }
    if len(created) != 1:
        raise RuntimeError(f"ArithMark 3.0 produced {len(created)} new result files")
    result = created.pop()
    _record_lm_eval_result(result, local_identity)
    print(f"ArithMark 3.0 result: {result}")
    return result


def _one_result(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _selected_lm_eval_result(directory, expected_model_sha256=None):
    directory = Path(directory)
    selection = directory / "selected-result.json"
    if not selection.is_file():
        if expected_model_sha256 is not None:
            raise FileNotFoundError(f"local result has no export-bound selection: {selection}")
        return _one_result(directory, "results_*.json")
    value = json.loads(selection.read_text(encoding="utf-8"))
    name = value.get("path")
    if (
        value.get("format_version") != 1
        or not isinstance(name, str)
        or Path(name).name != name
        or not isinstance(value.get("sha256"), str)
    ):
        raise ValueError(f"invalid lm-eval result selection: {selection}")
    if (
        expected_model_sha256 is not None
        and value.get("local_model_sha256") != expected_model_sha256
    ):
        raise ValueError("selected result is bound to a different local export")
    result = directory / name
    if not result.is_file():
        raise FileNotFoundError(f"selected lm-eval result does not exist: {result}")
    _verify_checksum(result, value["sha256"], "selected lm-eval result")
    return result


def _normalize_from_chance(score, chance):
    return 100 * (score - chance) / (100 - chance)


def _summarize(
    config,
    lm_eval_path,
    arithmark_2_path,
    arithmark_3_path,
    local_identity=None,
):
    lm_result = json.loads(lm_eval_path.read_text(encoding="utf-8"))
    arithmark_2 = json.loads(arithmark_2_path.read_text(encoding="utf-8"))
    arithmark_3 = json.loads(arithmark_3_path.read_text(encoding="utf-8"))
    evaluation = config["lm_eval"]

    if lm_result["config"]["limit"] is not None:
        raise ValueError("cannot summarize a limited lm-eval run")
    if (
        local_identity is None
        and lm_result["config"]["model_revision"] != config["model"]["revision"]
    ):
        raise ValueError("lm-eval model revision does not match the evaluation config")
    if lm_result["lm_eval_version"] != evaluation["version"]:
        raise ValueError("lm-eval version does not match the evaluation config")
    if lm_result["transformers_version"] != evaluation["transformers_version"]:
        raise ValueError("Transformers version does not match the evaluation config")

    scores = {}
    for task in evaluation["tasks"]:
        samples = lm_result["n-samples"][task]
        expected_samples = evaluation["expected_samples"][task]
        if samples["original"] != expected_samples or samples["effective"] != expected_samples:
            raise ValueError(f"{task} result is not a complete {expected_samples}-sample run")
        scores[task] = round(lm_result["results"][task]["acc_norm,none"] * 100, 2)

    arithmark_2_result = arithmark_2["results"]["arithmark_2.0"]
    arithmark_3_result = arithmark_3["results"]["arithmark-3"]
    if arithmark_2_result["total"] != config["arithmark_2"]["samples"]:
        raise ValueError("ArithMark 2.0 result is incomplete")
    if arithmark_3_result["total"] != config["arithmark_3"]["samples"]:
        raise ValueError("ArithMark 3.0 result is incomplete")
    scores["arithmark_2"] = round(arithmark_2_result["acc"], 2)
    scores["arithmark_3"] = round(arithmark_3_result["acc_norm"], 2)

    combined_arc = round((scores["arc_easy"] + scores["arc_challenge"]) / 2, 2)
    average = round(
        (scores["hellaswag"] + combined_arc + scores["piqa"] + scores["arithmark_3"]) / 4,
        2,
    )
    normalized = {
        "hellaswag": _normalize_from_chance(scores["hellaswag"], 25),
        "arc": _normalize_from_chance(combined_arc, 25),
        "piqa": _normalize_from_chance(scores["piqa"], 50),
        "arithmark_3": _normalize_from_chance(scores["arithmark_3"], 25),
    }
    intelligence_index = round(
        (
            normalized["hellaswag"]
            + normalized["arc"]
            + normalized["piqa"]
            + 0.65 * normalized["arithmark_3"]
        )
        / 3.65,
        2,
    )
    summary = {
        "model": config["model"],
        "leaderboard": config["leaderboard"],
        "scores_percent": {
            **scores,
            "combined_arc": combined_arc,
            "average": average,
            "intelligence_index": intelligence_index,
        },
        "evaluation": {
            "zero_shot": True,
            "lm_eval_revision": evaluation["revision"],
            "lm_eval_version": evaluation["version"],
            "transformers_version": evaluation["transformers_version"],
            "dataset_revisions": evaluation["datasets"],
            "arithmark_2_revision": config["arithmark_2"]["revision"],
            "arithmark_3_revision": config["arithmark_3"]["revision"],
        },
        "artifacts": {
            "lm_eval_sha256": _sha256(lm_eval_path),
            "arithmark_2_sha256": _sha256(arithmark_2_path),
            "arithmark_3_sha256": _sha256(arithmark_3_path),
        },
    }
    if local_identity is not None:
        summary["model"] = {
            "type": "local_export",
            "path": local_identity["path"],
            "directory_sha256": local_identity["sha256"],
            "parameters": config["model"]["parameters"],
        }
        summary["local_model"] = local_identity
    return summary


def _summarize_output(config, output_dir, local_identity=None):
    expected_identity = local_identity["sha256"] if local_identity is not None else None
    lm_eval_path = _selected_lm_eval_result(output_dir / "lm-eval", expected_identity)
    model_tag = config["model"]["repo"].replace("/", "_")
    arithmark_2_dir = output_dir / "arithmark-2"
    arithmark_3_dir = output_dir / "arithmark-3"
    if (arithmark_2_dir / "selected-result.json").is_file():
        arithmark_2_path = _selected_lm_eval_result(arithmark_2_dir, expected_identity)
    else:
        arithmark_2_path = arithmark_2_dir / (f"{model_tag}_arithmark_2.0_results.json")
    if (arithmark_3_dir / "selected-result.json").is_file():
        arithmark_3_path = _selected_lm_eval_result(arithmark_3_dir, expected_identity)
    else:
        arithmark_3_path = arithmark_3_dir / (f"{model_tag}_arithmark-3_results.json")
    for path in (arithmark_2_path, arithmark_3_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = _summarize(
        config,
        lm_eval_path,
        arithmark_2_path,
        arithmark_3_path,
        local_identity,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Open SLM summary: {summary_path}")
    return summary_path


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("lm-eval", "arithmark-2", "arithmark-3", "summary", "all"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=float, help="lm-eval smoke-test sample limit")
    parser.add_argument(
        "--local-model",
        type=Path,
        help="parity-validated local Transformers export for every benchmark stage",
    )
    args = parser.parse_args()
    if args.limit is not None and (args.limit <= 0 or args.stage not in ("lm-eval",)):
        parser.error("--limit must be positive and is only supported by the lm-eval stage")
    return args


def main():
    args = _parse_args()
    config = _load_config(args.config)
    local_model = _validate_local_export(args.local_model) if args.local_model is not None else None
    output_dir = (
        (args.output_dir or _default_output_dir(config, local_model)).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    local_identity = _bind_local_model(output_dir, local_model) if local_model is not None else None

    if args.stage in ("lm-eval", "all"):
        _run_lm_eval(config, output_dir, args.device, args.limit, local_model)
    if args.stage in ("arithmark-2", "arithmark-3", "all"):
        files = _download_benchmark_files(config)
        if args.stage in ("arithmark-2", "all"):
            _run_arithmark_2(
                config,
                files["arithmark_2"],
                output_dir,
                local_model,
                local_identity,
            )
        if args.stage in ("arithmark-3", "all"):
            _run_arithmark_3(
                config,
                files["arithmark_3"],
                output_dir,
                args.device,
                local_model,
                local_identity,
            )
    if args.stage in ("summary", "all"):
        _summarize_output(config, output_dir, local_identity)


if __name__ == "__main__":
    main()
