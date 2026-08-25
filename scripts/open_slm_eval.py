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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments" / "Speck1-140M" / "open_slm.json"
DEFAULT_OUTPUT_DIR = (
    Path(os.environ.get("speck_base_dir", Path.home() / ".cache" / "speck"))
    / "evaluations"
    / "open-slm"
    / "Speck1-140M"
)


def _load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"leaderboard", "model", "lm_eval", "arithmark_2", "arithmark_3"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Open SLM config is missing: {', '.join(sorted(missing))}")
    return config


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


def _assert_implicit_revisions(config):
    api = HfApi()
    model = config["model"]
    _assert_revision(api, model["repo"], "model", model["revision"], require_main=True)
    for repo, revision in config["lm_eval"]["datasets"].items():
        _assert_revision(api, repo, "dataset", revision, require_main=True)


def _lm_eval_command(config, output_path, device, limit=None):
    model = config["model"]
    evaluation = config["lm_eval"]
    model_args = [
        f"pretrained={model['repo']}",
        f"revision={model['revision']}",
        "trust_remote_code=True",
        f"dtype={evaluation['dtype']}",
        "use_cache=False",
    ]
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


def _run_lm_eval(config, output_dir, device, limit=None):
    _assert_implicit_revisions(config)
    name = "smoke-results.json" if limit is not None else "results.json"
    output_path = output_dir / "lm-eval" / name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before = set(output_path.parent.glob(f"{output_path.stem}_*.json"))
    subprocess.run(_lm_eval_command(config, output_path, device, limit), check=True)
    _assert_implicit_revisions(config)
    result = _new_result(output_path, before)
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


def _run_arithmark_2(config, files, output_dir):
    _assert_implicit_revisions(config)
    benchmark = config["arithmark_2"]
    results_dir = output_dir / "arithmark-2"
    results_dir.mkdir(parents=True, exist_ok=True)
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
            config["model"]["repo"],
            "--batch-size",
            str(benchmark["batch_size"]),
            "--data-path",
            str(files["data"]),
        ]
        official.main()
    finally:
        sys.argv = original_argv
    _assert_implicit_revisions(config)
    result = results_dir / (
        f"{config['model']['repo'].replace('/', '_')}_arithmark_2.0_results.json"
    )
    if not result.is_file():
        raise RuntimeError(f"ArithMark 2.0 did not produce {result}")
    print(f"ArithMark 2.0 result: {result}")
    return result


def _run_arithmark_3(config, files, output_dir, device):
    _assert_implicit_revisions(config)
    benchmark = config["arithmark_3"]
    results_dir = output_dir / "arithmark-3"
    results_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(files["runner"]),
        "--model",
        config["model"]["repo"],
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
    _assert_implicit_revisions(config)
    result = results_dir / (f"{config['model']['repo'].replace('/', '_')}_arithmark-3_results.json")
    if not result.is_file():
        raise RuntimeError(f"ArithMark 3.0 did not produce {result}")
    print(f"ArithMark 3.0 result: {result}")
    return result


def _one_result(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _normalize_from_chance(score, chance):
    return 100 * (score - chance) / (100 - chance)


def _summarize(config, lm_eval_path, arithmark_2_path, arithmark_3_path):
    lm_result = json.loads(lm_eval_path.read_text(encoding="utf-8"))
    arithmark_2 = json.loads(arithmark_2_path.read_text(encoding="utf-8"))
    arithmark_3 = json.loads(arithmark_3_path.read_text(encoding="utf-8"))
    evaluation = config["lm_eval"]

    if lm_result["config"]["limit"] is not None:
        raise ValueError("cannot summarize a limited lm-eval run")
    if lm_result["config"]["model_revision"] != config["model"]["revision"]:
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
    return {
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


def _summarize_output(config, output_dir):
    lm_eval_path = _one_result(output_dir / "lm-eval", "results_*.json")
    model_tag = config["model"]["repo"].replace("/", "_")
    arithmark_2_path = output_dir / "arithmark-2" / (f"{model_tag}_arithmark_2.0_results.json")
    arithmark_3_path = output_dir / "arithmark-3" / (f"{model_tag}_arithmark-3_results.json")
    for path in (arithmark_2_path, arithmark_3_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = _summarize(config, lm_eval_path, arithmark_2_path, arithmark_3_path)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=float, help="lm-eval smoke-test sample limit")
    args = parser.parse_args()
    if args.limit is not None and (args.limit <= 0 or args.stage not in ("lm-eval",)):
        parser.error("--limit must be positive and is only supported by the lm-eval stage")
    return args


def main():
    args = _parse_args()
    config = _load_config(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("lm-eval", "all"):
        _run_lm_eval(config, output_dir, args.device, args.limit)
    if args.stage in ("arithmark-2", "arithmark-3", "all"):
        files = _download_benchmark_files(config)
        if args.stage in ("arithmark-2", "all"):
            _run_arithmark_2(config, files["arithmark_2"], output_dir)
        if args.stage in ("arithmark-3", "all"):
            _run_arithmark_3(config, files["arithmark_3"], output_dir, args.device)
    if args.stage in ("summary", "all"):
        _summarize_output(config, output_dir)


if __name__ == "__main__":
    main()
