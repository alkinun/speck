import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import open_slm_eval
from speck.checkpoint import directory_identity


def load_config():
    return open_slm_eval._load_config(open_slm_eval.DEFAULT_CONFIG)


def test_lm_eval_command_pins_model_and_numerical_settings(tmp_path):
    config = load_config()

    command = open_slm_eval._lm_eval_command(config, tmp_path / "result.json", "cuda")

    model_args_start = command.index("--model_args") + 1
    tasks_start = command.index("--tasks")
    model_args = command[model_args_start:tasks_start]
    assert "revision=155b759545645cc694545fab85cd7d4c385fd965" in model_args
    assert "dtype=bfloat16" in model_args
    assert "use_cache=False" in model_args
    assert command[command.index("--num_fewshot") + 1] == "0"
    assert command[command.index("--batch_size") + 1] == "32"


def test_lm_eval_command_accepts_a_local_export_without_a_model_revision(tmp_path):
    config = load_config()
    model = tmp_path / "export"
    command = open_slm_eval._lm_eval_command(
        config, tmp_path / "result.json", "cuda", local_model=model
    )

    model_args_start = command.index("--model_args") + 1
    tasks_start = command.index("--tasks")
    model_args = command[model_args_start:tasks_start]
    assert f"pretrained={model}" in model_args
    assert not any(argument.startswith("revision=") for argument in model_args)


def test_local_model_identity_changes_with_export_contents(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "weights.bin").write_bytes(b"first")
    first = directory_identity(model)

    (model / "weights.bin").write_bytes(b"second")
    second = directory_identity(model)

    assert first["path"] == str(model.resolve())
    assert first["sha256"] != second["sha256"]
    assert [entry["path"] for entry in first["files"]] == ["config.json", "weights.bin"]


def test_local_lm_eval_identity_is_bound_to_one_successful_result(tmp_path, monkeypatch):
    config = load_config()
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    result = tmp_path / "output" / "lm-eval" / "results_2026.json"

    monkeypatch.setattr(open_slm_eval, "_assert_implicit_revisions", lambda *args, **kwargs: None)

    def run(command, check):
        assert check
        result.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(open_slm_eval.subprocess, "run", run)
    returned = open_slm_eval._run_lm_eval(config, tmp_path / "output", "cpu", local_model=model)

    assert returned == result
    identity_path = result.parent / "local-identities" / f"{result.name}.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assert identity["local_model"] == directory_identity(model)
    assert identity["result"] == {
        "path": result.name,
        "sha256": open_slm_eval._sha256(result),
    }
    selection = json.loads((result.parent / "selected-result.json").read_text(encoding="utf-8"))
    assert selection == {
        "format_version": 1,
        "local_model_sha256": directory_identity(model)["sha256"],
        "path": result.name,
        "sha256": open_slm_eval._sha256(result),
    }


def test_local_default_output_is_collision_free_by_directory_hash(tmp_path):
    config = load_config()
    model = tmp_path / "M1-step-7630"
    model.mkdir()
    (model / "weights").write_bytes(b"first")

    first = open_slm_eval._default_output_dir(config, model)
    (model / "weights").write_bytes(b"second")
    second = open_slm_eval._default_output_dir(config, model)

    assert first.name.startswith("M1-step-7630--")
    assert first != second


def test_local_export_requires_successful_parity_attestation(tmp_path):
    model = tmp_path / "export"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"expected_parameters": 3}), encoding="utf-8"
    )
    (model / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="parity artifacts"):
        open_slm_eval._validate_local_export(model)

    (model / "speck_parity.json").write_text(
        json.dumps(
            {
                "format": "speck_export_parity",
                "passed": True,
                "parameters": 3,
            }
        ),
        encoding="utf-8",
    )
    assert open_slm_eval._validate_local_export(model) == model.resolve()


def test_output_directory_binding_rejects_changed_local_export(tmp_path):
    model = tmp_path / "model"
    output = tmp_path / "output"
    model.mkdir()
    output.mkdir()
    (model / "weights").write_bytes(b"first")
    identity = open_slm_eval._bind_local_model(output, model)

    assert identity == directory_identity(model)
    (model / "weights").write_bytes(b"second")
    with pytest.raises(ValueError, match="different local export"):
        open_slm_eval._bind_local_model(output, model)


def test_lm_eval_result_selection_survives_reruns_and_checks_integrity(tmp_path):
    directory = tmp_path / "lm-eval"
    directory.mkdir()
    first = directory / "results_first.json"
    second = directory / "results_second.json"
    first.write_text('{"run": 1}', encoding="utf-8")
    second.write_text('{"run": 2}', encoding="utf-8")

    open_slm_eval._record_lm_eval_result(second)

    assert open_slm_eval._selected_lm_eval_result(directory) == second
    second.write_text('{"run": "changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        open_slm_eval._selected_lm_eval_result(directory)


def test_selected_result_rejects_a_different_local_export_identity(tmp_path):
    result = tmp_path / "results_local.json"
    result.write_text("{}", encoding="utf-8")
    open_slm_eval._record_lm_eval_result(result, {"sha256": "first"})

    assert open_slm_eval._selected_lm_eval_result(tmp_path, "first") == result
    with pytest.raises(ValueError, match="different local export"):
        open_slm_eval._selected_lm_eval_result(tmp_path, "second")


def test_legacy_single_lm_eval_result_remains_summarizable(tmp_path):
    result = tmp_path / "results_only.json"
    result.write_text("{}", encoding="utf-8")

    assert open_slm_eval._selected_lm_eval_result(tmp_path) == result


@pytest.mark.parametrize(
    ("path", "repo", "revision"),
    [
        (
            "experiments/Speck1-140M-Instruct/open_slm.json",
            "specklabs/Speck1-140M-Instruct",
            "686350e82db5996f9ab65bdadca70c6d41d49227",
        ),
        (
            "experiments/Speck1.1-140M-Instruct/open_slm.json",
            "specklabs/Speck1.1-140M-Instruct",
            "4ed4c6824b8dd37ecaa72df5dbbc531f55871588",
        ),
    ],
)
def test_instruct_configs_inherit_benchmarks_and_select_model(path, repo, revision):
    config = open_slm_eval._load_config(open_slm_eval.REPOSITORY_ROOT / path)

    assert config["model"] == {
        "repo": repo,
        "revision": revision,
        "parameters": 140654208,
    }
    assert config["lm_eval"]["tasks"] == [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "piqa",
    ]
    assert open_slm_eval._default_output_dir(config).name == repo.rsplit("/", 1)[-1]


def test_arithmark_2_shim_only_disables_model_cache():
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    tokenizer = object()
    calls = []
    official = SimpleNamespace(
        load_hf_model=lambda *args, **kwargs: calls.append((args, kwargs)) or (model, tokenizer)
    )

    open_slm_eval._disable_arithmark_2_cache(official)
    loaded_model, loaded_tokenizer = official.load_hf_model("model", "tokenizer", "cuda")

    assert calls == [(("model", "tokenizer", "cuda"), {})]
    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert model.config.use_cache is False


def test_arithmark_2_accepts_and_binds_a_local_export(tmp_path, monkeypatch):
    config = load_config()
    model = tmp_path / "export"
    model.mkdir()
    (model / "weights").write_bytes(b"weights")
    identity = directory_identity(model)
    seen = {}

    class Official:
        CACHE_DIR = None

        @staticmethod
        def load_hf_model():
            raise AssertionError("not called by fake runner")

        @staticmethod
        def main():
            import sys

            seen["argv"] = sys.argv[:]
            path = Path(Official.CACHE_DIR) / "local_arithmark_2.0_results.json"
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(open_slm_eval, "_load_module", lambda *args: Official)
    monkeypatch.setattr(open_slm_eval, "_disable_arithmark_2_cache", lambda official: None)
    monkeypatch.setattr(open_slm_eval, "_assert_implicit_revisions", lambda *a, **kw: None)
    files = {"runner": tmp_path / "runner.py", "data": tmp_path / "data.jsonl"}

    result = open_slm_eval._run_arithmark_2(
        config,
        files,
        tmp_path / "output",
        local_model=model,
        local_identity=identity,
    )

    assert seen["argv"][seen["argv"].index("--model") + 1] == str(model)
    selection = json.loads((result.parent / "selected-result.json").read_text())
    assert selection["local_model_sha256"] == identity["sha256"]


def test_arithmark_3_accepts_and_binds_a_local_export(tmp_path, monkeypatch):
    config = load_config()
    model = tmp_path / "export"
    model.mkdir()
    (model / "weights").write_bytes(b"weights")
    identity = directory_identity(model)
    seen = {}

    def run(command, check):
        assert check
        seen["command"] = command
        directory = Path(command[command.index("--results-dir") + 1])
        (directory / "local_arithmark-3_results.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(open_slm_eval.subprocess, "run", run)
    monkeypatch.setattr(open_slm_eval, "_assert_implicit_revisions", lambda *a, **kw: None)
    files = {"runner": tmp_path / "runner.py", "data": tmp_path / "data.jsonl"}

    result = open_slm_eval._run_arithmark_3(
        config,
        files,
        tmp_path / "output",
        "cpu",
        local_model=model,
        local_identity=identity,
    )

    assert seen["command"][seen["command"].index("--model") + 1] == str(model)
    selection = json.loads((result.parent / "selected-result.json").read_text())
    assert selection["local_model_sha256"] == identity["sha256"]


def test_summary_matches_leaderboard_formula(tmp_path):
    config = load_config()
    lm_eval_path = tmp_path / "lm-eval.json"
    arithmark_2_path = tmp_path / "arithmark-2.json"
    arithmark_3_path = tmp_path / "arithmark-3.json"
    task_scores = {
        "hellaswag": 0.40,
        "arc_easy": 0.50,
        "arc_challenge": 0.30,
        "piqa": 0.60,
    }
    lm_eval_path.write_text(
        json.dumps(
            {
                "config": {"limit": None, "model_revision": config["model"]["revision"]},
                "lm_eval_version": config["lm_eval"]["version"],
                "transformers_version": config["lm_eval"]["transformers_version"],
                "results": {task: {"acc_norm,none": score} for task, score in task_scores.items()},
                "n-samples": {
                    task: {"original": total, "effective": total}
                    for task, total in config["lm_eval"]["expected_samples"].items()
                },
            }
        ),
        encoding="utf-8",
    )
    arithmark_2_path.write_text(
        json.dumps({"results": {"arithmark_2.0": {"acc": 31.0, "total": 2500}}}),
        encoding="utf-8",
    )
    arithmark_3_path.write_text(
        json.dumps({"results": {"arithmark-3": {"acc_norm": 35.0, "total": 1000}}}),
        encoding="utf-8",
    )

    summary = open_slm_eval._summarize(config, lm_eval_path, arithmark_2_path, arithmark_3_path)

    assert summary["scores_percent"] == {
        "hellaswag": 40.0,
        "arc_easy": 50.0,
        "arc_challenge": 30.0,
        "piqa": 60.0,
        "arithmark_2": 31.0,
        "arithmark_3": 35.0,
        "combined_arc": 40.0,
        "average": 43.75,
        "intelligence_index": 18.81,
    }


def test_local_summary_is_bound_to_export_identity_not_hub_revision(tmp_path):
    config = load_config()
    lm_eval_path = tmp_path / "lm-eval.json"
    arithmark_2_path = tmp_path / "arithmark-2.json"
    arithmark_3_path = tmp_path / "arithmark-3.json"
    lm_eval_path.write_text(
        json.dumps(
            {
                "config": {"limit": None, "model_revision": None},
                "lm_eval_version": config["lm_eval"]["version"],
                "transformers_version": config["lm_eval"]["transformers_version"],
                "results": {
                    task: {"acc_norm,none": 0.5} for task in config["lm_eval"]["tasks"]
                },
                "n-samples": {
                    task: {"original": count, "effective": count}
                    for task, count in config["lm_eval"]["expected_samples"].items()
                },
            }
        ),
        encoding="utf-8",
    )
    arithmark_2_path.write_text(
        json.dumps({"results": {"arithmark_2.0": {"acc": 50.0, "total": 2500}}}),
        encoding="utf-8",
    )
    arithmark_3_path.write_text(
        json.dumps(
            {"results": {"arithmark-3": {"acc_norm": 50.0, "total": 1000}}}
        ),
        encoding="utf-8",
    )
    identity = {"path": "/export", "sha256": "directory-hash", "files": []}

    summary = open_slm_eval._summarize(
        config,
        lm_eval_path,
        arithmark_2_path,
        arithmark_3_path,
        local_identity=identity,
    )

    assert summary["local_model"] == identity
    assert summary["model"] == {
        "type": "local_export",
        "path": "/export",
        "directory_sha256": "directory-hash",
        "parameters": config["model"]["parameters"],
    }


def test_summary_rejects_limited_lm_eval_result(tmp_path):
    config = load_config()
    path = tmp_path / "limited.json"
    path.write_text(json.dumps({"config": {"limit": 2}}), encoding="utf-8")

    with pytest.raises(ValueError, match="limited"):
        open_slm_eval._summarize(config, path, path, path)
