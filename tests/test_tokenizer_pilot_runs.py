import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from speck.tokenizer_pilot_runs import (
    build_screen_run_manifests,
    materialize_screen_runs,
    validate_run_materialization_plan,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture_plan(tmp_path):
    repository = tmp_path / "repository"
    flagship = repository / "research" / "flagship"
    tokenizers = (
        ("mistral-32k", "mistral-32k", 32_000, 32, 16, 32, 32),
        (
            "compression_endpoint",
            "speck-bpe-40960-whitespace",
            40_960,
            28,
            20,
            40,
            48,
        ),
        ("compact_endpoint", "speck-bpe-32000-whitespace", 32_000, 30, 18, 44, 48),
    )
    models = {
        tokenizer_id: write_file(repository / "tokenizers" / tokenizer_id / "tokenizer.model", role)
        for role, tokenizer_id, *_ in tokenizers
    }
    preflight = {
        "tokenizers": [
            {
                "id": tokenizer_id,
                "vocab_size": vocab_size,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "fixed_flop_token_stop": fixed_flop,
                "run_stop_aligned_tokens": run_stop,
            }
            for _, tokenizer_id, vocab_size, _, _, fixed_flop, run_stop in tokenizers
        ]
    }
    preflight_identity = write_json(flagship / "preflight.json", preflight)
    scale = {
        "format": "speck_flagship_scale_target_spec",
        "targets": [
            {
                "id": "ladder-60m",
                "embedding_size": 16,
                "hidden_size": 16,
                "depth": 4,
                "intermediate_size": 32,
                "head_dim": 4,
                "num_key_heads": 1,
                "num_value_heads": 2,
                "num_key_value_heads": 1,
            }
        ],
    }
    scale_identity = write_json(flagship / "scale.json", scale)

    def packed_entries(offset):
        return [
            {
                "id": category,
                "tokens": index + 1,
                "shards": [{"path": "train.bin", "sha256": f"shard-{index}", "tokens": 1}],
            }
            for index, category in enumerate(
                ("web", "code", "math", "synthetic", "science", "reference"), start=offset
            )
        ]

    fixed = {
        "format": "speck_tokenizer_pilot_stream_result",
        "status": "whole_document_stream_and_tokenizer_packs_complete_not_training_authority",
        "document_stream": {"sha256": "fixed-document-stream"},
        "tokenizers": [
            {
                "id": tokenizer_id,
                "model": models[tokenizer_id],
                "tokens": fixed_tokens,
                "categories": packed_entries(position),
            }
            for position, (_, tokenizer_id, _, fixed_tokens, _, _, _) in enumerate(tokenizers)
        ],
    }
    fixed_identity = write_json(repository / "runtime" / "fixed" / "manifest.json", fixed)
    continuation = {
        "format": "speck_tokenizer_pilot_continuation_result",
        "status": "shared_equal_flop_continuation_complete_not_training_authority",
        "fixed_stream": {"sha256": fixed_identity["sha256"]},
        "tokenizers": [
            {
                "id": tokenizer_id,
                "model": models[tokenizer_id],
                "tokens": continuation_tokens,
                "categories": packed_entries(position + 10),
            }
            for position, (_, tokenizer_id, _, _, continuation_tokens, _, _) in enumerate(
                tokenizers
            )
        ],
    }
    continuation_identity = write_json(
        repository / "runtime" / "continuation" / "manifest.json", continuation
    )
    categories = []
    for category in ("web", "code", "math", "synthetic", "science", "reference"):
        evaluation_file = write_file(
            repository / "evaluation" / f"eval-{category}.jsonl", '{"text":"value"}\n'
        )
        categories.append(
            {
                "id": category,
                "splits": {
                    "eval": {
                        "path": Path(evaluation_file["path"]).name,
                        "sha256": evaluation_file["sha256"],
                        "documents": 1,
                        "utf8_bytes": 5,
                    }
                },
            }
        )
    evaluation_identity = write_json(
        repository / "evaluation" / "manifest.json",
        {"format": "speck_tokenizer_sample", "categories": categories},
    )
    pilot = {
        "format": "speck_tokenizer_pilot_plan",
        "format_version": 10,
        "screen": {
            "seed": 42,
            "runs": ["mistral-32k", "compression_endpoint", "compact_endpoint"],
        },
        "execution_preflight": {
            "status": "pass_screen_only",
            "plan": {
                "path": "research/flagship/preflight.json",
                "sha256": preflight_identity["sha256"],
            },
        },
        "model_geometry": {"source_spec": {"sha256": scale_identity["sha256"]}},
        "fixed_stream": {"runtime_manifest": {"sha256": fixed_identity["sha256"]}},
        "fixed_flop_materialization": {
            "continuation": {"runtime_manifest": {"sha256": continuation_identity["sha256"]}}
        },
        "final_selection_authority": False,
    }
    pilot_identity = write_json(flagship / "pilot.json", pilot)
    implementation = {
        name: write_file(repository / "implementation" / f"{name}.py", name)
        for name in ("module", "cli", "tests")
    }
    plan = {
        "format": "speck_tokenizer_pilot_run_materialization",
        "format_version": 1,
        "status": "screen_materialization_authorized_no_model_outputs",
        "pilot_plan": pilot_identity,
        "scale_spec": scale_identity,
        "target_id": "ladder-60m",
        "fixed_stream": fixed_identity,
        "continuation": continuation_identity,
        "evaluation_sample": evaluation_identity,
        "implementation": implementation,
        "tokenizers": [
            {
                "role": role,
                "id": tokenizer_id,
                "model": models[tokenizer_id],
                "vocab_size": vocab_size,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "fixed_flop_token_stop": fixed_flop,
                "run_stop_aligned_tokens": run_stop,
            }
            for role, tokenizer_id, vocab_size, _, _, fixed_flop, run_stop in tokenizers
        ],
        "screen": {
            "seed": 42,
            "runs": ["mistral-32k", "compression_endpoint", "compact_endpoint"],
        },
        "settings": {
            "sequence_length": 4,
            "device_batch_size": 2,
            "batch_tokens": 16,
            "accumulation": 2,
            "optimizer": "muon",
            "learning_rate": 0.0015,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "lr_schedule": "cosine",
            "warmup_steps": 1,
            "min_lr": 0.1,
            "loss_backend": "torch",
            "compile": True,
            "checkpoint_every_steps": 2,
            "learning_curve_every_steps": 1,
        },
        "output_directory": str(repository / "runs"),
        "authority": {
            "screen_execution": True,
            "confirmation_execution": False,
            "D5_opening": False,
            "final_selection": False,
            "flagship_training": False,
        },
    }
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True)
    return plan


def test_builds_three_role_resolved_screen_manifests(tmp_path):
    plan = validate_run_materialization_plan(fixture_plan(tmp_path))

    runs = build_screen_run_manifests(plan, repository_revision="fixture-revision")

    assert [run["tokenizer_id"] for run in runs] == [
        "mistral-32k",
        "speck-bpe-40960-whitespace",
        "speck-bpe-32000-whitespace",
    ]
    assert {run["seed"] for run in runs} == {42}
    assert {run["repository_revision"] for run in runs} == {"fixture-revision"}
    assert set(runs[0]["materializer_implementation"]) == {"module", "cli", "tests"}
    assert len({run["model"]["backbone_sha256"] for run in runs}) == 1
    assert runs[0]["stops"]["final_step"] == 2
    assert runs[1]["stops"]["final_step"] == 3
    assert all(run["authority"]["D5_opening"] is False for run in runs)


def test_materialization_is_exclusive_and_keeps_runs_unstarted(tmp_path):
    plan = validate_run_materialization_plan(fixture_plan(tmp_path))

    result = materialize_screen_runs(plan)

    assert result["status"] == "three_screen_runs_materialized_not_started"
    for entry in result["runs"]:
        path = Path(plan["output_directory"]) / entry["path"]
        assert json.loads(path.read_text())["status"] == "materialized_not_started"
    with pytest.raises(FileExistsError, match="already exists"):
        materialize_screen_runs(plan)


def test_rejects_stream_lineage_or_capacity_mismatch(tmp_path):
    plan = fixture_plan(tmp_path)
    continuation_path = Path(plan["continuation"]["path"])
    continuation = json.loads(continuation_path.read_text())
    continuation["fixed_stream"]["sha256"] = "wrong"
    plan["continuation"] = write_json(continuation_path, continuation)

    with pytest.raises(ValueError, match="inputs differ from the v10 pilot plan"):
        validate_run_materialization_plan(plan)

    plan = fixture_plan(tmp_path / "capacity")
    plan["tokenizers"][1]["run_stop_aligned_tokens"] = 64
    with pytest.raises(ValueError, match="not executable"):
        validate_run_materialization_plan(plan)


def test_rejects_authority_expansion_or_batch_mismatch(tmp_path):
    plan = fixture_plan(tmp_path)
    plan["authority"]["D5_opening"] = True
    with pytest.raises(ValueError, match="screen-only"):
        validate_run_materialization_plan(plan)

    plan = fixture_plan(tmp_path / "batch")
    plan["settings"]["batch_tokens"] = 32
    with pytest.raises(ValueError, match="batch geometry"):
        validate_run_materialization_plan(plan)
