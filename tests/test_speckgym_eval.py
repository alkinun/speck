import json
from types import SimpleNamespace

import pytest
import torch

from scripts import open_slm_eval, speckgym_eval
from speck.speckgym import load_speckgym_config
from speck.speckgym_eval import (
    EVALUATION_FAMILIES,
    _validate_language_checkpoint,
    cases_fingerprint,
    generate_cases,
    score_cases,
    summarize_scores,
)


class CharacterTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, bos=False, eos=False):
        values = [
            ord(character) - ord("a") + 3 for character in text.lower() if character.isalpha()
        ]
        return ([self.bos_id] if bos else []) + values + ([self.eos_id] if eos else [])


class EchoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_position_embeddings=32)

    def forward(self, tokens):
        logits = torch.zeros((*tokens.shape, 32), device=tokens.device) + self.anchor
        return logits.scatter(-1, tokens.unsqueeze(-1), 10.0)


def test_heldout_cases_are_deterministic_balanced_and_well_formed():
    first = generate_cases(20250829, 8)
    second = generate_cases(20250829, 8)

    assert first == second
    assert first != generate_cases(20250830, 8)
    assert len(first) == len(EVALUATION_FAMILIES) * 8
    assert {case["family"] for case in first} == set(EVALUATION_FAMILIES)
    for case in first:
        assert len(case["choices"]) == len(set(case["choices"])) == 4
        assert 0 <= case["answer"] < 4
        assert case["prompt"]
        if case["family"] == "set_union":
            assert len({choice.count(",") for choice in case["choices"]}) == 1


def test_conditional_scoring_selects_the_best_continuation():
    cases = [
        {
            "id": "echo-0000",
            "family": "hierarchy",
            "prompt": "a",
            "choices": ["a", "b", "c", "d"],
            "answer": 0,
        }
    ]
    scores = score_cases(EchoModel(), CharacterTokenizer(), cases, batch_size=2)
    assert scores[0][0] > max(scores[0][1:])


def test_score_summary_reports_family_and_overall_accuracy():
    cases = generate_cases(9, 1)
    scores = []
    for case in cases:
        values = [0.0] * 4
        values[case["answer"]] = 1.0
        scores.append(values)

    metrics, predictions = summarize_scores(cases, scores)

    assert metrics["overall"] == {"accuracy": 1.0, "chance_accuracy": 0.25, "samples": 6}
    assert all(metrics[family]["accuracy"] == 1.0 for family in EVALUATION_FAMILIES)
    assert all(
        prediction["prediction"] == case["answer"] for prediction, case in zip(predictions, cases)
    )


def test_checkpoint_validation_rejects_a_different_run_label():
    train = {
        "run": "SpeckGym-v0-E",
        "batch_tokens": 40,
        "train_tokens": 80,
        "global_token_offset": 80,
        "checkpoint_tokens": [120, 160],
        "initialization": {"kind": "backbone_checkpoint"},
    }
    metadata = {
        "step": 2,
        "training_phase": "language",
        "milestone_tokens": 160,
        "global_tokens": 160,
        "validation_global_tokens": 160,
        "resolved": {**train, "run": "SpeckGym-v0-B"},
    }
    with pytest.raises(ValueError, match="selected SpeckGym run: run"):
        _validate_language_checkpoint(metadata, train, 2, 160)


def test_speckgym_eval_parser_is_import_safe():
    args = speckgym_eval.parse_args(["E", "500000000", "procedural", "--device", "cpu"])
    assert args.run == "E"
    assert args.tokens == 500_000_000
    assert args.stage == "procedural"
    assert args.device == "cpu"


def test_procedural_summary_rejects_a_different_checkpoint():
    suite = load_speckgym_config()
    cases = generate_cases(
        suite["evaluation"]["seed"],
        suite["evaluation"]["cases_per_family"],
        suite["evaluation"]["families"],
    )
    checkpoint = {"model_sha256": "current"}
    report = {
        "run": "E",
        "requested_tokens": 500_000_000,
        "actual_tokens": 500_039_680,
        "checkpoint": checkpoint,
        "cases": {
            "seed": suite["evaluation"]["seed"],
            "cases_per_family": suite["evaluation"]["cases_per_family"],
            "families": suite["evaluation"]["families"],
            "sha256": cases_fingerprint(cases),
        },
        "metrics": {"overall": {"accuracy": 0.5}},
    }
    assert speckgym_eval._procedural_scores(
        suite, report, "E", 500_000_000, 500_039_680, checkpoint
    ) == {"overall": {"accuracy": 0.5}}

    report["checkpoint"] = {"model_sha256": "stale"}
    with pytest.raises(ValueError, match="selected checkpoint"):
        speckgym_eval._procedural_scores(suite, report, "E", 500_000_000, 500_039_680, checkpoint)


def test_standard_summary_uses_the_identified_result_when_old_results_exist(tmp_path):
    suite = load_speckgym_config()
    config_path = suite["evaluation"]["standard_config"]
    config = open_slm_eval._load_config(config_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    result_dir = tmp_path / "standard" / "lm-eval"
    result_dir.mkdir(parents=True)
    result = result_dir / "results_current.json"
    result.write_text(
        json.dumps(
            {
                "config": {"limit": None},
                "lm_eval_version": config["lm_eval"]["version"],
                "transformers_version": config["lm_eval"]["transformers_version"],
                "results": {task: {"acc_norm,none": 0.5} for task in config["lm_eval"]["tasks"]},
                "n-samples": {
                    task: {"original": samples, "effective": samples}
                    for task, samples in config["lm_eval"]["expected_samples"].items()
                },
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "results_old.json").write_text("stale", encoding="utf-8")
    checkpoint = {"model_sha256": "current"}
    identity = {
        "checkpoint": checkpoint,
        "evaluation_config_sha256": open_slm_eval._sha256(config_path),
        "limit": None,
        "local_model": open_slm_eval._local_model_identity(model),
        "result": {
            "path": result.relative_to(tmp_path).as_posix(),
            "sha256": open_slm_eval._sha256(result),
        },
    }
    (tmp_path / "standard_identity.json").write_text(json.dumps(identity), encoding="utf-8")

    summary = speckgym_eval._standard_scores(suite, tmp_path, checkpoint)

    assert summary["scores"] == {task: 0.5 for task in config["lm_eval"]["tasks"]}
