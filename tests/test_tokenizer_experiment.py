import hashlib
import json
import shutil
from pathlib import Path

import pytest

import speck.tokenizer_experiment as tokenizer_experiment
from speck.chat import ChatTokenizer
from speck.tokenizer import Tokenizer
from speck.tokenizer_experiment import (
    evaluate_tokenizers,
    prepare_baseline,
    prepare_sample,
    train_candidate,
    validate_experiment_config,
)

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _config(tmp_path, output_name="run"):
    inputs = []
    for category in ("web", "code"):
        path = tmp_path / f"{category}.jsonl"
        rows = [
            {
                "text": (
                    f"{category} document {index}: value_{index} = {index} * {index + 1}\n"
                    f"A deterministic second line for tokenizer training number {index}."
                )
            }
            for index in range(500)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        inputs.append(
            {
                "id": category,
                "inputs": [
                    {
                        "id": f"{category}-fixture",
                        "path": str(path),
                        "format": "jsonl",
                        "text_column": "text",
                        "sha256": _sha256(path),
                        "training_bytes": 4_000,
                        "evaluation_bytes": 1_000,
                    }
                ],
            }
        )
    return {
        "format": "speck_tokenizer_experiment",
        "format_version": 1,
        "seed": 42,
        "output_dir": str(tmp_path / output_name),
        "sample": {
            "training_bytes_per_category": 4_000,
            "evaluation_bytes_per_category": 1_000,
            "evaluation_modulus": 2,
            "evaluation_remainders": [0],
            "min_chars": 1,
            "max_chars": 1_000,
            "categories": inputs,
        },
        "trainer": {
            "model_type": "bpe",
            "character_coverage": 1.0,
            "byte_fallback": True,
            "normalization_rule_name": "identity",
            "remove_extra_whitespaces": False,
            "add_dummy_prefix": False,
            "split_digits": True,
            "split_by_unicode_script": True,
            "split_by_whitespace": True,
            "split_by_number": True,
            "max_sentence_length": 1_000,
            "num_threads": 1,
            "hard_vocab_limit": False,
        },
        "candidates": [{"id": "speck-test-300", "vocab_size": 300}],
        "baselines": [],
        "evaluation": {
            "embedding_width": 64,
            "tied_embeddings": False,
            "chat_added_tokens": 3,
            "probe_strings": [
                "def f(x):\n\treturn x + 1\n",
                "x² + y₁ = 42",
                "two  spaces\n\nnext",
            ],
        },
    }


def _sample_hashes(output, split):
    values = set()
    for path in (output / "sample").glob(f"{split}-*.jsonl"):
        values.update(
            json.loads(line)["content_sha256"]
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    return values


def test_sample_is_balanced_disjoint_and_checksum_bound(tmp_path):
    config = validate_experiment_config(_config(tmp_path))
    manifest = prepare_sample(config)
    output = Path(config["output_dir"])

    assert [category["id"] for category in manifest["categories"]] == ["web", "code"]
    for category in manifest["categories"]:
        assert category["splits"]["train"]["utf8_bytes"] >= 4_000
        assert category["splits"]["eval"]["utf8_bytes"] >= 1_000
        assert category["splits"]["train"]["sha256"] == _sha256(
            output / "sample" / category["splits"]["train"]["path"]
        )
    assert _sample_hashes(output, "train").isdisjoint(_sample_hashes(output, "eval"))

    changed = _config(tmp_path, "changed")
    fixture = tmp_path / "web.jsonl"
    fixture.write_text(fixture.read_text() + json.dumps({"text": "changed"}) + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_sample(validate_experiment_config(changed))


def test_training_is_path_independent_and_static_evaluation_is_non_authoritative(tmp_path):
    first = validate_experiment_config(_config(tmp_path, "first"))
    second = validate_experiment_config(_config(tmp_path, "second"))
    prepare_sample(first)
    prepare_sample(second)
    first_manifest = train_candidate(first, "speck-test-300")
    second_manifest = train_candidate(second, "speck-test-300")

    assert first_manifest["model"]["sha256"] == second_manifest["model"]["sha256"]
    assert first_manifest["model"]["vocab_size"] <= 300
    assert first_manifest["model"]["unk_id"] == 0
    report = evaluate_tokenizers(first)
    assert evaluate_tokenizers(first) == report
    result = report["tokenizers"][0]

    assert report["selection_authority"] is False
    assert report["primary_baseline"] is None
    assert result["static_hard_gates"] == {
        "uint16_with_chat_tokens": True,
        "zero_unknown_tokens": True,
        "probe_roundtrip_exact": True,
    }
    assert set(result["per_category"]) == {"web", "code"}
    assert result["embedding_and_head_parameters"] == (result["vocab_size"] + 3) * 64 * 2


def test_config_rejects_vocabularies_that_overflow_packed_chat_ids(tmp_path):
    config = _config(tmp_path)
    config["candidates"][0]["vocab_size"] = 65_534
    with pytest.raises(ValueError, match="exceeds uint16 capacity"):
        validate_experiment_config(config)


def test_config_requires_explicit_input_quotas_to_cover_each_category(tmp_path):
    config = _config(tmp_path)
    config["sample"]["categories"][0]["inputs"][0]["training_bytes"] -= 1
    with pytest.raises(ValueError, match="input training bytes must sum"):
        validate_experiment_config(config)


def test_whitespace_only_piece_setting_is_explicit_hash_bound_and_forwarded(tmp_path):
    legacy = validate_experiment_config(_config(tmp_path, "legacy"))
    raw = _config(tmp_path, "whitespace")
    raw["trainer"]["allow_whitespace_only_pieces"] = True
    config = validate_experiment_config(raw)

    assert "allow_whitespace_only_pieces" not in legacy["trainer"]
    assert config["trainer"]["allow_whitespace_only_pieces"] is True
    assert config["plan_fingerprint"] != legacy["plan_fingerprint"]
    prepare_sample(config)
    manifest = train_candidate(config, "speck-test-300")
    assert manifest["trainer"]["allow_whitespace_only_pieces"] is True
    model_path = Path(config["output_dir"]) / "candidates/speck-test-300/tokenizer.model"
    tokenizer = Tokenizer(model_path)
    assert (
        tokenizer.decode(tokenizer.encode("two  spaces\n\tand code")) == "two  spaces\n\tand code"
    )
    export = tmp_path / "tokenizer-export"
    ChatTokenizer(tokenizer).save_pretrained(export)
    metadata = json.loads((export / "tokenizer_metadata.json").read_text())
    assert metadata["base_fingerprint"] == manifest["model"]["sha256"]
    assert metadata["vocab_size"] == tokenizer.vocab_size + 3
    assert (export / "tokenizer.model").read_bytes() == model_path.read_bytes()

    raw["trainer"]["allow_whitespace_only_pieces"] = 1
    with pytest.raises(ValueError, match="allow_whitespace_only_pieces"):
        validate_experiment_config(raw)


def test_normalized_config_cannot_change_after_its_fingerprint_is_frozen(tmp_path):
    config = validate_experiment_config(_config(tmp_path))
    config["candidates"][0]["vocab_size"] = 301
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        prepare_sample(config)


def test_declared_baseline_is_pinned_and_becomes_the_static_reference(tmp_path, monkeypatch):
    raw = _config(tmp_path)
    raw["trainer"]["hard_vocab_limit"] = True
    raw["baselines"] = [
        {
            "id": "baseline-test-300",
            "repo": "example/tokenizer",
            "revision": "a" * 40,
            "filename": "tokenizer.model",
            "expected_vocab_size": 300,
        }
    ]
    config = validate_experiment_config(raw)
    prepare_sample(config)
    train_candidate(config, "speck-test-300")
    source = Path(config["output_dir"]) / "candidates" / "speck-test-300" / "tokenizer.model"

    def download(*, filename, local_dir, **_):
        destination = Path(local_dir) / filename
        shutil.copy2(source, destination)
        return str(destination)

    monkeypatch.setattr(tokenizer_experiment, "hf_hub_download", download)
    baseline = prepare_baseline(config, "baseline-test-300")
    report = evaluate_tokenizers(config)

    assert baseline["model"]["sha256"] == _sha256(source)
    assert report["primary_baseline"] == "baseline-test-300"
    assert {result["kind"] for result in report["tokenizers"]} == {"candidate", "baseline"}


def test_flagship_tokenizer_plan_is_balanced_and_requires_an_lm_pilot():
    plan = json.loads((ROOT / "research" / "flagship" / "tokenizer_plan.json").read_text())

    assert plan["format"] == "speck_flagship_tokenizer_plan"
    assert plan["status"] == "tooling_ready_source_inputs_pending"
    assert plan["categories"] == [
        "web",
        "code",
        "math",
        "synthetic",
        "science",
        "reference",
    ]
    assert plan["sample"]["training_bytes_per_category"] * len(plan["categories"]) == 600_000_000
    assert plan["sample"]["evaluation_bytes_per_category"] * len(plan["categories"]) == 60_000_000
    assert [candidate["vocab_size"] for candidate in plan["candidates"]] == [
        32_768,
        40_960,
        49_152,
    ]
    assert plan["language_model_pilot"]["total_runs"] == (
        plan["language_model_pilot"]["screen_runs"]
        + plan["language_model_pilot"]["confirmation_runs"]
    )
    assert plan["static_evaluation"]["selection_authority"] is False
    assert plan["language_model_pilot"]["fallback"] == "mistral-32k"
