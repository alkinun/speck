import hashlib
import json
from pathlib import Path

import speck.tokenizer_pilot_continuation as continuation
from speck.tokenizer_pilot_continuation import (
    materialize_continuation,
    validate_continuation_plan,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Tokenizer:
    def __init__(self, path):
        self.extra = "custom" in Path(path).name

    def encode_batch(self, texts, *, bos, eos):
        assert bos and eos
        return [[1, *([3] * (len(text.split()) + int(self.extra))), 2] for text in texts]


def test_continuation_starts_after_fixed_rows_and_meets_tokenizer_minimums(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "REFERENCE_CONTINUATION_TOKENS", 60)
    monkeypatch.setattr(continuation, "Tokenizer", _Tokenizer)
    reference = tmp_path / "reference.model"
    custom = tmp_path / "custom.model"
    reference.write_text("reference")
    custom.write_text("custom")
    models = {
        "mistral-32k": {"path": str(reference), "sha256": _sha256(reference)},
        "custom": {"path": str(custom), "sha256": _sha256(custom)},
    }
    fixed_categories = []
    categories = []
    for category in continuation.CATEGORIES:
        source = tmp_path / f"{category}.jsonl"
        rows = []
        for index in range(3):
            text = f"{category} document {index} has seven fixed training words"
            rows.append(
                {
                    "text": text,
                    "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        index_path = tmp_path / f"documents-{category}.jsonl"
        index_path.write_text(json.dumps({"source_row": 0}) + "\n")
        identity = {"path": str(source), "sha256": _sha256(source)}
        source_id = f"pilot_train__{category}"
        fixed_categories.append(
            {
                "id": category,
                "source_id": source_id,
                "source": identity,
                "index": {"path": index_path.name, "sha256": _sha256(index_path)},
            }
        )
        categories.append(
            {
                "id": category,
                "source_id": source_id,
                "input": identity,
                "reference_token_target": 10,
            }
        )
    fixed_path = tmp_path / "fixed.json"
    fixed_path.write_text(
        json.dumps(
            {
                "format": "speck_tokenizer_pilot_stream_result",
                "status": "whole_document_stream_and_tokenizer_packs_complete_not_training_authority",
                "model_training_authority": False,
                "document_stream": {"sha256": "a" * 64, "categories": fixed_categories},
                "tokenizers": [
                    {"id": tokenizer_id, "model": model} for tokenizer_id, model in models.items()
                ],
            }
        )
    )
    plan = validate_continuation_plan(
        {
            "format": "speck_tokenizer_pilot_continuation",
            "format_version": 1,
            "status": "materialization_authorized_not_model_training_authority",
            "fixed_stream": {"path": str(fixed_path), "sha256": _sha256(fixed_path)},
            "reference_tokenizer_id": "mistral-32k",
            "tokenizers": [
                {
                    "id": "mistral-32k",
                    "model": models["mistral-32k"],
                    "minimum_additional_tokens": 0,
                },
                {
                    "id": "custom",
                    "model": models["custom"],
                    "minimum_additional_tokens": 66,
                },
            ],
            "categories": categories,
            "shard_tokens": 100,
            "output_directory": str(tmp_path / "continuation"),
        }
    )
    first = materialize_continuation(plan)
    second = materialize_continuation(plan)

    assert first == second
    assert first["continuation"]["actual_reference_tokens"] == 60
    assert all(item["start_after_source_row"] == 0 for item in first["continuation"]["categories"])
    assert {item["tokens"] for item in first["tokenizers"]} == {60, 66}
    assert first["gates"]["minimum_equal_flop_tokens"] == "pass"
    assert first["model_training_authority"] is False
