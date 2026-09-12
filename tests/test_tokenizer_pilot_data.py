import hashlib
import json
from pathlib import Path

import speck.tokenizer_pilot_data as pilot_data
from speck.tokenizer_pilot_data import (
    materialize_pilot_stream,
    validate_pilot_stream_plan,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Tokenizer:
    def __init__(self, path):
        self.extra = "custom" in Path(path).name

    def encode_batch(self, texts, *, bos, eos):
        assert bos and eos
        return [[1, *([3] * (len(text.split()) + int(self.extra))), 2] for text in texts]


def test_whole_document_stream_is_shared_and_packed_under_each_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot_data, "REFERENCE_TOKENS", 60)
    monkeypatch.setattr(pilot_data, "Tokenizer", _Tokenizer)
    categories = []
    outputs = {}
    for category in ("web", "code", "math", "synthetic", "science", "reference"):
        path = tmp_path / f"{category}.jsonl"
        rows = []
        for index in range(3):
            text = f"{category} document {index} has seven fixed training words"
            rows.append(
                {
                    "text": text,
                    "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        source_id = f"pilot_train__{category}"
        outputs[source_id] = {"path": path.name, "sha256": _sha256(path)}
        categories.append(
            {
                "id": category,
                "source_id": source_id,
                "input": {"path": str(path), "sha256": _sha256(path)},
                "reference_token_target": 10,
            }
        )
    dedup = tmp_path / "manifest.json"
    dedup.write_text(
        json.dumps(
            {
                "format": "speck_production_text_preprocess_result",
                "gates": {
                    "global_exact_deduplication": "pass",
                    "disk_backed_Minhash_candidates_and_verified_near_deduplication": "pass",
                },
                "outputs": outputs,
            }
        )
    )
    reference = tmp_path / "reference.model"
    custom = tmp_path / "custom.model"
    reference.write_text("reference")
    custom.write_text("custom")
    plan = validate_pilot_stream_plan(
        {
            "format": "speck_tokenizer_pilot_stream",
            "format_version": 1,
            "status": "materialization_authorized_not_model_training_authority",
            "seed": 42,
            "dedup_manifest": {"path": str(dedup), "sha256": _sha256(dedup)},
            "reference_tokenizer_id": "mistral-32k",
            "tokenizers": [
                {
                    "id": "mistral-32k",
                    "model": {"path": str(reference), "sha256": _sha256(reference)},
                },
                {
                    "id": "custom",
                    "model": {"path": str(custom), "sha256": _sha256(custom)},
                },
            ],
            "categories": categories,
            "shard_tokens": 100,
            "output_directory": str(tmp_path / "stream"),
        }
    )
    first = materialize_pilot_stream(plan)
    second = materialize_pilot_stream(plan)

    assert first == second
    assert first["document_stream"]["actual_reference_tokens"] == 60
    assert sum(x["documents"] for x in first["document_stream"]["categories"]) == 6
    assert {item["tokens"] for item in first["tokenizers"]} == {60, 66}
    assert first["gates"]["shared_document_identity"] == "pass"
    assert first["model_training_authority"] is False
