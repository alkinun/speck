import hashlib
import json
from pathlib import Path

from speck.tokenizer import Tokenizer
from speck.tokenizer_fixture import prepare_fullsize_fixture
from speck.tokenizer_inputs import load_tokenizer_inputs_freeze

ROOT = Path(__file__).parents[1]


def test_real_tokenizer_inputs_are_complete_hash_bound_and_blocked():
    path = ROOT / "research/flagship/tokenizer_inputs_blocked_v1.json"
    value = load_tokenizer_inputs_freeze(path, verify_files=True)

    assert [category["id"] for category in value["categories"]] == [
        "web",
        "code",
        "math",
        "synthetic",
        "science",
        "reference",
    ]
    assert sum(len(category["inputs"]) for category in value["categories"]) == 30
    assert value["targets"]["total_training_bytes"] == 600_000_000
    assert value["targets"]["total_evaluation_bytes"] == 60_000_000
    assert value["training_authority"] == "blocked"
    assert value["executable_tokenizer_config"].startswith("forbidden")


def test_mistral_baseline_payload_matches_pinned_identity_and_special_ids():
    value = json.loads((ROOT / "research/flagship/mistral_tokenizer_baseline.json").read_text())
    path = Path(value["path"])
    tokenizer = Tokenizer(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == value["sha256"]
    assert path.stat().st_size == value["bytes"]
    assert tokenizer.vocab_size == value["vocab_size"] == 32_000
    assert (tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id) == (0, 1, 2)
    assert value["selection_authority"] is False


def test_fullsize_fixture_generation_is_content_deterministic_and_non_authoritative(tmp_path):
    first = prepare_fullsize_fixture(
        tmp_path / "first",
        rows_per_category=100,
        training_bytes_per_category=1_000,
        evaluation_bytes_per_category=100,
    )
    second = prepare_fullsize_fixture(
        tmp_path / "second",
        rows_per_category=100,
        training_bytes_per_category=1_000,
        evaluation_bytes_per_category=100,
    )

    assert [value["sha256"] for value in first["categories"]] == [
        value["sha256"] for value in second["categories"]
    ]
    assert first["selection_authority"] is second["selection_authority"] is False
    assert first["training_authority"] == "generated_fixture_only"
