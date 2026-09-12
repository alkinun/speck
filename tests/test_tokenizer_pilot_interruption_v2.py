import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RESULT = ROOT / "results/systems/tokenizer-pilot-mistral-interruption-v2-20260912.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_second_interruption_preserves_both_attempts_and_same_checkpoint():
    result = json.loads(RESULT.read_text())

    assert result["status"] == "second_recoverable_interruption_same_step_3662_checkpoint"
    assert sha256(ROOT / result["predecessor"]["path"]) == result["predecessor"]["sha256"]
    assert result["system"]["reboot_observed"] is True
    assert result["system"]["data_volume_restored"] is True
    assert result["resume_checkpoint"]["step"] == 3662
    assert result["comparison"]["byte_identical"] is False
    assert result["comparison"]["document_id_and_utf8_byte_pairs_identical"] is True
    assert result["comparison"]["documents_compared"] == 12_470
    assert result["second_orphan_evaluation"]["scientific_use"] is False
    assert result["authority"]["mechanical_retry"] is True
    assert result["authority"]["D5_opening"] is False


def test_second_orphan_bytes_match_when_available():
    result = json.loads(RESULT.read_text())
    path = Path(result["second_orphan_evaluation"]["preserved_path"])
    if not path.is_file():
        pytest.skip("requires maintainer-local second interrupted evaluation")
    assert path.stat().st_size == result["second_orphan_evaluation"]["bytes"]
    assert sha256(path) == result["second_orphan_evaluation"]["sha256"]
