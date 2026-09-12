import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RESULT = ROOT / "results/systems/tokenizer-pilot-mistral-interruption-20260912.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_interruption_is_checkpoint_bound_and_fail_closed():
    result = json.loads(RESULT.read_text())

    assert result["status"] == "recoverable_from_step_3662_orphan_evaluation_preserved"
    assert sha256(ROOT / result["execution"]["path"]) == result["execution"]["sha256"]
    assert result["observation"]["latest_completed_checkpoint_step"] == 3662
    assert result["observation"]["latest_evaluation_step"] == 3664
    assert result["resume_checkpoint"]["next_token_offset"] == 239_992_832
    assert result["orphan_evaluation"]["scientific_use"] is False
    assert result["recovery"]["reuse_execution_manifest"] is True
    assert result["recovery"]["new_seed_or_arm"] is False
    assert result["authority"]["mechanical_retry"] is True
    assert result["authority"]["D5_opening"] is False


def test_interrupted_evaluation_is_preserved_when_available():
    result = json.loads(RESULT.read_text())
    path = Path(result["orphan_evaluation"]["preserved_path"])
    if not path.is_file():
        pytest.skip("requires maintainer-local interrupted tokenizer evaluation")
    assert path.stat().st_size == result["orphan_evaluation"]["bytes"]
    assert sha256(path) == result["orphan_evaluation"]["sha256"]
    assert not Path(result["orphan_evaluation"]["original_path"]).exists()
