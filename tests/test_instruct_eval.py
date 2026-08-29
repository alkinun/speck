import json
from pathlib import Path

from scripts.instruct_eval import MODEL_NAMES

root = Path(__file__).parents[1]
expected_steps = {
    "Speck1-140M-Instruct": 4_835,
    "Speck1.1-140M-Instruct": 8_534,
    "Speck1.1-140M-Instruct-2ep": 17_068,
}


def test_instruction_evaluation_model_names_match_checkpoint_steps():
    assert set(MODEL_NAMES) == expected_steps.keys()
    artifacts = (
        ("instruct-eval-15.json", "step"),
        ("bananamind-instruct-comparison.json", "checkpoint_step"),
    )
    for filename, step_key in artifacts:
        data = json.loads((root / "experiments" / filename).read_text(encoding="utf-8"))
        assert {model["name"]: model[step_key] for model in data["models"]} == expected_steps
