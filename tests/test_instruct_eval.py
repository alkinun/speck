import json
from pathlib import Path

import pytest

from scripts.instruct_eval import MODEL_STEPS, score

root = Path(__file__).parents[1]
expected_steps = {
    "Speck1-140M-Instruct": 4_835,
    "Speck1.1-140M-Instruct": 8_534,
    "Speck1.1-140M-Instruct-2ep": 17_068,
}
source_runs = {
    "Speck1-140M-Instruct": "Speck1-140M-Instruct",
    "Speck1.1-140M-Instruct": "Speck1.1-140M-Instruct-Light",
    "Speck1.1-140M-Instruct-2ep": "Speck1.1-140M-Instruct",
}


def test_instruction_evaluation_model_names_match_checkpoint_steps():
    assert MODEL_STEPS == expected_steps
    artifacts = (
        ("instruct-eval-15.json", "step"),
        ("bananamind-instruct-comparison.json", "checkpoint_step"),
    )
    for filename, step_key in artifacts:
        data = json.loads((root / "experiments" / filename).read_text(encoding="utf-8"))
        assert {model["name"]: model[step_key] for model in data["models"]} == expected_steps
        assert {model["name"]: model["source_run"] for model in data["models"]} == source_runs
        if filename.startswith("bananamind"):
            assert all(
                Path(model["report"]).parent.name == model["source_run"] for model in data["models"]
            )


@pytest.mark.parametrize(
    "answer",
    ("-45", "−45", "- 45", "4.45", ".45", "4.5e2", "1,045", "id45", "90/45", "4,45", "--45"),
)
def test_numeric_scoring_preserves_signs_and_complete_numbers(answer):
    assert score(answer, ("45",), "final_number") == (False, False)


@pytest.mark.parametrize(
    "answer,accepted,expected",
    (
        ("45", ("45",), (True, True)),
        ("+45", ("45",), (True, True)),
        ("45.0", ("45",), (True, True)),
        ("4.5e1", ("45",), (True, True)),
        ("1,045", ("1045",), (True, True)),
        ("−45", ("-45",), (True, True)),
        ("- 45", ("-45",), (True, True)),
        ("  -0.5  ", ("-0.5",), (True, True)),
        ("The answer is 45.", ("45",), (True, False)),
        ("First 12, then 45", ("45",), (True, False)),
        ("45, then 12", ("45",), (False, False)),
        ("**45**", ("45",), (True, False)),
        ("No number here", ("45",), (False, False)),
    ),
)
def test_numeric_scoring_uses_final_value_and_separate_exact_format(answer, accepted, expected):
    assert score(answer, accepted, "final_number") == expected


def test_text_scoring_retains_normalization_and_word_boundaries():
    assert score("Buenos días!", ("buenos dias",)) == (True, True)
    assert score("The city is Paris.", ("Paris",)) == (True, False)
    assert score("Parisian", ("Paris",)) == (False, False)
    assert score("blue triangle", ("blue triangle",), "exact") == (True, True)


def test_unknown_scoring_mode_is_rejected():
    with pytest.raises(ValueError, match="scoring"):
        score("45", ("45",), "final_numbr")
