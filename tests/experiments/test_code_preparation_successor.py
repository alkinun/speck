import copy
import json
from pathlib import Path

import pytest

from speck.experiments.code_languages import load_code_languages
from speck.experiments.code_preparation_decision import (
    validate_language_successor,
    validate_source_successor,
)
from speck.experiments.first_wave_preparation import load_preparation

ROOT = Path(__file__).resolve().parents[2] / "research/flagship"


def read(name):
    return json.loads((ROOT / name).read_text())


def test_selected_successor_preserves_design_and_moves_only_common_code_background():
    prior = read("first_wave_preparation_v2.json")
    wave = load_preparation(ROOT / "first_wave_proposal_v3.json")
    languages = load_code_languages(ROOT / "code_language_preparation_v2.json")
    assert len(wave["logical_slots"]) == 29
    assert wave["logical_training_tokens"] == 130000000000
    assert wave["source_capacity_total_tokens"] == 17700000000
    assert wave["budget_gpu_hours"] == prior["budget_gpu_hours"]
    assert wave["category_weights_percent"] == prior["category_weights_percent"]
    for current, old in zip(wave["logical_slots"], prior["logical_slots"], strict=True):
        assert {k: v for k, v in current.items() if k != "required_source_tokens"} == {
            k: v for k, v in old.items() if k != "required_source_tokens"
        }
    for name, recipe in wave["recipes"].items():
        for category, sources in recipe.items():
            if category != "code" or name.startswith("code:"):
                assert sources == prior["recipes"][name][category]
            else:
                assert sources == {"stack_v3_train_permissive": 1}
    for source, total in (("stack_edu", 300000000), ("stack_v3_train_permissive", 1200000000)):
        rows = [
            row
            for row in languages["source_language_capacity_envelope"]
            if row["source_id"] == source
        ]
        assert sum(row["nominal_tokens"] for row in rows) == total
        assert sum(row["preparation_target_tokens"] for row in rows) == total * 120 // 100
    for slot in languages["logical_slot_code_quotas"]:
        if slot["recipe"] is None:
            assert slot["source_language_tokens"] is None
            continue
        for quotas in slot["source_language_tokens"].values():
            assert all(
                quotas[lang] * 100 == sum(quotas.values()) * weight
                for lang, weight in languages["language_weights_percent"].items()
            )
        if slot["recipe"] == "code:equal_blend":
            assert (
                slot["source_language_tokens"]["stack_edu"]
                == slot["source_language_tokens"]["stack_v3_train_permissive"]
            )
    assert "D5 selection" not in wave["remaining_before_launch"]
    assert "D5-token" not in wave["capacity_boundary"]
    assert wave["training_authority"] is False


@pytest.mark.parametrize("change", ["category", "arm", "seed", "background"])
def test_source_successor_cannot_hide_other_design_changes(change):
    previous = read("first_wave_proposal_v2.json")
    spec = read("first_wave_proposal_v3.json")
    decision = read("code_preparation_decision_v2.json")
    if change == "category":
        spec["incumbent"]["math"] = {"megamath_web_pro": 1}
    elif change == "arm":
        spec["treatments"]["code"].pop()
    elif change == "seed":
        spec["screen_seed"] = 43
    else:
        spec["incumbent"]["code"] = {"stack_edu": 1}
    with pytest.raises(ValueError):
        validate_source_successor(spec, previous, decision)


def test_language_successor_is_bound_to_decision_and_preserves_headroom_and_filters():
    spec = read("code_language_preparation_v2.json")
    previous = read("code_language_preparation_v1.json")
    decision = read("code_preparation_decision_v2.json")
    wave = read("first_wave_preparation_v3.json")
    for key, value in (
        ("headroom_percent", 10),
        ("policy_boundary", "relaxed"),
        ("language_weights_percent", previous["language_weights_percent"]),
    ):
        changed = copy.deepcopy(spec)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_language_successor(changed, previous, wave, decision)
