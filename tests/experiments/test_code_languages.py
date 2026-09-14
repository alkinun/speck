import copy
import json
from pathlib import Path

import pytest

from speck.experiments.code_languages import compile_code_languages, load_code_languages

ROOT = Path(__file__).resolve().parents[2]


def inputs():
    spec = json.loads((ROOT / "research/flagship/code_language_preparation_v1.json").read_text())
    wave = json.loads((ROOT / "research/flagship/first_wave_preparation_v2.json").read_text())
    return spec, wave


def test_code_background_and_blend_preserve_matched_source_language_proportions():
    result = load_code_languages(ROOT / "research/flagship/code_language_preparation_v1.json")
    spec, wave = inputs()
    assert len(result["logical_slot_code_quotas"]) == len(wave["logical_slots"]) == 29
    for slot, original in zip(
        result["logical_slot_code_quotas"], wave["logical_slots"], strict=True
    ):
        if original["recipe"] is None:
            assert slot["recipe"] is None
            assert slot["source_language_tokens"] is None
            assert slot["selection_ref"] == original["selection_ref"]
            continue
        for source, languages in slot["source_language_tokens"].items():
            total = original["required_source_tokens"][source]
            assert sum(languages.values()) == total
            assert all(
                languages[lang] * 100 == total * share
                for lang, share in spec["language_weights_percent"].items()
            )
        if slot["recipe"] == "code:equal_blend":
            assert (
                slot["source_language_tokens"]["stack_edu"]
                == slot["source_language_tokens"]["stack_v3_train_permissive"]
            )
    for source, nominal in [("stack_edu", 1200000000), ("stack_v3_train_permissive", 300000000)]:
        rows = [
            row for row in result["source_language_capacity_envelope"] if row["source_id"] == source
        ]
        assert sum(row["nominal_tokens"] for row in rows) == nominal
        assert sum(row["preparation_target_tokens"] for row in rows) == nominal * 120 // 100


@pytest.mark.parametrize(
    "change",
    ["missing_language", "fractional_weight", "wrong_sum", "source", "rounding", "envelope"],
)
def test_code_allocation_rejects_undeclared_language_source_or_quota_changes(change):
    spec, wave = inputs()
    if change == "missing_language":
        spec["language_weights_percent"].pop("SQL")
    elif change == "fractional_weight":
        spec["language_weights_percent"]["SQL"] = 1.0
    elif change == "wrong_sum":
        spec["language_weights_percent"]["SQL"] = 2
    elif change == "source":
        wave["recipes"]["code:equal_blend"]["code"] = {"common_pile_stackv2_edu": 1}
    elif change == "rounding":
        wave["logical_slots"][0]["required_source_tokens"]["stack_edu"] += 1
    else:
        for row in wave["source_capacity_envelope"]:
            if row["source_id"] == "stack_edu":
                row["required_token_capacity"] += 1000000
    with pytest.raises(ValueError):
        compile_code_languages(spec, wave)


def test_code_requirements_do_not_modify_first_wave_assignments_or_budgets():
    spec, wave = inputs()
    original = copy.deepcopy(wave)
    compile_code_languages(spec, wave)
    assert wave == original
    assert sum(wave["budget_gpu_hours"].values()) == 192
    assert wave["logical_training_tokens"] == 130000000000
