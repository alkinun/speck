import json

import pytest

from speck.experiments.first_wave import compile_first_wave, load_first_wave
from speck.provenance.repository import repository_root

ROOT = repository_root(__file__)
PROPOSAL = ROOT / "research/flagship/first_wave_proposal_v1.json"


def inputs():
    proposal = json.loads(PROPOSAL.read_text())
    return proposal, *[
        json.loads((PROPOSAL.parent / proposal[name]["path"]).read_text())
        for name in ("data_plan", "source_registry", "source_use")
    ]


def test_all_source_screens_preserve_category_exposure_and_change_only_the_tested_category():
    result = load_first_wave(PROPOSAL)
    registry = inputs()[2]
    category = {source["id"]: source["category"] for source in registry["sources"]}
    incumbent = result["recipes"]["incumbent"]
    for slot in result["logical_slots"]:
        if slot["stage"] != "screen":
            continue
        tested = slot["recipe"].split(":")[0]
        recipe = result["recipes"][slot["recipe"]]
        for name in incumbent:
            if name != tested:
                assert recipe[name] == incumbent[name]
            actual = sum(
                tokens
                for source, tokens in slot["required_source_tokens"].items()
                if category[source] == name
            )
            assert (
                actual == slot["training_tokens"] * result["category_weights_percent"][name] // 100
            )
        assert sum(slot["required_source_tokens"].values()) == slot["training_tokens"]


def test_repetition_changes_unique_pool_without_changing_training_horizon():
    result = load_first_wave(PROPOSAL)
    slots = [slot for slot in result["logical_slots"] if slot["family"] == "E3"]
    assert len(slots) == 6
    for slot in slots:
        assert slot["training_tokens"] == 6_000_000_000
        assert slot["unique_pool_tokens"] * slot["effective_epochs"] == slot["training_tokens"]
        assert sum(slot["required_source_tokens"].values()) == slot["unique_pool_tokens"]


def test_budget_slots_and_reusable_source_envelope_are_not_total_training_exposure():
    result = load_first_wave(PROPOSAL)
    assert len(result["logical_slots"]) == 29
    assert sum(result["budget_gpu_hours"].values()) == 192
    assert result["logical_training_tokens"] == 130_000_000_000
    assert result["source_capacity_total_tokens"] == 17_500_000_000
    assert len(result["source_capacity_envelope"]) == 11
    confirmation = [slot for slot in result["logical_slots"] if slot["stage"] == "confirmation"]
    assert len(confirmation) == 10
    assert all(slot["recipe"] is None and slot["selection_ref"] for slot in confirmation)
    assert result["status"] == "proposed_for_review_not_launchable"
    assert result["training_authority"] is False


@pytest.mark.parametrize("bad", ["unapproved", "category", "tokenizer_only"])
def test_ineligible_source_cannot_silently_enter_the_proposal(bad):
    proposal, plan, registry, rights = inputs()
    if bad == "unapproved":
        rights["approved_source_ids"].remove("stack_edu")
    elif bad == "category":
        proposal["incumbent"]["science"] = {"finewiki_en": 1}
    else:
        proposal["incumbent"]["code"] = {"common_pile_python_peps": 1}
    with pytest.raises(ValueError, match="invalid/unapproved"):
        compile_first_wave(proposal, plan, registry, rights)
