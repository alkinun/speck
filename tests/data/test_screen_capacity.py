import pytest

from speck.data.screen_capacity import screen_capacity


def inputs():
    weights = {"web": 55, "code": 15, "math": 10, "synthetic": 10, "science": 5, "reference": 5}
    plan = {
        "categories": [{"id": key, "prior_percent": value} for key, value in weights.items()],
        "experiments": [
            {"id": "E3", "tokens": 6_000_000_000},
            {"id": "E1W", "tokens": 8_000_000_000},
            {"id": "E1S", "tokens": 2_000_000_000},
        ],
    }
    registry = {
        "sources": [{"id": key, "category": key, "priority": "primary_screen"} for key in weights]
    }
    capacities = {
        f"pilot_train__{key}": {"mistral_tokens": 6_000_000_000 * value // 100, "documents": 1}
        for key, value in weights.items()
    }
    record = {
        "pilot_training_capacity": {
            "per_source": capacities,
            "total_mistral_reference_tokens": 6_000_000_000,
            "tokenizer": {"path": "fixture", "sha256": "0" * 64},
        }
    }
    return plan, registry, record


def test_balanced_capacity_is_limited_by_category_not_total():
    plan, registry, record = inputs()
    record["pilot_training_capacity"]["per_source"]["pilot_train__science"]["mistral_tokens"] = (
        10_000_000
    )
    record["pilot_training_capacity"]["total_mistral_reference_tokens"] -= 290_000_000
    result = screen_capacity(plan, registry, record)
    assert result["total_available_reference_tokens"] == 5_710_000_000
    assert result["balanced_prior_pool_upper_bound_tokens"] == 200_000_000
    assert result["limiting_category"] == "science"
    four_epochs = result["E3_conditional_prior_requirements"][2]
    assert four_epochs["unique_pool_tokens"] == 1_500_000_000
    assert four_epochs["missing_by_category"]["science"] == 65_000_000
    assert not four_epochs["capacity_sufficient"]
    assert not result["training_authority"]


def test_epoch_reuse_does_not_multiply_unique_supply_by_seed_or_exposure():
    result = screen_capacity(*inputs())
    assert [r["unique_pool_tokens"] for r in result["E3_conditional_prior_requirements"]] == [
        6_000_000_000,
        3_000_000_000,
        1_500_000_000,
    ]
    assert all(r["capacity_sufficient"] for r in result["E3_conditional_prior_requirements"])
    assert (
        result["E1_conditional_requirements"]["E1W"][
            "conditional_prior_substitution_category_tokens"
        ]["web"]
        == 4_400_000_000
    )
    assert not result["E1_conditional_requirements"]["E1W"]["scientific_recipe_frozen"]


def test_unmeasured_source_is_unknown_instead_of_zero_capacity():
    plan, registry, record = inputs()
    registry["sources"].append(
        {"id": "another_web", "category": "web", "priority": "primary_screen"}
    )
    result = screen_capacity(plan, registry, record)
    assert result["primary_screen_sources_without_capacity_in_this_evidence"] == [
        {
            "source_id": "another_web",
            "category": "web",
            "measured_tokens_in_this_evidence": None,
        }
    ]
    record["pilot_training_capacity"]["total_mistral_reference_tokens"] += 1
    with pytest.raises(ValueError, match="total"):
        screen_capacity(plan, registry, record)
