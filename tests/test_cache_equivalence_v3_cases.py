import pytest

from scripts.cache_equivalence_v3_cases import _assert_disjoint


def group(*cases):
    return {"cases": list(cases)}


def case(identifier, source, start, length):
    return {
        "id": identifier,
        "source": source,
        "source_offset": start,
        "base_length": length,
    }


def test_v3_case_intervals_accept_boundaries_and_other_sources():
    current = [group(case("new-a", "a", 8, 4), case("new-b", "b", 0, 8))]
    prior = [group(case("old-a", "a", 0, 8))]

    _assert_disjoint(current, prior)


def test_v3_case_intervals_reject_prior_and_internal_overlap():
    with pytest.raises(ValueError, match="overlap"):
        _assert_disjoint([group(case("new", "a", 7, 4))], [group(case("old", "a", 0, 8))])
    with pytest.raises(ValueError, match="overlap"):
        _assert_disjoint(
            [group(case("left", "a", 0, 8), case("right", "a", 4, 8))],
            [],
        )
