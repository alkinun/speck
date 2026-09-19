"""Sampling must preserve rare cells, denominators and holds without claiming eligibility."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "code_yield_audit",
    Path(__file__).resolve().parents[1] / "experiments/corpus-audit/audit_code_yield.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def rows():
    return [
        {"id": str(i), "language": "Python", "role_hint": "other", "tokens": 100 + i}
        for i in range(20)
    ] + [
        {"id": "rare", "language": "Go", "role_hint": "test", "tokens": 4097},
        {"id": "boundary", "language": "Go", "role_hint": "test", "tokens": 4096},
    ]


def test_order_independent_sample_preserves_small_cells_and_population_weights():
    population = rows()
    strata, sample = audit.select(population, "fixed-before-review", 2)
    assert audit.select(reversed(population), "fixed-before-review", 2) == (strata, sample)
    assert {r["id"] for r in sample} >= {"rare", "boundary"}
    assert len(sample) == 4
    assert sum(s["sample_documents"] * s["expansion_weight"] for s in strata.values()) == 22
    assert sum(s["tokens"] for s in strata.values()) == sum(r["tokens"] for r in population)
    assert strata["Python/other/le_4096"]["inclusion_probability"] == 0.1
    assert strata["Go/test/gt_4096"]["inclusion_probability"] == 1


def test_holds_are_outcomes_not_silent_resampling_filters():
    population = rows()
    original = audit.select(population, "frozen", 2)
    marked = [{**r, "known_hold": True} for r in population]
    strata, sample = audit.select(marked, "frozen", 2)
    assert strata == original[0]
    assert [r["id"] for r in sample] == [r["id"] for r in original[1]]


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_sample_sizes_fail(count):
    with pytest.raises(ValueError, match="positive integer"):
        audit.select(rows(), "frozen", count)


def test_duplicate_ids_cannot_distort_sampling_denominators():
    with pytest.raises(ValueError, match="duplicate"):
        audit.select(rows() + [rows()[0]], "frozen", 2)


def test_transitive_alias_holds_propagate_independently_of_edge_order():
    aliases = [("Fork/Project", "renamed/project"), ("source/project", "fork/project")]
    expected = {"source/project", "fork/project", "renamed/project"}
    assert audit.held_closure(["SOURCE/PROJECT"], aliases) == expected
    assert audit.held_closure(["SOURCE/PROJECT"], reversed(aliases)) == expected
