"""Prevent leakage through forks, derivatives, duplicates and missing lineage."""

import pytest

from speck.data.code_families import partition_code_families


def assign(rows, **kwargs):
    return partition_code_families(rows, seed="qualification-v1", **kwargs)


def test_hold_propagates_through_alias_parent_and_duplicate():
    rows = [
        {"id": "source", "repository": "fork/lib"},
        {"id": "exercise", "repository": "maker/tasks", "parents": ["source"]},
        {"id": "copy", "repository": "copy/lib", "duplicate_group": "same"},
        {"id": "copy2", "repository": "maker/tasks", "duplicate_group": "same"},
    ]
    result = assign(
        rows, aliases=[("fork/lib", "upstream/lib")], held_repositories=["UPSTREAM/lib"]
    )
    assert {r["candidate_partition"] for r in result} == {"quarantine"}
    assert all("benchmark_family_or_content_overlap" in r["hold_reasons"] for r in result)
    assert len({r["family_component_sha256"] for r in result}) == 1


@pytest.mark.parametrize("bad", [{"parents": ["missing"]}, {"repository": None}])
def test_incomplete_lineage_quarantines_connected_records(bad):
    rows = [
        {"id": "a", "repository": "owner/lib", **bad},
        {"id": "b", "repository": "other/lib", "parents": ["a"]},
    ]
    assert all(r["candidate_partition"] == "quarantine" for r in assign(rows))


def test_content_overlap_holds_entire_repository():
    rows = [
        {"id": "a", "repository": "owner/lib", "benchmark_overlap": True},
        {"id": "b", "repository": "OWNER/lib"},
    ]
    assert all(r["candidate_partition"] == "quarantine" for r in assign(rows))


def test_order_independent_and_never_admits_data():
    rows = [
        {"id": "a", "repository": "owner/lib"},
        {"id": "b", "repository": "fork/lib", "parents": ["a"]},
        {"id": "c", "repository": "unrelated/lib"},
    ]
    aliases = [("owner/lib", "old/lib"), ("old/lib", "fork/lib")]
    assert assign(rows, aliases=aliases) == assign(rows[::-1], aliases=aliases[::-1])
    assert all(r["training_admitted"] is False for r in assign(rows))


def test_unrelated_family_is_not_held():
    result = assign([{"id": "a", "repository": "other/lib"}], held_repositories=["benchmark/lib"])
    assert result[0]["candidate_partition"] in {"train", "development", "final"}
    assert result[0]["hold_reasons"] == []


@pytest.mark.parametrize(
    "rows",
    [
        [{"id": "a", "repository": "not-a-repo"}],
        [{"id": "a", "repository": "o/r", "benchmark_overlap": "false"}],
        [{"id": "a"}, {"id": "a"}],
        [{"id": "a", "parents": "missing"}],
    ],
)
def test_malformed_inputs_fail_closed(rows):
    with pytest.raises(ValueError):
        assign(rows)
