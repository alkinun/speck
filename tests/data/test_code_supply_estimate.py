import math
from copy import deepcopy

import pytest

from scripts.qualify_ordered_code_fetch import additional_metadata_units, estimate_language


def test_file_strata_have_separate_populations_and_rejections_contribute_zero():
    targets = [
        {
            "language": "Java",
            "metadata_sha256": file,
            "stratum": 0,
            "population": population,
            "sample_size": 2,
        }
        for file, population in [("first", 10), ("additional", 100)]
        for _ in range(2)
    ]
    estimate, se, observations = estimate_language(targets, {1: 2, 2: 1, 3: 3}, "Java")
    assert estimate == 210
    assert se == pytest.approx(math.sqrt(9880))
    assert len(observations) == 2
    with pytest.raises(ValueError, match="cover"):
        estimate_language(targets[:-1], {1: 2, 2: 1}, "Java")


def test_complete_small_stratum_has_zero_sampling_variance():
    target = {
        "language": "C",
        "metadata_sha256": "first",
        "stratum": 0,
        "population": 1,
        "sample_size": 1,
    }
    estimate, se, _ = estimate_language([target], {0: 37}, "C")
    assert (estimate, se) == (37, 0)


@pytest.mark.parametrize("version,start,count", [(2, 11, 26), (3, 26, 28)])
def test_supply_probe_excludes_all_previously_sampled_files(version, start, count):
    identities = {
        "code_languages": {"sha256": "a" * 64},
        "source_qualification": {"sha256": "b" * 64},
    }
    units = [{"id": f"file_{i}", "expected_file_rows": 100 + i} for i in range(count)]
    intake = {"format_version": version, "inputs": identities, "units": units}
    metadata = {
        "status": "complete_metadata_verified_not_code_stock",
        "training_authority": False,
        "inputs": identities,
        "files": [{"unit": unit, "raw": {"path": unit["id"]}} for unit in units],
    }
    spec = {"format_version": version, "selected_unit_ids": [u["id"] for u in units[start:]]}
    selected = additional_metadata_units(spec, metadata, intake, identities)
    assert [u["id"] for u in selected] == spec["selected_unit_ids"]
    assert all(u["start_row"] == 0 and u["stop_row"] == u["expected_file_rows"] for u in selected)
    for selection in [units, units[start:][::-1], units[start + 1 :]]:
        changed = {**spec, "selected_unit_ids": [u["id"] for u in selection]}
        with pytest.raises(ValueError, match="exact additional"):
            additional_metadata_units(changed, metadata, intake, identities)
    changed = deepcopy(metadata)
    changed["files"][0]["unit"]["expected_file_rows"] += 1
    with pytest.raises(ValueError, match="exact additional"):
        additional_metadata_units(spec, changed, intake, identities)
    changed = deepcopy(identities)
    changed["code_languages"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="exact additional"):
        additional_metadata_units(spec, metadata, intake, changed)
    with pytest.raises(ValueError, match="exact additional"):
        additional_metadata_units(spec, metadata, {**intake, "format_version": 1}, identities)
