import math

import pytest

from scripts.qualify_ordered_code_fetch import estimate_language


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
