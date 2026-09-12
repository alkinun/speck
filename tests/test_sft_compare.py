from collections import Counter

from scripts.sft_compare import questions


def test_pilot_cases_are_fixed_unique_and_category_balanced():
    cases, qualitative = questions()
    assert (cases, qualitative) == questions()
    assert len(cases) == 215 and len(qualitative) == 12
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({case["prompt"] for case in cases}) == len(cases)
    counts = Counter(case["category"] for case in cases if not case["id"].startswith("legacy-"))
    assert counts == {
        "arithmetic": 40,
        "sorting": 40,
        "grounding": 40,
        "format": 40,
        "code_trace": 40,
    }
    for case in cases:
        assert case["accepted"]
        if case["category"] == "grounding":
            assert f": {case['accepted'][0]}" in case["prompt"]
