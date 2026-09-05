import pytest

from scripts.paper_finalist_qualify import unique_windows, window_points


def pairs(second_offset=1_610_612_736):
    tokens = 1_539_833_856
    result = []
    for pair, (seed, offset) in enumerate(
        (seed, offset) for seed in (42, 43, 44) for offset in (0, second_offset)
    ):
        result.append(
            {
                "pair": pair,
                "seed": seed,
                "data_token_offset": offset,
                "end_token_offset": offset + tokens,
            }
        )
    return result


def test_finalist_unique_windows_are_disjoint_and_crossed():
    tokens = 1_539_833_856
    windows = unique_windows(pairs(), tokens)

    assert windows == [
        {"start": 0, "end": tokens, "seeds": [42, 43, 44]},
        {
            "start": 1_610_612_736,
            "end": 3_150_446_592,
            "seeds": [42, 43, 44],
        },
    ]
    assert window_points(windows[0], tokens) == [
        0,
        384_958_464,
        769_916_928,
        1_154_875_392,
        1_539_833_856,
    ]


def test_finalist_unique_windows_reject_overlap():
    with pytest.raises(ValueError, match="overlap"):
        unique_windows(pairs(second_offset=1_000_000_000), 1_539_833_856)


def test_finalist_unique_windows_require_every_seed():
    with pytest.raises(ValueError, match="every seed"):
        unique_windows(pairs()[:-1], 1_539_833_856)
