from argparse import Namespace

import pytest

from scripts.long_context_eval import arguments, parse_depths, report_config, resolved_eval_settings


def config():
    return {
        "lengths": [4_096, 8_192],
        "depths": [0.1, 0.9],
        "samples_per_depth": 2,
        "effective_threshold": 0.85,
    }


def test_pilot_overrides_produce_a_complete_resolved_config():
    args = Namespace(lengths=(4_096, 32_768), depths="0.5", samples_per_depth=1)
    settings = resolved_eval_settings(config(), args)

    assert report_config(settings) == {
        "lengths": [4_096, 32_768],
        "depths": [0.5],
        "samples_per_depth": 1,
        "effective_threshold": 0.85,
        "kv_cache_dtype": "bfloat16",
    }


def test_long_context_eval_arguments_parse_pilot_lengths():
    args = arguments(
        [
            "experiment",
            "--lengths",
            "4096,32768",
            "--depths",
            "0.1,0.9",
            "--samples-per-depth",
            "1",
            "--warmup-each-length",
            "--counterfactual",
        ]
    )
    assert args.lengths == (4_096, 32_768)
    assert parse_depths(args.depths) == [0.1, 0.9]
    assert args.samples_per_depth == 1
    assert args.warmup_each_length
    assert args.counterfactual


@pytest.mark.parametrize("value", ("", "one", "0.1,nope"))
def test_long_context_depth_parser_rejects_non_numeric_values(value):
    with pytest.raises(ValueError, match="depths"):
        parse_depths(value)
