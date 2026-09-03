from argparse import Namespace
from types import SimpleNamespace

import pytest

from scripts.long_context_eval import (
    arguments,
    parse_depths,
    positional_regime,
    report_config,
    resolved_eval_settings,
)
from speck.architecture import AttentionSpec, BlockConfig, BlockInvocation, StageConfig


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


def model_with_attention(*attention):
    block = BlockConfig(8, tuple(StageConfig((branch,)) for branch in attention))
    return SimpleNamespace(
        execution_plan=(BlockInvocation(block, 0, "block_0"),),
        config=SimpleNamespace(rope_scaling_factor=8.0),
    )


def test_positional_regime_only_reports_extrapolation_for_active_global_rope():
    nope = model_with_attention(AttentionSpec(4, 1, rope_dim=0))
    rope = model_with_attention(AttentionSpec(4, 1, rope_dim=2))
    mixed = model_with_attention(
        AttentionSpec(4, 1, scope="global", rope_dim=0),
        AttentionSpec(4, 1, scope="sliding", window_size=32, rope_dim=2),
    )

    assert positional_regime(nope, 32, 128) == {
        "attention_scopes": ["global"],
        "rope_dimensions_by_scope": {"global": [0]},
        "rope_scaling_factor": 8.0,
        "training_sequence_length": 32,
        "extrapolates_global_rope": False,
    }
    assert positional_regime(rope, 32, 128)["extrapolates_global_rope"] is True
    assert positional_regime(mixed, 32, 128)["extrapolates_global_rope"] is False
