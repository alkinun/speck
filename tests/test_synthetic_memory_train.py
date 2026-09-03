from argparse import Namespace

import pytest

from scripts.synthetic_memory_train import (
    architecture_for_variant,
    arguments,
    cosine_scale,
    resolved_settings,
    source_provenance,
    task_batch,
)
from speck.architecture import GatedDeltaNetSpec, KimiDeltaAttentionSpec


def test_synthetic_memory_arguments_pin_paper_geometry_defaults():
    args = arguments(["--variant", "kda-sigmoid", "--lr", "0.001", "--output", "x.json"])
    assert args.sequence_length == 1_024
    assert args.vocab_size == 8_192
    assert args.num_pairs == 256
    assert args.max_steps == 20_000
    assert args.eval_every == 250


@pytest.mark.parametrize(
    ("variant", "kind", "activation"),
    (
        ("gdn-silu", GatedDeltaNetSpec, "silu"),
        ("gdn-sigmoid", GatedDeltaNetSpec, "sigmoid"),
        ("kda-sigmoid", KimiDeltaAttentionSpec, None),
    ),
)
def test_synthetic_architectures_change_only_declared_mixer_axis(
    variant,
    kind,
    activation,
):
    architecture = architecture_for_variant(variant, 1_024, 8_192)
    assert architecture.logical_depth == 2
    mixers = [invocation.block.stages[0].branches[0] for invocation in architecture.execution_plan]
    assert all(isinstance(mixer, kind) for mixer in mixers)
    assert all(mixer.key_head_dim == 128 for mixer in mixers)
    assert all(mixer.num_key_heads == mixer.num_value_heads == 2 for mixer in mixers)
    if activation is not None:
        assert all(mixer.output_gate_activation == activation for mixer in mixers)


def test_task_batch_has_requested_shape_and_targets():
    settings = {
        "task": "mqar",
        "sequence_length": 64,
        "vocab_size": 256,
        "num_pairs": 8,
        "num_stacks": 8,
    }
    inputs, targets = task_batch(settings, 3, 9)
    assert inputs.shape == targets.shape == (3, 64)
    assert (targets != -100).sum().item() == 24


def test_cosine_scale_has_fixed_endpoints():
    assert cosine_scale(0, 100) == 1
    assert cosine_scale(100, 100) == 0
    with pytest.raises(ValueError, match="outside"):
        cosine_scale(101, 100)


def test_resolved_settings_rejects_incomplete_evaluation_batch():
    args = Namespace(
        task="mqar",
        variant="gdn-silu",
        sequence_length=64,
        vocab_size=256,
        num_pairs=8,
        num_stacks=8,
        batch_size=4,
        eval_batch_size=4,
        eval_examples=10,
        max_steps=2,
        eval_every=1,
        log_every=1,
        lr=1e-3,
        weight_decay=0.1,
        grad_clip=1.0,
        seed=1,
        validation_seed=2,
        early_stop_accuracy=0.99,
        device="cpu",
        no_compile=True,
    )
    with pytest.raises(ValueError, match="divisible"):
        resolved_settings(args)


def test_source_provenance_distinguishes_tracked_changes():
    provenance = source_provenance()
    assert provenance["git_revision"]
    assert isinstance(provenance["git_tracked_dirty"], bool)
    assert isinstance(provenance["untracked_files"], list)
