import pytest
import torch

from speck.synthetic import IGNORE_INDEX, mqar_batch, palindrome_batch, stack_batch


@pytest.mark.parametrize(
    ("generator", "kwargs"),
    (
        (palindrome_batch, {}),
        (mqar_batch, {"num_pairs": 8}),
        (stack_batch, {"num_stacks": 8}),
    ),
)
def test_synthetic_batches_are_locally_deterministic(generator, kwargs):
    settings = {"batch_size": 3, "sequence_length": 64, "vocab_size": 256, **kwargs}
    first = generator(**settings, seed=17)
    repeated = generator(**settings, seed=17)
    different = generator(**settings, seed=18)

    assert all(torch.equal(left, right) for left, right in zip(first, repeated))
    assert not torch.equal(first[0], different[0])
    assert first[0].shape == first[1].shape == (3, 64)


def test_palindrome_targets_are_the_exact_reverse():
    inputs, targets = palindrome_batch(2, 16, 32, 5)
    content_length = inputs.size(1) // 2

    assert torch.equal(inputs[:, content_length], torch.zeros(2, dtype=torch.long))
    assert torch.equal(targets[:, content_length:], inputs[:, :content_length].flip(1))
    assert (targets[:, :content_length] == IGNORE_INDEX).all()


def test_mqar_targets_match_unique_context_associations():
    num_pairs = 16
    inputs, targets = mqar_batch(4, 128, 512, num_pairs, 11)
    context = inputs[:, : 2 * num_pairs]

    for example_inputs, example_targets, example_context in zip(inputs, targets, context):
        mapping = dict(zip(example_context[0::2].tolist(), example_context[1::2].tolist()))
        supervised = (example_targets != IGNORE_INDEX).nonzero().flatten()
        assert supervised.numel() == num_pairs
        assert len(mapping) == num_pairs
        for position in supervised:
            assert example_targets[position].item() == mapping[example_inputs[position].item()]


def test_stack_targets_follow_last_in_first_out_state():
    inputs, targets = stack_batch(3, 192, 256, 23, num_stacks=8)

    for example_inputs, example_targets in zip(inputs, targets):
        stacks = [[] for _ in range(8)]
        for start in range(0, 192, 3):
            operation, stack_token, value = example_inputs[start : start + 3].tolist()
            stack_index = stack_token - 2
            if operation == 0:
                stacks[stack_index].append(value)
                assert example_targets[start + 1] == IGNORE_INDEX
            else:
                assert value == stacks[stack_index].pop()
                assert example_targets[start + 1] == value


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    (
        (palindrome_batch, {}, "must be even"),
        (mqar_batch, {"num_pairs": 8}, "must be even"),
    ),
)
def test_even_length_tasks_reject_odd_lengths(function, kwargs, message):
    with pytest.raises(ValueError, match=message):
        function(batch_size=2, sequence_length=63, vocab_size=256, seed=1, **kwargs)


def test_mqar_rejects_impossible_pair_count():
    with pytest.raises(ValueError, match="too short"):
        mqar_batch(2, 64, 256, 17, 1)


def test_stack_requires_operation_stack_and_value_vocabularies():
    with pytest.raises(ValueError, match="operation, stack, and value"):
        stack_batch(2, 96, 66, 1, num_stacks=64)
