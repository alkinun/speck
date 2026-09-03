"""Generate deterministic, target-masked synthetic sequence-memory tasks."""

import numpy as np
import torch

IGNORE_INDEX = -100
ZOOLOGY_MQAR_REVISION = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _base_settings(batch_size, sequence_length, vocab_size):
    return (
        _positive_integer(batch_size, "batch size"),
        _positive_integer(sequence_length, "sequence length"),
        _positive_integer(vocab_size, "vocabulary size"),
    )


def palindrome_batch(batch_size, sequence_length, vocab_size, seed):
    """Return random strings followed by a separator and their exact reversal."""

    batch_size, sequence_length, vocab_size = _base_settings(
        batch_size,
        sequence_length,
        vocab_size,
    )
    if sequence_length % 2:
        raise ValueError("palindrome sequence length must be even")
    if vocab_size < 3:
        raise ValueError("palindrome vocabulary must contain a separator and content tokens")
    rng = np.random.RandomState(seed)
    content_length = sequence_length // 2
    content = rng.randint(
        1,
        vocab_size,
        size=(batch_size, content_length),
        dtype=np.int64,
    )
    separator = np.zeros((batch_size, 1), dtype=np.int64)
    stream = np.concatenate((content, separator, content[:, ::-1]), axis=1)
    inputs = stream[:, :-1].copy()
    targets = np.full(inputs.shape, IGNORE_INDEX, dtype=np.int64)
    targets[:, content_length:] = stream[:, content_length + 1 :]
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def mqar_batch(
    batch_size,
    sequence_length,
    vocab_size,
    num_pairs,
    seed,
    *,
    power_a=0.01,
    random_non_queries=True,
):
    """Generate Zoology-style multi-query associative recall examples."""

    batch_size, sequence_length, vocab_size = _base_settings(
        batch_size,
        sequence_length,
        vocab_size,
    )
    num_pairs = _positive_integer(num_pairs, "number of key-value pairs")
    if sequence_length % 2:
        raise ValueError("MQAR sequence length must be even")
    if vocab_size <= sequence_length:
        raise ValueError("MQAR vocabulary size must exceed sequence length")
    if not np.isfinite(power_a) or power_a <= 0:
        raise ValueError("MQAR power must be positive and finite")
    context_size = 2 * num_pairs
    space = (sequence_length - context_size) // 2
    if num_pairs > space:
        raise ValueError("MQAR sequence is too short for the requested associations and queries")
    key_vocab_size = vocab_size // 2
    if num_pairs >= key_vocab_size or num_pairs > vocab_size - key_vocab_size:
        raise ValueError("MQAR vocabulary is too small for unique keys and values")

    rng = np.random.RandomState(seed)
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)
    keys = np.stack(
        [rng.choice(key_choices, replace=False, size=num_pairs) for _ in range(batch_size)]
    )
    values = np.stack(
        [rng.choice(value_choices, replace=False, size=num_pairs) for _ in range(batch_size)]
    )
    context = np.empty((batch_size, context_size), dtype=np.int64)
    context[:, 0::2] = keys
    context[:, 1::2] = values

    probabilities = power_a * np.arange(1, space + 1) ** (power_a - 1)
    probabilities /= probabilities.sum()
    gaps = np.stack(
        [
            rng.choice(space, replace=False, p=probabilities, size=num_pairs)
            for _ in range(batch_size)
        ]
    )
    query_region = np.zeros((batch_size, sequence_length - context_size + 1), dtype=np.int64)
    np.put_along_axis(query_region, gaps * 2, keys, axis=1)
    stream = np.concatenate((context, query_region), axis=1)
    inputs = stream[:, :-1].copy()
    targets = np.full(inputs.shape, IGNORE_INDEX, dtype=np.int64)
    np.put_along_axis(targets, gaps * 2 + context_size, values, axis=1)

    if random_non_queries:
        filler = rng.randint(0, vocab_size, size=inputs.shape, dtype=np.int64)
        inputs[inputs == 0] = filler[inputs == 0]
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def stack_batch(
    batch_size,
    sequence_length,
    vocab_size,
    seed,
    *,
    num_stacks=64,
    pop_probability=0.5,
):
    """Generate interleaved PUSH/POP traces and supervise only valid POP answers."""

    batch_size, sequence_length, vocab_size = _base_settings(
        batch_size,
        sequence_length,
        vocab_size,
    )
    num_stacks = _positive_integer(num_stacks, "number of stacks")
    if not np.isfinite(pop_probability) or not 0 < pop_probability < 1:
        raise ValueError("stack pop probability must be strictly between zero and one")
    first_value = 2 + num_stacks
    if vocab_size <= first_value:
        raise ValueError("stack vocabulary must contain operation, stack, and value tokens")
    operations = sequence_length // 3
    if operations < 2:
        raise ValueError("stack sequence must contain at least two operations")

    rng = np.random.RandomState(seed)
    inputs = rng.randint(0, vocab_size, size=(batch_size, sequence_length), dtype=np.int64)
    targets = np.full(inputs.shape, IGNORE_INDEX, dtype=np.int64)
    push_token, pop_token = 0, 1
    for batch_index in range(batch_size):
        stacks = [[] for _ in range(num_stacks)]
        for operation_index in range(operations):
            start = 3 * operation_index
            nonempty = [index for index, stack in enumerate(stacks) if stack]
            should_pop = bool(nonempty) and rng.rand() < pop_probability
            if should_pop:
                stack_index = int(rng.choice(nonempty))
                value = stacks[stack_index].pop()
                inputs[batch_index, start : start + 3] = (
                    pop_token,
                    2 + stack_index,
                    value,
                )
                targets[batch_index, start + 1] = value
            else:
                stack_index = int(rng.randint(num_stacks))
                value = int(rng.randint(first_value, vocab_size))
                stacks[stack_index].append(value)
                inputs[batch_index, start : start + 3] = (
                    push_token,
                    2 + stack_index,
                    value,
                )
    if not (targets != IGNORE_INDEX).any():
        raise RuntimeError("stack generator produced no supervised POP operation")
    return torch.from_numpy(inputs), torch.from_numpy(targets)
