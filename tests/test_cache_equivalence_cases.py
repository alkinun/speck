from pathlib import Path

import torch

from scripts.cache_equivalence_cases import arguments, token_sha256


def test_cache_equivalence_case_arguments_require_output():
    args = arguments(["experiment", "--output", "cases.json"])

    assert args.data_experiment == Path("experiment")
    assert args.output == Path("cases.json")


def test_cache_equivalence_token_identity_is_content_and_shape_stable():
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)

    assert token_sha256(tokens) == token_sha256(tokens.clone())
    assert token_sha256(tokens) != token_sha256(tokens.flip(0))
    assert token_sha256(tokens) != token_sha256(tokens.flatten())
