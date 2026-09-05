import numpy as np

from scripts.contamination_audit import _batch_signatures, find_subsequence, prefix_signature


def test_prefix_signature_is_exact_uint16_packing():
    tokens = [1, 2, 3, 4]
    assert prefix_signature(tokens) == 1 | (2 << 16) | (3 << 32) | (4 << 48)


def test_batch_signatures_matches_scalar_packing():
    rows = np.array([[1, 2, 3, 4, 5]], dtype=np.uint16)
    signatures = _batch_signatures(rows)
    assert signatures.tolist() == [
        [prefix_signature([1, 2, 3, 4]), prefix_signature([2, 3, 4, 5])]
    ]


def test_find_subsequence_returns_every_occurrence():
    assert find_subsequence([1, 2, 1, 2, 3], [1, 2]) == [0, 2]
