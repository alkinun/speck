import argparse

import pytest

from scripts.inference_benchmark import _batch_sizes, _percentile


def test_batch_sizes():
    assert _batch_sizes("1,32") == (1, 32)
    with pytest.raises(argparse.ArgumentTypeError):
        _batch_sizes("1,0")


def test_percentile_uses_nearest_rank():
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.25) == 2.0
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.75) == 3.0
