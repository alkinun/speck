import argparse

import pytest
import torch

from scripts.gdn_kernel_qualify import arguments, command_output, maximum_error


def test_kernel_qualification_arguments_are_strict():
    args = arguments(["--output", "report.json"])
    assert args.lengths == (64, 512, 4_096)
    with pytest.raises((argparse.ArgumentError, SystemExit, ValueError)):
        arguments(["--lengths", "512,64", "--output", "report.json"])


def test_maximum_error_compares_in_float32():
    actual = torch.tensor([1.0, 2.5], dtype=torch.bfloat16)
    expected = torch.tensor([1.0, 2.0])
    assert maximum_error(actual, expected) == 0.5


def test_qualification_records_command_provenance():
    assert command_output(["git", "rev-parse", "--show-toplevel"]) is not None
    assert command_output(["false"]) is None
