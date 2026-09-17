"""Provide small helpers shared by diagnostic and qualification commands."""

import subprocess

import torch


def command_output(command):
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    return result.stdout.strip() or None


def maximum_error(actual, expected):
    return (actual.float() - expected.float()).abs().max().item()


def nearest_percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_lengths(value):
    try:
        lengths = tuple(int(item) for item in value.split(","))
    except (AttributeError, ValueError) as error:
        raise ValueError("lengths must be comma-separated integers") from error
    if not lengths or any(length < 32 for length in lengths):
        raise ValueError("context lengths must be at least 32 tokens")
    if tuple(sorted(set(lengths))) != lengths:
        raise ValueError("context lengths must be sorted and unique")
    return lengths
