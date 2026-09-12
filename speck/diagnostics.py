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
