"""Provide helpers for experiment directories."""

import json
from pathlib import Path


def load_experiment(directory, *names):
    directory = Path(directory)
    configs = {}
    for name in names:
        path = directory / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing experiment config: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"experiment config must be an object: {path}")
        configs[name] = value
    return configs
