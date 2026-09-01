"""Provide helpers for experiment directories."""

import json
from pathlib import Path


def _load_json_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"cyclic experiment config inheritance: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"missing experiment config: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"experiment config must be an object: {path}")
    parent = value.pop("extends", None)
    if parent is None:
        return value
    if not isinstance(parent, str) or not parent:
        raise ValueError(f"experiment config extends must be a path string: {path}")
    inherited = _load_json_config(path.parent / parent, seen | {path})
    inherited.update(value)
    return inherited


def load_experiment(directory, *names):
    directory = Path(directory)
    configs = {}
    for name in names:
        path = directory / f"{name}.json"
        configs[name] = _load_json_config(path)
        if name == "train":
            runtime_path = directory / "runtime.json"
            if runtime_path.is_file():
                runtime = _load_json_config(runtime_path)
                if set(runtime) != {"device_batch_size"}:
                    raise ValueError("runtime.json must contain exactly device_batch_size")
                configs[name].update(runtime)
    return configs
