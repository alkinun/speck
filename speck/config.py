"""Provide helpers for experiment directories."""

import json
from copy import deepcopy
from pathlib import Path


def _apply_operation_overrides(value, path):
    overrides = value.pop("operation_overrides", None)
    if overrides is None:
        return value
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError(f"operation_overrides must be a non-empty object: {path}")
    blocks = value.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError(f"operation_overrides requires architecture blocks: {path}")
    found = {kind: 0 for kind in overrides}
    for group in blocks:
        for stage in group["block"]["stages"]:
            for operation in stage["branches"]:
                kind = operation.get("kind")
                if kind in overrides:
                    settings = overrides[kind]
                    if not isinstance(settings, dict) or "kind" in settings:
                        raise ValueError(f"invalid {kind} operation override: {path}")
                    operation.update(settings)
                    found[kind] += 1
    missing = [kind for kind, count in found.items() if count == 0]
    if missing:
        raise ValueError(f"operation override matched no branches: {', '.join(missing)}")
    return value


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
        return _apply_operation_overrides(value, path)
    if not isinstance(parent, str) or not parent:
        raise ValueError(f"experiment config extends must be a path string: {path}")
    inherited = deepcopy(_load_json_config(path.parent / parent, seen | {path}))
    inherited.update(value)
    return _apply_operation_overrides(inherited, path)


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
