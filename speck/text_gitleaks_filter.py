"""Generalize the frozen Gitleaks exclusion contract to text categories."""

import json
from pathlib import Path

from speck.gitleaks_filter import (
    _fingerprint,
    apply_gitleaks_filter,
    validate_gitleaks_filter_config,
)

FORMAT = "speck_text_gitleaks_filter"
FORMAT_VERSION = 1
TEXT_CATEGORIES = {"web", "math", "synthetic", "science", "reference"}


def validate_text_gitleaks_config(config, *, config_dir=None):
    """Validate a text-category filter through the frozen exclusion implementation."""

    if not isinstance(config, dict) or config.get("format") != FORMAT:
        raise ValueError("unsupported text Gitleaks filter format")
    partition = config.get("downstream_partition")
    if not isinstance(partition, dict) or partition.get("category") not in TEXT_CATEGORIES:
        raise ValueError("text Gitleaks filter requires a supported text category")
    category = partition["category"]
    base = {
        **config,
        "format": "speck_gitleaks_filter",
        "downstream_partition": {**partition, "category": "code"},
    }
    normalized = validate_gitleaks_filter_config(base, config_dir=config_dir)
    normalized["format"] = FORMAT
    normalized["downstream_partition"]["category"] = category
    normalized.pop("plan_fingerprint")
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_text_gitleaks_config(path):
    path = Path(path).resolve()
    return validate_text_gitleaks_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def apply_text_gitleaks_filter(config, *, restart=False):
    """Run the frozen exclusion implementation with a text-category partition."""

    if "plan_fingerprint" not in config:
        config = validate_text_gitleaks_config(config)
    return apply_gitleaks_filter(config, restart=restart)
