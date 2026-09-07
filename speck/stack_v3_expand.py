"""Materialize a bounded Stack v3 extraction successor from a hash-pinned base plan."""

import hashlib
import json
from pathlib import Path

from speck.stack_v3 import qualify_stack_v3, validate_stack_v3_config

FORMAT = "speck_stack_v3_expansion"
FORMAT_VERSION = 1


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_expansion_config(config, *, config_dir=None):
    """Validate a small successor overlay without copying the base shard inventory."""

    expected = {
        "format",
        "format_version",
        "status",
        "base_config",
        "base_config_sha256",
        "language_sample_bytes",
        "output_directory",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f"Stack v3 expansion must contain exactly: {', '.join(sorted(expected))}")
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported Stack v3 expansion format")
    if config["status"] != "bounded_successor_authorized_not_training_authority":
        raise ValueError("Stack v3 expansion must remain non-authoritative")
    config_dir = Path(config_dir or ".").resolve()
    base_path = Path(config["base_config"]).expanduser()
    if not base_path.is_absolute():
        base_path = (config_dir / base_path).resolve()
    digest = config["base_config_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("base_config_sha256 must be lowercase hexadecimal")
    languages = config["language_sample_bytes"]
    if not isinstance(languages, dict) or not languages:
        raise ValueError("language_sample_bytes must be a non-empty object")
    normalized_languages = {}
    for language, size in languages.items():
        if not isinstance(language, str) or not language:
            raise ValueError("language names must be non-empty strings")
        normalized_languages[language] = _integer(size, f"language {language} bytes", 1)
    output = Path(config["output_directory"]).expanduser()
    if not output.is_absolute():
        output = (config_dir / output).resolve()
    return {
        **config,
        "base_config": str(base_path),
        "language_sample_bytes": normalized_languages,
        "output_directory": str(output),
    }


def load_expansion_config(path):
    path = Path(path).resolve()
    return validate_expansion_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def materialize_expansion_config(expansion):
    """Create the full frozen v1-compatible plan from a verified overlay."""

    expansion = validate_expansion_config(expansion)
    base_path = Path(expansion["base_config"])
    if not base_path.is_file() or _sha256(base_path) != expansion["base_config_sha256"]:
        raise ValueError("Stack v3 expansion base plan identity mismatch")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base["filters"]["language_sample_bytes"] = expansion["language_sample_bytes"]
    base["output"]["directory"] = expansion["output_directory"]
    return validate_stack_v3_config(base, config_dir=base_path.parent)


def qualify_stack_v3_expansion(expansion, *, restart=False):
    """Run a hash-bound bounded successor with the existing qualifier."""

    return qualify_stack_v3(materialize_expansion_config(expansion), restart=restart)
