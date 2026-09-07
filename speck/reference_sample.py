"""Apply the frozen document-quality sampler to bounded reference sources."""

import json
from pathlib import Path

from speck.science_sample import sample_science_source, validate_science_sample_config
from speck.web_sample import _fingerprint, _write_json

FORMAT = "speck_reference_sample"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_reference_sample_result"


def validate_reference_sample_config(config, *, config_dir=None):
    """Validate a reference plan through the immutable document-quality contract."""

    if not isinstance(config, dict) or config.get("format") != FORMAT:
        raise ValueError("unsupported reference sample format")
    partition = config.get("downstream_partition")
    if not isinstance(partition, dict) or partition.get("category") != "reference":
        raise ValueError("reference sample requires the reference partition category")
    base = {
        **config,
        "format": "speck_science_sample",
        "downstream_partition": {**partition, "category": "science"},
    }
    normalized = validate_science_sample_config(base, config_dir=config_dir)
    normalized["format"] = FORMAT
    normalized["downstream_partition"]["category"] = "reference"
    normalized.pop("plan_fingerprint")
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_reference_sample_config(path):
    path = Path(path).resolve()
    return validate_reference_sample_config(json.loads(path.read_text()), config_dir=path.parent)


def sample_reference_source(config, *, restart=False):
    """Build a bounded reference sample and retain reference-category partitioning."""

    if "plan_fingerprint" not in config:
        config = validate_reference_sample_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized reference sample fingerprint mismatch")
    reference_fingerprint = config["plan_fingerprint"]
    payload = {
        **{key: value for key, value in config.items() if key != "plan_fingerprint"},
        "format": "speck_science_sample",
    }
    payload["plan_fingerprint"] = _fingerprint(payload)
    report = sample_science_source(payload, restart=restart)
    report["format"] = REPORT_FORMAT
    report["plan_fingerprint"] = reference_fingerprint
    report["gates"]["English_and_reference_content"] = report["gates"].pop(
        "English_and_science_content"
    )
    _write_json(Path(config["output_directory"]) / "report.json", report)
    return report
