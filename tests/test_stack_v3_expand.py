import hashlib
import json
from pathlib import Path

import pytest

from speck.stack_v3_expand import (
    materialize_expansion_config,
    validate_expansion_config,
)

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_expansion_overlay_hash_binds_base_and_changes_only_quota_and_output(tmp_path):
    base = json.loads((ROOT / "research" / "flagship" / "stack_v3_qualification.json").read_text())
    base["output"]["directory"] = str(tmp_path / "old")
    base["output"]["download_directory"] = str(tmp_path / "downloads")
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    overlay = validate_expansion_config(
        {
            "format": "speck_stack_v3_expansion",
            "format_version": 1,
            "status": "bounded_successor_authorized_not_training_authority",
            "base_config": str(base_path),
            "base_config_sha256": _sha256(base_path),
            "language_sample_bytes": {"Python": 123},
            "output_directory": str(tmp_path / "new"),
        }
    )
    materialized = materialize_expansion_config(overlay)

    assert materialized["filters"]["language_sample_bytes"] == {"Python": 123}
    assert materialized["output"]["directory"] == str((tmp_path / "new").resolve())
    assert materialized["output"]["download_directory"] == str((tmp_path / "downloads").resolve())
    assert materialized["source"]["files"] == base["source"]["files"]

    base_path.write_text(base_path.read_text() + "\n")
    with pytest.raises(ValueError, match="base plan identity mismatch"):
        materialize_expansion_config(overlay)


def test_real_expansion_is_bounded_and_adds_no_source_shards():
    base_path = ROOT / "research" / "flagship" / "stack_v3_qualification.json"
    path = ROOT / "research" / "flagship" / "stack_v3_expansion_v1b.json"
    overlay = validate_expansion_config(json.loads(path.read_text()), config_dir=path.parent)
    materialized = materialize_expansion_config(overlay)
    base = json.loads(base_path.read_text())

    assert sum(overlay["language_sample_bytes"].values()) == 85_000_000
    assert materialized["source"]["files"] == base["source"]["files"]
    assert materialized["output"]["download_directory"] == base["output"]["download_directory"]
