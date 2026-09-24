import json

import pytest

from speck.data.production_data import preprocess_sources, validate_preprocess_config
from speck.data.sqlite_settings import validate_sqlite_settings
from tests.data.test_production_data import _config

SETTINGS = {
    "journal_mode": "WAL",
    "synchronous": "FULL",
    "wal_autocheckpoint_pages": 65536,
    "page_size": 4096,
    "cache_size_kib": 2000,
}


def test_v2_records_requested_and_actual_settings_and_keeps_v1_compatible(tmp_path):
    raw = _config(tmp_path, "legacy")
    legacy = preprocess_sources(raw)["manifest"]
    assert legacy["format_version"] == 1
    assert "sqlite" not in legacy
    current = _config(tmp_path, "bound")
    current.update(format_version=2, sqlite=SETTINGS)
    result = preprocess_sources(current)["manifest"]
    assert result["format_version"] == 2
    assert result["sqlite"] == SETTINGS
    assert result["sqlite_runtime"]["wal_autocheckpoint"] == 65536
    assert result["sqlite_runtime"]["synchronous"] == 2
    assert result["sqlite_runtime"]["cache_size"] == -2000
    for key in ("outputs", "removals", "counts"):
        assert result[key] == legacy[key]
    assert preprocess_sources(current)["manifest"] == result


@pytest.mark.parametrize("when", ["checkpoint", "published"])
def test_policy_change_cannot_reuse_existing_checkpoint_or_output(tmp_path, when):
    config = _config(tmp_path)
    config.update(format_version=2, sqlite=dict(SETTINGS))
    if when == "checkpoint":
        with pytest.raises(RuntimeError, match="injected"):
            preprocess_sources(config, crash_after_records=3)
    else:
        preprocess_sources(config)
    original = validate_preprocess_config(config)["plan_fingerprint"]
    config["sqlite"]["wal_autocheckpoint_pages"] = 1000
    assert validate_preprocess_config(config)["plan_fingerprint"] != original
    with pytest.raises(ValueError, match="contract changed|manifest identity"):
        preprocess_sources(config)


def test_tampered_runtime_observation_is_rejected_on_reopen(tmp_path):
    config = _config(tmp_path)
    config.update(format_version=2, sqlite=SETTINGS)
    preprocess_sources(config)
    path = tmp_path / "production/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sqlite_runtime"]["synchronous"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="actual SQLite"):
        preprocess_sources(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("synchronous", "NORMAL"),
        ("wal_autocheckpoint_pages", 0),
        ("wal_autocheckpoint_pages", True),
        ("page_size", 8192),
        ("cache_size_kib", 4000),
    ],
)
def test_unqualified_settings_fail_closed(field, value):
    with pytest.raises(ValueError, match="qualified"):
        validate_sqlite_settings({**SETTINGS, field: value})


def test_sqlite_declaration_requires_config_v2(tmp_path):
    config = _config(tmp_path)
    config["sqlite"] = SETTINGS
    with pytest.raises(ValueError, match="exactly"):
        validate_preprocess_config(config)
    del config["sqlite"]
    config["format_version"] = 2
    with pytest.raises(ValueError, match="exactly"):
        validate_preprocess_config(config)
