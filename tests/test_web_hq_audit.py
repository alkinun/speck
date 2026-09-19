"""A corrupt or oversized download must never publish an inspection receipt."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "web_hq_audit", Path(__file__).resolve().parents[1] / "experiments/corpus-audit/audit_web_hq.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.mark.parametrize("payload", [b"", b"wrong", b"expected plus excess"])
def test_corrupt_download_is_not_published(tmp_path, monkeypatch, payload):
    expected = b"valid"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    item = {
        "source_path": "sample.parquet",
        "url": "https://huggingface.co/datasets/example/corpus/resolve/pin/sample.parquet",
        "bytes": len(expected),
        "sha256": hashlib.sha256(expected).hexdigest(),
    }
    monkeypatch.setattr(
        audit,
        "load_plan",
        lambda _: {"repository": "example/corpus", "revision": "pin", "files": [item]},
    )
    monkeypatch.setattr(audit, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    output = tmp_path / "download"
    with pytest.raises(ValueError, match="identity mismatch|exceeded pinned"):
        audit.acquire(plan_path, output)
    assert not (output / "sample.parquet").exists()
    assert not (output / "acquisition.json").exists()


@pytest.mark.parametrize("score", [None, True, float("nan"), float("inf"), 0.49, 1.01])
def test_unexpected_score_cannot_disappear_from_the_census(score):
    with pytest.raises(ValueError, match="unexpected HQ score"):
        audit.score_band(score)


def test_exact_score_boundaries_match_the_frozen_plan():
    assert [audit.score_band(v) for v in (0.5, 0.649, 0.65, 0.799, 0.8, 0.949, 0.95, 1.0)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
