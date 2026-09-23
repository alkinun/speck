"""The restored Stack-Edu screen must reject in the retained acquisition's order."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "stack_edu_content",
    Path(__file__).resolve().parents[1] / "experiments/corpus-audit/stack_edu_content.py",
)
content = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(content)

PROSE = {
    "detector": "py3langid==0.3.0",
    "minimum_alphabetic_characters": 80,
    "minimum_probability": 0.8,
}
CODE = "\n".join(f"value_{index} = compute({index}, offset={index * 3})" for index in range(12))


class Benchmarks:
    def matches(self, text):
        return ["bench"] if "benchmark_marker" in text else []


def row(raw, **change):
    return {
        "blob_id": hashlib.sha1(raw).hexdigest(),
        "length_bytes": len(raw),
        "src_encoding": "UTF-8",
        "language": "Python",
    } | change


@pytest.mark.parametrize(
    ("text", "change", "reason"),
    [
        (CODE, {"blob_id": "0" * 40}, "content_hash_mismatch"),
        (CODE, {"length_bytes": 1}, "content_length_metadata_mismatch"),
        (CODE + "\nkey = 'AKIA" + "A" * 16 + "'", {}, "code_high_confidence_secret"),
        ("x = 1\n", {}, "code_character_envelope"),
        (CODE + "\n# contact someone@company.org", {}, "raw_email_or_ipv4"),
        ("\n".join(["same_line_of_code = 1"] * 20), {}, "duplicate_lines"),
        (CODE + "\nbenchmark_marker = 1", {}, "benchmark_contamination"),
        (CODE, {}, None),
    ],
)
def test_screen_reasons(text, change, reason):
    raw = text.encode()
    assert content.screen(row(raw, **change), raw, PROSE, Benchmarks())[0] == reason


def test_missing_blob_is_counted_not_raised():
    assert content.screen(row(b""), None, PROSE, Benchmarks()) == ("blob_missing_404", None)
