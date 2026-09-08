import json
import os
import subprocess
import sys

import pytest

from scripts.ruler_case_prepare import (
    NETWORK_GUARD,
    case_summary,
    convert_cases,
    directory_identity,
    raw_case_summary,
)


def test_convert_cases_matches_nemo_skills_shape(tmp_path):
    original = tmp_path / "raw.jsonl"
    converted = tmp_path / "cases" / "test.jsonl"
    original.write_text(
        json.dumps(
            {
                "index": 7,
                "input": "Question?",
                "outputs": ["answer"],
                "length": 4000,
                "length_w_model_temp": 4005,
                "answer_prefix": " Answer:",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    convert_cases(original, converted)

    assert json.loads(converted.read_text(encoding="utf-8")) == {
        "index": 7,
        "question": "Question? Answer:",
        "expected_answer": ["answer"],
        "length": 4000,
    }


def test_case_summary_enforces_reserved_context_ceiling(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps({"index": 0, "question": "q", "expected_answer": ["a"], "length": 4047}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="violates the case contract"):
        case_summary(path, "qa_1", samples=1, length=4096)


def test_raw_case_summary_accounts_for_both_template_reserves(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps(
            {
                "index": 0,
                "input": "q",
                "outputs": ["a"],
                "length": 4041,
                "length_w_model_temp": 4046,
                "answer_prefix": " Answer:",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert raw_case_summary(path, "qa_1", samples=1, length=4096) == {
        "model_template_tokens": 5,
        "maximum_accounted_length": 4096,
    }


def test_directory_identity_is_path_order_independent(tmp_path):
    (tmp_path / "b").write_text("second", encoding="utf-8")
    (tmp_path / "a").write_text("first", encoding="utf-8")

    first = directory_identity(tmp_path)
    second = directory_identity(tmp_path)

    assert first == second
    assert [entry["path"] for entry in first["files"]] == ["a", "b"]


def test_network_guard_denies_ipv4_connection(tmp_path):
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(NETWORK_GUARD, encoding="utf-8")
    audit = tmp_path / "audit.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(guard),
            "SPECK_NETWORK_AUDIT_LOG": str(audit),
            "SPECK_NETWORK_AUDIT_PHASE": "test",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", "import socket; socket.create_connection(('127.0.0.1', 9))"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Speck offline RULER guard denied network access" in result.stderr
    assert json.loads(audit.read_text(encoding="utf-8"))["event"] == "network_attempt_denied"
