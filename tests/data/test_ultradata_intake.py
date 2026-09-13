import json

import pytest

import speck.data.ultradata_intake as intake

SETTINGS = {
    "minimum_alphabetic_characters": 80,
    "minimum_english_probability": 0.8,
    "maximum_characters": 50000,
}


class Response:
    status_code = 200

    def __init__(self, revision="a" * 40, partial=False):
        self.headers = {"x-revision": revision}
        self.payload = json.dumps(
            {
                "partial": partial,
                "num_rows_total": 10,
                "rows": [
                    {"row_idx": index, "row": {"content": "example"}, "truncated_cells": []}
                    for index in range(2)
                ],
            }
        ).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, size):
        yield self.payload


@pytest.mark.parametrize("case", ["revision", "partial", "size", "rows"])
def test_unverifiable_or_unbounded_viewer_response_is_rejected(tmp_path, monkeypatch, case):
    response = Response(
        revision="b" * 40 if case == "revision" else "a" * 40, partial=case == "partial"
    )
    monkeypatch.setattr(intake.requests, "get", lambda *args, **kwargs: response)
    view = {"repo": "fixture/data", "config": "sample", "split": "train", "revision": "a" * 40}
    path = tmp_path / "response.json"
    with pytest.raises(ValueError):
        intake.fetch_window(
            view, 0, 3 if case == "rows" else 2, 1 if case == "size" else 10000, path
        )
    assert not path.exists()


def test_revision_checked_response_retains_exact_bytes(tmp_path, monkeypatch):
    response = Response()
    monkeypatch.setattr(intake.requests, "get", lambda *args, **kwargs: response)
    view = {"repo": "fixture/data", "config": "sample", "split": "train", "revision": "a" * 40}
    path = tmp_path / "response.json"
    result, receipt = intake.fetch_window(view, 0, 2, 10000, path)
    assert path.read_bytes() == response.payload
    assert receipt["revision_header"] == view["revision"]
    assert len(result["rows"]) == 2


def test_code_serialization_and_truncation_are_reported_without_imputing_reasoning(monkeypatch):
    class Language:
        def classify(self, text):
            return "en", 0.99

    monkeypatch.setattr(intake, "_language_identifier", Language)
    task = "Solve this programming task carefully. " * 5
    analysis = "Reason through boundary cases and computational complexity. " * 5
    row = {
        "uuid": "example",
        "task": task,
        "analysis": analysis,
        "solution": "def solve(): return 1",
        "test": "assert solve() == 1",
        "content": task + "\n" + "def solve(): return 1",
        "full_content": task + analysis + "def solve(): return 1" + "assert solve() == 1",
        "raw_content": json.dumps({"task": task, "analysis": analysis}),
        "content_format": "task_solution",
    }
    records = [{"row": row, "truncated_cells": []}, {"row": row, "truncated_cells": ["analysis"]}]
    result, ids = intake.summarize_rows({"kind": "code_exercise"}, records, SETTINGS)
    assert result["counts"]["complete_rows"] == 1
    assert result["counts"]["truncated_rows_excluded"] == 1
    assert result["counts"].get("content_exactly_contains_analysis", 0) == 0
    assert result["counts"]["full_content_exactly_contains_analysis"] == 1
    assert result["language_diagnostic"] == {"confident_English": 1}
    assert ids == {"example"}


def test_python_language_diagnostic_uses_prose_rather_than_string_literals(monkeypatch):
    observed = []

    class Language:
        def classify(self, text):
            observed.append(text)
            return "en", 0.99

    monkeypatch.setattr(intake, "_language_identifier", Language)
    content = (
        "# " + "This algorithm handles a sequence of input values. " * 5 + "\nvalue = '其他语言'\n"
    )
    result, _ = intake.summarize_rows(
        {"kind": "python_source"}, [{"row": {"content": content}, "truncated_cells": []}], SETTINGS
    )
    assert result["language_diagnostic"] == {"confident_English": 1}
    assert "其他语言" not in observed[0]
