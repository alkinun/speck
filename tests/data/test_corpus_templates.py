import hashlib
import json

import pytest

from scripts.corpus_templates import audit, role_hint, template_digest
from speck.provenance.io import file_sha256


def fixture(tmp_path):
    source, index, manifest = (
        tmp_path / name for name in ("text.jsonl", "index.jsonl", "manifest.json")
    )
    rows, spans, offset = [], [], 0
    for ordinal, (host, text) in enumerate(
        [
            ("a.test", "Convert 123.5 meters."),
            ("a.test", "Convert 246.0 meters."),
            ("b.test", "Convert 100.0 meters."),
            ("a.test", "A proof with an extra condition."),
        ]
    ):
        digest = hashlib.sha256(text.encode()).hexdigest()
        rows.append(
            {
                "content_id": str(ordinal),
                "text": text,
                "url": f"https://{host}/{ordinal}",
                "released_content_sha256": digest,
            }
        )
        tokens = 10 + ordinal
        spans.append(
            {
                "ordinal": ordinal,
                "content_id": str(ordinal),
                "token_start": offset,
                "token_count": tokens,
                "utf8_bytes": len(text.encode()),
                "released_content_sha256": digest,
            }
        )
        offset += tokens
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    index.write_text("".join(json.dumps(row) + "\n" for row in spans))
    manifest.write_text(
        json.dumps(
            {
                "status": "complete_document_token_cache_not_training_view",
                "plan": {"input": {"path": str(source), "sha256": file_sha256(source)}},
                "documents": {"path": index.name, "sha256": file_sha256(index)},
                "document_count": 4,
                "token_count": offset,
            }
        )
    )
    return manifest


def test_numeric_groups_are_host_scoped_and_never_rejection(tmp_path):
    manifest = fixture(tmp_path)
    before = (tmp_path / "text.jsonl").read_bytes()
    result = audit(manifest, file_sha256(manifest), tmp_path / "out")
    assert result["counts"] == {"documents": 4, "tokens": 46}
    assert result["repeated_families"] == 1
    assert result["documents_in_repeated_families"] == 2
    assert result["tokens_in_repeated_families"] == 21
    assert not result["training_authority"]
    assert result["status"] == "census_complete_not_a_selection_policy"
    examples = [
        json.loads(line) for line in (tmp_path / "out/examples.jsonl").read_text().splitlines()
    ]
    assert {row["ordinal"] for row in examples} == {0, 1}
    assert (tmp_path / "text.jsonl").read_bytes() == before
    with pytest.raises(FileExistsError):
        audit(manifest, file_sha256(manifest), tmp_path / "out")


def test_erasing_numbers_can_merge_different_math_problems():
    assert template_digest("Solve x + 2 = 4.") == template_digest("Solve x + 3 = 8.")
    assert template_digest("Solve x + 2 = 4.") != template_digest("Solve x * 2 = 4.")


@pytest.mark.parametrize("file", ["text.jsonl", "index.jsonl", "manifest.json"])
def test_input_tampering_never_publishes_a_report(tmp_path, file):
    manifest = fixture(tmp_path)
    expected = file_sha256(manifest)
    with (tmp_path / file).open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError):
        audit(manifest, expected, tmp_path / "out")
    assert not (tmp_path / "out/report.json").exists()


def test_directory_hint_requires_host_path_and_content():
    row = {
        "url": "https://nrich.maths.org/public/leg.php?code=1",
        "text": "Search by Topic. There are 66 results",
    }
    assert role_hint(row) == "nrich_topic_directory"
    assert role_hint({**row, "url": "https://nrich.maths.org/real-lesson"}) == "unclassified"
    assert role_hint({**row, "text": "A fully worked example"}) == "unclassified"
