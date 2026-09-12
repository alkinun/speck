from pathlib import Path

from speck.tokenizer_pilot_orchestration import load_execution_record

ROOT = Path(__file__).parents[1]
EXECUTIONS = ROOT / "research/flagship/tokenizer_pilot_executions_v1"


def test_three_screen_execution_records_resolve_without_expanding_authority():
    records = [load_execution_record(path) for path in sorted(EXECUTIONS.glob("*.json"))]

    assert len(records) == 3
    assert {record["run"]["tokenizer_id"] for record in records} == {
        "mistral-32k",
        "speck-bpe-40960-whitespace",
        "speck-bpe-32000-whitespace",
    }
    assert {record["run"]["seed"] for record in records} == {42}
    assert {record["repository_revision"] for record in records} == {
        "058ad3023a3bb1060d49eb92e2fb0ddfd69166e9"
    }
    for record in records:
        assert record["authority"] == {
            "screen_execution": True,
            "confirmation_execution": False,
            "D5_opening": False,
            "final_selection": False,
            "flagship_training": False,
        }


def test_execution_records_bind_all_required_qualifications_and_code():
    record = load_execution_record(EXECUTIONS / "mistral-32k-seed-42.json")

    assert set(record["qualifications"]) == {
        "runtime_inputs",
        "checkpoint_resume",
        "document_nll",
    }
    assert set(record["implementation"]) == {
        "orchestration",
        "trainer",
        "evaluation",
        "cli",
        "tests",
    }
    assert record["run"]["status"] == "corrected_materialized_not_started_execution_blocked"
