import copy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.sft_verify import main
from speck.training.sft_verify import (
    select_verified_sft_rows,
    verify_sft_outcome_receipt,
    verify_sft_outcomes,
)


def _tool_row(result=None):
    result = {"value": 6} if result is None else result
    return {
        "id": "tool-1",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Add 2 and 4."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": json.dumps({"operation": "add", "a": 2, "b": 4}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "calculate",
                "content": json.dumps(result),
            },
            {"role": "assistant", "content": "6"},
        ],
        "verification": {"kind": "tool", "expected": "6"},
    }


def test_outcome_receipt_separates_exact_code_tool_and_unverified(monkeypatch):
    def runner(code, test, entry_point, *, seconds):
        assert (code, test, entry_point, seconds) == ("x", "y", "f", 15)
        return {"status": "pass", "wall_seconds": 0.01}

    rows = [
        {
            "id": "exact",
            "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
            "verification": {"kind": "exact_text", "expected": "A"},
        },
        {
            "id": "code",
            "messages": [],
            "verification": {"kind": "code", "code": "x", "test": "y", "entry_point": "f"},
        },
        _tool_row(),
        {"id": "plain", "messages": []},
    ]
    receipt = verify_sft_outcomes(rows, runner=runner)
    assert [row["status"] for row in receipt["rows"]] == ["verified"] * 3 + ["unverified"]
    assert receipt["summary"] == {"input": 4, "by_status": {"unverified": 1, "verified": 3}}
    assert verify_sft_outcome_receipt(receipt)


def test_tool_outcome_rejects_tampered_recorded_result():
    row = _tool_row({"value": 7})
    receipt = verify_sft_outcomes([row])
    assert receipt["rows"][0]["status"] == "rejected"
    assert receipt["rows"][0]["reason"] == "tool result mismatch"


def test_receipt_validator_recomputes_summary_and_rejects_tampering():
    receipt = verify_sft_outcomes(
        [
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ],
                "verification": {"kind": "exact_text", "expected": "A"},
            }
        ]
    )
    verify_sft_outcome_receipt(receipt)
    tampered = copy.deepcopy(receipt)
    tampered["summary"]["by_status"]["verified"] = 2
    with pytest.raises(ValueError, match="summary"):
        verify_sft_outcome_receipt(tampered)


def test_sft_verify_cli_records_file_identity(tmp_path):
    source = tmp_path / "rows.parquet"
    output = tmp_path / "receipt.json"
    row = {
        "id": "one",
        "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
        "verification": {"kind": "exact_text", "expected": "A"},
    }
    pq.write_table(pa.Table.from_pylist([row]), source)
    main([str(source), "--output", str(output)])
    receipt = json.loads(output.read_text())
    assert receipt["inputs"][0]["sha256"]
    assert receipt["summary"]["by_status"] == {"verified": 1}


def test_verified_selector_binds_exact_input_order_and_content():
    rows = [
        {
            "id": "good",
            "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
            "verification": {"kind": "exact_text", "expected": "A"},
        },
        {
            "id": "bad",
            "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "B"}],
            "verification": {"kind": "exact_text", "expected": "A"},
        },
    ]
    receipt = verify_sft_outcomes(rows)
    assert [row["id"] for row in select_verified_sft_rows(rows, receipt)] == ["good"]
    with pytest.raises(ValueError, match="fingerprint"):
        select_verified_sft_rows(list(reversed(rows)), receipt)
    swapped = copy.deepcopy(receipt)
    swapped["rows"] = list(reversed(swapped["rows"]))
    with pytest.raises(ValueError, match="row fingerprint"):
        select_verified_sft_rows(rows, swapped)
