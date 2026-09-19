import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.tokenization.chat import ChatTokenizer
from speck.training.sft_audit import audit_sft, decode_conversation
from tests.tokenization.test_tools import fixture as tool_fixture
from tests.training.test_sft import BaseTokenizer


def rows():
    return [
        {
            "source": "fixture",
            "subset": "text",
            "source_id": str(index),
            "tools": [],
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A" * (index + 1), "weight": 1},
            ],
        }
        for index in range(20)
    ]


def test_arrow_json_and_parquet_have_identical_stratified_samples(tmp_path):
    data = rows()
    parquet = tmp_path / "stock.parquet"
    arrow = tmp_path / "stock.arrow"
    pq.write_table(pa.Table.from_pylist(data), parquet)
    encoded = [
        {**row, "messages": [json.dumps(message) for message in row["messages"]]}
        for row in reversed(data)
    ]
    table = pa.Table.from_pylist(encoded)
    with pa.OSFile(str(arrow), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    tokenizer = ChatTokenizer(BaseTokenizer(tmp_path / "tokenizer.model"))
    a = audit_sft([parquet], tokenizer, samples_per_subset=5, lengths=(16, 64))
    b = audit_sft([arrow], tokenizer, samples_per_subset=5, lengths=(16, 64))
    assert a["rows"] == b["rows"] == 20
    assert a["subsets"] == b["subsets"]
    summary = a["subsets"][0]
    assert summary["sampled_rows"] == summary["serializable_rows"] == 5
    assert summary["complete_rows_fitting_context"]["64"] == 5
    assert a["inputs"][0]["sha256"] != b["inputs"][0]["sha256"]


def test_audit_counts_unsupported_and_context_only_rows(tmp_path):
    data = rows()[:2]
    data[0]["messages"][1]["weight"] = 0
    data[1]["tools"] = [{"type": "function", "function": {"name": "lookup"}}]
    path = tmp_path / "stock.parquet"
    pq.write_table(pa.Table.from_pylist(data), path)
    result = audit_sft([path], ChatTokenizer(BaseTokenizer(tmp_path / "tokenizer.model")))
    summary = result["subsets"][0]
    assert summary["census"]["with_context_only_assistant"] == 1
    assert summary["census"]["with_tools"] == 1
    assert summary["serializable_rows"] == 0
    assert sum(summary["rejections"].values()) == 2
    assert summary["tokens"] is None
    assert summary["complete_rows_fitting_context"] == {"4096": 0}


def test_invalid_json_features_fail_loudly():
    with pytest.raises(ValueError, match="invalid JSON"):
        decode_conversation({"messages": ["{broken"]})
    with pytest.raises(ValueError, match="objects"):
        decode_conversation({"messages": ["null"]})


def test_tool_audit_counts_complete_context_and_masks_observations(tmp_path):
    valid = {**tool_fixture(), "source": "fixture", "subset": "tools"}
    valid["messages"][1]["content"] = ""
    valid["messages"][1]["reasoning_content"] = "Use the tool."
    broken = json.loads(json.dumps(valid))
    broken["messages"].pop(2)
    path = tmp_path / "tools.parquet"
    # JSON feature encoding preserves heterogeneous message/tool fields exactly.
    encoded = [
        {
            **row,
            "messages": [json.dumps(m) for m in row["messages"]],
            "tools": [json.dumps(t) for t in row["tools"]],
        }
        for row in (valid, broken)
    ]
    pq.write_table(pa.Table.from_pylist(encoded), path)
    tokenizer = ChatTokenizer(BaseTokenizer(tmp_path / "tokenizer.model"))
    result = audit_sft([path], tokenizer, lengths=(16, 4096))
    subset = result["subsets"][0]
    assert result["tool_protocol"] == "speck_tools_v1"
    assert subset["census"]["with_thinking"] == 2
    assert subset["serializable_rows"] == 1
    assert subset["rejections"] == {"all tool calls must receive results before the next turn": 1}
    assert subset["complete_rows_fitting_context"] == {"16": 0, "4096": 1}
    measurement = next(m for m in subset["sample_measurements"] if "tokens" in m)
    # Only the final answer plus EOS is supervised; definitions, calls and results add context.
    assert measurement["supervised_tokens"] == len(tokenizer.base.encode("3 and 6")) + 1
    assert measurement["tokens"] > measurement["supervised_tokens"]


def test_shuffle_index_caches_are_not_accepted_as_conversations():
    with pytest.raises(ValueError, match="index caches"):
        decode_conversation({"indices": 42})
