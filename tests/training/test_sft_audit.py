import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.tokenization.chat import ChatTokenizer
from speck.training.sft_audit import audit_sft, decode_conversation
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


def test_shuffle_index_caches_are_not_accepted_as_conversations():
    with pytest.raises(ValueError, match="index caches"):
        decode_conversation({"indices": 42})
