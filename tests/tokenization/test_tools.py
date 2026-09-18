import json

import pytest

from speck.evaluation.tools import TOOLS, golden_checks
from speck.tokenization.chat import ChatFormatError
from speck.tokenization.tools import adapt_conversation, parse_tool_calls


def fixture():
    return {
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "Calculate twice."},
            {
                "role": "assistant",
                "content": "<think>Use the tool.</think>",
                "weight": 0,
                "tool_calls": [
                    {
                        "id": "first",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"operation":"add","a":1,"b":2}',
                        },
                    },
                    {
                        "id": "second",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"operation":"multiply","a":2,"b":3}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "second", "content": "6"},
            {"role": "tool", "tool_call_id": "first", "content": "3"},
            {"role": "assistant", "content": "3 and 6", "weight": 1},
        ],
    }


def test_adapter_preserves_parallel_results_and_context_only_assistant():
    row = adapt_conversation(fixture())
    assert row["tools"] == []
    assert [m["role"] for m in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    message = row["messages"][2]
    assert message["weight"] == 0
    calls = parse_tool_calls(message["content"], {"calculate"})
    assert calls[0]["arguments"] == {"operation": "add", "a": 1, "b": 2}
    content = row["messages"][3]["content"]
    results = json.loads(content.removeprefix("<tool_results>").removesuffix("</tool_results>"))
    assert [result["id"] for result in results] == ["second", "first"]


def test_adapter_rejects_lost_or_unbound_tool_results():
    row = fixture()
    row["messages"].pop(2)
    with pytest.raises(ChatFormatError, match="all tool calls"):
        adapt_conversation(row)
    row = fixture()
    row["messages"][2]["tool_call_id"] = "other"
    with pytest.raises(ChatFormatError, match="pending"):
        adapt_conversation(row)


@pytest.mark.parametrize(
    "payload",
    [
        '[{"id":"1","name":"calculate","arguments":{"a":NaN}}]',
        '[{"id":"1","name":"calculate","arguments":{"a":1,"a":2}}]',
        '[{"id":"1","name":"undeclared","arguments":{}}]',
    ],
)
def test_parser_rejects_ambiguous_or_undeclared_calls(payload):
    with pytest.raises(ChatFormatError):
        parse_tool_calls(f"<tool_calls>{payload}</tool_calls>", {"calculate"})


def test_deterministic_tool_environment_requires_observed_completion():
    assert len(golden_checks()) == 5
