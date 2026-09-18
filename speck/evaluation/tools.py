"""Small deterministic tool episodes with bounded calls and observable task completion."""

import json
import math

from speck.tokenization.chat import ChatFormatError
from speck.tokenization.tools import INSTRUCTION, parse_tool_calls

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate using add, multiply, or divide.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "multiply", "divide"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
                "additionalProperties": False,
            },
        },
    }
]


def execute(call):
    args = call["arguments"]
    if set(args) != {"operation", "a", "b"} or args["operation"] not in {
        "add",
        "multiply",
        "divide",
    }:
        return {"error": "invalid_arguments"}
    if any(
        type(args[key]) not in (int, float) or not math.isfinite(args[key]) or abs(args[key]) > 1e9
        for key in ("a", "b")
    ):
        return {"error": "invalid_arguments"}
    a, b = args["a"], args["b"]
    if args["operation"] == "divide" and b == 0:
        return {"error": "division_by_zero"}
    result = (
        a + b if args["operation"] == "add" else a * b if args["operation"] == "multiply" else a / b
    )
    return {"value": result}


SCENARIOS = [
    {
        "id": "calculation",
        "prompt": "Use calculate to multiply 17 by 23. Reply with only the number.",
        "answer": "391",
        "required_success": True,
    },
    {
        "id": "no_tool",
        "prompt": "Reply exactly HELLO. Do not use tools.",
        "answer": "HELLO",
        "forbid_tools": True,
    },
    {
        "id": "missing_information",
        "prompt": "I need the sum of two numbers but have not supplied them. Reply exactly NEED_NUMBERS and do not call a tool.",
        "answer": "NEED_NUMBERS",
        "forbid_tools": True,
    },
    {
        "id": "tool_failure",
        "prompt": "Use calculate to divide 1 by 0. If it reports division_by_zero, reply exactly UNDEFINED.",
        "answer": "UNDEFINED",
        "required_error": "division_by_zero",
    },
    {
        "id": "correction",
        "prompt": "First use calculate to divide 1 by 0. After the error, use calculate to add 2 and 2, then reply with only the result.",
        "answer": "4",
        "required_error": "division_by_zero",
        "required_success": True,
    },
]


def run_episode(respond, scenario, *, max_turns=4):
    messages = [
        {"role": "system", "content": INSTRUCTION + "\n<tools>" + json.dumps(TOOLS) + "</tools>"},
        {"role": "user", "content": scenario["prompt"]},
    ]
    calls, errors, invalid, successful, final, used_ids = 0, [], 0, 0, None, set()
    for _ in range(max_turns):
        response = respond(messages)
        messages.append({"role": "assistant", "content": response})
        if "<tool_calls>" not in response:
            final = response.strip()
            break
        try:
            parsed = parse_tool_calls(response, {"calculate"})
            if len(parsed) > 4 or any(call["id"] in used_ids for call in parsed):
                raise ChatFormatError("call limit or duplicate call ID")
        except ChatFormatError:
            invalid += 1
            break
        results = []
        for call in parsed:
            used_ids.add(call["id"])
            calls += 1
            result = execute(call)
            successful += "value" in result
            if "error" in result:
                errors.append(result["error"])
            results.append({"id": call["id"], "name": call["name"], "content": json.dumps(result)})
        messages.append(
            {"role": "user", "content": "<tool_results>" + json.dumps(results) + "</tool_results>"}
        )
    passed = (
        final == scenario["answer"]
        and invalid == 0
        and (not scenario.get("forbid_tools") or calls == 0)
        and (not scenario.get("required_success") or successful > 0)
        and (not scenario.get("required_error") or scenario["required_error"] in errors)
    )
    return {
        "id": scenario["id"],
        "task_completion": passed,
        "calls": calls,
        "invalid_envelopes": invalid,
        "successful_calls": successful,
        "tool_errors": errors,
        "final": final,
        "transcript": messages,
    }


def golden_checks():
    def call(operation, a, b, identity="1"):
        return (
            "<tool_calls>"
            + json.dumps(
                [
                    {
                        "id": identity,
                        "name": "calculate",
                        "arguments": {"operation": operation, "a": a, "b": b},
                    }
                ]
            )
            + "</tool_calls>"
        )

    responses = [
        [call("multiply", 17, 23), "391"],
        ["HELLO"],
        ["NEED_NUMBERS"],
        [call("divide", 1, 0), "UNDEFINED"],
        [call("divide", 1, 0), call("add", 2, 2, "2"), "4"],
    ]
    outcomes = []
    for scenario, values in zip(SCENARIOS, responses):
        iterator = iter(values)
        result = run_episode(lambda messages: next(iterator), scenario)
        assert result["task_completion"], result
        outcomes.append(result)
    assert not run_episode(lambda messages: "391", SCENARIOS[0])["task_completion"]
    assert (
        run_episode(lambda messages: "<tool_calls>broken</tool_calls>", SCENARIOS[0])[
            "invalid_envelopes"
        ]
        == 1
    )
    return outcomes
