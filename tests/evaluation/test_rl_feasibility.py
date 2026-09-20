import json

from speck.evaluation.rl_feasibility import assert_feasibility_receipt, run_fixed_policy
from speck.evaluation.tools import SCENARIOS


def _call(operation, a, b, identity="1"):
    return "<tool_calls>" + json.dumps(
        [{"id": identity, "name": "calculate", "arguments": {"operation": operation, "a": a, "b": b}}]
    ) + "</tool_calls>"


def test_fixed_policy_receipt_covers_success_failure_and_no_tool_cases():
    responses = {
        "calculation": iter([_call("multiply", 17, 23), "391"]),
        "no_tool": iter(["HELLO"]),
        "missing_information": iter(["NEED_NUMBERS"]),
        "tool_failure": iter([_call("divide", 1, 0), "UNDEFINED"]),
        "correction": iter([_call("divide", 1, 0), _call("add", 2, 2, "2"), "4"]),
    }

    # Use prompt routing only for this fixed-policy fixture; production policies receive full messages.
    def policy(messages):
        prompt = messages[1]["content"]
        if "multiply" in prompt:
            key = "calculation"
        elif "HELLO" in prompt:
            key = "no_tool"
        elif "not supplied" in prompt:
            key = "missing_information"
        elif "divide 1 by 0. If" in prompt:
            key = "tool_failure"
        else:
            key = "correction"
        return next(responses[key])

    receipt = run_fixed_policy(policy, SCENARIOS, policy_id="golden-fixed-policy")
    assert receipt["summary"]["completed"] == 5
    assert receipt["summary"]["invalid_envelopes"] == 0
    assert receipt["status"] == "feasibility_only_no_policy_updates"
    assert_feasibility_receipt(receipt)


def test_fixed_policy_receipt_detects_transcript_tampering():
    receipt = run_fixed_policy(lambda messages: "391", [SCENARIOS[0]])
    receipt["episodes"][0]["transcript"].append({"role": "assistant", "content": "tampered"})
    try:
        assert_feasibility_receipt(receipt)
    except ValueError as error:
        assert "fingerprint" in str(error)
    else:
        raise AssertionError("tampered transcript was accepted")


def test_fixed_policy_receipt_detects_summary_tampering():
    receipt = run_fixed_policy(lambda messages: "391", [SCENARIOS[0]])
    receipt["summary"]["completion_rate"] = 1.0
    try:
        assert_feasibility_receipt(receipt)
    except ValueError as error:
        assert "summary" in str(error)
    else:
        raise AssertionError("tampered summary was accepted")
