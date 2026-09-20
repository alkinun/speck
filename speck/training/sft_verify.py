"""Verify declared SFT outcomes without conflating them with serialization validity.

Rows are only verified when they carry an explicit ``verification`` object.  This is
deliberate: a structurally valid answer is not evidence that the answer is correct.
The receipt is deterministic and can be used to select verified subsets downstream.
"""

import hashlib
import json
import re
from collections import Counter

from speck.evaluation.code_runner import run_python
from speck.evaluation.tools import execute
from speck.tokenization.chat import ChatFormatError
from speck.tokenization.tools import adapt_conversation, parse_tool_calls

_TOOL_RESULTS = re.compile(r"\s*<tool_results>(.*?)</tool_results>\s*", re.S)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _last_answer(messages):
    answers = [message["content"] for message in messages if message.get("role") == "assistant"]
    if not answers:
        raise ChatFormatError("conversation has no assistant answer")
    return answers[-1].strip()


def _verify_exact(row, verification):
    expected = verification.get("expected")
    if not isinstance(expected, str):
        raise ValueError("exact_text verification requires a string expected value")
    adapted = adapt_conversation(row)
    actual = _last_answer(adapted["messages"])
    return {"status": "verified" if actual == expected.strip() else "rejected", "actual": actual}


def _verify_code(verification, runner):
    required = ("code", "test", "entry_point")
    if any(not isinstance(verification.get(key), str) or not verification[key] for key in required):
        raise ValueError("code verification requires non-empty code, test, and entry_point")
    result = runner(
        verification["code"], verification["test"], verification["entry_point"],
        seconds=verification.get("seconds", 15),
    )
    return {"status": "verified" if result.get("status") == "pass" else "rejected", "result": result}


def _parse_results(content):
    match = _TOOL_RESULTS.fullmatch(content)
    if match is None:
        raise ChatFormatError("tool call is missing a complete tool_results envelope")
    try:
        values = json.loads(match.group(1))
    except (TypeError, ValueError) as error:
        raise ChatFormatError("tool results must be valid JSON") from error
    if not isinstance(values, list):
        raise ChatFormatError("tool results must be a list")
    return values


def _verify_tool(row, verification):
    adapted = adapt_conversation(row)
    messages = adapted["messages"]
    names = {tool["function"]["name"] for tool in row.get("tools", [])}
    expected_final = verification.get("expected")
    if expected_final is not None and not isinstance(expected_final, str):
        raise ValueError("tool verification expected value must be a string")
    executed = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or "<tool_calls>" not in message["content"]:
            continue
        calls = parse_tool_calls(message["content"], names)
        if index + 1 >= len(messages) or messages[index + 1].get("role") != "user":
            raise ChatFormatError("tool call must be followed by tool results")
        recorded = _parse_results(messages[index + 1]["content"])
        actual = [
            {"id": call["id"], "name": call["name"], "content": json.dumps(execute(call))}
            for call in calls
        ]
        if recorded != actual:
            return {"status": "rejected", "reason": "tool result mismatch", "tool_results": actual}
        executed.extend(actual)
    if expected_final is not None and _last_answer(messages) != expected_final.strip():
        return {"status": "rejected", "reason": "final answer mismatch", "actual": _last_answer(messages)}
    return {"status": "verified", "tool_results": executed, "actual": _last_answer(messages)}


def verify_sft_outcomes(rows, *, runner=run_python):
    """Return a deterministic receipt for explicitly declared exact/code/tool checks."""
    if not isinstance(rows, list):
        rows = list(rows)
    results, counts = [], Counter()
    for index, row in enumerate(rows):
        identity = row.get("id") or row.get("sha256") or _digest(row)
        verification = row.get("verification")
        result = {
            "index": index,
            "id": identity,
            "input_sha256": _digest(row),
            "status": "unverified",
        }
        if not isinstance(verification, dict):
            result["reason"] = "missing verification declaration"
        else:
            kind = verification.get("kind")
            result["kind"] = kind
            try:
                if kind == "exact_text":
                    result.update(_verify_exact(row, verification))
                elif kind == "code":
                    result.update(_verify_code(verification, runner))
                elif kind == "tool":
                    result.update(_verify_tool(row, verification))
                else:
                    result["status"] = "unverified"
                    result["reason"] = "unsupported verification kind"
            except (ChatFormatError, ValueError, KeyError, TypeError) as error:
                result["status"] = "rejected"
                result["reason"] = str(error)
        counts[result["status"]] += 1
        results.append(result)
    return {
        "format": "speck_sft_outcome_verification",
        "format_version": 1,
        "status": "completed",
        "input_sha256": _digest(rows),
        "rows": results,
        "summary": {"input": len(rows), "by_status": dict(sorted(counts.items()))},
        "interpretation": (
            "Structural adaptation and outcome verification are separate gates. Only rows with "
            "an explicit exact_text, code, or tool declaration can be verified; unverified rows "
            "must not be described as correctness-checked."
        ),
    }


def verify_sft_outcome_receipt(receipt):
    """Fail closed on receipt edits and summary/fingerprint tampering."""
    if (
        not isinstance(receipt, dict)
        or receipt.get("format") != "speck_sft_outcome_verification"
        or receipt.get("format_version") != 1
    ):
        raise ValueError("invalid SFT outcome receipt format")
    rows = receipt.get("rows")
    if not isinstance(rows, list) or receipt.get("status") != "completed":
        raise ValueError("invalid SFT outcome receipt rows")
    counts = Counter()
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("status") not in {"verified", "rejected", "unverified"}
            or not isinstance(row.get("input_sha256"), str)
            or len(row["input_sha256"]) != 64
        ):
            raise ValueError("invalid SFT outcome row")
        counts[row["status"]] += 1
    if receipt.get("summary") != {"input": len(rows), "by_status": dict(sorted(counts.items()))}:
        raise ValueError("SFT outcome summary does not match rows")
    if not isinstance(receipt.get("input_sha256"), str) or len(receipt["input_sha256"]) != 64:
        raise ValueError("SFT outcome receipt is missing input fingerprint")
    return True


def select_verified_sft_rows(rows, receipt):
    """Select only verified rows after binding the receipt to the exact input sequence."""
    if not isinstance(rows, list):
        rows = list(rows)
    verify_sft_outcome_receipt(receipt)
    if receipt["input_sha256"] != _digest(rows):
        raise ValueError("SFT outcome receipt input fingerprint does not match rows")
    if len(receipt["rows"]) != len(rows):
        raise ValueError("SFT outcome receipt row count does not match rows")
    selected = []
    for row, outcome in zip(rows, receipt["rows"], strict=True):
        if outcome["input_sha256"] != _digest(row):
            raise ValueError("SFT outcome row fingerprint does not match rows")
        if outcome["status"] == "verified":
            selected.append(row)
    return selected
