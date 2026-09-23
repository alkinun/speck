"""Score assistant responses against checked math answers and sandboxed stdin/stdout tests.

Verification is conservative: an unusual but correct rendering may score 0, but a response
scores 1 only when its final answer matches the reference or its program passes every case.
"""

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from speck.evaluation.code_runner import run_stdio

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)
_FRACTION = re.compile(r"^(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}$")


def final_text(response):
    """Drop the reasoning block; an unclosed block leaves no final answer."""

    response = _THINK.sub("", response)
    return "" if "<think>" in response else response


def last_boxed(text):
    """Return the contents of the last \\boxed{...}, honoring nested braces."""

    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, index = 0, start + len("\\boxed")
    for position in range(index, len(text)):
        depth += {"{": 1, "}": -1}.get(text[position], 0)
        if depth == 0:
            return text[index + 1 : position]
    return None


def normalize_math(answer):
    value = answer.strip().removeprefix("\\[").removesuffix("\\]").strip().strip("$").strip()
    value = re.sub(r"\\text\{(.*?)\}", r"\1", value)
    for old, new in (
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("^\\circ", ""),
        ("^{\\circ}", ""),
        ("\\%", "%"),
    ):
        value = value.replace(old, new)
    value = re.sub(r"\s+", "", value).rstrip(".")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", value):
        value = value.replace(",", "")
    return value


def _number(value):
    fraction = _FRACTION.match(value)
    if fraction:
        sign, numerator, denominator = fraction.groups()
        if int(denominator) == 0:
            return None
        return Fraction(int(numerator), int(denominator)) * (-1 if sign else 1)
    try:
        return Fraction(Decimal(value))
    except (InvalidOperation, ValueError):
        return None


def math_reward(response, ground_truth):
    answer = last_boxed(final_text(response))
    if answer is None:
        return 0.0
    # A few references carry their own \boxed{} or display-math wrapper.
    reference = last_boxed(ground_truth) or ground_truth
    candidate, reference = normalize_math(answer), normalize_math(reference)
    candidate_number, reference_number = _number(candidate), _number(reference)
    if candidate_number is not None and reference_number is not None:
        return float(candidate_number == reference_number)
    return float(candidate == reference and bool(reference))


def code_reward(response, tests, *, seconds=15):
    """Run the last fenced code block against every stdin/stdout case in a fresh sandbox."""

    blocks = _FENCE.findall(final_text(response))
    if not blocks:
        return 0.0
    cases = list(zip(tests["inputs"], tests["outputs"], strict=True))
    return float(run_stdio(blocks[-1], cases, seconds=seconds)["status"] == "pass")
