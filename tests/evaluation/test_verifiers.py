import shutil

import pytest

from speck.evaluation.verifiers import code_reward, last_boxed, math_reward, normalize_math


def test_last_boxed_honors_nested_braces_and_takes_the_last_answer():
    assert last_boxed(r"first \boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed("no answer") is None
    assert last_boxed(r"\boxed{unclosed") is None


@pytest.mark.parametrize(
    "response,truth,reward",
    [
        (r"<think>maybe \boxed{3}</think>So \boxed{4}.", "4", 1.0),
        (r"<think>\boxed{4}</think>No final answer.", "4", 0.0),
        (r"<think>unfinished \boxed{4}", "4", 0.0),
        (r"\boxed{\dfrac{1}{2}}", "0.5", 1.0),
        (r"\boxed{1,000}", "1000", 1.0),
        (r"\boxed{$2\sqrt{5}$}", r"2\sqrt{5}", 1.0),
        (r"\boxed{2\sqrt{3}}", r"2\sqrt{5}", 0.0),
        (r"\boxed{45^\circ}", "45", 1.0),
        (r"\boxed{\text{3:30 PM}}", "3:30 PM", 1.0),
        (r"\boxed{}", "", 0.0),
    ],
)
def test_math_reward_is_conservative(response, truth, reward):
    assert math_reward(response, truth) == reward


def test_normalize_math_strips_presentation_only():
    assert normalize_math(r" $\left( 1, 2 \right)$ ") == "(1,2)"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap unavailable")
def test_code_reward_runs_the_final_program_against_every_case():
    tests = {"inputs": ["1 2\n", "5 5\n"], "outputs": ["3\n", "10\n"]}
    good = "<think>add</think>\n```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    wrong = "```python\nprint(3)\n```"
    assert code_reward(good, tests) == 1.0
    assert code_reward(wrong, tests) == 0.0
    assert code_reward("no code", tests) == 0.0
