import pytest

from scripts.tokenizer_budget_fallback import require_budget_stop
from speck.tokenization.pilot_screen import analyze_tokenizer_screen
from tests.tokenization.test_tokenizer_pilot_screen import inputs


def test_incomplete_or_passing_budget_never_freezes_baseline():
    plan, nominations, runs, accounting = inputs()
    with pytest.raises(ValueError, match="decisive budget stop"):
        require_budget_stop(analyze_tokenizer_screen(plan, nominations, runs, accounting))
    accounting["complete"] = False
    with pytest.raises(ValueError, match="decisive budget stop"):
        require_budget_stop(analyze_tokenizer_screen(plan, nominations, runs, accounting))


def test_frozen_budget_stop_freezes_only_the_declared_fallback():
    plan, nominations, runs, accounting = inputs()
    accounting["other_spent_gpu_hours"] = 30
    result = analyze_tokenizer_screen(plan, nominations, runs, accounting)
    require_budget_stop(result)
    result["fallback"] = "custom-small"
    with pytest.raises(ValueError):
        require_budget_stop(result)
