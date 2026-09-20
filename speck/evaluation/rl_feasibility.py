"""Fixed-policy RL feasibility reporting over deterministic tool episodes.

This module evaluates a responder; it never updates policy parameters. It is intentionally
small so the same receipt format can later wrap math, code and sandbox environments.
"""

import hashlib
import json
from collections.abc import Callable, Iterable

from speck.evaluation.tools import SCENARIOS, run_episode


def _fingerprint(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def score_episode(outcome):
    """Return a bounded diagnostic reward; correctness remains the primary outcome."""

    reward = 1.0 if outcome["task_completion"] else 0.0
    reward -= 0.25 * outcome["invalid_envelopes"]
    reward -= 0.05 * max(0, outcome["calls"] - 4)
    return max(-1.0, min(1.0, reward))


def run_fixed_policy(
    respond: Callable,
    scenarios: Iterable[dict] = SCENARIOS,
    *,
    max_turns=4,
    policy_id="unidentified",
):
    """Evaluate one fixed responder and return a reproducible, non-training receipt."""

    scenarios = list(scenarios)
    identifiers = [scenario.get("id") for scenario in scenarios]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("every feasibility scenario requires a non-empty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("feasibility scenario IDs must be unique")
    outcomes = []
    for scenario in scenarios:
        outcome = run_episode(respond, scenario, max_turns=max_turns)
        outcome = {**outcome, "reward": score_episode(outcome)}
        outcomes.append(outcome)
    completed = sum(outcome["task_completion"] for outcome in outcomes)
    invalid = sum(outcome["invalid_envelopes"] for outcome in outcomes)
    receipt = {
        "format": "speck_fixed_policy_rl_feasibility",
        "format_version": 1,
        "status": "feasibility_only_no_policy_updates",
        "policy_id": policy_id,
        "scenario_ids": identifiers,
        "max_turns": max_turns,
        "episodes": outcomes,
        "summary": {
            "episodes": len(outcomes),
            "completed": completed,
            "completion_rate": completed / len(outcomes) if outcomes else 0.0,
            "invalid_envelopes": invalid,
            "mean_reward": (
                sum(outcome["reward"] for outcome in outcomes) / len(outcomes)
                if outcomes
                else 0.0
            ),
        },
        "transcript_sha256": _fingerprint([outcome["transcript"] for outcome in outcomes]),
        "boundary": "Fixed-policy evaluation only. No optimizer, rollout learner or reward-model update is performed.",
    }
    return receipt


def assert_feasibility_receipt(receipt):
    """Reject receipts that claim feasibility without complete, deterministic evidence."""

    if receipt.get("format") != "speck_fixed_policy_rl_feasibility":
        raise ValueError("unsupported RL feasibility receipt")
    if receipt.get("status") != "feasibility_only_no_policy_updates":
        raise ValueError("RL feasibility must not claim policy training")
    episodes = receipt.get("episodes")
    summary = receipt.get("summary")
    if not isinstance(episodes, list) or not episodes or not isinstance(summary, dict):
        raise ValueError("RL feasibility receipt requires episodes and summary")
    if summary.get("episodes") != len(episodes):
        raise ValueError("RL feasibility episode count mismatch")
    completed = sum(bool(outcome.get("task_completion")) for outcome in episodes)
    invalid = sum(int(outcome.get("invalid_envelopes", 0)) for outcome in episodes)
    rewards = [float(outcome.get("reward")) for outcome in episodes]
    expected_summary = {
        "episodes": len(episodes),
        "completed": completed,
        "completion_rate": completed / len(episodes),
        "invalid_envelopes": invalid,
        "mean_reward": sum(rewards) / len(rewards),
    }
    if summary != expected_summary:
        raise ValueError("RL feasibility summary does not match episode outcomes")
    if receipt.get("transcript_sha256") != _fingerprint(
        [outcome.get("transcript") for outcome in episodes]
    ):
        raise ValueError("RL feasibility transcript fingerprint mismatch")
    if not receipt.get("boundary", "").startswith("Fixed-policy evaluation only"):
        raise ValueError("RL feasibility boundary must prohibit policy updates")
    return receipt
