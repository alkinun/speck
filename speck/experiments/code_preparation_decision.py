"""Validate a supply-informed preparation successor without changing scientific run budgets."""

import json
from pathlib import Path

from speck.provenance.io import file_sha256


def bound_json(binding, root):
    path = (Path(root) / binding["path"]).resolve()
    if file_sha256(path) != binding["sha256"]:
        raise ValueError("code preparation decision input identity mismatch")
    return json.loads(path.read_text())


def load_decision(binding, root):
    decision = bound_json(binding, root)
    if (
        decision.get("format") != "speck_code_preparation_decision"
        or decision.get("format_version") != 1
        or decision.get("status") != "selected_for_preparation_not_launchable"
        or decision.get("training_authority") is not False
        or decision.get("headroom_percent") != 20
        or decision["invariants"].get("model_outputs_observed") is not False
    ):
        raise ValueError("unsupported pre-results code preparation decision")
    for evidence in decision["evidence"].values():
        bound_json(evidence, root)
    return decision


def validate_source_successor(proposal, previous, decision):
    if any(
        proposal.get(key) != value
        for key, value in previous.items()
        if key not in {"incumbent", "preparation_decision"}
    ) or set(proposal) != set(previous) | {"supersedes", "code_preparation_decision"}:
        raise ValueError("code successor changed another first-wave contract field")
    expected = {**previous["incumbent"], "code": decision["shared_code_background"]}
    if proposal["incumbent"] != expected:
        raise ValueError("code background differs from the recorded decision")
    if proposal["supersedes"] != decision["evidence"]["previous_first_wave"]:
        raise ValueError("code successor belongs to another predecessor")


def validate_language_successor(spec, previous, wave, decision):
    allowed = {"first_wave", "language_weights_percent", "decision_basis"}
    if any(spec.get(key) != value for key, value in previous.items() if key not in allowed):
        raise ValueError("language successor changed a preserved policy")
    if set(spec) != set(previous) | {"supersedes", "code_preparation_decision"}:
        raise ValueError("unexpected language successor fields")
    if (
        spec["language_weights_percent"] != decision["language_weights_percent"]
        or spec["supersedes"] != decision["evidence"]["previous_language_plan"]
        or wave["recipes"]["incumbent"]["code"] != decision["shared_code_background"]
        or len(wave["logical_slots"]) != decision["invariants"]["logical_runs"]
        or wave["logical_training_tokens"] != decision["invariants"]["logical_training_tokens"]
        or wave["source_capacity_total_tokens"]
        != decision["invariants"]["source_capacity_total_tokens"]
        or sum(wave["budget_gpu_hours"].values()) != decision["invariants"]["data_gpu_hour_ceiling"]
        or wave["category_weights_percent"] != decision["invariants"]["category_weights_percent"]
    ):
        raise ValueError("language/source allocation or scientific envelope differs from decision")
