import json
from pathlib import Path

import pytest

from speck.source_rights import (
    assess_rights_template,
    finalize_human_acceptance,
    validate_rights_template,
)

ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "research/flagship/source_rights_acceptance_template.json"


def _template():
    return json.loads(TEMPLATE.read_text())


def test_checked_template_covers_every_selected_source_and_approves_nothing():
    value = validate_rights_template(_template(), config_dir=TEMPLATE.parent, verify_evidence=True)
    assessment = assess_rights_template(
        _template(), config_dir=TEMPLATE.parent, verify_evidence=True
    )

    assert len(value["required_source_ids"]) == 30
    assert assessment["counts"] == {"pending": 30, "approve": 0, "reject": 0}
    assert assessment["automated_approval_made"] is False
    assert assessment["signed_at_pending"] is True


def test_finalization_requires_named_human_complete_scope_and_all_approvals(tmp_path):
    pending = _template()
    with pytest.raises(ValueError, match="intended scope is incomplete"):
        finalize_human_acceptance(pending, tmp_path / "acceptance.json", config_dir=TEMPLATE.parent)

    complete = _template()
    complete["intended_scope"] = {key: False for key in complete["intended_scope"]}
    complete["intended_scope"].update(
        {"research_training": True, "paper_publication": True, "public_model_weight_release": True}
    )
    complete["authority"] = {
        "authority_type": "human",
        "name": "Fixture Human",
        "role": "Fixture Approver",
        "organization": "Fixture Organization",
    }
    complete["signed_at"] = "2026-09-07T00:00:00Z"
    for decision in complete["source_decisions"]:
        decision.update(
            {
                "decision": "approve",
                "attribution_plan": "fixture attribution",
                "redistribution_policy": "no source data redistribution",
                "removal_policy": "fixture deny ledger and rebuild",
                "rationale": "fixture-only test decision",
            }
        )
    acceptance = finalize_human_acceptance(
        complete, tmp_path / "acceptance.json", config_dir=TEMPLATE.parent
    )

    assert acceptance["status"] == "all_sources_human_approved"
    assert acceptance["authority"]["authority_type"] == "human"
    assert len(acceptance["approved_source_ids"]) == 30
    assert acceptance["automated_approval_made"] is False


def test_rejection_blocks_acceptance_and_requires_source_replacement(tmp_path):
    value = _template()
    value["intended_scope"] = {key: False for key in value["intended_scope"]}
    value["authority"] = {
        "authority_type": "human",
        "name": "Fixture Human",
        "role": "Fixture Approver",
        "organization": "Fixture Organization",
    }
    value["signed_at"] = "2026-09-07T00:00:00Z"
    for decision in value["source_decisions"]:
        decision.update(
            {
                "decision": "approve",
                "attribution_plan": "fixture attribution",
                "redistribution_policy": "none",
                "removal_policy": "fixture removal",
                "rationale": "fixture rationale",
            }
        )
    value["source_decisions"][0]["decision"] = "reject"

    with pytest.raises(ValueError, match="all selected sources"):
        finalize_human_acceptance(value, tmp_path / "acceptance.json", config_dir=TEMPLATE.parent)
