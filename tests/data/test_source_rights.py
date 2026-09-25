import json

import pytest

from speck.data.rights import (
    CATEGORIES,
    SCOPE_FIELDS,
    assess_rights_template,
    finalize_human_acceptance,
    validate_rights_template,
)
from speck.provenance.io import file_sha256

SOURCES = {"web_source": "web", "code_source": "code", "math_source": "math"}


def _template(root):
    """A pending template over a small registry, with every evidence packet on disk."""
    registry = root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "sources": [{"id": key, "category": value} for key, value in SOURCES.items()],
                "tokenizer_sample_allocations": [{"source_id": key} for key in SOURCES],
            }
        )
    )
    packets = []
    for category in CATEGORIES:
        path = root / f"{category}-review.json"
        path.write_text(json.dumps({"category": category}))
        packets.append(
            {
                "category": category,
                "path": path.name,
                "sha256": file_sha256(path),
                "linear_issue": f"FIXTURE#{category}",
            }
        )
    return {
        "format": "speck_source_rights_acceptance_template",
        "format_version": 1,
        "status": "awaiting_human_decision",
        "source_registry": registry.name,
        "source_registry_sha256": file_sha256(registry),
        "evidence_packets": packets,
        "intended_scope": {key: None for key in SCOPE_FIELDS},
        "authority": {"authority_type": None, "name": None, "role": None, "organization": None},
        "source_decisions": [
            {
                "source_id": key,
                "category": value,
                "decision": "pending",
                "evidence_category": value,
                "attribution_plan": None,
                "redistribution_policy": None,
                "removal_policy": None,
                "conditions": [],
                "rationale": None,
            }
            for key, value in SOURCES.items()
        ],
        "signed_at": None,
    }


def _signed(value, decision="approve"):
    value["intended_scope"] = {key: False for key in value["intended_scope"]}
    value["intended_scope"].update(
        {"research_training": True, "paper_publication": True, "public_model_weight_release": True}
    )
    value["authority"] = {
        "authority_type": "human",
        "name": "Fixture Human",
        "role": "Fixture Approver",
        "organization": "Fixture Organization",
    }
    value["signed_at"] = "2026-09-07T00:00:00Z"
    for item in value["source_decisions"]:
        item.update(
            {
                "decision": decision,
                "attribution_plan": "fixture attribution",
                "redistribution_policy": "no source data redistribution",
                "removal_policy": "fixture deny ledger and rebuild",
                "rationale": "fixture-only test decision",
            }
        )
    return value


def test_pending_template_covers_every_selected_source_and_approves_nothing(tmp_path):
    value = validate_rights_template(_template(tmp_path), config_dir=tmp_path, verify_evidence=True)
    assessment = assess_rights_template(
        _template(tmp_path), config_dir=tmp_path, verify_evidence=True
    )

    assert value["required_source_ids"] == list(SOURCES)
    assert assessment["counts"] == {"pending": 3, "approve": 0, "reject": 0}
    assert assessment["scope_fields_pending"] == list(SCOPE_FIELDS)
    assert assessment["automated_approval_made"] is False
    assert assessment["signed_at_pending"] is True


def test_evidence_identity_and_decision_order_are_enforced(tmp_path):
    value = _template(tmp_path)
    (tmp_path / "code-review.json").write_text("changed")
    validate_rights_template(value, config_dir=tmp_path)
    with pytest.raises(ValueError, match="code evidence packet identity mismatch"):
        validate_rights_template(value, config_dir=tmp_path, verify_evidence=True)

    value = _template(tmp_path)
    value["source_decisions"].reverse()
    with pytest.raises(ValueError, match="tokenizer allocation order"):
        validate_rights_template(value, config_dir=tmp_path)

    value = _template(tmp_path)
    value["source_decisions"][0]["category"] = "code"
    with pytest.raises(ValueError, match="category mismatch: web_source"):
        validate_rights_template(value, config_dir=tmp_path)


def test_finalization_requires_named_human_complete_scope_and_all_approvals(tmp_path):
    with pytest.raises(ValueError, match="intended scope is incomplete"):
        finalize_human_acceptance(
            _template(tmp_path), tmp_path / "acceptance.json", config_dir=tmp_path
        )

    acceptance = finalize_human_acceptance(
        _signed(_template(tmp_path)), tmp_path / "acceptance.json", config_dir=tmp_path
    )

    assert acceptance["status"] == "all_sources_human_approved"
    assert acceptance["authority"]["authority_type"] == "human"
    assert acceptance["approved_source_ids"] == list(SOURCES)
    assert acceptance["automated_approval_made"] is False
    assert json.loads((tmp_path / "acceptance.json").read_text()) == acceptance
    with pytest.raises(FileExistsError):
        finalize_human_acceptance(
            _signed(_template(tmp_path)), tmp_path / "acceptance.json", config_dir=tmp_path
        )


def test_rejection_blocks_acceptance_and_requires_source_replacement(tmp_path):
    value = _signed(_template(tmp_path))
    value["source_decisions"][0]["decision"] = "reject"

    with pytest.raises(ValueError, match="all selected sources"):
        finalize_human_acceptance(value, tmp_path / "acceptance.json", config_dir=tmp_path)
