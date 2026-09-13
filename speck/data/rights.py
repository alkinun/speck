"""Validate, but never make, human source-rights decisions."""

import json
import re
from pathlib import Path

from speck.provenance.io import durable_json as _write_json
from speck.provenance.io import file_sha256 as _sha256

TEMPLATE_FORMAT = "speck_source_rights_acceptance_template"
ACCEPTANCE_FORMAT = "speck_human_source_rights_acceptance"
FORMAT_VERSION = 1
CATEGORIES = ("web", "code", "math", "synthetic", "science", "reference")
SCOPE_FIELDS = (
    "research_training",
    "paper_publication",
    "public_model_weight_release",
    "commercial_use",
    "source_data_redistribution",
    "derived_packed_shard_redistribution",
)


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def _resolve(value, config_dir, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def validate_rights_template(template, *, config_dir=None, verify_evidence=False):
    """Validate complete source coverage while permitting only explicit pending decisions."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        template,
        {
            "format",
            "format_version",
            "status",
            "source_registry",
            "source_registry_sha256",
            "evidence_packets",
            "intended_scope",
            "authority",
            "source_decisions",
            "signed_at",
        },
        "rights template",
    )
    if (
        template["format"] != TEMPLATE_FORMAT
        or template["format_version"] != FORMAT_VERSION
        or template["status"] != "awaiting_human_decision"
    ):
        raise ValueError("unsupported or non-pending rights template")
    registry_path = _resolve(template["source_registry"], config_dir, "source registry")
    if not isinstance(template["source_registry_sha256"], str):
        raise ValueError("source registry hash must be a string")
    if verify_evidence and (
        not registry_path.is_file() or _sha256(registry_path) != template["source_registry_sha256"]
    ):
        raise ValueError("source registry identity mismatch")
    registry = json.loads(registry_path.read_text())
    allocations = registry.get("tokenizer_sample_allocations")
    sources = {source["id"]: source for source in registry.get("sources", [])}
    if not isinstance(allocations, list) or not allocations:
        raise ValueError("source registry has no tokenizer allocations")
    required_ids = [allocation["source_id"] for allocation in allocations]
    if len(required_ids) != len(set(required_ids)) or any(
        value not in sources for value in required_ids
    ):
        raise ValueError("source registry tokenizer allocations are invalid")
    packets = template["evidence_packets"]
    if not isinstance(packets, list) or [packet.get("category") for packet in packets] != list(
        CATEGORIES
    ):
        raise ValueError("evidence packets must cover categories in frozen order")
    normalized_packets = []
    for packet in packets:
        _exact_keys(packet, {"category", "path", "sha256", "linear_issue"}, "evidence packet")
        path = _resolve(packet["path"], config_dir, f"{packet['category']} evidence packet")
        if verify_evidence and (not path.is_file() or _sha256(path) != packet["sha256"]):
            raise ValueError(f"{packet['category']} evidence packet identity mismatch")
        if not isinstance(packet["linear_issue"], str) or not packet["linear_issue"]:
            raise ValueError("evidence packet must name its human review issue")
        normalized_packets.append({**packet, "path": str(path)})
    scope = template["intended_scope"]
    _exact_keys(scope, set(SCOPE_FIELDS), "intended_scope")
    if any(value is not None and not isinstance(value, bool) for value in scope.values()):
        raise ValueError("scope fields must be null or boolean")
    authority = template["authority"]
    _exact_keys(authority, {"authority_type", "name", "role", "organization"}, "authority")
    if any(value is not None and not isinstance(value, str) for value in authority.values()):
        raise ValueError("authority fields must be null or strings")
    decisions = template["source_decisions"]
    if (
        not isinstance(decisions, list)
        or [decision.get("source_id") for decision in decisions] != required_ids
    ):
        raise ValueError("source decisions must exactly follow tokenizer allocation order")
    normalized_decisions = []
    for decision in decisions:
        _exact_keys(
            decision,
            {
                "source_id",
                "category",
                "decision",
                "evidence_category",
                "attribution_plan",
                "redistribution_policy",
                "removal_policy",
                "conditions",
                "rationale",
            },
            "source decision",
        )
        source_id = decision["source_id"]
        category = sources[source_id]["category"]
        if decision["category"] != category or decision["evidence_category"] != category:
            raise ValueError(f"source decision category mismatch: {source_id}")
        if decision["decision"] not in {"pending", "approve", "reject"}:
            raise ValueError(f"invalid source decision: {source_id}")
        for key in ("attribution_plan", "redistribution_policy", "removal_policy", "rationale"):
            if decision[key] is not None and (
                not isinstance(decision[key], str) or not decision[key]
            ):
                raise ValueError(f"source {source_id} {key} must be null or non-empty")
        if not isinstance(decision["conditions"], list) or any(
            not isinstance(value, str) or not value for value in decision["conditions"]
        ):
            raise ValueError(f"source {source_id} conditions must be strings")
        normalized_decisions.append(dict(decision))
    if template["signed_at"] is not None and not isinstance(template["signed_at"], str):
        raise ValueError("signed_at must be null or a string")
    return {
        **template,
        "source_registry": str(registry_path),
        "evidence_packets": normalized_packets,
        "source_decisions": normalized_decisions,
        "required_source_ids": required_ids,
    }


def assess_rights_template(template, *, config_dir=None, verify_evidence=False):
    """Report human work remaining without converting any decision."""

    value = validate_rights_template(
        template, config_dir=config_dir, verify_evidence=verify_evidence
    )
    counts = {decision: 0 for decision in ("pending", "approve", "reject")}
    incomplete = []
    for decision in value["source_decisions"]:
        counts[decision["decision"]] += 1
        if decision["decision"] == "approve" and any(
            not decision[key]
            for key in ("attribution_plan", "redistribution_policy", "removal_policy", "rationale")
        ):
            incomplete.append(decision["source_id"])
    return {
        "status": "human_decision_required",
        "counts": counts,
        "incomplete_approved_sources": incomplete,
        "scope_fields_pending": [
            key for key, value in value["intended_scope"].items() if value is None
        ],
        "authority_fields_pending": [key for key, value in value["authority"].items() if not value],
        "signed_at_pending": not bool(value["signed_at"]),
        "automated_approval_made": False,
    }


def finalize_human_acceptance(template, output_path, *, config_dir=None):
    """Canonicalize only a fully completed, all-approved human record."""

    value = validate_rights_template(template, config_dir=config_dir, verify_evidence=True)
    counts = {decision: 0 for decision in ("pending", "approve", "reject")}
    incomplete = []
    for decision in value["source_decisions"]:
        counts[decision["decision"]] += 1
        if decision["decision"] == "approve" and any(
            not decision[key]
            for key in (
                "attribution_plan",
                "redistribution_policy",
                "removal_policy",
                "rationale",
            )
        ):
            incomplete.append(decision["source_id"])
    if any(value is None for value in value["intended_scope"].values()):
        raise ValueError("human intended scope is incomplete")
    authority = value["authority"]
    if authority["authority_type"] != "human" or any(
        not authority[key] for key in ("name", "role", "organization")
    ):
        raise ValueError("a named human authority is required")
    if not value["signed_at"]:
        raise ValueError("human acceptance requires signed_at")
    if counts["pending"] or counts["reject"]:
        raise ValueError("all selected sources must be explicitly approved")
    if incomplete:
        raise ValueError("approved source decisions require complete operational policies")
    scope = value["intended_scope"]
    acceptance = {
        "format": ACCEPTANCE_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "all_sources_human_approved",
        "authority": {
            "name": authority["name"],
            "role": authority["role"],
            "organization": authority["organization"],
            "authority_type": "human",
        },
        "signed_at": value["signed_at"],
        "scope": ", ".join(f"{key}={str(scope[key]).lower()}" for key in SCOPE_FIELDS),
        "scope_details": scope,
        "source_registry": value["source_registry"],
        "source_registry_sha256": value["source_registry_sha256"],
        "evidence_packets": value["evidence_packets"],
        "approved_source_ids": value["required_source_ids"],
        "source_decisions": value["source_decisions"],
        "automated_approval_made": False,
    }
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"human acceptance record already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, acceptance)
    return acceptance


def load_source_use_extension(path):
    """Validate an explicitly approved additive source decision, without changing D5 inputs."""

    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        value.get("format") != "speck_human_source_use_extension"
        or value.get("format_version") != 1
        or value.get("status") != "human_approved_source_extension"
        or value.get("decision") != "approve"
        or value.get("automated_approval_made") is not False
        or value.get("training_authority") is not False
    ):
        raise ValueError("source extension requires an explicit human source-use decision")
    parent_identity = value["parent_acceptance"]
    parent_path = _resolve(parent_identity["path"], path.parent, "parent acceptance")
    if _sha256(parent_path) != parent_identity["sha256"]:
        raise ValueError("source extension parent identity mismatch")
    parent = json.loads(parent_path.read_text())
    if (
        parent.get("status") != "all_sources_human_approved"
        or parent.get("automated_approval_made") is not False
    ):
        raise ValueError("source extension parent is not a human acceptance")
    if value.get("scope_details") != parent.get("scope_details"):
        raise ValueError("source extension cannot silently expand the parent scope")
    authority = value.get("authority", {})
    if authority.get("authority_type") != "human" or any(
        not isinstance(authority.get(key), str) or not authority[key]
        for key in ("name", "role", "organization")
    ):
        raise ValueError("source extension needs a named human authority")
    if not value.get("signed_at") or not value.get("approval_basis"):
        raise ValueError("source extension needs its human approval basis and date")
    source = value.get("source", {})
    if (
        not source.get("id")
        or source["id"] in parent["approved_source_ids"]
        or source.get("category") not in CATEGORIES
        or any(
            not isinstance(source.get(key), str) or not source[key]
            for key in ("id", "repo", "revision", "config", "content_field")
        )
        or not re.fullmatch(r"[0-9a-f]{40}", source["revision"])
    ):
        raise ValueError("source extension must identify exactly one new source")
    for key in (
        "attribution_plan",
        "redistribution_policy",
        "removal_policy",
        "conditions",
        "rationale",
    ):
        if not value.get(key):
            raise ValueError(f"source extension is missing {key}")
    for identity in value.get("evidence", []):
        evidence = _resolve(identity["path"], path.parent, "source evidence")
        if _sha256(evidence) != identity["sha256"]:
            raise ValueError("source extension evidence identity mismatch")
    if not value.get("evidence"):
        raise ValueError("source extension needs checked evidence")
    return {**value, "identity": {"path": str(path), "sha256": _sha256(path)}}
