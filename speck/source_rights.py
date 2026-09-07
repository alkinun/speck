"""Validate, but never make, human source-rights decisions."""

import hashlib
import json
import os
from pathlib import Path

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


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def _resolve(value, config_dir, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
