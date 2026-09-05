"""Validate versioned architecture-research decision contracts."""

import json
import math
import re
from pathlib import Path

CONTRACT_FILES = (
    "policy.json",
    "cost_envelopes.json",
    "evaluation_manifest.json",
    "evidence_matrix.json",
)
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _load_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load research contract file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"research contract file must contain an object: {path}")
    return value


def _require_keys(value, keys, context):
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")


def _probability(value, context, *, allow_one=False):
    maximum = 1 if allow_one else 1.0
    valid_upper = value <= maximum if allow_one else value < maximum
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or not valid_upper
    ):
        boundary = "(0, 1]" if allow_one else "(0, 1)"
        raise ValueError(f"{context} must be in {boundary}")


def _positive(value, context, *, allow_zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{context} must be a finite {qualifier} number")


def _unique_ids(values, context):
    identifiers = [value.get("id") for value in values]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError(f"every {context} requires a non-empty id")
    duplicates = sorted(
        {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
    )
    if duplicates:
        raise ValueError(f"duplicate {context} ids: {', '.join(duplicates)}")
    return set(identifiers)


def _validate_policy(policy):
    _require_keys(
        policy,
        {
            "format",
            "format_version",
            "policy_id",
            "status",
            "comparison_contract",
            "statistical_contract",
            "replication_stages",
            "promotion_gate",
            "change_control",
        },
        "promotion policy",
    )
    if policy["format"] != "speck_architecture_promotion_policy":
        raise ValueError("promotion policy has the wrong format")
    if policy["format_version"] != 1 or policy["status"] != "active":
        raise ValueError("promotion policy must use active format version 1")

    statistics = policy["statistical_contract"]
    _probability(statistics["alpha"], "statistical alpha")
    _probability(statistics["confidence_level"], "confidence level")
    if not math.isclose(statistics["confidence_level"], 1 - statistics["alpha"]):
        raise ValueError("confidence level must equal one minus alpha")

    loss = statistics["language_loss"]
    margin = loss["default_non_inferiority_margin_nats"]
    _positive(margin, "language-loss non-inferiority margin")
    if not math.isclose(loss["perplexity_ratio_at_margin"], math.exp(margin), rel_tol=1e-12):
        raise ValueError("perplexity ratio must equal exp(language-loss margin)")
    _positive(loss["source_guardrail_nats"], "language-loss source guardrail")

    tasks = statistics["bounded_task_scores"]
    _probability(tasks["default_non_inferiority_margin_absolute"], "task score margin")
    _probability(tasks["critical_task_guardrail_absolute"], "critical-task guardrail")

    systems = statistics["systems"]
    for key in (
        "simple_component_minimum_primary_improvement",
        "custom_runtime_component_minimum_primary_improvement",
        "minimum_state_reduction_for_memory_claim",
    ):
        _probability(systems[key], f"systems {key}")
    if (
        systems["custom_runtime_component_minimum_primary_improvement"]
        <= systems["simple_component_minimum_primary_improvement"]
    ):
        raise ValueError("custom-runtime components must clear a larger systems threshold")
    _positive(systems["repeats"], "systems repeats")
    if not isinstance(systems["repeats"], int) or systems["repeats"] < 5:
        raise ValueError("systems comparisons require at least five repeats")

    stages = policy["replication_stages"]
    stage_ids = _unique_ids(stages, "replication stage")
    required_stages = {
        "correctness",
        "discovery",
        "proxy_confirmation",
        "finalist_replication",
        "medium_scale_transfer",
        "target_scale_sentinel",
    }
    if stage_ids != required_stages:
        raise ValueError("promotion policy replication stages are incomplete")
    for stage in stages:
        runs = stage.get("minimum_paired_runs")
        _positive(runs, f"{stage['id']} minimum paired runs", allow_zero=True)
        if not isinstance(runs, int):
            raise ValueError(f"{stage['id']} minimum paired runs must be an integer")
        if not isinstance(stage.get("promotion_authority"), bool):
            raise ValueError(f"{stage['id']} promotion authority must be boolean")
        if not stage.get("requirements"):
            raise ValueError(f"{stage['id']} requires explicit requirements")
    if next(stage for stage in stages if stage["id"] == "discovery")["promotion_authority"]:
        raise ValueError("discovery cannot have promotion authority")


def _validate_envelopes(envelopes, policy_id):
    _require_keys(
        envelopes,
        {
            "format",
            "format_version",
            "policy_id",
            "hardware",
            "measurement_controls",
            "training_profiles",
            "serving_profiles",
            "cost_accounting",
            "architecture_freeze_blocker",
        },
        "cost envelopes",
    )
    if envelopes["format"] != "speck_cost_envelopes" or envelopes["format_version"] != 1:
        raise ValueError("cost envelopes must use format version 1")
    if envelopes["policy_id"] != policy_id:
        raise ValueError("cost envelopes reference a different promotion policy")
    hardware = envelopes["hardware"]
    if not hardware:
        raise ValueError("cost envelopes require at least one named hardware profile")
    for identifier, profile in hardware.items():
        _require_keys(profile, {"gpu", "gpu_memory_gib", "ownership"}, f"hardware {identifier}")
        _positive(profile["gpu_memory_gib"], f"hardware {identifier} memory")

    training_ids = _unique_ids(envelopes["training_profiles"], "training profile")
    serving_ids = _unique_ids(envelopes["serving_profiles"], "serving profile")
    if training_ids & serving_ids:
        raise ValueError("training and serving profile ids must be distinct")
    for kind, profiles in (
        ("training", envelopes["training_profiles"]),
        ("serving", envelopes["serving_profiles"]),
    ):
        for profile in profiles:
            reference = profile.get("hardware")
            if reference not in hardware and reference != "to_be_named_before_launch":
                raise ValueError(f"{kind} profile {profile['id']} references unknown hardware")
            if (
                reference == "to_be_named_before_launch"
                and profile.get("hard_envelope") is not None
            ):
                raise ValueError(f"unbound {kind} profile {profile['id']} cannot set an envelope")
            envelope = profile.get("hard_envelope")
            if envelope is not None:
                if not envelope:
                    raise ValueError(f"{kind} profile {profile['id']} has an empty envelope")
                for key, value in envelope.items():
                    _positive(
                        value,
                        f"{kind} profile {profile['id']} envelope {key}",
                        allow_zero=key == "non_finite_steps",
                    )
            if not profile.get("primary_cost_metric"):
                raise ValueError(f"{kind} profile {profile['id']} needs a primary cost metric")
    accounting = envelopes["cost_accounting"]
    _require_keys(
        accounting,
        {
            "training_usd",
            "serving_usd_per_million_output_tokens",
            "required_physical_metrics",
            "relative_decision_rule",
        },
        "cost accounting",
    )


def _validate_evaluations(manifest, policy_id, repository_root):
    _require_keys(
        manifest,
        {
            "format",
            "format_version",
            "policy_id",
            "manifest_id",
            "status",
            "repository_baseline_revision",
            "rules",
            "internal_suites",
            "external_suites",
            "release_gate",
        },
        "evaluation manifest",
    )
    if (
        manifest["format"] != "speck_architecture_evaluation_manifest"
        or manifest["format_version"] != 1
    ):
        raise ValueError("evaluation manifest must use format version 1")
    if manifest["policy_id"] != policy_id:
        raise ValueError("evaluation manifest references a different promotion policy")
    if not COMMIT_PATTERN.fullmatch(manifest["repository_baseline_revision"]):
        raise ValueError("evaluation manifest repository revision must be a full commit")

    internal_ids = _unique_ids(manifest["internal_suites"], "internal evaluation suite")
    external_ids = _unique_ids(manifest["external_suites"], "external evaluation suite")
    if internal_ids & external_ids:
        raise ValueError("internal and external evaluation ids must be distinct")
    for suite in manifest["internal_suites"]:
        source = suite.get("source")
        if source is not None and not (repository_root / source).is_file():
            raise ValueError(f"evaluation suite {suite['id']} source does not exist: {source}")
        revision = suite.get("generator_revision")
        if revision is not None and not COMMIT_PATTERN.fullmatch(revision):
            raise ValueError(f"evaluation suite {suite['id']} generator revision is invalid")
        if suite["id"] in {"structured_retrieval_v2", "symbolic_composition_v2"}:
            samples = suite.get("samples_per_length_load_cell", suite.get("samples_per_view", 0))
            if not isinstance(samples, int) or samples < 200:
                raise ValueError(f"evaluation suite {suite['id']} requires at least 200 samples")
    for suite in manifest["external_suites"]:
        if not COMMIT_PATTERN.fullmatch(suite.get("revision", "")):
            raise ValueError(f"external suite {suite['id']} must pin a full commit")
        if not suite.get("repository", "").startswith("https://github.com/"):
            raise ValueError(f"external suite {suite['id']} must name its upstream repository")
        if not suite.get("release_required"):
            raise ValueError(f"external suite {suite['id']} must remain release-required")

    gate = manifest["release_gate"]
    if set(gate["required_internal"]) != internal_ids:
        raise ValueError("release gate must include every internal suite exactly once")
    if set(gate["required_external"]) != external_ids:
        raise ValueError("release gate must include every external suite exactly once")


def _validate_evidence(matrix, policy_id, repository_root):
    _require_keys(
        matrix,
        {
            "format",
            "format_version",
            "policy_id",
            "status_values",
            "components",
        },
        "evidence matrix",
    )
    if matrix["format"] != "speck_architecture_evidence_matrix" or matrix["format_version"] != 1:
        raise ValueError("evidence matrix must use format version 1")
    if matrix["policy_id"] != policy_id:
        raise ValueError("evidence matrix references a different promotion policy")
    status_values = set(matrix["status_values"])
    if len(status_values) != len(matrix["status_values"]):
        raise ValueError("evidence matrix status values must be unique")
    _unique_ids(matrix["components"], "evidence component")
    required = {
        "id",
        "claim",
        "status",
        "alternatives",
        "evidence",
        "quality",
        "systems",
        "scale_transfer",
        "runtime_support",
        "unresolved_risks",
        "next_gate",
    }
    for component in matrix["components"]:
        _require_keys(component, required, f"evidence component {component.get('id')}")
        if component["status"] not in status_values:
            raise ValueError(f"evidence component {component['id']} has an unknown status")
        for reference in component["evidence"]:
            if not (repository_root / reference).is_file():
                raise ValueError(
                    f"evidence component {component['id']} reference does not exist: {reference}"
                )


def validate_research_contract(directory):
    """Validate one complete contract directory and return a compact inventory."""

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"research contract directory does not exist: {directory}")
    missing = [name for name in CONTRACT_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"research contract is missing files: {', '.join(missing)}")
    values = {name: _load_json(directory / name) for name in CONTRACT_FILES}
    policy = values["policy.json"]
    _validate_policy(policy)
    policy_id = policy["policy_id"]
    repository_root = directory.parents[1]
    _validate_envelopes(values["cost_envelopes.json"], policy_id)
    _validate_evaluations(values["evaluation_manifest.json"], policy_id, repository_root)
    _validate_evidence(values["evidence_matrix.json"], policy_id, repository_root)
    evaluations = values["evaluation_manifest.json"]
    return {
        "policy_id": policy_id,
        "contract_files": list(CONTRACT_FILES),
        "replication_stages": len(policy["replication_stages"]),
        "training_profiles": len(values["cost_envelopes.json"]["training_profiles"]),
        "serving_profiles": len(values["cost_envelopes.json"]["serving_profiles"]),
        "internal_evaluation_suites": len(evaluations["internal_suites"]),
        "external_evaluation_suites": len(evaluations["external_suites"]),
        "evidence_components": len(values["evidence_matrix.json"]["components"]),
        "status": "valid",
    }
