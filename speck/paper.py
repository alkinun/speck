"""Validate the preregistered Speck Paper 1 research program."""

import hashlib
import json
from pathlib import Path

PROGRAM_FILES = (
    "README.md",
    "claims.json",
    "baseline_matrix.json",
    "experiment_program.json",
    "paper_outline.md",
    "reference_audit.md",
    "reporting_checklist.md",
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load paper program file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"paper program file must contain an object: {path}")
    return value


def _require(value, keys, context):
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")


def _unique_ids(values, context):
    identifiers = [value.get("id") for value in values]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError(f"every {context} requires an id")
    duplicates = sorted(
        {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
    )
    if duplicates:
        raise ValueError(f"duplicate {context} ids: {', '.join(duplicates)}")
    return set(identifiers)


def _validate_claims(claims):
    _require(
        claims,
        {
            "format",
            "format_version",
            "paper_id",
            "status",
            "central_claim_id",
            "claims",
            "non_claims",
        },
        "paper claims",
    )
    if claims["format"] != "speck_paper_claims" or claims["format_version"] != 1:
        raise ValueError("paper claims must use format version 1")
    if claims["status"] != "hypotheses_only":
        raise ValueError("Paper 1 claims must remain hypotheses before confirmatory experiments")
    claim_ids = _unique_ids(claims["claims"], "paper claim")
    if claims["central_claim_id"] not in claim_ids:
        raise ValueError("central paper claim does not exist")
    required = {
        "id",
        "type",
        "statement",
        "status",
        "required_evidence",
        "falsifiers",
        "paper_sections",
    }
    for claim in claims["claims"]:
        _require(claim, required, f"paper claim {claim.get('id')}")
        if claim["status"] != "hypothesis":
            raise ValueError(f"unconfirmed paper claim {claim['id']} cannot be promoted")
        if not claim["required_evidence"] or not claim["falsifiers"] or not claim["paper_sections"]:
            raise ValueError(f"paper claim {claim['id']} lacks evidence or falsification criteria")
    central = next(claim for claim in claims["claims"] if claim["id"] == claims["central_claim_id"])
    if central["type"] != "central":
        raise ValueError("central paper claim must have type central")
    if len(claims["non_claims"]) < 5:
        raise ValueError("paper program requires explicit non-claims")
    return claim_ids


def _validate_program(program, paper_id, claim_ids, repository_root):
    _require(
        program,
        {
            "format",
            "format_version",
            "paper_id",
            "policy_id",
            "status",
            "baseline_evidence",
            "controls",
            "matching_views",
            "scales",
            "axes",
            "interaction_program",
            "scaling_program",
            "evaluation_program",
            "analysis_program",
            "large_pretraining_gate",
        },
        "paper experiment program",
    )
    if program["format"] != "speck_paper_experiment_program" or program["format_version"] != 1:
        raise ValueError("paper experiment program must use format version 1")
    if program["paper_id"] != paper_id:
        raise ValueError("claims and experiment program use different paper ids")
    evidence = program["baseline_evidence"]
    _require(
        evidence,
        {
            "status",
            "matrix",
            "matrix_sha256",
            "materialization",
            "materialization_sha256",
            "audit",
            "audit_sha256",
            "runner_revision",
        },
        "paper baseline evidence",
    )
    if evidence["status"] != "historical_evidence_qualified_proxy_launch_blocked":
        raise ValueError("paper baseline evidence cannot claim an unexecuted launch")
    for path_key, hash_key in (
        ("matrix", "matrix_sha256"),
        ("materialization", "materialization_sha256"),
        ("audit", "audit_sha256"),
    ):
        path = repository_root / evidence[path_key]
        if not path.is_file() or _file_sha256(path) != evidence[hash_key]:
            raise ValueError(f"paper baseline {path_key} does not match its pin")
    audit = _load_json(repository_root / evidence["audit"])
    if (
        audit.get("format") != "speck_paper_baseline_audit"
        or audit.get("status") != evidence["status"]
        or audit.get("paper_id") != paper_id
        or audit.get("runner_revision") != evidence["runner_revision"]
        or audit.get("contract_sha256") != evidence["matrix_sha256"]
    ):
        raise ValueError("paper baseline audit is invalid")
    policy = repository_root / "research" / program["policy_id"] / "policy.json"
    if not policy.is_file():
        raise ValueError("paper experiment program references a missing promotion policy")
    controls = program["controls"]
    _require(
        controls, {"primary_conventional", "primary_hybrid", "mechanism_controls"}, "paper controls"
    )
    if len(program["matching_views"]) < 5:
        raise ValueError("paper program does not preserve enough matching views")

    scale_ids = _unique_ids(program["scales"], "paper scale")
    required_scales = {
        "mechanism",
        "proxy_discovery",
        "proxy_finalist",
        "medium_transfer",
        "target_sentinel",
        "paper_scale",
    }
    if scale_ids != required_scales:
        raise ValueError("paper program scale ladder is incomplete")
    paper_scale = next(scale for scale in program["scales"] if scale["id"] == "paper_scale")
    if (
        paper_scale.get("active_parameters") is not None
        or paper_scale.get("training_tokens") is not None
    ):
        raise ValueError("paper-scale geometry cannot be selected before the launch gate")

    axes = program["axes"]
    axis_ids = _unique_ids(axes, "paper axis")
    if axis_ids != {"sequence", "depth", "width"}:
        raise ValueError("paper program must isolate sequence, depth, and width")
    referenced_claims = set()
    for axis in axes:
        axis_claims = set(axis.get("claim_ids", ()))
        if not axis_claims or not axis_claims <= claim_ids:
            raise ValueError(f"paper axis {axis['id']} references invalid claims")
        referenced_claims |= axis_claims
        families = axis.get("ordered_families", ())
        _unique_ids(families, f"{axis['id']} experiment family")
        if len(families) < 3 or len(axis.get("required_diagnostics", ())) < 5:
            raise ValueError(f"paper axis {axis['id']} lacks experiments or diagnostics")
        for family in families:
            _require(
                family, {"id", "fixed", "arms", "decision"}, f"experiment family {family.get('id')}"
            )
            if len(family["arms"]) < 2:
                raise ValueError(f"experiment family {family['id']} needs at least two arms")
    if not {"C1", "C2", "C3", "C4", "C6"} <= referenced_claims:
        raise ValueError("paper axes do not cover all component and systems claims")

    interaction = program["interaction_program"]
    if len(interaction.get("pairwise_cells", ())) != 3 or "2^3" not in interaction.get(
        "full_factorial_requirement", ""
    ):
        raise ValueError("paper program lacks the complete tri-axis interaction design")
    scaling = program["scaling_program"]
    if scaling.get("minimum_points_per_architecture", 0) < 5 or "C5" not in scaling.get(
        "claim_ids", ()
    ):
        raise ValueError("paper scaling claim lacks five-point evidence")
    if len(program["analysis_program"]) < 4:
        raise ValueError("paper program lacks mechanistic analysis coverage")

    gate = program["large_pretraining_gate"]
    if gate.get("status") != "blocked" or len(gate.get("required", ())) < 10:
        raise ValueError("paper-scale pretraining must remain blocked on explicit prerequisites")
    return {"scales": len(scale_ids), "axes": len(axis_ids)}


def _validate_baseline_matrix(matrix, program, paper_id, repository_root):
    _require(
        matrix,
        {
            "format",
            "format_version",
            "paper_id",
            "policy_id",
            "status",
            "historical_evidence",
            "planned_primary_baselines",
            "future_finalist_design",
            "storage_contract",
            "launch_gates",
        },
        "paper baseline matrix",
    )
    if matrix["format"] != "speck_paper_baseline_matrix" or matrix["format_version"] != 1:
        raise ValueError("paper baseline matrix must use format version 1")
    if matrix["paper_id"] != paper_id or matrix["policy_id"] != program["policy_id"]:
        raise ValueError("paper baseline matrix references a different paper or policy")
    if matrix["status"] != "historical_evidence_audited_new_launch_blocked":
        raise ValueError("paper baseline launch must remain blocked before paired preflight")

    historical = matrix["historical_evidence"]
    _require(
        historical,
        {"authority", "shared_expected", "source_artifacts", "arms", "known_limits"},
        "historical baseline evidence",
    )
    historical_ids = _unique_ids(historical["arms"], "historical baseline arm")
    if historical_ids != {
        "dense_global",
        "swa_2048",
        "gdn_global_silu_rope",
        "gdn_global_sigmoid_nope",
        "kda_global_sigmoid_nope",
    }:
        raise ValueError("historical baseline inventory is incomplete")
    required_arm = {
        "id",
        "role",
        "experiment",
        "checkpoint_run",
        "parameters",
        "flops_per_token_at_4096",
        "validation_loss",
        "mixer_counts",
    }
    for arm in historical["arms"]:
        _require(arm, required_arm, f"historical baseline {arm.get('id')}")
        experiment = repository_root / arm["experiment"]
        if not experiment.is_dir() or any(
            not (experiment / f"{name}.json").is_file()
            for name in ("data", "model", "tokenizer", "train")
        ):
            raise ValueError(f"historical baseline {arm['id']} experiment is missing")
        if arm["parameters"] < 130_000_000 or arm["parameters"] > 170_000_000:
            raise ValueError(f"historical baseline {arm['id']} is outside the proxy scale")
    for artifact in historical["source_artifacts"]:
        _require(artifact, {"path", "sha256"}, "historical baseline source artifact")
        path = repository_root / artifact["path"]
        if not path.is_file() or _file_sha256(path) != artifact["sha256"]:
            raise ValueError("historical baseline source artifact does not match its pin")

    planned = matrix["planned_primary_baselines"]
    _require(
        planned,
        {
            "family_id",
            "output_root",
            "scientific_scope",
            "arms",
            "parameter_matching",
            "shared_training",
            "proxy_confirmation_pairs",
            "matching_views",
            "decision_rule",
        },
        "planned primary baselines",
    )
    planned_ids = _unique_ids(planned["arms"], "planned baseline arm")
    if planned_ids != {"dense_global_param_match", "five_cache_kda_gqa"}:
        raise ValueError("planned primary baseline arms are incomplete")
    control_ids = {arm["control_id"] for arm in planned["arms"]}
    if control_ids != {
        program["controls"]["primary_conventional"]["id"],
        program["controls"]["primary_hybrid"]["id"],
    }:
        raise ValueError("planned baseline arms do not implement the paper controls")
    for arm in planned["arms"]:
        _require(
            arm,
            {
                "id",
                "control_id",
                "template_experiment",
                "transform",
                "parameters",
                "flops_per_token_at_4096",
                "device_batch_size",
            },
            f"planned baseline {arm.get('id')}",
        )
        if not (repository_root / arm["template_experiment"]).is_dir():
            raise ValueError(f"planned baseline {arm['id']} template is missing")
    parameters = [arm["parameters"] for arm in planned["arms"]]
    relative = (max(parameters) - min(parameters)) / min(parameters)
    matching = planned["parameter_matching"]
    if (
        relative > matching.get("maximum_relative_difference", 0)
        or abs(relative - matching.get("actual_relative_difference", -1)) > 1e-15
        or max(parameters) - min(parameters) != matching.get("actual_parameter_difference")
    ):
        raise ValueError("planned primary baseline parameter match is invalid")
    training = planned["shared_training"]
    batch_tokens = training.get("batch_tokens")
    training_tokens = training.get("training_tokens")
    if (
        batch_tokens != 65_536
        or training_tokens != 131_072_000
        or training_tokens % batch_tokens
        or training.get("sequence_length") != 4_096
    ):
        raise ValueError("planned baseline training geometry is invalid")
    pairs = planned["proxy_confirmation_pairs"]
    if [pair.get("seed") for pair in pairs] != [42, 43, 44]:
        raise ValueError("planned baseline confirmation requires seeds 42, 43, and 44")
    offsets = [pair.get("data_token_offset") for pair in pairs]
    if (
        len(set(offsets)) != 3
        or any(
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or offset % batch_tokens
            for offset in offsets
        )
        or any(left + training_tokens > right for left, right in zip(offsets, offsets[1:]))
    ):
        raise ValueError("planned baseline packed-data windows must be aligned and disjoint")
    compute = planned["matching_views"].get("compute_matched", {})
    if (
        compute.get("reference_tokens") != training_tokens
        or compute.get("dense_global_tokens", 0) % batch_tokens
        or not 0 < compute.get("dense_global_tokens", 0) < training_tokens
    ):
        raise ValueError("planned baseline compute-matched view is invalid")

    finalist = matrix["future_finalist_design"]
    if (
        finalist.get("status") != "not_materialized_until_proxy_confirmation"
        or finalist.get("seeds") != [42, 43, 44]
        or len(finalist.get("data_token_offsets", ())) != 2
        or finalist.get("paired_runs") != 6
        or finalist.get("total_model_runs") != 12
        or finalist.get("training_tokens_per_run", 0) < max(parameters) * 10
    ):
        raise ValueError("future finalist baseline design is invalid")
    storage = matrix["storage_contract"]
    if (
        storage.get("proxy_confirmation_model_runs") != 2 * len(pairs)
        or storage.get("future_finalist_model_runs") != finalist["total_model_runs"]
        or storage.get("minimum_free_bytes_before_proxy_launch", 0)
        < storage.get("estimated_proxy_checkpoint_bytes", 0)
        or storage.get("minimum_free_bytes_before_finalist_launch", 0)
        < storage.get("estimated_finalist_checkpoint_bytes", 0)
    ):
        raise ValueError("paper baseline storage contract is insufficient")
    if len(matrix["launch_gates"]) < 6:
        raise ValueError("paper baseline launch gates are incomplete")
    return {
        "historical_arms": len(historical_ids),
        "planned_arms": len(planned_ids),
        "proxy_pairs": len(pairs),
    }


def _validate_markdown(directory):
    required_headings = {
        "README.md": ("## Working thesis", "## Novelty gate", "## Current state"),
        "paper_outline.md": (
            "## 3. Speck architecture",
            "## 11. Mechanistic analysis",
            "## 14. Limitations",
        ),
        "reference_audit.md": (
            "## Structural comparison",
            "## Lessons incorporated into Speck",
            "## Standard Speck intentionally raises",
        ),
        "reporting_checklist.md": (
            "## Claim integrity",
            "## Correctness and systems",
            "## Reproducibility and release",
        ),
    }
    for name, headings in required_headings.items():
        content = (directory / name).read_text(encoding="utf-8")
        missing = [heading for heading in headings if heading not in content]
        if missing:
            raise ValueError(f"{name} is missing required sections: {', '.join(missing)}")


def validate_paper_program(directory, repository_root=None):
    """Validate Paper 1 claims, experiment coverage, launch gate, and manuscript structure."""

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"paper program directory does not exist: {directory}")
    missing = [name for name in PROGRAM_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"paper program is missing files: {', '.join(missing)}")
    claims = _load_json(directory / "claims.json")
    claim_ids = _validate_claims(claims)
    program = _load_json(directory / "experiment_program.json")
    repository_root = (
        directory.parents[1]
        if repository_root is None
        else Path(repository_root).expanduser().resolve()
    )
    inventory = _validate_program(program, claims["paper_id"], claim_ids, repository_root)
    baseline = _validate_baseline_matrix(
        _load_json(directory / "baseline_matrix.json"),
        program,
        claims["paper_id"],
        repository_root,
    )
    _validate_markdown(directory)
    return {
        "paper_id": claims["paper_id"],
        "status": "valid_hypotheses_only",
        "program_files": list(PROGRAM_FILES),
        "claims": len(claim_ids),
        "non_claims": len(claims["non_claims"]),
        "scales": inventory["scales"],
        "axes": inventory["axes"],
        "historical_baseline_arms": baseline["historical_arms"],
        "planned_primary_baseline_arms": baseline["planned_arms"],
        "proxy_confirmation_pairs": baseline["proxy_pairs"],
        "paper_scale_pretraining": "blocked",
    }
