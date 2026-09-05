"""Validate the preregistered Speck Paper 1 research program."""

import json
from pathlib import Path

PROGRAM_FILES = (
    "README.md",
    "claims.json",
    "experiment_program.json",
    "paper_outline.md",
    "reference_audit.md",
    "reporting_checklist.md",
)


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
    _validate_markdown(directory)
    return {
        "paper_id": claims["paper_id"],
        "status": "valid_hypotheses_only",
        "program_files": list(PROGRAM_FILES),
        "claims": len(claim_ids),
        "non_claims": len(claims["non_claims"]),
        "scales": inventory["scales"],
        "axes": inventory["axes"],
        "paper_scale_pretraining": "blocked",
    }
