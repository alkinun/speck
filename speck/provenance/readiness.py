"""Cross-check selected pre-access contracts without treating coherence as launch authority."""

import json
from fractions import Fraction
from pathlib import Path

from speck.provenance.catalog import validate_research_catalog
from speck.provenance.io import file_sha256
from speck.provenance.repository import repository_root


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relocated_source_path(source_path, original_plan_path, relative_plan_path):
    """Recover a source's relative path using the checked plan's original checkout."""
    original_root = Path(original_plan_path)
    for _ in Path(relative_plan_path).parts:
        original_root = original_root.parent
    require(
        original_root / relative_plan_path == Path(original_plan_path),
        "R0 original checkout cannot be recovered from its plan identity",
    )
    relative = Path(source_path).relative_to(original_root)
    require(".." not in relative.parts, "R0 source identity escapes its checkout")
    return relative


def check_contracts(values, claims, status, shapes):
    """Check cross-file invariants beyond each contract's individual identity/schema."""
    execution, model = values["execution"], values["model"]
    study, architecture = values["integration"], values["architecture"]
    data, evaluation, capability = (
        values["data"],
        values["heldout_evaluation"],
        values["capability"],
    )
    tokenizer = values["tokenizer_decision"]
    phases = {p["id"]: p for p in execution["phases"]}
    budget, flagship = execution["budget"], model["flagship"]
    require(len(phases) == len(execution["phases"]), "duplicate execution phase")
    require(
        all(type(p["gpu_hours"]) is int and p["gpu_hours"] >= 0 for p in phases.values()),
        "invalid phase budget",
    )
    require(
        sum(p["gpu_hours"] for p in phases.values()) == budget["gpu_hours"],
        "total allocation does not balance",
    )
    require(
        budget["mandatory_gpu_hours"] + budget["reserve_gpu_hours"] == budget["gpu_hours"],
        "mandatory/reserve allocation differs",
    )
    require(
        phases["RESERVE"]["gpu_hours"] == budget["reserve_gpu_hours"]
        and phases["RESERVE"]["conditional"] is True,
        "protected reserve differs",
    )
    require(
        sum(p["gpu_hours"] for p in phases.values() if not p["conditional"])
        == budget["mandatory_gpu_hours"],
        "mandatory phase total differs",
    )
    require(
        sum(phases[f"R{i}"]["gpu_hours"] for i in range(5)) == budget["research_gpu_hours"],
        "research subtotal differs",
    )
    require(
        sum(phases[f"V{i}"]["gpu_hours"] for i in range(1, 4))
        == budget["release_validation_gpu_hours"],
        "release subtotal differs",
    )
    require(
        phases["B0"]["gpu_hours"]
        == budget["base_gpu_hours"]
        == flagship["pretraining_gpu_hours_ceiling"],
        "base budget differs between contracts",
    )
    require(
        phases["L0"]["gpu_hours"]
        == budget["capability_gpu_hours"]
        == capability["gpu_hours_ceiling"]
        == sum(s["gpu_hours"] for s in capability["stages"]),
        "capability subtotal differs",
    )
    resolved = set()
    while len(resolved) < len(phases):
        ready = {
            key
            for key, phase in phases.items()
            if key not in resolved and set(phase["depends_on"]) <= resolved
        }
        require(bool(ready), "execution dependency cycle or missing phase")
        resolved.update(ready)
    for phase in phases.values():
        days = phase["target_days"]
        require(
            len(days) == 2 and 0 <= days[0] <= days[1] <= execution["calendar_days"],
            "phase calendar outside allocation window",
        )
    horizon = execution["base_horizon_policy"]
    require(
        (flagship["base_tokens"], flagship["stretch_tokens"])
        == (horizon["default_tokens"], horizon["stretch_tokens"]),
        "base/stretch horizons differ",
    )
    require(study["factors"] == architecture["factors"], "study and architecture factors differ")
    require(
        all(len(levels) == len(set(levels)) == 2 for levels in study["factors"].values()),
        "study must retain all four factorial cells",
    )
    for key in ("core", "transfer"):
        block = study[key]
        seeds = block["paired_seeds"]
        require(len(seeds) == len(set(seeds)), "duplicate paired seed")
        require(
            block["parent_runs"] == 2 * len(seeds) and block["branch_runs"] == 4 * len(seeds),
            "study parent/branch counts differ from paired seeds",
        )
        require(
            block["gpu_hours_ceiling"] == phases[block["phase"]]["gpu_hours"],
            "study budget differs from execution",
        )
    require(
        len(study["core"]["paired_seeds"]) == 3 and len(study["transfer"]["paired_seeds"]) == 1,
        "selected three-block study or single transfer block changed",
    )
    require(
        set(study["core"]["paired_seeds"]).isdisjoint(study["transfer"]["paired_seeds"]),
        "transfer seed overlaps core study",
    )
    lengths = model["lengths"]
    require(
        evaluation["lengths"]
        == [
            lengths[k]
            for k in (
                "base_training",
                "mandatory_capability_milestone",
                "intermediate_evaluation",
                "target",
            )
        ],
        "model and evaluation context lengths differ",
    )
    require(
        study["core"]["common_context"] == lengths["mandatory_capability_milestone"],
        "study primary context differs from useful-context milestone",
    )
    require(
        len(evaluation["primary_families"]) == len(set(evaluation["primary_families"])) == 2,
        "primary family identities differ",
    )
    require(
        sum(data["base"]["category_percent"].values()) == 100
        and set(data["base"]["category_percent"]) == set(data["base"]["initial_sources"]),
        "broad category weights/source assignments differ",
    )
    fractions = data["long_context"]
    require(
        sum(
            Fraction(str(fractions[k]))
            for k in (
                "default_targeted_fraction_of_assistant_tokens",
                "general_replay_fraction_of_assistant_tokens",
            )
        )
        == 1,
        "supervised-token fractions do not conserve exposure",
    )
    require(
        tokenizer["status"] == "tokenizer_selected_and_frozen"
        and tokenizer["D5_opened_by_this_decision"] is False,
        "tokenizer stop or D5 boundary changed",
    )
    reserved = tokenizer["reserved_role_capacity"]
    require(
        tokenizer["base_vocab_size"] + reserved["additional_ids"]
        == reserved["effective_model_vocab_size"],
        "base/reserved model vocabulary differs",
    )
    expected_cases = {
        (a, n)
        for a in ("hybrid", "dense")
        for n in (
            lengths["base_training"],
            lengths["mandatory_capability_milestone"],
            lengths["target"],
        )
    }
    require(
        len(shapes["cases"]) == len(expected_cases)
        and {(c["architecture"], c["sequence_length"]) for c in shapes["cases"]} == expected_cases,
        "R0 shape coverage differs",
    )
    require(shapes["r0_gpu_hour_ceiling"] == phases["R0"]["gpu_hours"], "R0 shape budget differs")
    for case in shapes["cases"]:
        require(
            case["model_vocab_size"] == reserved["effective_model_vocab_size"]
            and case["synthetic_input_vocab_size"] == tokenizer["base_vocab_size"],
            "R0 vocabulary differs from frozen tokenizer",
        )
        require(
            case["model"]["expected_parameters"]
            == case["instantiated_parameters"]
            == case["active_parameters"],
            "R0 parameter counts disagree",
        )
        if case["architecture"] == "hybrid":
            require(
                case["instantiated_parameters"] == flagship["reference_parameters"],
                "selected flagship count differs from R0 construction",
            )
        require(
            case["physically_tied_embeddings_pass"]
            == flagship["physically_tied_embeddings"]
            is True,
            "R0 embedding tie differs",
        )
        require(
            all(
                case[key] is None
                for key in (
                    "actual_gpu_peak_bytes",
                    "measured_tokens_per_second",
                    "gpu_fit_pass",
                    "backward_pass",
                    "resume_parity_pass",
                    "four_gpu_ddp_pass",
                )
            ),
            "meta-only R0 evidence cannot certify GPU execution",
        )
    require(
        claims["status"] == "pre_results"
        and all(c["status"] == "planned" and not c["evidence"] for c in claims["claims"]),
        "pre-access paper status exceeds current planned evidence",
    )
    require(
        all(
            values[role].get("training_authority") is False
            for role in (
                "model",
                "data",
                "architecture",
                "integration",
                "heldout_evaluation",
                "capability",
                "execution",
            )
        ),
        "selected design is not training authority",
    )
    require(study["execution_ready"] is False, "study freeze is not yet execution-qualified")
    allocation = status["allocation_status"]
    require(
        allocation["state"] == "proposal_under_evaluation"
        and allocation["access_confirmed"] is False
        and allocation["compute_access_date"] is None,
        "review-pending audit cannot certify an access date",
    )
    return {
        "status": "pre_access_contracts_consistent_execution_pending",
        "training_authority": False,
        "allocated_node_days_if_awarded": budget["gpu_hours"] / execution["hardware"]["gpus"] / 24,
        "core_parent_runs": study["core"]["parent_runs"],
        "core_branch_runs": study["core"]["branch_runs"],
        "shape_cases": len(shapes["cases"]),
        "pending": [
            "allocation decision, site access and scheduler details",
            "bounded R0 executor and actual GH200 kernel/fit/backward/DDP/resume/throughput checks",
            "successor source capacity, exposure limits, delivery and jointly eligible coherent units",
            "family-separated pilot/development/final tasks and pinned benchmark/scorer identities",
            "numerical study floors/margins, endpoints, optimizer settings, cost forecast and transfer route",
            "independent novelty/design review and new-partition launch compatibility",
        ],
        "boundary": "Static cross-contract coherence under the current pending-review phase, not exhaustive software correctness, scientific novelty, source availability, GPU readiness or model-launch permission. An award or qualified execution requires a successor readiness audit rather than editing this historical result.",
    }


def audit_readiness(root=None):
    root = Path(root or repository_root()).resolve()
    catalog_result = validate_research_catalog(root / "research/catalog.json", repository_root=root)
    catalog = json.loads((root / "research/catalog.json").read_text())
    selected = catalog["active_contracts"]
    values = {role: json.loads((root / row["path"]).read_text()) for role, row in selected.items()}
    for role, path in values["execution"]["active_contracts"].items():
        require(
            selected[role]["path"] == path,
            "execution selects a different contract from the catalog",
        )
    status = json.loads((root / catalog["status_record"]).read_text())
    claims = json.loads((root / catalog["paper"]["claims"]).read_text())
    hardware = next(row for row in status["work"] if row["id"] == "lc_hardware")
    shape_path = root / hardware["shape_evidence"]
    shapes = json.loads(shape_path.read_text())
    shape_plan = json.loads((root / hardware["shape_plan"]).read_text())
    require(
        shapes["plan"]["sha256"] == file_sha256(root / hardware["shape_plan"]),
        "R0 evidence uses a different shape plan",
    )
    for role, binding in shape_plan["inputs"].items():
        require(
            file_sha256(root / binding["path"]) == binding["sha256"], "R0 shape-plan input changed"
        )
        require(
            shapes["inputs"][role]["sha256"] == binding["sha256"],
            "R0 result input differs from its plan",
        )
        if role != "retained_geometry":
            require(
                binding["path"] == selected[role]["path"]
                and binding["sha256"] == selected[role]["sha256"],
                "R0 evidence is stale against selected contracts",
            )
    for identity in shapes["implementation"]:
        # Derive the original checkout from the bound plan, independent of checkout directory names.
        relative = relocated_source_path(
            identity["path"], shapes["plan"]["path"], hardware["shape_plan"]
        )
        require(
            file_sha256(root / relative) == identity["sha256"],
            "R0 implementation has changed without successor qualification",
        )
    result = check_contracts(values, claims, status, shapes)
    paths = [
        "research/catalog.json",
        catalog["status_record"],
        catalog["paper"]["claims"],
        hardware["shape_plan"],
        hardware["shape_evidence"],
        *[row["path"] for row in selected.values()],
    ]
    return {
        "format": "speck_pre_access_coherence_audit",
        "format_version": 1,
        **result,
        "catalog_validation": catalog_result,
        "inputs": [{"path": p, "sha256": file_sha256(root / p)} for p in dict.fromkeys(paths)],
        "audit_implementation": [
            {"path": p, "sha256": file_sha256(root / p)}
            for p in ("speck/provenance/readiness.py", "scripts/readiness_audit.py")
        ],
    }
