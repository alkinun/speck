"""Validate versioned architecture-research decision contracts."""

import hashlib
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
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PROMOTION_TASKS = {
    "multi_key",
    "two_hop_route",
    "two_hop_payload",
    "two_hop_symbolic",
}


def _repository_root(path):
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError(f"cannot resolve repository root from promotion protocol: {path}")


def _load_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load research contract file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"research contract file must contain an object: {path}")
    return value


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _validate_route_vocabulary(path, expected_tokenizer, tokenizer=None):
    vocabulary = _load_json(path)
    _require_keys(
        vocabulary,
        {"format", "format_version", "vocabulary_id", "tokenizer", "selection", "values"},
        "symbolic route vocabulary",
    )
    if (
        vocabulary["format"] != "speck_symbolic_route_vocabulary"
        or vocabulary["format_version"] != 1
    ):
        raise ValueError("symbolic route vocabulary must use format version 1")
    if vocabulary["tokenizer"] != expected_tokenizer:
        raise ValueError("symbolic route vocabulary tokenizer does not match its protocol")
    values = vocabulary["values"]
    declared_count = vocabulary["selection"].get("count")
    if not isinstance(values, list) or len(values) != declared_count or len(values) < 100:
        raise ValueError("symbolic route vocabulary must contain its declared 100+ values")
    texts = [entry.get("text") for entry in values]
    token_ids = [entry.get("token_id") for entry in values]
    if (
        any(not isinstance(text, str) or not text for text in texts)
        or len(set(texts)) != len(texts)
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in token_ids
        )
        or len(set(token_ids)) != len(token_ids)
    ):
        raise ValueError("symbolic route texts and token ids must be valid and unique")
    if tokenizer is not None:
        if tokenizer.fingerprint() != expected_tokenizer["fingerprint"]:
            raise ValueError("prepared tokenizer fingerprint does not match the route vocabulary")
        mismatches = [
            text for text, token_id in zip(texts, token_ids) if tokenizer.encode(text) != [token_id]
        ]
        if mismatches:
            raise ValueError(
                "symbolic route values do not round-trip as declared one-token ids: "
                + ", ".join(mismatches[:5])
            )
    return tuple(texts)


def _validate_promotion_protocol(path, repository_root, policy_id, tokenizer=None):
    from speck.long_context import ANSWER_SETS, RETRIEVAL_TEMPLATES

    protocol = _load_json(path)
    _require_keys(
        protocol,
        {
            "format",
            "format_version",
            "protocol_id",
            "status",
            "policy_id",
            "comparison",
            "tokenizer",
            "adaptation",
            "validation",
            "leakage_controls",
        },
        f"promotion protocol {path.name}",
    )
    if (
        protocol["format"] != "speck_promotion_retrieval_protocol"
        or protocol["format_version"] != 1
    ):
        raise ValueError(f"promotion protocol {path.name} must use format version 1")
    if protocol["policy_id"] != policy_id:
        raise ValueError(f"promotion protocol {path.name} references a different policy")
    if protocol["status"] != "frozen_runner_integration_pending":
        raise ValueError(f"promotion protocol {path.name} has an unexpected status")

    comparison = protocol["comparison"]
    seeds = comparison.get("base_model_seeds")
    offsets = comparison.get("training_stream_seed_offsets")
    if (
        seeds != [42, 43, 44]
        or not isinstance(offsets, dict)
        or set(offsets) != {str(seed) for seed in seeds}
        or len(set(offsets.values())) != len(offsets)
    ):
        raise ValueError(f"promotion protocol {path.name} has an invalid paired seed design")
    validation_offset = comparison.get("fixed_validation_seed_offset")
    _positive(validation_offset, f"promotion protocol {path.name} validation seed offset")
    if validation_offset in offsets.values():
        raise ValueError(f"promotion protocol {path.name} overlaps train and validation seeds")

    adaptation = protocol["adaptation"]
    validation = protocol["validation"]
    train_tasks = set(adaptation.get("tasks", ()))
    validation_tasks = set(validation.get("tasks", ()))
    if (
        not train_tasks
        or not validation_tasks
        or not (train_tasks | validation_tasks) <= PROMOTION_TASKS
    ):
        raise ValueError(f"promotion protocol {path.name} has unsupported tasks")
    train_templates = adaptation.get("train_templates")
    validation_templates = validation.get("templates")
    if (
        not train_templates
        or not validation_templates
        or not set(train_templates).isdisjoint(validation_templates)
        or not (set(train_templates) | set(validation_templates)) <= set(RETRIEVAL_TEMPLATES)
    ):
        raise ValueError(f"promotion protocol {path.name} must keep valid templates disjoint")
    train_answer_sets = adaptation.get("train_answer_sets")
    validation_answer_sets = validation.get("answer_sets")
    if (
        not train_answer_sets
        or not validation_answer_sets
        or not set(train_answer_sets).isdisjoint(validation_answer_sets)
        or not (set(train_answer_sets) | set(validation_answer_sets)) <= set(ANSWER_SETS)
    ):
        raise ValueError(f"promotion protocol {path.name} must keep valid answer sets disjoint")
    train_answers = {answer for name in train_answer_sets for answer in ANSWER_SETS[name]}
    validation_answers = {answer for name in validation_answer_sets for answer in ANSWER_SETS[name]}
    if not train_answers.isdisjoint(validation_answers):
        raise ValueError(f"promotion protocol {path.name} answer values leak across the split")
    if tokenizer is not None:
        if tokenizer.fingerprint() != protocol["tokenizer"]["fingerprint"]:
            raise ValueError(f"promotion protocol {path.name} tokenizer fingerprint does not match")
        for name in set(train_answer_sets) | set(validation_answer_sets):
            sequences = [tokenizer.encode(answer) for answer in ANSWER_SETS[name]]
            if any(not sequence for sequence in sequences) or len(
                {sequence[0] for sequence in sequences}
            ) != len(sequences):
                raise ValueError(
                    f"promotion protocol {path.name} answer set {name} lacks unique first tokens"
                )
            if (
                name in validation_answer_sets
                and len({len(sequence) for sequence in sequences}) != 1
            ):
                raise ValueError(
                    f"promotion protocol {path.name} validation answer set {name} "
                    "does not have a fixed token length"
                )
    replay = repository_root / adaptation.get("replay_data_experiment", "")
    if not replay.is_dir():
        raise ValueError(f"promotion protocol {path.name} replay experiment does not exist")
    for key in ("sequence_length", "steps", "batch_size", "accumulation", "eval_every"):
        _positive(adaptation.get(key), f"promotion protocol {path.name} adaptation {key}")

    sample_key = (
        "samples_per_condition"
        if protocol["protocol_id"] == "structured_retrieval_v2"
        else "samples_per_view"
    )
    samples = validation.get(sample_key)
    if not isinstance(samples, int) or samples < 200:
        raise ValueError(f"promotion protocol {path.name} requires at least 200 validation cases")

    route_count = 0
    if protocol["protocol_id"] == "structured_retrieval_v2":
        if adaptation.get("train_record_counts") != [2, 8] or validation.get("record_counts") != [
            2,
            8,
        ]:
            raise ValueError("structured retrieval v2 must report two- and eight-record loads")
        length_evaluation = protocol.get("length_evaluation", {})
        if length_evaluation.get("lengths") != [4096, 32768, 131072]:
            raise ValueError("structured retrieval v2 must freeze 4K, 32K, and 128K")
        if length_evaluation.get("samples_per_condition", 0) < 200:
            raise ValueError("structured retrieval v2 length cells require at least 200 cases")
    elif protocol["protocol_id"] == "symbolic_composition_v2":
        route_reference = protocol.get("route_vocabulary")
        route_hash = protocol.get("route_vocabulary_sha256")
        if not isinstance(route_reference, str) or not SHA256_PATTERN.fullmatch(route_hash or ""):
            raise ValueError("symbolic composition v2 requires a route vocabulary")
        route_path = path.parent / route_reference
        if _file_sha256(route_path) != route_hash:
            raise ValueError("symbolic composition v2 route vocabulary hash does not match")
        route_values = _validate_route_vocabulary(
            route_path,
            protocol["tokenizer"],
            tokenizer,
        )
        route_count = len(route_values)
        required_tasks = {"two_hop_route", "two_hop_payload", "two_hop_symbolic"}
        if train_tasks != required_tasks or validation_tasks != required_tasks:
            raise ValueError("symbolic composition v2 must preserve all three task views")
    else:
        raise ValueError(f"unknown promotion protocol: {protocol['protocol_id']}")
    return {"id": protocol["protocol_id"], "route_values": route_count}


def _validate_internal_protocols(manifest, repository_root, policy_id, tokenizer=None):
    protocols = []
    for suite in manifest["internal_suites"]:
        reference = suite.get("protocol")
        if reference is None:
            continue
        path = repository_root / reference
        expected_hash = suite.get("protocol_sha256")
        if not path.is_file() or not SHA256_PATTERN.fullmatch(expected_hash or ""):
            raise ValueError(f"evaluation suite {suite['id']} has an invalid protocol pin")
        if _file_sha256(path) != expected_hash:
            raise ValueError(f"evaluation suite {suite['id']} protocol hash does not match")
        protocol = _validate_promotion_protocol(path, repository_root, policy_id, tokenizer)
        if protocol["id"] != suite["id"]:
            raise ValueError(f"evaluation suite {suite['id']} pins the wrong protocol")
        protocols.append(protocol)
    if {protocol["id"] for protocol in protocols} != {
        "structured_retrieval_v2",
        "symbolic_composition_v2",
    }:
        raise ValueError("evaluation manifest must pin both promotion protocols")
    return protocols


def load_promotion_protocol(path, tokenizer=None):
    """Load and validate one frozen retrieval protocol for an execution runner."""

    path = Path(path).expanduser().resolve()
    repository_root = _repository_root(path)
    protocol = _load_json(path)
    policy_path = repository_root / "research" / protocol.get("policy_id", "") / "policy.json"
    policy = _load_json(policy_path)
    _validate_policy(policy)
    summary = _validate_promotion_protocol(
        path,
        repository_root,
        policy["policy_id"],
        tokenizer,
    )
    route_values = None
    if route_reference := protocol.get("route_vocabulary"):
        route_values = tuple(
            entry["text"] for entry in _load_json(path.parent / route_reference)["values"]
        )
    return {
        "protocol": protocol,
        "identity": {
            "id": summary["id"],
            "path": str(path),
            "sha256": _file_sha256(path),
        },
        "repository_root": repository_root,
        "route_values": route_values,
    }


def resolve_adaptation_protocol(loaded, seed):
    """Map a frozen protocol to structured-retrieval adaptation settings."""

    protocol = loaded["protocol"]
    comparison = protocol["comparison"]
    if seed not in comparison["base_model_seeds"]:
        raise ValueError(f"seed {seed} is not declared by protocol {protocol['protocol_id']}")
    adaptation = protocol["adaptation"]
    validation = protocol["validation"]
    validation_samples = validation.get("samples_per_condition", validation.get("samples_per_view"))
    record_counts = tuple(adaptation.get("train_record_counts", (adaptation.get("records", 8),)))
    validation_record_counts = tuple(
        validation.get("record_counts", (adaptation.get("records", 8),))
    )
    return {
        "tasks": tuple(adaptation["tasks"]),
        "validation_tasks": tuple(validation["tasks"]),
        "after_switch_tasks": (),
        "task_switch_step": None,
        "sequence_length": adaptation["sequence_length"],
        "steps": adaptation["steps"],
        "batch_size": adaptation["batch_size"],
        "accumulation": adaptation["accumulation"],
        "validation_samples": validation_samples,
        "eval_every": adaptation["eval_every"],
        "records": max(record_counts),
        "chains": adaptation.get("chains", validation.get("chains", 6)),
        "lr": adaptation["learning_rate"],
        "warmup_steps": adaptation["warmup_steps"],
        "min_lr": adaptation["minimum_lr_multiplier"],
        "weight_decay": adaptation["weight_decay"],
        "grad_clip": adaptation["gradient_clip"],
        "optimizer": adaptation["optimizer"],
        "seed": seed,
        "train_seed_offset": comparison["training_stream_seed_offsets"][str(seed)],
        "validation_seed_offset": comparison["fixed_validation_seed_offset"],
        "train_templates": tuple(adaptation["train_templates"]),
        "validation_templates": tuple(validation["templates"]),
        "train_answer_sets": tuple(adaptation["train_answer_sets"]),
        "validation_answer_sets": tuple(validation["answer_sets"]),
        "train_record_counts": record_counts,
        "validation_record_counts": validation_record_counts,
        "train_response_cue": adaptation["train_response_cue"],
        "validation_response_cue": validation["response_cue"],
        "replay_fraction": adaptation["language_replay_fraction"],
        "candidate_loss_weight": adaptation["candidate_loss_weight"],
        "route_values": loaded["route_values"] or tuple("KLMNOPQRST"),
        "replay_data_experiment": (
            loaded["repository_root"] / adaptation["replay_data_experiment"]
        ),
    }


def resolve_evaluation_protocol(loaded):
    """Resolve the exact condition grid for a frozen multi-length evaluation."""

    protocol = loaded["protocol"]
    evaluation = protocol.get("length_evaluation")
    if evaluation is None:
        raise ValueError(f"protocol {protocol['protocol_id']} has no length evaluation")
    validation = protocol["validation"]
    tasks = tuple(validation["tasks"])
    templates = tuple(validation["templates"])
    answer_sets = tuple(validation["answer_sets"])
    record_counts = tuple(evaluation.get("record_counts", (evaluation.get("records", 8),)))
    conditions = []
    for task in tasks:
        task_record_counts = record_counts if task == "multi_key" else (max(record_counts),)
        for template in templates:
            for answer_set in answer_sets:
                for records in task_record_counts:
                    conditions.append(
                        {
                            "task": task,
                            "template": template,
                            "answer_set": answer_set,
                            "records": records,
                            "chains": evaluation.get(
                                "chains", protocol["adaptation"].get("chains", 6)
                            ),
                            "response_cue": evaluation["response_cue"],
                        }
                    )
    return {
        "lengths": tuple(evaluation["lengths"]),
        "samples": evaluation["samples_per_condition"],
        "seed_offset": protocol["comparison"]["fixed_validation_seed_offset"],
        "kv_cache_dtype": evaluation["kv_cache_dtype"],
        "effective_threshold": evaluation.get("effective_threshold", 0.85),
        "conditions": tuple(conditions),
        "route_values": loaded["route_values"],
    }


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


def validate_research_contract(directory, tokenizer=None):
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
    protocols = _validate_internal_protocols(
        values["evaluation_manifest.json"], repository_root, policy_id, tokenizer
    )
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
        "promotion_protocols": len(protocols),
        "symbolic_route_values": sum(protocol["route_values"] for protocol in protocols),
        "tokenizer_qualified": tokenizer is not None,
        "evidence_components": len(values["evidence_matrix.json"]["components"]),
        "status": "valid",
    }
