"""Materialize tokenizer-pilot runs with independently checked fixed-FLOP stops."""

import json
import math
from pathlib import Path

from speck.io import atomic_json, file_sha256
from speck.scale_targets import flop_accounting
from speck.tokenizer_pilot_runs import (
    _by_id,
    _clean_git_revision,
    _fingerprint,
    _identity,
    build_screen_run_manifests,
    validate_run_materialization_plan,
)

FORMAT_VERSION = 2
STATUS = "corrected_screen_materialization_authorized_no_model_outputs"


def validate_flop_correction(correction, scale, preflight):
    """Recompute the target and every tokenizer stop from independent inputs."""

    if (
        not isinstance(correction, dict)
        or correction.get("format") != "speck_tokenizer_pilot_fixed_flop_correction"
        or correction.get("format_version") != 1
        or correction.get("status") != "corrected_pre_output_accounting_v2_materialization_required"
    ):
        raise ValueError("unsupported tokenizer pilot FLOP correction")
    if scale.get("format") != "speck_flagship_scale_target_spec":
        raise ValueError("invalid scale target input to the FLOP correction")
    target = _by_id(scale.get("targets"), "scale targets").get(correction.get("target_id"))
    if target is None:
        raise ValueError("FLOP correction target is absent from the scale specification")
    measured = _by_id(preflight.get("tokenizers"), "preflight tokenizers")
    corrected = _by_id(correction.get("tokenizers"), "corrected tokenizers")
    reference = correction.get("reference")
    if not isinstance(reference, dict) or reference.get("id") not in corrected:
        raise ValueError("FLOP correction reference is invalid")
    target_flops = (
        reference["fixed_document_tokens"] * reference["analytic_training_flops_per_token"]
    )
    if reference.get("analytic_flop_target") != target_flops:
        raise ValueError("corrected analytic FLOP target does not match its equation")
    batch_tokens = correction.get("batch_tokens")
    if isinstance(batch_tokens, bool) or not isinstance(batch_tokens, int) or batch_tokens < 1:
        raise ValueError("FLOP correction batch tokens must be positive")
    for tokenizer_id, declaration in corrected.items():
        preflight_entry = measured.get(tokenizer_id)
        if preflight_entry is None or declaration.get("vocab_size") != preflight_entry.get(
            "vocab_size"
        ):
            raise ValueError(f"corrected tokenizer is absent from preflight: {tokenizer_id}")
        accounting = flop_accounting(
            target,
            declaration["vocab_size"],
            correction["sequence_length"],
            contract_version=2,
        )["analytic_training_flops_per_token"]
        if (
            declaration.get("analytic_training_flops_per_token") != accounting
            or preflight_entry.get("analytic_training_flops_per_token") != accounting
        ):
            raise ValueError(f"corrected tokenizer FLOPs differ from accounting: {tokenizer_id}")
        fixed_flop_stop = math.ceil(target_flops / accounting)
        aligned = (
            math.ceil(max(declaration["fixed_document_tokens"], fixed_flop_stop) / batch_tokens)
            * batch_tokens
        )
        if (
            declaration.get("fixed_flop_token_stop") != fixed_flop_stop
            or declaration.get("run_stop_aligned_tokens") != aligned
        ):
            raise ValueError(f"corrected tokenizer stop differs from its equation: {tokenizer_id}")
    if correction.get("authority") != {
        "v2_run_materialization": True,
        "screen_execution": False,
        "confirmation_execution": False,
        "D5_opening": False,
        "final_selection": False,
        "flagship_training": False,
    }:
        raise ValueError("FLOP correction authority is invalid")
    return {
        "analytic_flop_target": target_flops,
        "tokenizers": corrected,
    }


def _load_json_identity(value, root, context):
    identity = _identity(value, root, context)
    document = json.loads(Path(identity["path"]).read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{context} must contain a JSON object")
    return identity, document


def _legacy_plan(value, config_dir):
    """Build a v1-shaped copy solely for its existing identity and capacity checks."""

    root = Path(config_dir or ".").resolve()
    _, pilot = _load_json_identity(value["pilot_plan"], root, "pilot plan")
    pilot_path = _identity(value["pilot_plan"], root, "pilot plan")["path"]
    repository = Path(pilot_path).parents[2]
    _, preflight = _load_json_identity(
        pilot["execution_preflight"]["plan"], repository, "preflight plan"
    )
    legacy_stops = _by_id(preflight["tokenizers"], "preflight tokenizers")
    tokenizers = []
    for declaration in value["tokenizers"]:
        legacy = dict(declaration)
        previous = legacy_stops[declaration["id"]]
        legacy["fixed_flop_token_stop"] = previous["fixed_flop_token_stop"]
        legacy["run_stop_aligned_tokens"] = previous["run_stop_aligned_tokens"]
        tokenizers.append(legacy)
    authority = dict(value["authority"])
    authority.pop("v2_run_materialization", None)
    authority["screen_execution"] = True
    return {
        **value,
        "format_version": 1,
        "status": "screen_materialization_authorized_no_model_outputs",
        "tokenizers": tokenizers,
        "authority": authority,
    }


def validate_corrected_materialization_plan(value, *, config_dir=None):
    """Validate v2 inputs, the predecessor checks, and corrected FLOP equations."""

    if not isinstance(value, dict):
        raise ValueError("corrected tokenizer pilot materialization must be an object")
    expected = {
        "format",
        "format_version",
        "status",
        "pilot_plan",
        "scale_spec",
        "target_id",
        "fixed_stream",
        "continuation",
        "evaluation_sample",
        "flop_correction",
        "implementation",
        "tokenizers",
        "screen",
        "settings",
        "output_directory",
        "authority",
    }
    if (
        set(value) != expected
        or value["format_version"] != FORMAT_VERSION
        or value["status"] != STATUS
    ):
        raise ValueError("corrected tokenizer pilot materialization fields are invalid")
    root = Path(config_dir or ".").resolve()
    correction_identity, correction = _load_json_identity(
        value["flop_correction"], root, "FLOP correction"
    )
    legacy = _legacy_plan(
        {key: item for key, item in value.items() if key != "flop_correction"}, root
    )
    normalized_legacy = validate_run_materialization_plan(legacy, config_dir=root)
    scale = json.loads(Path(normalized_legacy["scale_spec"]["path"]).read_text())
    pilot = json.loads(Path(normalized_legacy["pilot_plan"]["path"]).read_text())
    repository = Path(normalized_legacy["pilot_plan"]["path"]).parents[2]
    for name, identity in correction["inputs"].items():
        _identity(identity, repository, f"FLOP correction {name}")
    _identity(correction["supersedes_section"]["plan"], repository, "superseded pilot plan")
    preflight_result_identity = correction["inputs"]["preflight_result"]
    if (
        file_sha256(repository / preflight_result_identity["path"])
        != preflight_result_identity["sha256"]
    ):
        raise ValueError("corrected preflight result identity mismatch")
    preflight = json.loads((repository / preflight_result_identity["path"]).read_text())
    geometry = validate_flop_correction(correction, scale, preflight)
    if (
        correction["supersedes_section"]["plan"]["sha256"]
        != normalized_legacy["pilot_plan"]["sha256"]
        or correction["inputs"]["scale_spec"]["sha256"] != normalized_legacy["scale_spec"]["sha256"]
        or value["target_id"] != correction["target_id"]
        or pilot["screen"] != value["screen"]
        or value["settings"]["sequence_length"] != correction["sequence_length"]
        or value["settings"]["batch_tokens"] != correction["batch_tokens"]
    ):
        raise ValueError("corrected materialization differs from its bound inputs")
    corrected = geometry["tokenizers"]
    fixed = json.loads(Path(normalized_legacy["fixed_stream"]["path"]).read_text())
    fixed_tokenizers = _by_id(fixed["tokenizers"], "fixed tokenizers")
    normalized_tokenizers = []
    for declaration in value["tokenizers"]:
        correction_entry = corrected.get(declaration["id"])
        if (
            correction_entry is None
            or declaration["vocab_size"] != correction_entry["vocab_size"]
            or declaration["fixed_flop_token_stop"] != correction_entry["fixed_flop_token_stop"]
            or declaration["run_stop_aligned_tokens"] != correction_entry["run_stop_aligned_tokens"]
            or fixed_tokenizers[declaration["id"]]["tokens"]
            != correction_entry["fixed_document_tokens"]
        ):
            raise ValueError(f"v2 run declaration differs from correction: {declaration['id']}")
        normalized = next(
            item for item in normalized_legacy["tokenizers"] if item["id"] == declaration["id"]
        )
        normalized_tokenizers.append(
            {
                **normalized,
                "fixed_flop_token_stop": declaration["fixed_flop_token_stop"],
                "run_stop_aligned_tokens": declaration["run_stop_aligned_tokens"],
            }
        )
    normalized = {
        **normalized_legacy,
        "format_version": FORMAT_VERSION,
        "status": STATUS,
        "flop_correction": correction_identity,
        "tokenizers": normalized_tokenizers,
        "authority": dict(value["authority"]),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_corrected_materialization_plan(path):
    path = Path(path).resolve()
    return validate_corrected_materialization_plan(
        json.loads(path.read_text()), config_dir=path.parent
    )


def _validated_plan(plan):
    if "plan_fingerprint" not in plan:
        return validate_corrected_materialization_plan(plan)
    payload = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    validated = validate_corrected_materialization_plan(payload)
    if validated["plan_fingerprint"] != plan["plan_fingerprint"]:
        raise ValueError("corrected materialization fingerprint mismatch")
    return validated


def build_corrected_screen_run_manifests(plan, *, repository_revision=None):
    """Build corrected screen records while preserving v1's validated non-FLOP inputs."""

    plan = _validated_plan(plan)
    repository = Path(plan["pilot_plan"]["path"]).parents[2]
    revision = repository_revision or _clean_git_revision(repository)
    legacy = _legacy_plan(
        {
            key: item
            for key, item in plan.items()
            if key not in {"flop_correction", "plan_fingerprint"}
        },
        repository,
    )
    predecessor_runs = build_screen_run_manifests(legacy, repository_revision=revision)
    declarations = _by_id(plan["tokenizers"], "corrected tokenizer declarations")
    runs = []
    for predecessor in predecessor_runs:
        declaration = declarations[predecessor["tokenizer_id"]]
        run = {
            **predecessor,
            "format": "speck_tokenizer_pilot_corrected_screen_run",
            "format_version": FORMAT_VERSION,
            "status": "corrected_materialized_not_started_execution_blocked",
            "materialization_plan_fingerprint": plan["plan_fingerprint"],
            "materializer_implementation": plan["implementation"],
            "flop_correction": plan["flop_correction"],
            "stops": {
                **predecessor["stops"],
                "fixed_flop_token_stop": declaration["fixed_flop_token_stop"],
                "run_stop_aligned_tokens": declaration["run_stop_aligned_tokens"],
                "final_step": declaration["run_stop_aligned_tokens"]
                // plan["settings"]["batch_tokens"],
            },
            "output_directory": str(
                Path(plan["output_directory"]) / predecessor["tokenizer_id"] / "seed-42"
            ),
            "authority": plan["authority"],
        }
        run.pop("run_fingerprint")
        run["run_fingerprint"] = _fingerprint(run)
        runs.append(run)
    return tuple(runs)


def materialize_corrected_screen_runs(plan):
    """Exclusively publish the corrected, still execution-blocked screen records."""

    plan = _validated_plan(plan)
    output = Path(plan["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError(f"corrected tokenizer pilot materialization already exists: {output}")
    staging.mkdir(parents=True)
    repository = Path(plan["pilot_plan"]["path"]).parents[2]
    runs = build_corrected_screen_run_manifests(
        plan, repository_revision=_clean_git_revision(repository)
    )
    for run in runs:
        atomic_json(staging / run["tokenizer_id"] / "seed-42" / "run.json", run)
    result = {
        "format": "speck_tokenizer_pilot_corrected_screen_materialization_result",
        "format_version": FORMAT_VERSION,
        "status": "three_corrected_screen_runs_materialized_execution_blocked",
        "plan_fingerprint": plan["plan_fingerprint"],
        "runs": [
            {
                "run_id": run["run_id"],
                "run_fingerprint": run["run_fingerprint"],
                "path": str(Path(run["tokenizer_id"]) / "seed-42" / "run.json"),
            }
            for run in runs
        ],
        "authority": plan["authority"],
    }
    atomic_json(staging / "manifest.json", result)
    staging.replace(output)
    return result
