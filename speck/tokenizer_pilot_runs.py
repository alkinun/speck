"""Materialize immutable tokenizer-pilot screen run manifests."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

from speck.io import atomic_json, file_sha256
from speck.scale_targets import flop_accounting, model_settings, parameter_accounting
from speck.validation import positive_integer, require_keys

FORMAT = "speck_tokenizer_pilot_run_materialization"
FORMAT_VERSION = 1
CATEGORIES = ("web", "code", "math", "synthetic", "science", "reference")


def _fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _identity(value, root, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain exactly path and sha256")
    require_keys(value, {"path", "sha256"}, context)
    if set(value) != {"path", "sha256"}:
        raise ValueError(f"{context} must contain exactly path and sha256")
    path = Path(value["path"]).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or file_sha256(path) != value["sha256"]:
        raise ValueError(f"{context} identity mismatch")
    return {"path": str(path), "sha256": value["sha256"]}


def _load_identity(value, root, context):
    identity = _identity(value, root, context)
    try:
        document = json.loads(Path(identity["path"]).read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{context} must contain a JSON object")
    return identity, document


def _validate_settings(settings):
    expected = {
        "sequence_length",
        "device_batch_size",
        "batch_tokens",
        "accumulation",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "lr_schedule",
        "warmup_steps",
        "min_lr",
        "loss_backend",
        "compile",
        "checkpoint_every_steps",
        "learning_curve_every_steps",
    }
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError("tokenizer pilot training settings are incomplete")
    for key in ("sequence_length", "device_batch_size", "batch_tokens", "accumulation"):
        positive_integer(settings[key], key.replace("_", " "))
    for key in ("checkpoint_every_steps", "learning_curve_every_steps"):
        positive_integer(settings[key], key.replace("_", " "))
    if settings["batch_tokens"] != (
        settings["sequence_length"] * settings["device_batch_size"] * settings["accumulation"]
    ):
        raise ValueError("tokenizer pilot batch geometry is inconsistent")
    if settings["optimizer"] != "muon" or settings["loss_backend"] != "torch":
        raise ValueError("tokenizer pilot must use the qualified optimizer and loss backend")
    if settings["lr_schedule"] != "cosine":
        raise ValueError("tokenizer pilot must use the frozen cosine schedule")
    if not isinstance(settings["compile"], bool):
        raise ValueError("tokenizer pilot compile setting must be boolean")
    warmup = settings["warmup_steps"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("tokenizer pilot warmup steps must be non-negative")
    for key in ("learning_rate", "grad_clip"):
        value = settings[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"tokenizer pilot {key} must be positive")
    weight_decay = settings["weight_decay"]
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("tokenizer pilot weight decay must be non-negative")
    minimum = settings["min_lr"]
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(minimum)
        or not 0 <= minimum <= 1
    ):
        raise ValueError("tokenizer pilot minimum learning-rate scale must be in [0, 1]")
    return dict(settings)


def _by_id(values, context):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{context} must be a non-empty list")
    result = {value.get("id"): value for value in values if isinstance(value, dict)}
    if len(result) != len(values) or None in result:
        raise ValueError(f"{context} IDs must be present and unique")
    return result


def _clean_git_revision(repository):
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("tokenizer pilot runs require a clean tracked Git tree")
    return revision


def validate_run_materialization_plan(value, *, config_dir=None):
    """Validate and resolve every pre-output input to the seed-42 screen."""

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
        "tokenizers",
        "screen",
        "settings",
        "output_directory",
        "authority",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("tokenizer pilot run materialization fields are invalid")
    if (
        value["format"] != FORMAT
        or value["format_version"] != FORMAT_VERSION
        or value["status"] != "screen_materialization_authorized_no_model_outputs"
    ):
        raise ValueError("unsupported tokenizer pilot run materialization contract")
    root = Path(config_dir or ".").resolve()
    pilot_identity, pilot = _load_identity(value["pilot_plan"], root, "pilot plan")
    scale_identity, scale = _load_identity(value["scale_spec"], root, "scale spec")
    fixed_identity, fixed = _load_identity(value["fixed_stream"], root, "fixed stream")
    continuation_identity, continuation = _load_identity(
        value["continuation"], root, "continuation"
    )
    evaluation_identity, evaluation = _load_identity(
        value["evaluation_sample"], root, "evaluation sample"
    )
    if (
        pilot.get("format") != "speck_tokenizer_pilot_plan"
        or pilot.get("format_version") != 10
        or pilot.get("execution_preflight", {}).get("status") != "pass_screen_only"
        or pilot.get("final_selection_authority") is not False
    ):
        raise ValueError("run materialization requires the screen-authorized v10 pilot plan")
    if (
        pilot.get("model_geometry", {}).get("source_spec", {}).get("sha256")
        != scale_identity["sha256"]
        or pilot.get("fixed_stream", {}).get("runtime_manifest", {}).get("sha256")
        != fixed_identity["sha256"]
        or pilot.get("fixed_flop_materialization", {})
        .get("continuation", {})
        .get("runtime_manifest", {})
        .get("sha256")
        != continuation_identity["sha256"]
    ):
        raise ValueError("run materialization inputs differ from the v10 pilot plan")
    if scale.get("format") != "speck_flagship_scale_target_spec":
        raise ValueError("invalid tokenizer pilot scale specification")
    targets = _by_id(scale.get("targets"), "scale targets")
    if value["target_id"] not in targets:
        raise ValueError("tokenizer pilot target is missing from the scale specification")
    if (
        fixed.get("format") != "speck_tokenizer_pilot_stream_result"
        or fixed.get("status")
        != "whole_document_stream_and_tokenizer_packs_complete_not_training_authority"
        or continuation.get("format") != "speck_tokenizer_pilot_continuation_result"
        or continuation.get("status")
        != "shared_equal_flop_continuation_complete_not_training_authority"
        or continuation.get("fixed_stream", {}).get("sha256") != fixed_identity["sha256"]
    ):
        raise ValueError("tokenizer pilot fixed and continuation streams are incompatible")
    evaluation_categories = evaluation.get("categories")
    if (
        evaluation.get("format") != "speck_tokenizer_sample"
        or tuple(item.get("id") for item in evaluation_categories or ()) != CATEGORIES
    ):
        raise ValueError("tokenizer pilot evaluation sample is incomplete")
    authority = value["authority"]
    if authority != {
        "screen_execution": True,
        "confirmation_execution": False,
        "D5_opening": False,
        "final_selection": False,
        "flagship_training": False,
    }:
        raise ValueError("tokenizer pilot authority must remain screen-only")
    settings = _validate_settings(value["settings"])
    if not isinstance(value["tokenizers"], list) or not value["tokenizers"]:
        raise ValueError("tokenizer declarations must be a non-empty list")
    declarations = {
        declaration.get("role"): declaration
        for declaration in value["tokenizers"]
        if isinstance(declaration, dict)
    }
    if len(declarations) != len(value["tokenizers"]) or None in declarations:
        raise ValueError("tokenizer declaration roles must be present and unique")
    fixed_tokenizers = _by_id(fixed.get("tokenizers"), "fixed-stream tokenizers")
    continuation_tokenizers = _by_id(continuation.get("tokenizers"), "continuation tokenizers")
    screen = value["screen"]
    if (
        not isinstance(screen, dict)
        or set(screen) != {"seed", "runs"}
        or screen["seed"] != pilot["screen"]["seed"]
        or screen["runs"] != pilot["screen"]["runs"]
        or set(declarations) != set(screen["runs"])
    ):
        raise ValueError("tokenizer pilot screen matrix differs from the v10 contract")
    repository_root = Path(pilot_identity["path"]).parents[2]
    _, preflight_plan = _load_identity(
        pilot["execution_preflight"]["plan"], repository_root, "preflight plan"
    )
    preflight = _by_id(preflight_plan["tokenizers"], "preflight tokenizers")
    normalized_declarations = []
    for role in screen["runs"]:
        declaration = declarations[role]
        tokenizer_id = declaration.get("id")
        if set(declaration) != {
            "role",
            "id",
            "model",
            "vocab_size",
            "bos_token_id",
            "eos_token_id",
            "fixed_flop_token_stop",
            "run_stop_aligned_tokens",
        }:
            raise ValueError("tokenizer pilot declaration fields are invalid")
        model = _identity(declaration["model"], root, f"tokenizer {tokenizer_id}")
        fixed_entry = fixed_tokenizers.get(tokenizer_id)
        continuation_entry = continuation_tokenizers.get(tokenizer_id)
        preflight_entry = preflight.get(tokenizer_id)
        if (
            fixed_entry is None
            or continuation_entry is None
            or preflight_entry is None
            or fixed_entry["model"] != model
            or continuation_entry["model"] != model
            or any(
                declaration[key] != preflight_entry[key]
                for key in (
                    "vocab_size",
                    "bos_token_id",
                    "eos_token_id",
                    "fixed_flop_token_stop",
                    "run_stop_aligned_tokens",
                )
            )
            or declaration["run_stop_aligned_tokens"]
            > fixed_entry["tokens"] + continuation_entry["tokens"]
            or declaration["run_stop_aligned_tokens"] % settings["batch_tokens"]
        ):
            raise ValueError(f"tokenizer pilot declaration is not executable: {tokenizer_id}")
        normalized_declarations.append({**declaration, "model": model})
    output = Path(value["output_directory"]).expanduser()
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    normalized = {
        **value,
        "pilot_plan": pilot_identity,
        "scale_spec": scale_identity,
        "fixed_stream": fixed_identity,
        "continuation": continuation_identity,
        "evaluation_sample": evaluation_identity,
        "tokenizers": normalized_declarations,
        "settings": settings,
        "output_directory": str(output),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_run_materialization_plan(path):
    path = Path(path).resolve()
    return validate_run_materialization_plan(json.loads(path.read_text()), config_dir=path.parent)


def _validated_plan(plan):
    if "plan_fingerprint" not in plan:
        return validate_run_materialization_plan(plan)
    payload = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    validated = validate_run_materialization_plan(payload)
    if plan["plan_fingerprint"] != validated["plan_fingerprint"]:
        raise ValueError("normalized tokenizer pilot run materialization fingerprint mismatch")
    return validated


def _category_shards(root, manifest, tokenizer_id):
    tokenizer = _by_id(manifest["tokenizers"], "stream tokenizers")[tokenizer_id]
    return [
        {
            **shard,
            "path": str(root / tokenizer_id / category["id"] / shard["path"]),
            "category": category["id"],
        }
        for category in tokenizer["categories"]
        for shard in category["shards"]
    ]


def build_screen_run_manifests(plan, *, repository_revision=None):
    """Build the exact three-run screen matrix without writing or executing it."""

    plan = _validated_plan(plan)
    fixed = json.loads(Path(plan["fixed_stream"]["path"]).read_text())
    continuation = json.loads(Path(plan["continuation"]["path"]).read_text())
    evaluation = json.loads(Path(plan["evaluation_sample"]["path"]).read_text())
    scale = json.loads(Path(plan["scale_spec"]["path"]).read_text())
    target = _by_id(scale["targets"], "scale targets")[plan["target_id"]]
    repository = Path(plan["pilot_plan"]["path"]).parents[2]
    repository_revision = repository_revision or _clean_git_revision(repository)
    fixed_root = Path(plan["fixed_stream"]["path"]).parent
    continuation_root = Path(plan["continuation"]["path"]).parent
    evaluation_root = Path(plan["evaluation_sample"]["path"]).parent
    settings = plan["settings"]
    manifests = []
    for declaration in plan["tokenizers"]:
        tokenizer_id = declaration["id"]
        vocab_size = declaration["vocab_size"]
        model = model_settings(target, vocab_size)
        fixed_tokens = _by_id(fixed["tokenizers"], "fixed tokenizers")[tokenizer_id]["tokens"]
        run = {
            "format": "speck_tokenizer_pilot_screen_run",
            "format_version": 1,
            "status": "materialized_not_started",
            "run_id": f"tokenizer-pilot-{tokenizer_id}-seed-{plan['screen']['seed']}",
            "role": declaration["role"],
            "tokenizer_id": tokenizer_id,
            "seed": plan["screen"]["seed"],
            "repository_revision": repository_revision,
            "materialization_plan_fingerprint": plan["plan_fingerprint"],
            "model": {
                "settings": model,
                "sha256": _fingerprint(model),
                "backbone_sha256": _fingerprint({**model, "vocab_size": "tokenizer_specific"}),
                "parameters": parameter_accounting(target, vocab_size, contract_version=2)[
                    "total_parameters"
                ],
                "analytic_training_flops_per_token": flop_accounting(
                    target, vocab_size, settings["sequence_length"], contract_version=2
                )["analytic_training_flops_per_token"],
            },
            "tokenizer": {
                "model": declaration["model"],
                "vocab_size": vocab_size,
                "bos_token_id": declaration["bos_token_id"],
                "eos_token_id": declaration["eos_token_id"],
            },
            "training_data": {
                "fixed_stream": plan["fixed_stream"],
                "continuation": plan["continuation"],
                "document_stream_sha256": fixed["document_stream"]["sha256"],
                "fixed_shards": _category_shards(fixed_root, fixed, tokenizer_id),
                "continuation_shards": _category_shards(
                    continuation_root, continuation, tokenizer_id
                ),
            },
            "stops": {
                "fixed_document_tokens": fixed_tokens,
                "fixed_document_step": (fixed_tokens + settings["batch_tokens"] - 1)
                // settings["batch_tokens"],
                "fixed_flop_token_stop": declaration["fixed_flop_token_stop"],
                "final_step": declaration["run_stop_aligned_tokens"] // settings["batch_tokens"],
                "run_stop_aligned_tokens": declaration["run_stop_aligned_tokens"],
            },
            "evaluation": {
                "sample": plan["evaluation_sample"],
                "categories": [
                    {
                        "id": category["id"],
                        "path": str(evaluation_root / category["splits"]["eval"]["path"]),
                        "sha256": category["splits"]["eval"]["sha256"],
                        "documents": category["splits"]["eval"]["documents"],
                        "utf8_bytes": category["splits"]["eval"]["utf8_bytes"],
                    }
                    for category in evaluation["categories"]
                ],
            },
            "settings": settings,
            "output_directory": str(Path(plan["output_directory"]) / tokenizer_id / "seed-42"),
            "authority": plan["authority"],
        }
        run["run_fingerprint"] = _fingerprint(run)
        manifests.append(run)
    return tuple(manifests)


def materialize_screen_runs(plan):
    """Publish all three screen manifests once through an isolated staging directory."""

    plan = _validated_plan(plan)
    output = Path(plan["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError(f"tokenizer pilot run materialization already exists: {output}")
    staging.mkdir(parents=True)
    repository = Path(plan["pilot_plan"]["path"]).parents[2]
    manifests = build_screen_run_manifests(
        plan, repository_revision=_clean_git_revision(repository)
    )
    for run in manifests:
        path = staging / run["tokenizer_id"] / "seed-42" / "run.json"
        atomic_json(path, run)
    summary = {
        "format": "speck_tokenizer_pilot_screen_materialization_result",
        "format_version": 1,
        "status": "three_screen_runs_materialized_not_started",
        "plan_fingerprint": plan["plan_fingerprint"],
        "runs": [
            {
                "run_id": run["run_id"],
                "run_fingerprint": run["run_fingerprint"],
                "path": str(Path(run["tokenizer_id"]) / "seed-42" / "run.json"),
            }
            for run in manifests
        ],
        "authority": plan["authority"],
    }
    atomic_json(staging / "manifest.json", summary)
    staging.replace(output)
    return summary
