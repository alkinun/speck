"""Define execution authority and completed records for tokenizer-pilot runs."""

import json
import math
import subprocess
from pathlib import Path

from speck.io import file_sha256
from speck.tokenizer_pilot_runtime import load_pilot_run_manifest


def _identity(value, root, context):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{context} must contain exactly path and sha256")
    path = Path(value["path"])
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if not path.is_file() or file_sha256(path) != value["sha256"]:
        raise ValueError(f"{context} identity mismatch")
    return {"path": str(path), "sha256": value["sha256"]}


def verify_execution_revision(root, revision):
    """Allow checked records after a freeze commit but reject later runtime changes."""

    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", revision, "HEAD"],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("tokenizer pilot implementation revision is not an ancestor of HEAD")
    changed = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", f"{revision}..HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    runtime_roots = ("speck/", "scripts/")
    runtime_files = {"pyproject.toml", "uv.lock", "Makefile"}
    forbidden = [
        candidate
        for candidate in changed
        if candidate.startswith(runtime_roots) or candidate in runtime_files
    ]
    if forbidden:
        raise ValueError(
            "tokenizer pilot runtime changed after its implementation revision: "
            + ", ".join(forbidden)
        )


def load_execution_record(path):
    """Load one screen authority that wraps an immutable execution-blocked v2 run."""

    path = Path(path).resolve()
    value = json.loads(path.read_text())
    expected = {
        "format",
        "format_version",
        "status",
        "repository_revision",
        "parent_run",
        "qualifications",
        "implementation",
        "output_directory",
        "authority",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value["format"] != "speck_tokenizer_pilot_screen_execution"
        or value["format_version"] != 1
        or value["status"] != "screen_execution_authorized_not_started"
    ):
        raise ValueError("unsupported tokenizer pilot screen execution record")
    root = path.parents[3]
    parent_identity = _identity(value["parent_run"], root, "parent run")
    parent = load_pilot_run_manifest(parent_identity["path"])
    qualifications = value["qualifications"]
    if not isinstance(qualifications, dict) or set(qualifications) != {
        "runtime_inputs",
        "checkpoint_resume",
        "document_nll",
    }:
        raise ValueError("tokenizer pilot execution qualifications are incomplete")
    normalized_qualifications = {
        name: _identity(identity, root, f"execution qualification {name}")
        for name, identity in qualifications.items()
    }
    statuses = {
        "runtime_inputs": "real_v2_streams_evaluation_and_resume_inputs_pass_model_execution_blocked",
        "checkpoint_resume": "exact_shape_bf16_checkpoint_resume_within_frozen_policy",
        "document_nll": "six_category_cuda_chunk_parity_pass_execution_records_pending",
    }
    for name, identity in normalized_qualifications.items():
        result = json.loads(Path(identity["path"]).read_text())
        if result.get("status") != statuses[name]:
            raise ValueError(f"tokenizer pilot execution qualification is not passing: {name}")
    implementation = value["implementation"]
    if not isinstance(implementation, dict) or set(implementation) != {
        "orchestration",
        "trainer",
        "evaluation",
        "cli",
        "tests",
    }:
        raise ValueError("tokenizer pilot execution implementation is incomplete")
    normalized_implementation = {
        name: _identity(identity, root, f"execution implementation {name}")
        for name, identity in implementation.items()
    }
    if value["authority"] != {
        "screen_execution": True,
        "confirmation_execution": False,
        "D5_opening": False,
        "final_selection": False,
        "flagship_training": False,
    }:
        raise ValueError("tokenizer pilot execution authority is invalid")
    verify_execution_revision(root, value["repository_revision"])
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("tokenizer pilot execution requires a clean tracked tree")
    output = Path(value["output_directory"]).expanduser().resolve()
    return {
        **value,
        "execution_record": {"path": str(path), "sha256": file_sha256(path)},
        "parent_run": parent_identity,
        "qualifications": normalized_qualifications,
        "implementation": normalized_implementation,
        "output_directory": str(output),
        "run": parent,
    }


def orchestration_boundaries(run):
    """Return predeclared evaluation and checkpoint boundaries for a full run."""

    final = run["stops"]["final_step"]
    fixed = run["stops"]["fixed_document_step"]
    curve_every = run["settings"]["learning_curve_every_steps"]
    checkpoint_every = run["settings"]["checkpoint_every_steps"]
    curve = set(range(curve_every, fixed, curve_every)) | {fixed}
    evaluations = curve | {final}
    checkpoints = set(range(checkpoint_every, final, checkpoint_every)) | {fixed, final}
    return {
        "learning_curve_steps": tuple(sorted(curve)),
        "evaluation_steps": tuple(sorted(evaluations)),
        "checkpoint_steps": tuple(sorted(checkpoints)),
        "fixed_document_step": fixed,
        "final_step": final,
    }


def _document_bpb(document):
    return document["nll_nats"] / (math.log(2) * document["utf8_bytes"])


def macro_bpb(categories):
    """Return the unweighted mean of six category document-mean BPBs."""

    expected = ("web", "code", "math", "synthetic", "science", "reference")
    if set(categories) != set(expected) or any(not categories[name] for name in expected):
        raise ValueError("tokenizer pilot evaluation must contain six non-empty categories")
    means = [
        sum(_document_bpb(document) for document in categories[name]) / len(categories[name])
        for name in expected
    ]
    return sum(means) / len(means)


def build_completed_run_record(run, evaluations, timing, peak_memory_bytes):
    """Construct the exact schema consumed by the frozen seven-run analyzer."""

    boundaries = orchestration_boundaries(run)
    if set(evaluations) != set(boundaries["evaluation_steps"]):
        raise ValueError("completed tokenizer pilot run has incomplete evaluation boundaries")
    if set(timing) != set(boundaries["evaluation_steps"]):
        raise ValueError("completed tokenizer pilot run has incomplete timing boundaries")
    fixed_step = boundaries["fixed_document_step"]
    final_step = boundaries["final_step"]
    fixed_categories = evaluations[fixed_step]
    final_categories = evaluations[final_step]
    flops_per_token = run["model"]["analytic_training_flops_per_token"]
    correction = json.loads(Path(run["flop_correction"]["path"]).read_text())
    reference_tokens = correction["reference"]["fixed_document_tokens"]
    target_flops = correction["reference"]["analytic_flop_target"]

    def view(tokenizer_tokens, analytic_flops, step, categories):
        active = timing[step]
        aligned_tokens = step * run["settings"]["batch_tokens"]
        return {
            "mistral_reference_tokens": reference_tokens,
            "tokenizer_tokens": tokenizer_tokens,
            "analytic_flops": analytic_flops,
            "target_analytic_flops": target_flops,
            "active_seconds": active,
            "peak_memory_bytes": peak_memory_bytes,
            "throughput_tokens_per_second": aligned_tokens / active,
            "categories": categories,
        }

    curve = [
        {
            "analytic_flops": step * run["settings"]["batch_tokens"] * flops_per_token,
            "active_seconds": timing[step],
            "macro_bpb": macro_bpb(evaluations[step]),
        }
        for step in boundaries["learning_curve_steps"]
    ]
    return {
        "format": "speck_tokenizer_pilot_run",
        "format_version": 1,
        "status": "complete",
        "tokenizer_id": run["tokenizer_id"],
        "seed": run["seed"],
        "tokenizer_model_sha256": run["tokenizer"]["model"]["sha256"],
        "model_manifest_sha256": run["model"]["sha256"],
        "backbone_manifest_sha256": run["model"]["backbone_sha256"],
        "document_stream_sha256": run["training_data"]["document_stream_sha256"],
        "vocab_size": run["tokenizer"]["vocab_size"],
        "total_parameters": run["model"]["parameters"],
        "fixed_document": view(
            run["stops"]["fixed_document_tokens"],
            run["stops"]["fixed_document_tokens"] * flops_per_token,
            fixed_step,
            fixed_categories,
        ),
        "fixed_flop": view(
            run["stops"]["fixed_flop_token_stop"],
            target_flops,
            final_step,
            final_categories,
        ),
        "learning_curve": curve,
    }
