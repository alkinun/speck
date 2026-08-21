"""validate and initialize version three search studies."""

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from speck.architecture import ArchitectureConfig
from speck.dataset import default_data_dir, load_manifest, verify_shards
from speck.model import build_model
from speck.search.architecture_v3 import parameter_count, quantized_weight_bytes
from speck.search.artifacts import ArtifactStore, file_digest
from speck.search.segments import (
    load_document_index,
    load_segment_plan,
    validate_segment_plan,
)
from speck.search.spec_v3 import V3SearchSettings
from speck.search.study_v3 import V3Study
from speck.dataloader import manifest_fingerprint


required_partitions = ("train", "monitor", "promotion", "audit", "final")


def _output_text(value):
    return value.decode(errors="replace")


def git_state(repository=None):
    repository = Path(repository or Path(__file__).resolve().parents[2])

    def run(*arguments):
        result = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            check=False,
            cwd=repository,
        )
        if result.returncode:
            message = _output_text(result.stderr).strip()
            raise RuntimeError(f"git {' '.join(arguments)} failed: {message}")
        return result.stdout

    revision = _output_text(run("rev-parse", "HEAD")).strip()
    status = run("status", "--porcelain=v1", "-z")
    difference = run("diff", "--binary", "HEAD")
    untracked = run("ls-files", "--others", "--exclude-standard", "-z")
    fingerprint = hashlib.sha256(status + difference)
    for name in sorted(item for item in untracked.split(b"\0") if item):
        fingerprint.update(name)
        fingerprint.update((repository / os.fsdecode(name)).read_bytes())
    return {
        "dirty": bool(status),
        "revision": revision or None,
        "working_tree": fingerprint.hexdigest(),
    }


def runtime_environment():
    return {
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
    }


def _configuration_digests(experiment, config_path=None):
    experiment = Path(experiment)
    paths = {
        name: experiment / f"{name}.json"
        for name in ("data", "model", "tokenizer")
    }
    if config_path is not None:
        paths["search_v3"] = Path(config_path)
    return {
        name: file_digest(path)
        for name, path in paths.items()
        if path.is_file()
    }


def initialize_study(
    study_path,
    artifact_root,
    settings,
    *,
    experiment,
    model_settings,
    tokenizer_settings,
    data_settings,
    tokenizer,
    data_dir=None,
    config_path=None,
    captured_git=None,
    environment=None,
):
    if not isinstance(settings, V3SearchSettings):
        raise TypeError("v3 initialization needs v3 search settings")
    expected_plan_digest = settings.segment_plan.expected_digest
    if expected_plan_digest is None:
        raise ValueError("v3 initialization needs a frozen segment plan digest")
    data_dir = Path(
        data_dir
        or data_settings.get("output_dir")
        or default_data_dir / "packed"
    ).expanduser()
    manifest = load_manifest(data_dir)
    verify_shards(data_dir, manifest)
    tokenizer_digest = tokenizer.fingerprint()
    if manifest["tokenizer"]["fingerprint"] != tokenizer_digest:
        raise ValueError("search dataset and tokenizer do not match")
    dataset_digest = manifest_fingerprint(manifest)
    plan = load_segment_plan(settings.segment_plan.path)
    if plan.digest != expected_plan_digest:
        raise ValueError("segment plan digest does not match the configuration")
    if plan.dataset_digest != dataset_digest:
        raise ValueError("segment plan dataset does not match the packed dataset")
    records = load_document_index(data_dir, manifest)
    validate_segment_plan(plan, records, required_partitions)
    protocol = settings.quality.resolve(
        dataset_digest,
        tokenizer_digest,
        plan.digest,
    )
    partitions = {partition.name: partition for partition in plan.partitions}
    if partitions["train"].tokens < protocol.target_tokens + 1:
        raise ValueError("training segment is shorter than the quality trajectory")

    with torch.device("meta"):
        model = build_model(
            model_settings,
            tokenizer.vocab_size,
            tokenizer.bos_id,
            tokenizer.eos_id,
        )
    baseline = (
        model.config
        if isinstance(model.config, ArchitectureConfig)
        else ArchitectureConfig.from_v2(model.config)
    )
    if protocol.sequence_length > baseline.max_position_embeddings:
        raise ValueError("quality sequence exceeds the baseline context")
    if any(
        profile.prompt_tokens + profile.generated_tokens
        > baseline.max_position_embeddings
        for profile in settings.profiles
    ):
        raise ValueError("profile request exceeds the baseline context")
    parameters = parameter_count(baseline)
    if parameters != model.parameter_count():
        raise ValueError("v3 static parameter accounting does not match the baseline")
    static = {
        "logical_depth": baseline.logical_depth,
        "parameters": parameters,
        "q4_weight_bytes": quantized_weight_bytes(baseline),
        "unique_parameter_blocks": baseline.unique_parameter_blocks,
    }
    provenance = {
        "configuration_digests": _configuration_digests(experiment, config_path),
        "dataset_dir": str(data_dir.resolve()),
        "dataset_manifest": dataset_digest,
        "environment": environment or runtime_environment(),
        "experiment": str(Path(experiment).resolve()),
        "git": captured_git or git_state(),
        "model_digest": baseline.digest,
        "resolved_protocol": asdict(protocol),
        "resolved_protocol_digest": protocol.digest,
        "segment_plan": {
            "digest": plan.digest,
            "path": str(Path(settings.segment_plan.path).resolve()),
        },
        "tokenizer": tokenizer_settings,
        "tokenizer_fingerprint": tokenizer_digest,
    }
    artifacts = ArtifactStore(artifact_root)
    segment_artifact = artifacts.put_json("segment_plan", plan.export())
    if segment_artifact.digest != plan.digest:
        raise RuntimeError("segment plan artifact identity changed")
    study = V3Study(study_path)
    try:
        initialized = study.initialize_bundle(
            settings.export(),
            provenance,
            objective_sets=settings.objective_sets,
            architecture=baseline,
            static=static,
            operation={"operator": "baseline"},
            artifacts=(segment_artifact,),
        )
    finally:
        study.close()
    return {
        "architecture_digest": baseline.digest,
        "dataset_digest": dataset_digest,
        "initialized": initialized,
        "protocol_digest": protocol.digest,
        "segment_plan_digest": plan.digest,
    }
