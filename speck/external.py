"""Validate and source-qualify pinned external evaluation suites."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load external suite config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"external suite config must contain an object: {path}")
    return value


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(value, keys, context):
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")


def _validate_source(source, context):
    _require(
        source,
        {"id", "repository", "revision", "ref", "license", "required_files"},
        context,
    )
    if not source["repository"].startswith(("https://github.com/", "https://huggingface.co/")):
        raise ValueError(f"{context} repository is not an approved upstream URL")
    if not COMMIT_PATTERN.fullmatch(source["revision"]):
        raise ValueError(f"{context} revision must be a full commit")
    if not source["required_files"]:
        raise ValueError(f"{context} requires source-file pins")
    paths = []
    for entry in source["required_files"]:
        _require(entry, {"path", "sha256"}, f"{context} file pin")
        if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts:
            raise ValueError(f"{context} file pin must stay inside its checkout")
        if not SHA256_PATTERN.fullmatch(entry["sha256"]):
            raise ValueError(f"{context} file pin has an invalid SHA-256")
        paths.append(entry["path"])
    if len(set(paths)) != len(paths):
        raise ValueError(f"{context} file pins contain duplicate paths")


def validate_external_suite(path):
    """Validate one checked external-suite contract without accessing its checkout."""

    path = Path(path).expanduser().resolve()
    config = _load_json(path)
    _require(
        config,
        {
            "format",
            "format_version",
            "suite_id",
            "suite_version",
            "status",
            "upstream",
            "dependencies",
            "benchmark",
            "data",
            "model_adapter",
            "release_use",
        },
        "external suite",
    )
    if config["format"] != "speck_external_evaluation_suite" or config["format_version"] != 1:
        raise ValueError("external suite must use format version 1")
    if config["suite_id"] not in {"ruler", "nolima", "helmet"}:
        raise ValueError("external suite id is not recognized")
    if "blocked" not in config["status"]:
        raise ValueError("unexecuted external suites must preserve their blocker status")
    sources = [config["upstream"], *config["dependencies"]]
    identifiers = [source.get("id") for source in sources]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("external suite source ids must be unique")
    for index, source in enumerate(sources):
        _validate_source(source, f"external source {index}")
    lengths = config["benchmark"].get("lengths")
    if (
        not isinstance(lengths, list)
        or not lengths
        or lengths != sorted(set(lengths))
        or any(
            isinstance(length, bool) or not isinstance(length, int) or length < 1
            for length in lengths
        )
    ):
        raise ValueError("external suite lengths must be sorted unique positive integers")
    data_status = config["data"].get("status", "")
    if "blocked" not in data_status and "qualified" not in data_status:
        raise ValueError("external suite data must state qualification or an unresolved blocker")
    if "blocked" not in config["model_adapter"].get("status", ""):
        raise ValueError("external suite model_adapter must state its unresolved blocker")
    data = config["data"]
    if config["suite_id"] == "ruler" and data["status"].startswith("source_bundle_qualified"):
        _require(
            data,
            {
                "source_manifest",
                "source_manifest_sha256",
                "bundle_identity_sha256",
                "generator_revision",
                "needle_repository_revision",
                "environment_group",
                "source_assets",
                "licenses",
            },
            "qualified RULER source bundle",
        )
        repository_root = path.parents[3]
        source_manifest_path = repository_root / data["source_manifest"]
        if (
            not source_manifest_path.is_file()
            or _file_sha256(source_manifest_path) != data["source_manifest_sha256"]
        ):
            raise ValueError("RULER source manifest does not match its pin")
        source_manifest = _load_json(source_manifest_path)
        asset_ids = {asset.get("id") for asset in source_manifest.get("assets", ())}
        if (
            source_manifest.get("format") != "speck_ruler_source_manifest"
            or source_manifest.get("status") != "offline_sources_complete_unredistributable"
            or source_manifest.get("bundle_identity_sha256") != data["bundle_identity_sha256"]
            or source_manifest.get("generator", {}).get("revision") != data["generator_revision"]
            or source_manifest.get("needle_repository", {}).get("revision")
            != data["needle_repository_revision"]
            or len(source_manifest.get("paul_graham_sources", ()))
            != data["source_assets"]["paul_graham_source_records"]
            or asset_ids != {"english_words", "squad_v2_dev", "hotpotqa_dev_distractor"}
            or source_manifest.get("paul_graham_output", {}).get("sha256")
            != data["source_assets"]["paul_graham_essays_sha256"]
        ):
            raise ValueError("RULER source manifest is invalid")
        manifest_assets = {asset["id"]: asset["sha256"] for asset in source_manifest["assets"]}
        expected_assets = {
            "english_words": data["source_assets"]["english_words_sha256"],
            "squad_v2_dev": data["source_assets"]["squad_v2_dev_sha256"],
            "hotpotqa_dev_distractor": data["source_assets"]["hotpotqa_dev_distractor_sha256"],
        }
        if manifest_assets != expected_assets:
            raise ValueError("RULER source asset hashes do not match")
        case_generation = data.get("case_generation")
        if case_generation is not None:
            _require(
                case_generation,
                {"status", "compatibility_patch", "qualified_lengths", "remaining_lengths"},
                "RULER case generation",
            )
            if case_generation["remaining_lengths"]:
                if "blocked" not in case_generation["status"]:
                    raise ValueError("partial RULER case generation must preserve its blocker status")
            elif case_generation["status"] != "qualified_all_declared_lengths":
                raise ValueError("complete RULER case generation must state full qualification")
            patch = case_generation["compatibility_patch"]
            _require(
                patch,
                {"path", "sha256", "target", "upstream_sha256", "patched_sha256"},
                "RULER compatibility patch",
            )
            patch_path = repository_root / patch["path"]
            if (
                not patch_path.is_file()
                or _file_sha256(patch_path) != patch["sha256"]
                or not all(
                    SHA256_PATTERN.fullmatch(patch[field])
                    for field in ("sha256", "upstream_sha256", "patched_sha256")
                )
            ):
                raise ValueError("RULER compatibility patch does not match its pin")
            qualifications = case_generation["qualified_lengths"]
            qualified_lengths = [entry.get("length") for entry in qualifications]
            remaining_lengths = case_generation["remaining_lengths"]
            if (
                not qualifications
                or qualified_lengths != sorted(set(qualified_lengths))
                or remaining_lengths != sorted(set(remaining_lengths))
                or set(qualified_lengths).intersection(remaining_lengths)
                or sorted((*qualified_lengths, *remaining_lengths)) != lengths
            ):
                raise ValueError("RULER qualified and remaining lengths do not partition the suite")
            scorer_sha256 = next(
                entry["sha256"]
                for dependency in config["dependencies"]
                if dependency["id"] == "nemo_skills"
                for entry in dependency["required_files"]
                if entry["path"] == "nemo_skills/dataset/ruler/ruler_score.py"
            )
            for qualification in qualifications:
                _require(
                    qualification,
                    {
                        "length",
                        "report",
                        "report_sha256",
                        "case_identity_sha256",
                        "tokenizer_identity_sha256",
                        "total_cases",
                    },
                    "RULER length qualification",
                )
                report_path = repository_root / qualification["report"]
                if (
                    not report_path.is_file()
                    or _file_sha256(report_path) != qualification["report_sha256"]
                ):
                    raise ValueError("RULER case qualification report does not match its pin")
                report = _load_json(report_path)
                case_tasks = [entry.get("task") for entry in report.get("cases", ())]
                if (
                    report.get("format") != "speck_ruler_case_qualification"
                    or report.get("status") != "qualified_offline_deterministic_case_stream"
                    or report.get("length") != qualification["length"]
                    or report.get("tasks") != config["benchmark"]["tasks"]
                    or case_tasks != config["benchmark"]["tasks"]
                    or report.get("samples_per_task")
                    != config["benchmark"]["samples_per_task_length"]
                    or report.get("total_cases") != qualification["total_cases"]
                    or any(
                        entry.get("rows") != config["benchmark"]["samples_per_task_length"]
                        or entry.get("maximum_accounted_length", qualification["length"] + 1)
                        > qualification["length"]
                        for entry in report.get("cases", ())
                    )
                    or report.get("case_identity_sha256")
                    != qualification["case_identity_sha256"]
                    or report.get("tokenizer", {}).get("identity_sha256")
                    != qualification["tokenizer_identity_sha256"]
                    or report.get("source_bundle", {}).get("identity_sha256")
                    != data["bundle_identity_sha256"]
                    or report.get("generator", {}).get("revision") != data["generator_revision"]
                    or report.get("generator", {}).get("compatibility_patch", {}).get("sha256")
                    != patch["sha256"]
                    or report.get("nemo_skills", {}).get("scorer_sha256") != scorer_sha256
                    or report.get("network_denial", {}).get("generation_attempts") != 0
                    or report.get("network_denial", {}).get("self_test") != "denied_as_expected"
                    or report.get("determinism", {}).get("complete_generations", 0) < 2
                    or not report.get("determinism", {}).get("all_task_hashes_equal")
                ):
                    raise ValueError("RULER case qualification report is invalid")
    if config["suite_id"] == "nolima" and data["status"].startswith(
        "metadata_and_license_audited"
    ):
        _require(
            data,
            {
                "license_decision",
                "license_decision_sha256",
                "license_audit_runner_revision",
                "dataset_license_sha256",
                "haystack_licenses_sha256",
                "metadata_audit",
            },
            "audited NoLiMa license decision",
        )
        repository_root = path.parents[3]
        decision_path = repository_root / data["license_decision"]
        if (
            not decision_path.is_file()
            or _file_sha256(decision_path) != data["license_decision_sha256"]
            or not SHA256_PATTERN.fullmatch(data["license_decision_sha256"])
            or not COMMIT_PATTERN.fullmatch(data["license_audit_runner_revision"])
            or not SHA256_PATTERN.fullmatch(data["dataset_license_sha256"])
            or not SHA256_PATTERN.fullmatch(data["haystack_licenses_sha256"])
        ):
            raise ValueError("NoLiMa license decision does not match its pin")
        decision = _load_json(decision_path)
        cache = decision.get("cache_audit", {})
        metadata = data["metadata_audit"]
        if (
            decision.get("format") != "speck_nolima_license_decision"
            or decision.get("status")
            != "metadata_and_license_audited_authorized_acceptance_blocked"
            or decision.get("runner_revision") != data["license_audit_runner_revision"]
            or decision.get("dataset", {}).get("revision") != data["revision"]
            or decision.get("dataset", {}).get("restricted_payload_bytes")
            != metadata["restricted_payload_bytes"]
            or len(decision.get("dataset", {}).get("restricted_payloads", ()))
            != metadata["restricted_payloads"]
            or decision.get("dataset", {})
            .get("optional_lfs_book_archive", {})
            .get("oid_sha256")
            != metadata["optional_book_archive_lfs_sha256"]
            or decision.get("dataset", {})
            .get("optional_lfs_book_archive", {})
            .get("bytes")
            != metadata["optional_book_archive_bytes"]
            or decision.get("adobe_research_license", {}).get("sha256")
            != data["dataset_license_sha256"]
            or decision.get("haystack_rights", {}).get("source_file_sha256")
            != data["haystack_licenses_sha256"]
            or cache.get("restricted_blobs_cached") != metadata["restricted_blobs_cached"]
            or cache.get("working_tree_materialized")
            != metadata["working_tree_materialized"]
            or cache.get("new_objects_fetched") != metadata["new_objects_fetched"]
            or decision.get("decision", {}).get("download_or_use_authorized") is not False
            or decision.get("decision", {}).get("raw_payload_git_redistribution_authorized")
            is not False
        ):
            raise ValueError("NoLiMa license decision artifact is invalid")
    adapter = config["model_adapter"]
    if adapter["status"].startswith("endpoint_protocol_qualified"):
        _require(
            adapter,
            {"qualification", "qualification_sha256", "runner_revision", "settings"},
            "qualified external model adapter",
        )
        if not SHA256_PATTERN.fullmatch(adapter["qualification_sha256"]):
            raise ValueError("external model-adapter qualification has an invalid SHA-256")
        if not COMMIT_PATTERN.fullmatch(adapter["runner_revision"]):
            raise ValueError("external model-adapter qualification has an invalid revision")
        repository_root = path.parents[3]
        qualification_path = repository_root / adapter["qualification"]
        if (
            not qualification_path.is_file()
            or _file_sha256(qualification_path) != adapter["qualification_sha256"]
        ):
            raise ValueError("external model-adapter qualification artifact does not match")
        qualification = _load_json(qualification_path)
        if (
            qualification.get("format") != "speck_evaluation_endpoint_qualification"
            or qualification.get("status")
            != "qualified_for_serialized_openai_correctness_evaluation"
            or qualification.get("runner_revision") != adapter["runner_revision"]
            or config["suite_id"] not in qualification.get("qualified_consumers", {})
            or not qualification.get("export", {}).get("parity", {}).get("passed")
            or qualification.get("export", {}).get("maximum_context", 0) < min(lengths)
        ):
            raise ValueError("external model-adapter qualification artifact is invalid")
    elif config["suite_id"] == "helmet" and adapter["status"].startswith(
        "native_hf_cpu_eager_qualified"
    ):
        _require(
            adapter,
            {
                "qualification",
                "qualification_sha256",
                "runner_revision",
                "environment_group",
                "export_identity_sha256",
                "settings",
                "scoring_scope",
            },
            "qualified HELMET model adapter",
        )
        repository_root = path.parents[3]
        qualification_path = repository_root / adapter["qualification"]
        if (
            not SHA256_PATTERN.fullmatch(adapter["qualification_sha256"])
            or not COMMIT_PATTERN.fullmatch(adapter["runner_revision"])
            or not SHA256_PATTERN.fullmatch(adapter["export_identity_sha256"])
            or not qualification_path.is_file()
            or _file_sha256(qualification_path) != adapter["qualification_sha256"]
        ):
            raise ValueError("HELMET model-adapter qualification does not match its pin")
        qualification = _load_json(qualification_path)
        result = qualification.get("results", {})
        model = result.get("model", {})
        scoring = result.get("scoring", {})
        settings = adapter["settings"]
        if (
            qualification.get("format") != "speck_helmet_adapter_qualification"
            or qualification.get("status") != "qualified_native_hf_cpu_eager_adapter"
            or qualification.get("runner_revision") != adapter["runner_revision"]
            or qualification.get("helmet", {}).get("revision")
            != config["upstream"]["revision"]
            or qualification.get("export", {}).get("identity_sha256")
            != adapter["export_identity_sha256"]
            or not qualification.get("export", {}).get("parity_passed")
            or qualification.get("runtime", {}).get("environment_group")
            != adapter["environment_group"]
            or model.get("device") != settings["device"]
            or model.get("dtype") != settings["dtype"]
            or model.get("attention_implementation") != settings["attn_implementation"]
            or model.get("compiled") != settings["torch_compile"]
            or model.get("device_map") != settings["device_map"]
            or result.get("tokenizer", {}).get("padding_side") != "left"
            or result.get("tokenizer", {}).get("truncation_side") != "left"
            or result.get("prepared_input", {}).get("maximum_tokens_after_generation_reserve")
            != settings["input_max_length"] - settings["generation_max_length"]
            or not result.get("truncation", {}).get("retained_prefix")
            or result.get("truncation", {}).get("prepared_tokens", settings["input_max_length"] + 1)
            > settings["input_max_length"] - settings["generation_max_length"]
            or not result.get("generation", {}).get("repeated_raw_outputs_identical")
            or scoring.get("ruler_full_recall") != 1.0
            or scoring.get("ruler_partial_recall") != 0.5
            or qualification.get("network_denial", {}).get("qualification_attempts") != 0
            or qualification.get("network_denial", {}).get("self_test")
            != "denied_as_expected"
        ):
            raise ValueError("HELMET model-adapter qualification artifact is invalid")
        if "scorer_runtime_qualified" in adapter["status"]:
            _require(adapter, {"scorer_runtime"}, "qualified HELMET scorer runtime")
            scorer_runtime = adapter["scorer_runtime"]
            _require(
                scorer_runtime,
                {
                    "qualification",
                    "qualification_sha256",
                    "runner_revision",
                    "wheel_sha256",
                    "runtime_identity_sha256",
                    "platform",
                },
                "qualified HELMET scorer runtime",
            )
            scorer_path = repository_root / scorer_runtime["qualification"]
            if (
                not scorer_path.is_file()
                or _file_sha256(scorer_path) != scorer_runtime["qualification_sha256"]
                or not COMMIT_PATTERN.fullmatch(scorer_runtime["runner_revision"])
                or not all(
                    SHA256_PATTERN.fullmatch(scorer_runtime[field])
                    for field in (
                        "qualification_sha256",
                        "wheel_sha256",
                        "runtime_identity_sha256",
                    )
                )
            ):
                raise ValueError("HELMET scorer runtime does not match its pin")
            scorer = _load_json(scorer_path)
            build = scorer.get("build", {})
            metrics = scorer.get("metrics", {}).get("metrics", {})
            if (
                scorer.get("format") != "speck_helmet_scorer_runtime_manifest"
                or scorer.get("status")
                != "qualified_offline_platform_specific_local_runtime_unredistributable"
                or scorer.get("runner_revision") != scorer_runtime["runner_revision"]
                or scorer.get("helmet", {}).get("revision") != config["upstream"]["revision"]
                or build.get("wheel", {}).get("sha256") != scorer_runtime["wheel_sha256"]
                or build.get("complete_rebuilds") != 2
                or not build.get("wheels_identical")
                or scorer.get("runtime", {}).get("identity_sha256")
                != scorer_runtime["runtime_identity_sha256"]
                or metrics.get("P@1") != 0.5
                or metrics.get("Recall@3") != 1.0
                or metrics.get("MRR") != 0.75
                or scorer.get("network_denial", {}).get("build_attempts") != 0
                or scorer.get("network_denial", {}).get("metric_attempts") != 0
            ):
                raise ValueError("HELMET scorer runtime qualification is invalid")
    return config


def _git_output(directory, *args):
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def qualify_external_suite(path, checkouts):
    """Verify exact commits and source-file hashes for one external-suite checkout set."""

    config = validate_external_suite(path)
    sources = [config["upstream"], *config["dependencies"]]
    if set(checkouts) != {source["id"] for source in sources}:
        raise ValueError("checkout ids must exactly match the suite's upstream and dependencies")
    qualified = []
    for source in sources:
        directory = Path(checkouts[source["id"]]).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"external checkout does not exist: {directory}")
        revision = _git_output(directory, "rev-parse", "HEAD")
        if revision != source["revision"]:
            raise ValueError(
                f"external checkout {source['id']} is at {revision}, expected {source['revision']}"
            )
        file_results = []
        for entry in source["required_files"]:
            file_path = directory / entry["path"]
            if not file_path.is_file() or _file_sha256(file_path) != entry["sha256"]:
                raise ValueError(
                    f"external checkout {source['id']} file pin failed: {entry['path']}"
                )
            file_results.append(entry)
        qualified.append(
            {
                "id": source["id"],
                "repository": source["repository"],
                "revision": revision,
                "required_files": file_results,
            }
        )
    return {
        "format": "speck_external_suite_source_qualification",
        "format_version": 1,
        "suite_id": config["suite_id"],
        "suite_version": config["suite_version"],
        "config_path": str(Path(path).expanduser().resolve()),
        "config_sha256": _file_sha256(Path(path).expanduser().resolve()),
        "status": "source_qualified",
        "sources": qualified,
        "remaining_data_status": config["data"]["status"],
        "remaining_model_adapter_status": config["model_adapter"]["status"],
    }
