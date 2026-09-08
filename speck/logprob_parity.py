"""Validate and compare offline, hash-bound per-token log-probability records."""

import hashlib
import json
import math
from pathlib import Path

from speck.io import file_sha256

SHA256_LENGTH = 64
RECORD_FORMAT = "speck_per_token_logprob_record"
PLAN_FORMAT = "speck_cross_backend_logprob_parity_plan"


def canonical_json(value):
    """Return the canonical encoding used by record identity hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def value_sha256(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _load_object(path, description):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _exact_keys(value, expected, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"{context} has unsupported fields: {', '.join(extra)}")


def _sha256(value, context):
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")


def _nonempty_string(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")


def _finite_number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{context} must be finite") from error
    if not math.isfinite(converted):
        raise ValueError(f"{context} must be finite")
    return converted


def _finite_json(value, context):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains a non-finite value")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{context}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} contains a non-string object key")
            _finite_json(item, f"{context}.{key}")


def _validate_named_identity(value, context):
    _exact_keys(value, {"id", "revision", "artifact_sha256"}, context)
    _nonempty_string(value["id"], f"{context}.id")
    _nonempty_string(value["revision"], f"{context}.revision")
    _sha256(value["artifact_sha256"], f"{context}.artifact_sha256")


def _identity_projection(records):
    return [
        {
            "case_id": case["case_id"],
            "payload_sha256": case["payload_sha256"],
            "token_ids": [token["token_id"] for token in case["tokens"]],
        }
        for case in records
    ]


def validate_record_artifact(path, *, expected_role=None):
    """Load one local artifact and reject incomplete or unhashed token records."""

    artifact = _load_object(path, "log-probability artifact")
    _exact_keys(
        artifact,
        {
            "format",
            "format_version",
            "producer",
            "identity",
            "records",
            "records_sha256",
            "payload_token_ids_sha256",
        },
        "log-probability artifact",
    )
    if (
        artifact["format"] != RECORD_FORMAT
        or type(artifact["format_version"]) is not int
        or artifact["format_version"] != 1
    ):
        raise ValueError("log-probability artifact must use format version 1")

    producer = artifact["producer"]
    _exact_keys(
        producer,
        {"role", "backend", "revision", "dtype", "environment_sha256", "synthetic_fixture"},
        "producer",
    )
    _nonempty_string(producer["role"], "producer.role")
    if producer["role"] not in {"trusted_reference", "optimized_backend"}:
        raise ValueError("producer.role is not recognized")
    if expected_role is not None and producer["role"] != expected_role:
        raise ValueError(f"producer.role must be {expected_role}")
    for field in ("backend", "revision", "dtype"):
        _nonempty_string(producer[field], f"producer.{field}")
    _sha256(producer["environment_sha256"], "producer.environment_sha256")
    if not isinstance(producer["synthetic_fixture"], bool):
        raise ValueError("producer.synthetic_fixture must be a boolean")

    identity = artifact["identity"]
    _exact_keys(identity, {"evaluation", "model", "tokenizer"}, "identity")
    evaluation = identity["evaluation"]
    _exact_keys(evaluation, {"id", "manifest_sha256"}, "identity.evaluation")
    _nonempty_string(evaluation["id"], "identity.evaluation.id")
    _sha256(evaluation["manifest_sha256"], "identity.evaluation.manifest_sha256")
    _validate_named_identity(identity["model"], "identity.model")
    _validate_named_identity(identity["tokenizer"], "identity.tokenizer")

    records = artifact["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    case_ids = []
    for case_index, case in enumerate(records):
        context = f"records[{case_index}]"
        _exact_keys(case, {"case_id", "payload", "payload_sha256", "tokens"}, context)
        _nonempty_string(case["case_id"], f"{context}.case_id")
        case_ids.append(case["case_id"])
        _finite_json(case["payload"], f"{context}.payload")
        _sha256(case["payload_sha256"], f"{context}.payload_sha256")
        if value_sha256(case["payload"]) != case["payload_sha256"]:
            raise ValueError(f"{context} payload does not match payload_sha256")
        tokens = case["tokens"]
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"{context}.tokens must be a non-empty list")
        for position, token in enumerate(tokens):
            token_context = f"{context}.tokens[{position}]"
            _exact_keys(token, {"position", "token_id", "logprob"}, token_context)
            if (
                isinstance(token["position"], bool)
                or not isinstance(token["position"], int)
                or token["position"] != position
            ):
                raise ValueError(f"{token_context} is misaligned: position must equal {position}")
            token_id = token["token_id"]
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError(f"{token_context}.token_id must be a non-negative integer")
            _finite_number(token["logprob"], f"{token_context}.logprob")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("record case IDs must be unique")

    _sha256(artifact["records_sha256"], "records_sha256")
    if value_sha256(records) != artifact["records_sha256"]:
        raise ValueError("records do not match records_sha256")
    _sha256(artifact["payload_token_ids_sha256"], "payload_token_ids_sha256")
    if value_sha256(_identity_projection(records)) != artifact["payload_token_ids_sha256"]:
        raise ValueError("payload/token IDs do not match payload_token_ids_sha256")
    return artifact


def load_plan(path):
    """Load the non-authoritative threshold plan used for one comparison."""

    plan = _load_object(path, "log-probability parity plan")
    required = {"format", "format_version", "status", "thresholds"}
    missing = sorted(required - set(plan))
    if missing:
        raise ValueError(f"log-probability parity plan is missing: {', '.join(missing)}")
    if (
        plan["format"] != PLAN_FORMAT
        or type(plan["format_version"]) is not int
        or plan["format_version"] != 1
    ):
        raise ValueError("log-probability parity plan must use format version 1")
    _nonempty_string(plan["status"], "log-probability parity plan status")
    if "non_authoritative" not in plan["status"]:
        raise ValueError("pre-access log-probability parity plan must remain non-authoritative")
    thresholds = plan["thresholds"]
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("log-probability parity plan requires dtype thresholds")
    for dtype, threshold in thresholds.items():
        _nonempty_string(dtype, "threshold dtype")
        _exact_keys(threshold, {"max_absolute_error", "mean_absolute_error"}, f"{dtype} threshold")
        for name, value in threshold.items():
            if _finite_number(value, f"{dtype} {name}") < 0:
                raise ValueError(f"{dtype} {name} must be finite and non-negative")
    return plan


def _assert_aligned(reference, candidate):
    if candidate["identity"]["model"] != reference["identity"]["model"]:
        raise ValueError("candidate model identity does not exactly match the reference")
    if candidate["identity"]["tokenizer"] != reference["identity"]["tokenizer"]:
        raise ValueError("candidate tokenizer identity does not exactly match the reference")
    if candidate["identity"]["evaluation"] != reference["identity"]["evaluation"]:
        raise ValueError(
            "candidate evaluation manifest identity does not exactly match the reference"
        )
    reference_records = reference["records"]
    candidate_records = candidate["records"]
    if [case["case_id"] for case in candidate_records] != [
        case["case_id"] for case in reference_records
    ]:
        raise ValueError("candidate case IDs/order are misaligned with the reference")
    for reference_case, candidate_case in zip(reference_records, candidate_records, strict=True):
        if (
            candidate_case["payload"] != reference_case["payload"]
            or candidate_case["payload_sha256"] != reference_case["payload_sha256"]
        ):
            raise ValueError(
                f"candidate payload is misaligned for case {reference_case['case_id']}"
            )
        reference_ids = [token["token_id"] for token in reference_case["tokens"]]
        candidate_ids = [token["token_id"] for token in candidate_case["tokens"]]
        if candidate_ids != reference_ids:
            raise ValueError(
                f"candidate token IDs are misaligned for case {reference_case['case_id']}"
            )
    if candidate["payload_token_ids_sha256"] != reference["payload_token_ids_sha256"]:
        raise ValueError("candidate payload/token identity hash does not match the reference")


def compare_artifacts(plan_path, reference_path, candidate_paths):
    """Compare local candidates with one declared trusted reference without network access."""

    plan_path = Path(plan_path)
    reference_path = Path(reference_path)
    candidate_paths = [Path(path) for path in candidate_paths]
    if not candidate_paths:
        raise ValueError("at least one optimized backend artifact is required")
    plan = load_plan(plan_path)
    reference = validate_record_artifact(reference_path, expected_role="trusted_reference")
    comparisons = []
    seen = set()
    for candidate_path in candidate_paths:
        candidate = validate_record_artifact(candidate_path, expected_role="optimized_backend")
        _assert_aligned(reference, candidate)
        producer = candidate["producer"]
        key = (producer["backend"], producer["dtype"])
        if key in seen:
            raise ValueError(f"duplicate backend/dtype candidate: {key[0]}/{key[1]}")
        seen.add(key)
        if producer["dtype"] not in plan["thresholds"]:
            raise ValueError(f"no threshold is frozen for dtype {producer['dtype']}")
        errors = []
        for reference_case, candidate_case in zip(
            reference["records"], candidate["records"], strict=True
        ):
            for reference_token, candidate_token in zip(
                reference_case["tokens"], candidate_case["tokens"], strict=True
            ):
                error = abs(float(candidate_token["logprob"]) - float(reference_token["logprob"]))
                if not math.isfinite(error):
                    raise ValueError("log-probability error is non-finite")
                errors.append(error)
        maximum = max(errors)
        mean = math.fsum(errors) / len(errors)
        threshold = plan["thresholds"][producer["dtype"]]
        max_pass = maximum <= threshold["max_absolute_error"]
        mean_pass = mean <= threshold["mean_absolute_error"]
        comparisons.append(
            {
                "backend": producer["backend"],
                "revision": producer["revision"],
                "dtype": producer["dtype"],
                "synthetic_fixture": producer["synthetic_fixture"],
                "artifact_sha256": file_sha256(candidate_path),
                "case_count": len(candidate["records"]),
                "token_count": len(errors),
                "max_absolute_error": maximum,
                "mean_absolute_error": mean,
                "thresholds": threshold,
                "checks": {
                    "max_absolute_error_pass": max_pass,
                    "mean_absolute_error_pass": mean_pass,
                },
                "passed": max_pass and mean_pass,
            }
        )
    passed = all(comparison["passed"] for comparison in comparisons)
    return {
        "format": "speck_cross_backend_logprob_parity_report",
        "format_version": 1,
        "status": "passed_thresholds" if passed else "failed_thresholds",
        "evidence_authority": "pre_access_diagnostic_only",
        "plan_sha256": file_sha256(plan_path),
        "reference": {
            **reference["producer"],
            "artifact_sha256": file_sha256(reference_path),
        },
        "identity": {
            **reference["identity"],
            "payload_token_ids_sha256": reference["payload_token_ids_sha256"],
        },
        "comparisons": comparisons,
        "passed": passed,
        "network_access": {"attempted": False, "mode": "local_files_only"},
    }
