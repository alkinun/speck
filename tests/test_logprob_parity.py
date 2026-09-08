import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.logprob_parity import main
from speck.logprob_parity import (
    compare_artifacts,
    file_sha256,
    validate_record_artifact,
    value_sha256,
)

root = Path(__file__).parents[1]
plan = root / "research/flagship/logprob_parity_plan.json"
fixtures = root / "tests/fixtures/logprob_parity"
reference = fixtures / "reference.json"
candidates = [fixtures / "vllm.json", fixtures / "sglang.json"]
checked_report = root / "results/evaluation/logprob-parity-fixture-20260908.json"
fixture_manifest = fixtures / "evaluation_manifest.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rehash(artifact):
    artifact["records_sha256"] = value_sha256(artifact["records"])
    artifact["payload_token_ids_sha256"] = value_sha256(
        [
            {
                "case_id": case["case_id"],
                "payload_sha256": case["payload_sha256"],
                "token_ids": [token["token_id"] for token in case["tokens"]],
            }
            for case in artifact["records"]
        ]
    )


def test_checked_synthetic_artifacts_are_deterministic_and_pass():
    first = compare_artifacts(plan, reference, candidates)
    second = compare_artifacts(plan, reference, candidates)

    assert first == second
    assert first["status"] == "passed_thresholds"
    assert first["passed"] is True
    assert first["network_access"] == {"attempted": False, "mode": "local_files_only"}
    assert [entry["backend"] for entry in first["comparisons"]] == ["vllm", "sglang"]
    assert [entry["dtype"] for entry in first["comparisons"]] == ["bfloat16", "float16"]
    assert all(entry["passed"] for entry in first["comparisons"])
    assert all(entry["synthetic_fixture"] for entry in first["comparisons"])
    assert first["identity"]["evaluation"]["manifest_sha256"] == file_sha256(fixture_manifest)
    assert first == _load(checked_report)


def test_unhashed_record_edit_is_rejected(tmp_path):
    value = _load(candidates[0])
    value["records"][0]["tokens"][0]["logprob"] = -0.5
    path = tmp_path / "tampered.json"
    _write(path, value)

    with pytest.raises(ValueError, match="records do not match records_sha256"):
        validate_record_artifact(path)


@pytest.mark.parametrize(
    "mutation,message",
    (
        (lambda value: value["records"][0]["tokens"][0].pop("logprob"), "missing.*logprob"),
        (
            lambda value: value["records"][0]["tokens"][0].update(position=1),
            "misaligned",
        ),
        (
            lambda value: value["records"][0]["tokens"][0].update(logprob=float("inf")),
            "finite",
        ),
    ),
)
def test_record_validation_rejects_missing_nonfinite_and_position_misalignment(
    tmp_path, mutation, message
):
    value = _load(reference)
    mutation(value)
    path = tmp_path / "invalid.json"
    _write(path, value)

    with pytest.raises(ValueError, match=message):
        validate_record_artifact(path)


@pytest.mark.parametrize("field", ("model", "tokenizer", "evaluation"))
def test_comparison_requires_exact_top_level_identity(tmp_path, field):
    value = _load(candidates[0])
    identity = value["identity"][field]
    changed = next(name for name in identity if name.endswith("sha256"))
    identity[changed] = "0" * 64
    path = tmp_path / "candidate.json"
    _write(path, value)

    with pytest.raises(ValueError, match=field):
        compare_artifacts(plan, reference, [path])


def test_comparison_rejects_payload_and_token_id_misalignment(tmp_path):
    payload_value = _load(candidates[0])
    case = payload_value["records"][0]
    case["payload"]["body"]["prompt"] = "different"
    case["payload_sha256"] = value_sha256(case["payload"])
    _rehash(payload_value)
    payload_path = tmp_path / "payload.json"
    _write(payload_path, payload_value)
    with pytest.raises(ValueError, match="payload is misaligned"):
        compare_artifacts(plan, reference, [payload_path])

    token_value = _load(candidates[0])
    token_value["records"][0]["tokens"][0]["token_id"] += 1
    _rehash(token_value)
    token_path = tmp_path / "token.json"
    _write(token_path, token_value)
    with pytest.raises(ValueError, match="token IDs are misaligned"):
        compare_artifacts(plan, reference, [token_path])


def test_comparison_reports_threshold_failure_without_reclassifying_validation(tmp_path):
    value = deepcopy(_load(candidates[0]))
    value["records"][0]["tokens"][0]["logprob"] = -0.5
    _rehash(value)
    path = tmp_path / "numerically-divergent.json"
    _write(path, value)

    report = compare_artifacts(plan, reference, [path])

    assert report["status"] == "failed_thresholds"
    assert report["passed"] is False
    assert report["comparisons"][0]["checks"] == {
        "max_absolute_error_pass": False,
        "mean_absolute_error_pass": False,
    }


def test_cli_refuses_to_overwrite_historical_report(tmp_path):
    output = tmp_path / "report.json"
    arguments = [
        str(plan),
        str(reference),
        *(str(path) for path in candidates),
        "--output",
        str(output),
    ]
    main(arguments)
    original = output.read_bytes()

    with pytest.raises(SystemExit, match="File exists"):
        main(arguments)
    assert output.read_bytes() == original
