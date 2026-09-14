import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from speck.provenance.io import file_sha256
from speck.tokenization.pilot_screen import (
    analyze_tokenizer_screen,
    load_screen_report,
    validate_screen_plan,
)
from tests.tokenization.test_tokenizer_pilot import _matrix, _nominations

ROOT = Path(__file__).resolve().parents[2]


def inputs():
    plan = json.loads((ROOT / "research/flagship/tokenizer_pilot_plan_v10.json").read_text())
    nominations = _nominations()
    nominations["status"] = "real_static_endpoints_nominated_for_lm_pilot_no_selection_authority"
    runs = copy.deepcopy(_matrix()[:3])
    horizon = plan["stopping"]["fixed_document_mistral_tokens"]
    for run in runs:
        for view in ("fixed_document", "fixed_flop"):
            run[view]["mistral_reference_tokens"] = horizon
            run[view]["active_seconds"] = 3600
        for index, point in enumerate(run["learning_curve"]):
            point["active_seconds"] = (index + 1) * 1800
    runs[0]["fixed_document"]["tokenizer_tokens"] = horizon
    accounting = {
        "format": "speck_tokenizer_pilot_screen_accounting",
        "format_version": 1,
        "complete": True,
        "other_spent_gpu_hours": 2,
        "runs": [
            {
                "tokenizer_id": run["tokenizer_id"],
                "spent_gpu_hours": 2,
                "confirmation_run_gpu_hours": 3,
            }
            for run in runs
        ],
    }
    return plan, nominations, runs, accounting


def test_measured_projection_counts_losing_arm_and_all_attempts_without_double_counting():
    result = analyze_tokenizer_screen(*inputs())

    assert result["screen"]["selected_custom"] == "custom-large"
    assert result["budget"] == {
        "spent_gpu_hours": 8,
        "projected_four_confirmation_gpu_hours": 12,
        "projected_total_gpu_hours": 20,
        "ceiling_gpu_hours": 30,
        "headroom_gpu_hours": 10,
        "fits": True,
    }
    assert not any(result["authority"].values())


@pytest.mark.parametrize("extra, fits", [(12, True), (12.000001, False)])
def test_ceiling_is_inclusive_and_excess_keeps_fallback(extra, fits):
    plan, nominations, runs, accounting = inputs()
    accounting["other_spent_gpu_hours"] = extra
    result = analyze_tokenizer_screen(plan, nominations, runs, accounting)

    assert result["budget"]["fits"] is fits
    assert result["fallback"] == "mistral-32k"
    assert not any(result["authority"].values())
    if not fits:
        assert result["status"] == "measured_budget_exceeded_retain_mistral_D5_unopened"


def test_expensive_winner_does_not_get_replaced_with_cheaper_loser():
    plan, nominations, runs, accounting = inputs()
    accounting["runs"][1]["confirmation_run_gpu_hours"] = 20
    result = analyze_tokenizer_screen(plan, nominations, runs, accounting)

    assert result["screen"]["selected_custom"] == "custom-large"
    assert result["budget"]["fits"] is False


def test_missing_accounting_blocks_projection_instead_of_using_active_time_as_total():
    plan, nominations, runs, accounting = inputs()
    accounting["complete"] = False
    accounting["runs"] = []
    result = analyze_tokenizer_screen(plan, nominations, runs, accounting)

    assert result["status"] == "accounting_incomplete_confirmation_blocked"
    assert result["budget"] is None
    assert result["projection_lower_bound"]["projected_total_gpu_hours"] == 7
    assert result["projection_lower_bound"]["exceeds_ceiling"] is False


def test_incomplete_spending_can_prove_excess_but_cannot_prove_a_pass():
    plan, nominations, runs, accounting = inputs()
    accounting.update(complete=False, runs=[], other_spent_gpu_hours=None)
    for run in runs:
        for view in ("fixed_document", "fixed_flop"):
            run[view]["active_seconds"] = 5 * 3600
    result = analyze_tokenizer_screen(plan, nominations, runs, accounting)
    assert result["status"] == "projection_lower_bound_exceeded_retain_mistral_D5_unopened"
    assert result["projection_lower_bound"]["projected_total_gpu_hours"] == 35
    assert result["projection_lower_bound"]["all_attempt_spending_complete"] is False
    assert result["budget"] is None
    assert not any(result["authority"].values())


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, None, 0.5])
@pytest.mark.parametrize("field", ["spent_gpu_hours", "confirmation_run_gpu_hours"])
def test_invalid_or_underreported_cost_is_rejected(value, field):
    plan, nominations, runs, accounting = inputs()
    accounting["runs"][0][field] = value
    with pytest.raises(ValueError, match="GPU-hours|omits measured"):
        analyze_tokenizer_screen(plan, nominations, runs, accounting)


@pytest.mark.parametrize("change", ["missing", "duplicate", "seed", "stream", "flops", "documents"])
def test_incomplete_or_unmatched_screen_is_rejected(change):
    plan, nominations, runs, accounting = inputs()
    if change == "missing":
        runs.pop()
    elif change == "duplicate":
        runs[2] = runs[1]
    elif change == "seed":
        runs[2]["seed"] = 43
    elif change == "stream":
        runs[2]["document_stream_sha256"] = "different"
    elif change == "flops":
        runs[2]["fixed_flop"]["analytic_flops"] += 1
    else:
        runs[2]["fixed_flop"]["categories"]["web"][0]["utf8_bytes"] += 1
        # Keep the curve consistent so the document-pairing gate is exercised.
        from speck.tokenization.pilot import _macro

        runs[2]["learning_curve"][-1]["macro_bpb"] = _macro(runs[2], "fixed_document")
    with pytest.raises(ValueError):
        analyze_tokenizer_screen(plan, nominations, runs, accounting)


def test_tie_uses_document_time_then_vocabulary():
    plan, nominations, runs, accounting = inputs()
    runs[2]["fixed_document"] = copy.deepcopy(runs[1]["fixed_document"])
    runs[2]["learning_curve"] = copy.deepcopy(runs[1]["learning_curve"])
    result = analyze_tokenizer_screen(plan, nominations, runs, accounting)
    assert result["screen"]["selected_custom"] == "custom-small"

    runs[1]["fixed_document"]["active_seconds"] -= 1
    result = analyze_tokenizer_screen(plan, nominations, runs, accounting)
    assert result["screen"]["selected_custom"] == "custom-large"


def write_review(tmp_path):
    plan, nominations, runs, accounting = inputs()

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": name, "sha256": file_sha256(path)}

    summaries = []
    for run in runs:
        key = run["tokenizer_id"]
        summary = {
            "format": "speck_tokenizer_pilot_run_summary",
            "format_version": 1,
            "status": "complete",
            "run_id": f"tokenizer-pilot-{key}-seed-42",
            "D5_opening": False,
            "active_seconds": run["fixed_flop"]["active_seconds"],
            "result": write(f"{key}-result.json", run),
        }
        summaries.append(write(f"{key}-summary.json", summary))
    accounting["evidence"] = [write("timing.json", {"fixture": True})]
    review = {
        "format": "speck_tokenizer_pilot_screen_review",
        "format_version": 1,
        "plan": write("plan.json", plan),
        "nominations": write("nominations.json", nominations),
        "summaries": summaries,
        "accounting": write("accounting.json", accounting),
    }
    write("review.json", review)
    return tmp_path / "review.json"


def test_review_verifies_inputs_and_retains_identities(tmp_path):
    review = write_review(tmp_path)
    result = load_screen_report(review)

    assert result["budget"]["fits"] is True
    assert len(result["inputs"]["artifacts"]) == 10
    assert result["inputs"]["review"]["sha256"] == file_sha256(review)


@pytest.mark.parametrize("name", ["plan", "timing", "custom-large-result", "custom-small-summary"])
def test_review_rejects_changed_evidence_or_outputs(tmp_path, name):
    review = write_review(tmp_path)
    (tmp_path / f"{name}.json").write_text("{}")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_screen_report(review)


def test_v10_analysis_accepts_whole_document_horizon_without_rewriting_plan():
    plan = inputs()[0]
    original = copy.deepcopy(plan)
    assert validate_screen_plan(plan) is plan
    assert plan == original
    plan["stopping"]["fixed_document_mistral_tokens"] = 1_200_000_000
    with pytest.raises(ValueError, match="frozen contract"):
        validate_screen_plan(plan)


def test_cli_records_implementation_and_refuses_to_overwrite(tmp_path):
    review = write_review(tmp_path)
    output = tmp_path / "analysis.json"
    command = [
        sys.executable,
        "-m",
        "scripts.tokenizer_pilot_screen",
        str(review),
        str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["budget"]["projected_total_gpu_hours"] == 20
    assert all(file_sha256(entry["path"]) == entry["sha256"] for entry in result["implementation"])
    original = output.read_bytes()
    rerun = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert rerun.returncode != 0
    assert "already exists" in rerun.stderr
    assert output.read_bytes() == original
