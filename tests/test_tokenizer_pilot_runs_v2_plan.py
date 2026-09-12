import hashlib
import json
from pathlib import Path

import pytest

from speck.tokenizer_pilot_runs_v2 import (
    build_corrected_screen_run_manifests,
    load_corrected_materialization_plan,
)

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "research/flagship/tokenizer_pilot_runs_v2/plan.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_checked_v2_plan_binds_correction_code_and_closed_execution():
    plan = json.loads(PLAN.read_text())

    assert plan["format_version"] == 2
    assert plan["status"] == "corrected_screen_materialization_authorized_no_model_outputs"
    assert (
        sha256(PLAN.parent / plan["flop_correction"]["path"]) == plan["flop_correction"]["sha256"]
    )
    for identity in plan["implementation"].values():
        assert sha256(PLAN.parent / identity["path"]) == identity["sha256"]
    assert plan["authority"]["v2_run_materialization"] is True
    assert plan["authority"]["screen_execution"] is False
    assert plan["authority"]["D5_opening"] is False


def test_checked_v2_plan_recomputes_real_records_when_inputs_are_available():
    plan = json.loads(PLAN.read_text())
    missing = [
        plan[key]["path"]
        for key in ("fixed_stream", "continuation", "evaluation_sample")
        if not Path(plan[key]["path"]).is_file()
    ]
    if missing:
        pytest.skip(f"requires maintainer-local tokenizer pilot input: {missing[0]}")

    normalized = load_corrected_materialization_plan(PLAN)
    runs = build_corrected_screen_run_manifests(
        normalized, repository_revision="pre-output-fixture"
    )

    assert len(runs) == 3
    by_id = {run["tokenizer_id"]: run for run in runs}
    assert by_id["speck-bpe-40960-whitespace"]["stops"]["fixed_flop_token_stop"] == 1_125_458_190
    assert [run["stops"]["final_step"] for run in runs] == [18_311, 17_174, 18_311]
    assert all(run["authority"]["screen_execution"] is False for run in runs)
