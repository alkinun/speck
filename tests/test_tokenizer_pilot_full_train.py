import json
import math

import pytest

from speck.tokenizer_pilot_full_train import run_authorized_training
from speck.tokenizer_pilot_runs import _fingerprint
from tests.test_tokenizer_pilot_train import run_manifest


def categories(bpb=1.0):
    return {
        name: [
            {
                "document_id": f"{name}-document",
                "utf8_bytes": 100,
                "nll_nats": bpb * math.log(2) * 100,
            }
        ]
        for name in ("web", "code", "math", "synthetic", "science", "reference")
    }


def execution(tmp_path):
    run = run_manifest(tmp_path / "data")
    run["settings"]["compile"] = False
    run["model"]["backbone_sha256"] = "backbone"
    run["training_data"]["document_stream_sha256"] = "documents"
    correction = tmp_path / "correction.json"
    correction.write_text(
        json.dumps(
            {
                "reference": {
                    "fixed_document_tokens": 16,
                    "analytic_flop_target": 16 * run["model"]["analytic_training_flops_per_token"],
                }
            }
        )
    )
    run["flop_correction"] = {"path": str(correction), "sha256": "correction"}
    run["run_fingerprint"] = _fingerprint(
        {key: value for key, value in run.items() if key != "run_fingerprint"}
    )
    return {
        "run": run,
        "output_directory": str(tmp_path / "output"),
        "authority": {
            "screen_execution": True,
            "confirmation_execution": False,
            "D5_opening": False,
            "final_selection": False,
            "flagship_training": False,
        },
    }


def fake_evaluator(run, model, tokenizer, device):
    assert tokenizer is None
    return categories()


def test_full_run_stops_and_resumes_from_immutable_boundaries(tmp_path):
    value = execution(tmp_path)

    stopped = run_authorized_training(
        value,
        device="cpu",
        evaluator=fake_evaluator,
        compile_override=False,
        stop_after_step=2,
    )
    assert stopped["status"] == "stopped_at_qualified_optimizer_boundary"
    assert stopped["step"] == 2

    result = run_authorized_training(
        value,
        device="cpu",
        evaluator=fake_evaluator,
        compile_override=False,
    )

    assert result["status"] == "complete"
    assert result["learning_curve"][-1]["macro_bpb"] == pytest.approx(1.0)
    assert result["fixed_document"]["categories"] == categories()
    assert result["fixed_flop"]["categories"] == categories()
    assert (tmp_path / "output/run-result.json").is_file()
    summary = json.loads((tmp_path / "output/run-summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["D5_opening"] is False
    assert (tmp_path / "output/evaluations/step_000001.json").is_file()
    assert (tmp_path / "output/evaluations/step_000002.json").is_file()
    assert (tmp_path / "output/evaluations/step_000004.json").is_file()


def test_full_run_requires_exact_screen_authority(tmp_path):
    value = execution(tmp_path)
    value["authority"]["screen_execution"] = False

    with pytest.raises(ValueError, match="exact screen authority"):
        run_authorized_training(
            value,
            device="cpu",
            evaluator=fake_evaluator,
            compile_override=False,
        )


def test_resume_rejects_changed_evaluation_boundary(tmp_path):
    value = execution(tmp_path)
    run_authorized_training(
        value,
        device="cpu",
        evaluator=fake_evaluator,
        compile_override=False,
        stop_after_step=2,
    )
    path = tmp_path / "output/evaluations/step_000001.json"
    path.write_text("{}\n")

    with pytest.raises(ValueError, match="evaluation boundary identity mismatch"):
        run_authorized_training(
            value,
            device="cpu",
            evaluator=fake_evaluator,
            compile_override=False,
        )
