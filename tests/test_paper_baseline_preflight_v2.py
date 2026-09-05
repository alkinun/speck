from copy import deepcopy
from pathlib import Path

from scripts.paper_baseline_preflight_v2 import (
    arguments,
    load_prerequisites,
    qualify_arm,
)

root = Path(__file__).parents[1]


def test_preflight_v2_prerequisites_preserve_failure_and_require_v3():
    result = load_prerequisites(root)

    assert result["failed_v1_preflight"]["status"] == "failed"
    assert result["cache_equivalence_v3"]["status"] == "qualified"
    assert result["kda_kernel"]["passed"] is True


def test_preflight_v2_changes_only_full_model_behavioral_authority():
    arm = {
        "native_incremental": {"passed": False},
        "compiled_training_step": {"within_peak_envelope": True},
        "transformers_export": {"passed": True},
        "passed": False,
    }
    cache = {"path": "v3.json", "sha256": "a" * 64, "status": "qualified"}

    result = qualify_arm(deepcopy(arm), cache)

    assert result["passed"] is True
    assert result["native_incremental"]["v1_elementwise_passed"] is False
    assert "without v2 pass/fail" in result["native_incremental"]["authority"]
    assert result["behavioral_cache_equivalence"]["passed"] is True


def test_preflight_v2_arguments_default_to_cuda():
    args = arguments(["matrix.json", "--output", "result.json"])

    assert args.matrix == Path("matrix.json")
    assert args.output == Path("result.json")
    assert args.device == "cuda"
