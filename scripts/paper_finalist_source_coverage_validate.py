"""Require complete finite source-loss coverage in collected finalist results."""

import argparse
import json
import math
from pathlib import Path

EXPECTED_SOURCES = (
    "finemath_4plus",
    "math_textbook_exercise",
    "math_multi_style",
    "cosmopedia_v2",
    "ufw_l3_multi_style",
    "pes2o",
    "wikimedia",
    "dclm_edu",
    "fineweb_edu",
    "ultra_fineweb",
    "dclm",
)
EXPECTED_STEPS = (0, 5874, 11748, 17622, 23496)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    return parser.parse_args(argv)


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_result(report):
    if (
        report.get("format") != "speck_paper_finalist_run_result"
        or report.get("format_version") != 2
        or report.get("status") != "complete_qualified"
        or report.get("training_tokens") != 1_539_833_856
        or report.get("non_finite_steps") != 0
    ):
        raise ValueError("finalist source coverage requires a qualified v2 run result")
    history = report.get("validation_history")
    if (
        not isinstance(history, list)
        or tuple(entry.get("step") for entry in history) != EXPECTED_STEPS
    ):
        raise ValueError("finalist source coverage has the wrong validation cadence")
    expected = set(EXPECTED_SOURCES)
    for index, entry in enumerate(history):
        source_losses = entry.get("validation_source_losses")
        if not isinstance(source_losses, dict) or set(source_losses) != expected:
            raise ValueError(
                f"finalist validation step {entry.get('step')} does not contain every expected source"
            )
        if any(not _finite_number(value) for value in source_losses.values()):
            raise ValueError(
                f"finalist validation step {entry.get('step')} contains a non-finite source loss"
            )
        expected_tokens = 19_988_480 if index == len(history) - 1 else 4_997_120
        if entry.get("validation_tokens") != expected_tokens:
            raise ValueError("finalist source coverage has the wrong validation token budget")
    if report.get("final_validation") != history[-1]:
        raise ValueError("finalist final source coverage does not match the validation history")
    return {
        "status": "complete_finite_source_coverage",
        "run": report.get("run"),
        "sources": len(EXPECTED_SOURCES),
        "validation_points": len(EXPECTED_STEPS),
        "intermediate_minimum_batches_per_source": 27,
        "final_minimum_batches_per_source": 110,
    }


def validate_file(path):
    return validate_result(json.loads(Path(path).read_text(encoding="utf-8")))


def main(argv=None):
    reports = [validate_file(path) for path in arguments(argv).results]
    print(
        "Finalist source coverage: "
        f"{len(reports)} result(s), {reports[0]['sources']} sources, complete and finite"
    )


if __name__ == "__main__":
    main()
