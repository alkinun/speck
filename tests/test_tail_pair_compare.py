import json

import pytest

from scripts.tail_pair_compare import compare
from speck.checkpoint import file_sha256

_VARIANTS = ("control-final", "constant-final", "control-average", "constant-average")


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def prepare_results(directory, protocols=None):
    finalization = directory / "finalization.json"
    write_json(finalization, {"format": "speck_tail_pair_finalization", "format_version": 1})
    results = directory / "results"
    results.mkdir()
    protocols = protocols or [{}] * 4
    for index, (variant, protocol) in enumerate(zip(_VARIANTS, protocols)):
        write_json(
            results / f"{variant}.json",
            {
                "format": "speck_tail_pair_result",
                "format_version": 1,
                "variant": variant,
                "finalization": {
                    "path": str(finalization.resolve()),
                    "sha256": file_sha256(finalization),
                },
                "open_slm": {
                    "protocol": protocol,
                    "scores_percent": {"hellaswag": 10.0 + index},
                },
                "bananamind": {
                    "protocol": {"dataset_revision": "same"},
                    "summary": {
                        "accuracy": 0.1 + index * 0.01,
                        "overall_elo": 900 + index * 10,
                        "categories": {"code": {"accuracy": 0.2 + index * 0.02}},
                    },
                },
            },
        )


def test_compare_writes_only_predeclared_score_deltas(tmp_path):
    prepare_results(tmp_path)

    result = compare(tmp_path)

    schedule = result["contrasts"]["constant_final_minus_control_final"]
    averaging = result["contrasts"]["constant_average_minus_constant_final"]
    assert schedule["open_slm"]["hellaswag"] == 1.0
    assert schedule["bananamind"]["overall_elo"] == 10
    assert schedule["bananamind"]["categories/code/accuracy"] == pytest.approx(0.02)
    assert averaging["open_slm"]["hellaswag"] == 2.0
    assert set(result) == {"format", "format_version", "finalization", "results", "contrasts"}
    assert (tmp_path / "comparison.json").is_file()


def test_compare_rejects_protocol_mismatch(tmp_path):
    prepare_results(tmp_path, [{}, {}, {"different": True}, {}])

    with pytest.raises(ValueError, match="protocols do not match"):
        compare(tmp_path)
