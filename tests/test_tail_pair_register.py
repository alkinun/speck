import json

import pytest

from scripts.tail_pair_register import register
from speck.checkpoint import directory_identity


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_register_binds_export_and_benchmark_evidence(tmp_path):
    finalization = tmp_path / "finalization"
    finalization.mkdir()
    checkpoint = {
        "directory": "/checkpoints/control",
        "step": 3,
        "model_sha256": "a" * 64,
        "optimizer_sha256": "b" * 64,
        "metadata_sha256": "c" * 64,
    }
    write_json(
        finalization / "finalization.json",
        {
            "format": "speck_tail_pair_finalization",
            "format_version": 1,
            "control": {
                "final_checkpoint": checkpoint,
                "average": {
                    "path": "control-average",
                    "model_sha256": "d" * 64,
                    "metadata_sha256": "e" * 64,
                },
            },
            "constant": {},
        },
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text("{}", encoding="utf-8")
    write_json(
        model / "speck_source.json",
        {
            "format": "speck_export_source",
            "format_version": 1,
            "type": "checkpoint",
            "checkpoint": checkpoint,
        },
    )
    identity = directory_identity(model)
    open_slm = tmp_path / "open-slm.json"
    write_json(
        open_slm,
        {
            "model": {"transformers_export": identity},
            "scores_percent": {"intelligence_index": 20.0},
        },
    )
    banana = tmp_path / "banana.json"
    write_json(
        banana,
        {
            "model": str(model),
            "summary": {"official_complete_run": True, "overall_elo": 1000},
        },
    )

    record = register(finalization, "control-final", model, open_slm, banana)

    assert record["model_export"] == identity
    assert record["open_slm"]["scores_percent"]["intelligence_index"] == 20.0
    assert record["bananamind"]["summary"]["overall_elo"] == 1000
    assert (finalization / "results" / "control-final.json").is_file()


def test_register_rejects_wrong_export_source(tmp_path):
    finalization = tmp_path / "finalization"
    finalization.mkdir()
    write_json(
        finalization / "finalization.json",
        {
            "format": "speck_tail_pair_finalization",
            "format_version": 1,
            "control": {
                "average": {
                    "path": "control-average",
                    "model_sha256": "a" * 64,
                    "metadata_sha256": "b" * 64,
                }
            },
            "constant": {},
        },
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    write_json(
        model / "speck_source.json",
        {
            "format": "speck_export_source",
            "format_version": 1,
            "type": "average",
            "average": {"model_sha256": "wrong", "metadata_sha256": "b" * 64},
        },
    )
    result = tmp_path / "open-slm.json"
    write_json(result, {"scores_percent": {}})

    with pytest.raises(ValueError, match="does not match"):
        register(finalization, "control-average", model, open_slm=result)
