"""A rental packet must stay executable, because it is discovered on metered hardware."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/qualification"))

from check_throughput_packet import validate  # noqa: E402

PACKETS = (
    ROOT / "experiments/qualification/throughput-h100.json",
    ROOT / "experiments/qualification/throughput-gh200.json",
)


@pytest.mark.parametrize("packet", PACKETS, ids=lambda path: path.stem)
def test_every_recorded_run_parses_and_matches_its_declared_configuration(packet):
    result = validate(packet)

    assert result["status"] == "every_run_parses_and_matches_its_declared_configuration"
    assert result["runs_checked"] == len(json.loads(packet.read_text())["runs"])
    assert result["gpu_hours_spent"] == 0
    assert result["device_touched"] is False


@pytest.mark.parametrize("packet", PACKETS, ids=lambda path: path.stem)
def test_recorded_argv_is_not_allowed_to_go_stale(packet, tmp_path):
    value = json.loads(packet.read_text())
    value["runs"][0]["argv"] = ["experiments/pilot", "--device", "cpu"]
    drifted = tmp_path / packet.name
    drifted.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="recorded argv is stale"):
        validate(drifted)


def test_end_to_end_run_without_a_data_directory_is_rejected(tmp_path):
    value = json.loads((ROOT / "experiments/qualification/throughput-h100.json").read_text())
    value["common"]["data_dir"] = ""
    broken = tmp_path / "throughput-h100.json"
    broken.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="end-to-end"):
        validate(broken)


def test_boolean_profile_flag_is_rejected_because_the_flag_takes_a_path(tmp_path):
    value = json.loads((ROOT / "experiments/qualification/throughput-h100.json").read_text())
    for run in value["runs"]:
        if run["overrides"].get("profile"):
            run["overrides"]["profile"] = True
            run.pop("argv")
    broken = tmp_path / "throughput-h100.json"
    broken.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="profile must be a trace path"):
        validate(broken)


@pytest.mark.parametrize("packet", PACKETS, ids=lambda path: path.stem)
@pytest.mark.parametrize("scope", ["common", "override"])
def test_production_training_output_cannot_be_disabled(packet, scope, tmp_path):
    value = json.loads(packet.read_text())
    config = value["common"] if scope == "common" else value["runs"][0]["overrides"]
    config["training_output"] = False
    for run in value["runs"]:
        run.pop("argv")
    broken = tmp_path / packet.name
    broken.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="training_output must match the production trainer"):
        validate(broken)
