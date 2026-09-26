"""The throughput packet must stay executable, because it is discovered on metered hardware."""

import json
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/qualification"))

from check_throughput_packet import main, validate  # noqa: E402

PACKETS = (
    ROOT / "experiments/qualification/throughput-gh200.json",
    ROOT / "experiments/qualification/throughput-h100.json",
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
    value = json.loads(PACKETS[0].read_text())
    value["common"]["data_dir"] = ""
    broken = tmp_path / "throughput-gh200.json"
    broken.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="end-to-end"):
        validate(broken)


def test_boolean_profile_flag_is_rejected_because_the_flag_takes_a_path(tmp_path):
    value = json.loads(PACKETS[0].read_text())
    for run in value["runs"]:
        if run["overrides"].get("profile"):
            run["overrides"]["profile"] = True
            run.pop("argv")
    broken = tmp_path / "throughput-gh200.json"
    broken.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="profile must be a trace path"):
        validate(broken)


def test_print_commands_validates_before_emitting_commands(tmp_path, monkeypatch, capsys):
    value = json.loads(PACKETS[0].read_text())
    value["runs"][0]["argv"] = ["stale"]
    broken = tmp_path / "throughput-gh200.json"
    broken.write_text(json.dumps(value))
    monkeypatch.setattr(sys, "argv", ["check_throughput_packet", str(broken), "--print-commands"])

    with pytest.raises(ValueError, match="recorded argv is stale"):
        main()

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("packet", PACKETS, ids=lambda path: path.stem)
def test_printed_commands_use_locked_environment_and_preserve_argv(packet, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_throughput_packet", str(packet), "--print-commands"])
    main()

    runs = json.loads(packet.read_text())["runs"]
    lines = capsys.readouterr().out.splitlines()
    for run, line in zip(runs, lines, strict=False):
        command = shlex.split(line, comments=True)
        assert command[:6] == ["uv", "run", "--no-sync", "python", "-m", "scripts.benchmark"]
        assert command[6:] == run["argv"]
