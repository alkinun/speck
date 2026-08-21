import json
from types import SimpleNamespace

from scripts.architecture_search_v3 import parser, status_command, study_dir
from speck.search.study_v3 import V3Study


def test_v3_search_cli_parses_explicit_initialization_inputs():
    args = parser().parse_args(
        [
            "init",
            "experiment",
            "--study",
            "calibration",
            "--config",
            "search.json",
            "--data-dir",
            "packed",
        ]
    )
    assert args.command == "init"
    assert args.config == "search.json"
    profile = parser().parse_args(
        [
            "schedule-profile",
            "calibration",
            "--profile",
            "cpu_short",
            "--estimated-cost",
            "30",
        ]
    )
    assert profile.profile == "cpu_short"


def test_v3_search_cli_reports_normalized_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("speck_base_dir", str(tmp_path))
    path = study_dir("calibration") / "study.sqlite3"
    study = V3Study(path)
    study.initialize({}, {})
    study.add_action("profile", 1.0, 1.0, {})
    study.close()
    output = tmp_path / "status.json"
    status_command(SimpleNamespace(study="calibration", output=output))
    result = json.loads(capsys.readouterr().out)
    assert result["actions"]["pending"] == 1
    assert result["runs"]["pending"] == 0
    assert json.loads(output.read_text()) == result
