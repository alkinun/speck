from scripts.base_train import arguments


def test_checkpoint_output_override_is_operational_cli_state(tmp_path):
    path = tmp_path / "checkpoints" / "run"
    parsed = arguments(["experiment", "--output-dir", str(path)])

    assert parsed.output_dir == path
