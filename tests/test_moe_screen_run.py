import getpass
import hashlib
import json
import tempfile

import pytest

from scripts.moe_screen_run import command_for_arm, load_and_verify_screen, runtime_environment


def test_command_resumes_the_exact_arm_checkpoint(tmp_path):
    command = command_for_arm(tmp_path / "screen", tmp_path / "output", "g8", resume=763)

    assert command[-2:] == ["--resume", "763"]
    assert command[3].endswith("screen/g8")
    assert command[5].endswith("output/SpeckLC-150M-MoEScreen-g8")


def test_runner_rejects_changed_materialized_artifact(tmp_path):
    screen = tmp_path / "screen"
    arm = screen / "dense"
    arm.mkdir(parents=True)
    artifact = arm / "model.json"
    artifact.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    contract = {
        "format": "speck_moe_design_screen",
        "format_version": 2,
        "launch_order": ["dense"],
        "arms": {"dense": {"artifacts": {"model.json": digest}}},
    }
    (screen / "screen.json").write_text(json.dumps(contract), encoding="utf-8")

    assert load_and_verify_screen(screen)[1] == contract
    artifact.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        load_and_verify_screen(screen)


def test_runtime_environment_separates_noexec_volume_data_from_compiled_libraries(tmp_path):
    expected = f"{tempfile.gettempdir()}/torchinductor_{getpass.getuser()}"
    environment = runtime_environment(tmp_path)

    assert environment["WANDB_DIR"] == str(tmp_path / "wandb")
    assert environment["TORCHINDUCTOR_CACHE_DIR"] == expected
    assert not environment["TORCHINDUCTOR_CACHE_DIR"].startswith(str(tmp_path))
