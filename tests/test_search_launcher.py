import json
import os
import shutil
import subprocess
from pathlib import Path


root = Path(__file__).parents[1]
launcher = root / "scripts" / "run_search_v3.sh"


def test_v3_search_launcher_is_portable_and_executable():
    assert os.access(launcher, os.X_OK)
    for shell in ("sh", "dash"):
        if shutil.which(shell):
            subprocess.run([shell, "-n", launcher], check=True)
    result = subprocess.run(
        [launcher, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Run or resume the version three calibration search" in result.stdout
    assert "--no-dashboard" in result.stdout
    if fish := shutil.which("fish"):
        fish_result = subprocess.run(
            [fish, "-c", f'"{launcher}" --help'],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "Run or resume the version three calibration search" in fish_result.stdout


def test_v3_search_launcher_rejects_unknown_options():
    result = subprocess.run(
        [launcher, "--unknown"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unknown option: --unknown" in result.stderr


def test_v3_planner_budget_covers_the_default_launcher_rates():
    config = json.loads(
        (root / "experiments" / "speck00-200m" / "search-v3.json").read_text()
    )
    calibration = config["calibration"]
    crossed_runs = (
        calibration["initialization_seeds"]
        * calibration["data_seeds"]
        * calibration["numerical_repeats"]
    )
    quality_tokens = (
        calibration["broad_architectures"] * calibration["broad_tokens"]
        + calibration["noise_architectures"]
        * (crossed_runs - 1)
        * calibration["noise_tokens"]
        + calibration["anchor_architectures"]
        * (calibration["anchor_tokens"] - calibration["broad_tokens"])
    )
    objective_names = {item["name"] for item in config["objective_sets"]}
    profile_repetitions = calibration["broad_architectures"] * sum(
        item["process_repetitions"]
        for item in config["profiles"]
        if item["name"] in objective_names
    )
    checkpoints = config["quality"]["checkpoint_tokens"]
    noise_checkpoints = sum(
        tokens <= calibration["noise_tokens"] for tokens in checkpoints
    )
    broad_checkpoints = sum(
        tokens <= calibration["broad_tokens"] for tokens in checkpoints
    )
    anchor_checkpoints = sum(
        tokens <= calibration["anchor_tokens"] for tokens in checkpoints
    )
    evaluation_actions = (
        calibration["broad_architectures"] * broad_checkpoints
        + calibration["noise_architectures"] * (crossed_runs - 1) * noise_checkpoints
        + calibration["anchor_architectures"]
        * (anchor_checkpoints - broad_checkpoints)
    )
    minimum_with_reserve = (
        quality_tokens / 10_000
        + evaluation_actions * 1_000_492 / 30_000
        + profile_repetitions * 600
    ) * 1.1
    assert config["planner"]["total_cost"] >= minimum_with_reserve
