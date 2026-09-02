import json
from pathlib import Path

import pytest

from scripts.noise_floor_prepare import arguments, prepare, repeat_seeds
from speck.config import load_experiment

repository = Path(__file__).parents[1]
source = repository / "experiments" / "SpeckLC-150M-MixerScreen-131M" / "gdn-global"


def test_noise_floor_prepare_varies_only_training_seed_and_run_identity(tmp_path):
    output = tmp_path / "SpeckLC-150M-NoiseFloor-131M"
    contract = prepare(source, output, (43, 44))

    assert contract["baseline"]["seed"] == 42
    assert contract["repeat_seeds"] == [43, 44]
    baseline = load_experiment(source, "data", "long_context", "model", "tokenizer", "train")
    for seed in (43, 44):
        directory = output / f"seed-{seed}"
        repeat = load_experiment(
            directory,
            "data",
            "long_context",
            "model",
            "tokenizer",
            "train",
        )
        for name in ("data", "long_context", "model", "tokenizer"):
            assert repeat[name] == baseline[name]
        changed = {
            key
            for key in set(baseline["train"]) | set(repeat["train"])
            if baseline["train"].get(key) != repeat["train"].get(key)
        }
        assert changed == {"run", "seed", "wandb_group"}
        assert repeat["train"]["seed"] == seed
        assert repeat["train"]["device_batch_size"] == 4
    assert json.loads((output / "noise_floor.json").read_text()) == contract


@pytest.mark.parametrize("seeds", ((43,), (43, 43), (42, 43)))
def test_noise_floor_requires_two_unique_new_seeds(seeds):
    with pytest.raises(ValueError, match="seed"):
        repeat_seeds(42, seeds)


def test_noise_floor_arguments_require_explicit_seeds():
    args = arguments(["source", "output", "--seeds", "7", "9"])
    assert args.seeds == [7, 9]
