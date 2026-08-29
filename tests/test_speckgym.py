import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from speck.dataloader import packed_loader
from speck.dataset import load_manifest, verify_shards
from speck.speckgym import (
    FAMILIES,
    generate_gym_block,
    generate_shuffle_dyck,
    load_speckgym_config,
    prepare_speckgym,
    symbol_ids,
)


class FakeTokenizer:
    vocab_size = 256
    bos_id = 1
    eos_id = 2

    def fingerprint(self):
        return "speckgym-test-tokenizer"


def tiny_config():
    return {
        "format_version": 1,
        "base_experiment": "unused",
        "batch_tokens": 320,
        "total_requested_tokens": 3_200,
        "checkpoint_tokens": [1_600, 3_200],
        "procedural": {
            "seed": 7,
            "updates": 5,
            "sequence_length": 32,
            "symbol_count": 128,
            "validation_sequences": 2,
            "reserve_sequences": 2,
            "shard_tokens": 64,
            "formal": {
                "citation": "test",
                "reference": "test",
                "k": 64,
                "p_open": 0.5,
                "max_depth": 16,
            },
        },
    }


def split_values(path, source, split):
    pieces = [
        np.fromfile(path / shard["path"], dtype="<u2")
        for shard in source["splits"][split]["shards"]
    ]
    return np.concatenate(pieces)


def test_checked_speckgym_contract_has_expected_token_geometry():
    config = load_speckgym_config()

    assert config["base_experiment"].endswith("experiments/Speck1.5-140M")
    assert config["batch_tokens"] == 65_536
    assert config["procedural"]["updates"] == 500
    assert config["warmup_tokens"] == 32_768_000
    assert config["total_requested_tokens"] == 500_000_000
    assert config["procedural"]["formal"] == {
        "citation": "https://aclanthology.org/2025.acl-long.478/",
        "k": 64,
        "max_depth": 16,
        "p_open": 0.5,
        "reference": "Hu et al. (ACL 2025), Between Circuits and Chomsky",
    }


def test_symbol_ids_are_unique_deterministic_and_avoid_special_tokens():
    first = symbol_ids(32_000, 128, 42)
    second = symbol_ids(32_000, 128, 42)

    assert first == second
    assert len(first) == len(set(first)) == 128
    assert min(first) >= 3 and max(first) < 32_000
    assert symbol_ids(32_000, 128, 43) != first


def test_shuffle_dyck_is_deterministic_and_never_closes_an_absent_type():
    sequence = generate_shuffle_dyck(4, 256, 0.5, 16, random.Random(11))
    assert sequence == generate_shuffle_dyck(4, 256, 0.5, 16, random.Random(11))
    assert sequence[:20] == [3, 7, 3, 7, 1, 5, 1, 2, 0, 5, 4, 6, 0, 1, 5, 1, 5, 0, 2, 0]

    counts = [0] * 4
    maximum_depth = 0
    for token in sequence:
        if token < 4:
            counts[token] += 1
        else:
            counts[token - 4] -= 1
            assert counts[token - 4] >= 0
        maximum_depth = max(maximum_depth, sum(counts))
    assert maximum_depth <= 16


@pytest.mark.parametrize("family", FAMILIES)
def test_speckgym_blocks_are_deterministic_and_use_the_logical_alphabet(family):
    first = generate_gym_block(family, 2_048, 42, "train", 3, 100)
    second = generate_gym_block(family, 2_048, 42, "train", 3, 100)

    assert first == second
    assert len(first) == 2_048
    assert min(first) >= 0 and max(first) < 128
    assert generate_gym_block(family, 2_048, 42, "val", 3, 100) != first


def test_prepare_speckgym_writes_loadable_matched_controls(tmp_path):
    output = tmp_path / "SpeckGym-v0"
    manifests = prepare_speckgym(tiny_config(), FakeTokenizer(), output_dir=output)

    assert set(manifests) == {"B", "C", "D", "E"}
    paths = {
        "B": output / "B-RandomSymbols",
        "C": output / "C-ShuffledGym",
        "D": output / "D-FormalStructure",
        "E": output / "E-SpeckGym",
    }
    for run, path in paths.items():
        manifest = load_manifest(path)
        verify_shards(path, manifest)
        assert manifest["format"] == "speck_procedural_tokens"
        assert manifest["requested_train_tokens"] == 1_600
        assert manifest["tokenizer"]["vocab_size"] == 256
        assert manifest["encoding"] == manifests[run]["encoding"]

    gym = load_manifest(paths["E"])
    shuffled = load_manifest(paths["C"])
    changed = False
    for gym_source, shuffled_source in zip(gym["sources"], shuffled["sources"]):
        assert gym_source["id"] == shuffled_source["id"]
        gym_values = split_values(paths["E"], gym_source, "train")
        shuffled_values = split_values(paths["C"], shuffled_source, "train")
        for offset in range(0, len(gym_values), 32):
            original = gym_values[offset : offset + 32]
            control = shuffled_values[offset : offset + 32]
            assert Counter(original) == Counter(control)
            changed |= not np.array_equal(original, control)
    assert changed

    random_manifest = load_manifest(paths["B"])
    random_values = split_values(paths["B"], random_manifest["sources"][0], "train")
    assert set(random_values).issubset(set(random_manifest["encoding"]["surface_token_ids"]))

    formal = load_manifest(paths["D"])
    assert formal["sources"][0]["generator"]["name"] == "k_shuffle_dyck"
    assert formal["sources"][0]["generator"]["k"] == 64

    inputs, targets, state = next(
        packed_loader(FakeTokenizer(), 1, 32, data_dir=paths["B"], device="cpu")
    )
    assert inputs.shape == targets.shape == (1, 32)
    assert state["global_consumed_tokens"] == 0
    assert np.array_equal(inputs.numpy()[:, 1:], targets.numpy()[:, :-1])


def test_prepare_speckgym_requires_restart_for_an_incomplete_suite(tmp_path):
    output = tmp_path / "SpeckGym-v0"
    staging = output.with_name(output.name + ".building")
    staging.mkdir()
    (staging / "partial.json").write_text(json.dumps({"partial": True}))

    with pytest.raises(FileExistsError, match="incomplete SpeckGym build"):
        prepare_speckgym(tiny_config(), FakeTokenizer(), output_dir=output)

    prepare_speckgym(tiny_config(), FakeTokenizer(), output_dir=output, restart=True)
    assert output.is_dir()
    assert not staging.exists()


def test_procedural_manifest_rejects_changed_encoding_fingerprint(tmp_path):
    output = tmp_path / "SpeckGym-v0"
    prepare_speckgym(tiny_config(), FakeTokenizer(), output_dir=output)
    path = output / "D-FormalStructure" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["encoding"]["surface_token_ids"].reverse()
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="encoding fingerprint"):
        load_manifest(path.parent)


def test_speckgym_prepare_parser_is_import_safe():
    from scripts import speckgym_prepare

    args = speckgym_prepare.parse_args(["custom-gym", "--output-dir", "/tmp/gym", "--restart"])
    assert args.experiment == "custom-gym"
    assert args.output_dir == Path("/tmp/gym")
    assert args.restart
